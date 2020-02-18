from __future__ import print_function

import argparse
import sys

import os
import random
import re
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torchvision import datasets, transforms

import models
from common import LossType
from compute_flops import count_model_param_flops
from models.resnet_expand import BasicBlock

# Training settings
parser = argparse.ArgumentParser(description='PyTorch CIFAR fine-tuning')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar100)')
parser.add_argument('--sparsity-regularization', '-sr', dest='sr', action='store_true',
                    help='train with channel sparsity regularization')
parser.add_argument('--s', type=float, default=0.0001,
                    help='scale sparse rate (default: 0.0001)')
parser.add_argument('--refine', type=str, metavar='PATH', required=True,
                    help='path to the pruned model to be fine tuned')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--epochs', type=int, default=40, metavar='N',
                    help='number of epochs to train (default: 160)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--gammas', type=float, nargs='+', default=[0.1, 0.1],
                    help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
parser.add_argument('--decay-epoch', type=float, nargs='*', default=[0.5, 0.75],
                    help="the epoch to decay the learning rate (default 0.5, 0.75)")
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, metavar='S', default=None,
                    help='random seed (default: a random int)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save', default='./logs', type=str, metavar='PATH',
                    help='path to save prune model (default: current directory)')
parser.add_argument('--log', default='./log', type=str, metavar='PATH',
                    help='path to tensorboard log (default: ./log)')
parser.add_argument('--arch', default='vgg', type=str,
                    help='architecture to use')
parser.add_argument("--expand", action="store_true",
                    help="use expanded addition in shortcut connections")

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.decay_epoch = sorted([int(args.epochs * i if i < 1 else i) for i in args.decay_epoch])
if not args.seed:
    args.seed = random.randint(500, 1000)
print(args)

# reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if not os.path.exists(args.save):
    os.makedirs(args.save)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
if args.dataset == 'cifar10':
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data.cifar10', train=True, download=True,
                         transform=transforms.Compose([
                             transforms.Pad(4),
                             transforms.RandomCrop(32),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                         ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
    num_classes = 10
elif args.dataset == "cifar100":
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('./data.cifar100', train=True, download=True,
                          transform=transforms.Compose([
                              transforms.Pad(4),
                              transforms.RandomCrop(32),
                              transforms.RandomHorizontalFlip(),
                              transforms.ToTensor(),
                              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                          ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('./data.cifar100', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
    num_classes = 100
else:
    raise NotImplementedError("do not support dataset {}".format(args.dataset))

if args.refine:
    checkpoint = torch.load(args.refine)
    if args.expand:
        if args.arch == "resnet56":
            model = models.resnet_expand.resnet56(cfg=checkpoint['cfg'], num_classes=num_classes, aux_fc=False)
        else:
            raise NotImplementedError()
    else:
        if re.match("vgg.+", args.arch):
            model = models.__dict__[args.arch](num_classes=num_classes, cfg=checkpoint['cfg'])
        else:
            raise NotImplementedError()
    model.load_state_dict(checkpoint['state_dict'])

    # there is no parameters in ChannelMask layers
    # we need to load it manually
    if args.expand:
        bn3_masks = checkpoint["bn3_masks"]
        bottleneck_modules = list(filter(lambda m: isinstance(m[1], BasicBlock), model.named_modules()))
        assert len(bn3_masks) == len(bottleneck_modules)
        for i, (name, m) in enumerate(bottleneck_modules):
            if isinstance(m, BasicBlock):
                if isinstance(m.expand_layer, nn.Identity):
                    continue
                mask = bn3_masks[i]
                assert mask[1].shape[0] == m.expand_layer.idx.shape[0]
                m.expand_layer.idx = np.argwhere(mask[1].clone().cpu().numpy()).squeeze().reshape(-1)
else:
    raise ValueError("--refine is required to fine-tune.")

if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
              .format(args.resume, checkpoint['epoch'], best_prec1))
    else:
        raise ValueError("=> no checkpoint found at '{}'".format(args.resume))


# additional subgradient descent on the sparsity-induced penalty term
def updateBN():
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.grad.data.add_(args.s * torch.sign(m.weight.data))  # L1


def train(epoch):
    model.train()
    global history_score
    avg_loss = 0.
    train_acc = 0.
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        if isinstance(output, tuple):
            output, output_aux = output
        loss = F.cross_entropy(output, target)

        avg_loss += loss.data.item()
        pred = output.data.max(1, keepdim=True)[1]
        train_acc += pred.eq(target.data.view_as(pred)).cpu().sum()
        loss.backward()
        if args.sr:
            updateBN()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data.item()))
    history_score[epoch][0] = avg_loss / len(train_loader)
    history_score[epoch][1] = train_acc / float(len(train_loader))


def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        if isinstance(output, tuple):
            output, output_aux = output
        test_loss += F.cross_entropy(output, target, size_average=False).data.item()  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return float(correct) / float(len(test_loader.dataset))


def save_checkpoint(state, is_best, filepath):
    state['args'] = args

    torch.save(state, os.path.join(filepath, 'checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'), os.path.join(filepath, 'model_best.pth.tar'))


def adjust_learning_rate(optimizer, epoch, gammas, schedule):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr
    assert len(gammas) == len(schedule), "length of gammas and schedule should be equal"
    for (gamma, step) in zip(gammas, schedule):
        if epoch >= step:
            lr = lr * gamma
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


# test loaded model
print("Testing the loaded model...")
test()

if args.evaluate:
    sys.exit(0)


def calculate_flops(current_model):
    if args.expand:
        if args.arch == "resnet56":
            model_ref = models.resnet_expand.resnet56(num_classes=num_classes)
        else:
            raise NotImplementedError()
    else:
        if re.match("vgg.+", args.arch):
            model_ref = models.__dict__[args.arch](num_classes=num_classes)
        else:
            raise NotImplementedError()
    current_flops = count_model_param_flops(current_model.cpu(), 32)
    ref_flops = count_model_param_flops(model_ref.cpu(), 32)
    flops_ratio = current_flops / ref_flops

    print("FLOPs remains {}".format(flops_ratio))


calculate_flops(model)

writer = SummaryWriter(logdir=args.log)
history_score = np.zeros((args.epochs - args.start_epoch + 1, 3))

best_prec1 = 0.
for epoch in range(args.start_epoch, args.epochs):
    current_learning_rate = adjust_learning_rate(optimizer, epoch, args.gammas, args.decay_epoch)
    train(epoch)
    prec1 = test()
    history_score[epoch][2] = prec1
    np.savetxt(os.path.join(args.save, 'record.txt'), history_score, fmt='%10.5f', delimiter=',')
    is_best = prec1 > best_prec1
    best_prec1 = max(prec1, best_prec1)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_prec1': best_prec1,
        'optimizer': optimizer.state_dict(),
    }, is_best, filepath=args.save)

    # write the tensorboard
    writer.add_scalar("train/average_loss", history_score[epoch][0], epoch)
    writer.add_scalar("train/train_acc", history_score[epoch][1], epoch)
    writer.add_scalar("train/lr", optimizer.param_groups[0]['lr'], epoch)
    writer.add_scalar("val/acc", prec1, epoch)
    writer.add_scalar("val/best_acc", best_prec1, epoch)

print("Best accuracy: " + str(best_prec1))
history_score[-1][0] = best_prec1
np.savetxt(os.path.join(args.save, 'record.txt'), history_score, fmt='%10.5f', delimiter=',')
