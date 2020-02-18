import argparse
import numpy as np
import os
import shutil
import time

import torch
import torch.nn
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tensorboardX import SummaryWriter

from models import ResNetExpand
from models.resnet import resnet50
from models.resnet_expand import Bottleneck
from vgg import slimmingvgg as vgg11

model_names = ['vgg11', 'resnet50']

parser = argparse.ArgumentParser(description='PyTorch ImageNet Fine-tuning')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=25, type=int, metavar='N',
                    help='number of data loading workers (default: 25)')
parser.add_argument('--epochs', default=5, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', type=float, nargs='*', default=[1e-3], metavar='LR',
                    help="the learning rate in each stage (default 1e-2, 1e-3)")
parser.add_argument('--decay-epoch', type=float, nargs='*', default=[],
                    help="the epoch to decay the learning rate (default 0.5, 0.75)")
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--save', default='.', type=str, metavar='PATH',
                    help='path to save model (default: current directory)')
parser.add_argument('--backup-path', default='.', type=str, metavar='PATH',
                    help='path to save model (default: current directory)')
parser.add_argument('--refine', default='', type=str, metavar='PATH',
                    help='path to the pruned model')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument("--debug", action="store_true",
                    help="enable debug mode")
parser.add_argument("--expand", action="store_true",
                    help="use expanded addition in shortcut connections")
parser.add_argument('--seed', type=int, metavar='S', default=666,
                    help='random seed (default: 666)')

best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()

    if len(args.lr) != len(args.decay_epoch) + 1:
        print("args.lr: {}".format(args.lr))
        print("args.decay-epoch: {}".format(args.decay_epoch))
        raise ValueError("inconsistent between lr-decay-gamma and decay-epoch")

    print(args)

    args.distributed = args.world_size > 1

    # reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.rank == 0:
        if not os.path.exists(args.save):
            os.makedirs(args.save)
        if not os.path.exists(args.backup_path):
            os.makedirs(args.backup_path)

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    if args.refine:
        checkpoint = torch.load(args.refine)
        print("loaded checkpoint. cfg:")
        print(checkpoint['cfg'])
        if args.arch == "vgg11":
            model = vgg11(config=checkpoint['cfg'])
        elif args.arch == "resnet50":
            if args.expand:
                if "downsample_cfg" in checkpoint:
                    downsample_cfg = checkpoint["downsample_cfg"]
                else:
                    downsample_cfg = None
                model = ResNetExpand(cfg=checkpoint['cfg'], aux_fc=False, downsample_cfg=downsample_cfg)
            else:
                raise NotImplementedError("Use --expand option.")

        else:
            raise NotImplementedError("{} is not supported".format(args.arch))

        model.load_state_dict(checkpoint['state_dict'])

        # there is no parameters in ChannelMask layers
        # we need to load it manually
        if args.expand:
            bn3_masks = checkpoint["bn3_masks"]
            bottleneck_modules = list(filter(lambda m: isinstance(m[1], Bottleneck), model.named_modules()))
            assert len(bn3_masks) == len(bottleneck_modules)
            for i, (name, m) in enumerate(bottleneck_modules):
                if isinstance(m, Bottleneck):
                    mask = bn3_masks[i]
                    assert mask[1].shape[0] == m.expand_layer.idx.shape[0]
                    m.expand_layer.idx = np.argwhere(mask[1].clone().cpu().numpy()).squeeze()

        if 'downsample_cfg' in checkpoint:
            # set downsample expand layer
            downsample_modules = list(
                filter(lambda m: isinstance(m[1], nn.Sequential) and 'downsample' in m[0], model.named_modules()))
            downsample_mask = checkpoint['downsample_mask']
            assert len(downsample_modules) == len(downsample_mask)
            for i, (name, m) in enumerate(downsample_modules):
                mask = downsample_mask[f"{name}.1"]
                assert mask.shape[0] == m[-1].idx.shape[0]
                m[-1].idx = np.argwhere(mask.clone().cpu().numpy()).squeeze()

        if not args.distributed:
            if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
                model.features = torch.nn.DataParallel(model.features)
                model.cuda()
            else:
                model = torch.nn.DataParallel(model).cuda()
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr[0],
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1'].item()

            for param_name, param in model.named_parameters():
                if param_name not in checkpoint['state_dict']:
                    checkpoint['state_dict'][param_name] = param.data
                    raise ValueError("Missing parameter {}, do not load!".format(param_name))

            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])

            # move optimizer buffer to GPU
            for p in optimizer.state.keys():
                param_state = optimizer.state[p]
                buf = param_state["momentum_buffer"]
                param_state["momentum_buffer"] = buf.cuda()  # move buf to device

            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            raise ValueError("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, epoch=0, writer=None)
        return

    # only master process write to disk
    writer = SummaryWriter(logdir=args.save, write_to_disk=args.rank == 0)

    if args.arch == "resnet50":
        summary = pruning_summary_resnet50(model, args.expand)
    else:
        print("WARNING: arch {} do not support pretty print".format(args.arch))
        summary = str(model)

    print(model)

    print("********** MODEL SUMMARY **********")
    print(summary)
    print("********** ************* **********")

    writer.add_text("model/summary", summary)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch)

        weights = bn_weights(model)
        for bn_name, bn_weight in weights:
            writer.add_histogram("bn/" + bn_name, bn_weight, global_step=epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, writer)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, epoch, writer)

        writer.add_scalar("train/lr", optimizer.param_groups[0]['lr'], epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best, args.save,
            save_backup=(epoch - args.start_epoch) % 5 == 0,
            backup_path=args.backup_path,
            epoch=epoch)

    writer.close()
    print("Best prec@1: {}".format(best_prec1))


def pruning_summary_resnet50(model, expand=False):
    model_ref = resnet50() if not expand else ResNetExpand()
    if hasattr(model, "module"):
        # remove parallel wrapper
        model = model.module

    pruning_layers = []
    for (name, m), (name_ref, m_ref) in zip(model.named_modules(), model_ref.named_modules()):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            assert len(m_ref.weight.shape) == 1, "NAME: {}, NAME_REF: {}, len: {}".format(name, name_ref,
                                                                                          len(m_ref.weight.shape))
            assert len(m.weight.shape) == 1, "NAME: {}, NAME_REF: {}, len: {}".format(name, name_ref,
                                                                                      len(m.weight.shape))
            pruning_layers.append("{}: original shape: {}, pruned shape: {}"
                                  .format(name, m_ref.weight.shape[0], m.weight.shape[0]))
    return "\n".join(pruning_layers)


def bn_weights(model):
    weights = []
    for name, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            weights.append((name, m.weight.data))

    return weights
    pass


def BN_grad_zero(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            mask = (m.weight.data != 0)
            mask = mask.float().cuda()
            m.weight.grad.data.mul_(mask)
            m.bias.grad.data.mul_(mask)


def clamp_bn(model):
    bn_modules = list(filter(lambda m: isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d), model.modules()))
    for m in bn_modules:
        m.weight.data.clamp_(0, 1)


def train(train_loader, model, criterion, optimizer, epoch, writer=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    # loss_aux_recorder = AverageMeter()
    # avg_sparsity_loss = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(non_blocking=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        if isinstance(output, tuple):
            output, out_aux = output
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'lr {3}\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), optimizer.param_groups[0]['lr'], batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
        if writer:
            writer.add_scalar("train/cross_entropy", losses.avg, epoch)
            writer.add_scalar("train/top1", top1.avg.item(), epoch)
            writer.add_scalar("train/top5", top5.avg.item(), epoch)


def validate(val_loader, model, criterion, epoch, writer=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(non_blocking=True)
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            # compute output
            output = model(input_var)
            if isinstance(output, tuple):
                output, out_aux = output
            loss = criterion(output, target_var)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.data.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))
            if args.debug and i >= 5:
                break

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    if writer is not None:
        writer.add_scalar("val/cross_entropy", losses.avg, epoch)
        writer.add_scalar("val/top1", top1.avg.item(), epoch)
    return top1.avg


def save_checkpoint(state, is_best, filepath, save_backup, backup_path, epoch, name='checkpoint.pth.tar'):
    if args.rank == 0:
        torch.save(state, os.path.join(filepath, name))
        if is_best:
            shutil.copyfile(os.path.join(filepath, name), os.path.join(filepath, 'model_best.pth.tar'))
        if save_backup:
            shutil.copyfile(os.path.join(filepath, name),
                            os.path.join(backup_path, 'checkpoint_{}.pth.tar'.format(epoch)))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    if epoch in args.decay_epoch:
        lr_idx = args.decay_epoch.index(epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr[lr_idx + 1]


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
