import argparse
import time

import numpy as np
import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torchvision import datasets, transforms

from compute_flops import count_model_param_flops
from prune_vgg_function import prune_vgg
from vgg import slimmingvgg as vgg11


def __search_threshold(weight, alg: str):
    if alg not in ["fixed", "grad", "search"]:
        raise NotImplementedError()

    hist_y, hist_x = np.histogram(weight.data.cpu().numpy(), bins=100, range=(0, 1))
    if alg == "search":
        for i in range(len(hist_x) - 1):
            if hist_y[i] == hist_y[i + 1]:
                return hist_x[i]
    elif alg == "grad":
        hist_y_diff = np.diff(hist_y)
        for i in range(len(hist_y_diff) - 1):
            if hist_y_diff[i] <= 0 <= hist_y_diff[i + 1]:
                return hist_x[i + 1]
    elif alg == "fixed":
        return hist_x[1]
    return 0


# Prune settings
parser = argparse.ArgumentParser(description='Pruning networks')
parser.add_argument('--data', type=str, default='',
                    help='Path to imagenet validation data')
parser.add_argument('--test-batch-size', type=int, default=16, metavar='N',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--percent', type=float, default=0.5,
                    help='scale sparse rate (default: 0.5)')
parser.add_argument('--model', required=True, type=str, metavar='PATH',
                    help='path to raw trained model (default: none)')
parser.add_argument('--save', default='.', type=str, metavar='PATH',
                    help='path to save prune model (default: none)')
parser.add_argument('-j', '--workers', default=20, type=int, metavar='N',
                    help='number of data loading workers (default: 20)')
parser.add_argument("--pruning-strategy", type=str,
                    choices=["percent", "fixed", "grad", "search"],
                    help="Pruning strategy", required=True)
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

print(args)
output_name = "pruned" if args.pruning_strategy != "percent" else "pruned_{}".format(args.percent)

save_path = os.path.dirname(args.model)

model = vgg11()
model.features = nn.DataParallel(model.features)
# model = torch.nn.DataParallel(model)
cudnn.benchmark = True

if args.cuda:
    model.cuda()

# load trained model
if args.model:
    if os.path.isfile(args.model):
        print("=> loading checkpoint '{}'".format(args.model))
        if args.cuda:
            checkpoint = torch.load(args.model)
        else:
            checkpoint = torch.load(args.model, map_location=lambda storage, location: storage)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
              .format(args.model, checkpoint['epoch'], best_prec1))
    else:
        raise ValueError("=> no checkpoint found at '{}'".format(args.model))
else:
    raise ValueError("model path missing!")

print(model)

if_dataparallel = hasattr(model.features, "module")
new_model, cfg = prune_vgg(model, args.pruning_strategy, args.cuda, if_dataparallel)

if if_dataparallel:
    new_model.features = new_model.features.module

print("Saving pruned model...")
torch.save({'cfg': cfg,
            'epoch': args.start_epoch,
            'best_prec1': checkpoint['best_prec1'],
            'optimizer': checkpoint['optimizer'],
            'state_dict': new_model.state_dict()},
           os.path.join(args.save, '{}.pth.tar'.format(output_name)))

print('Pruned model saved!')


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
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


# simple test model after Pre-processing prune (simple set BN scales to zeros)
def test():
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(os.path.join(args.data, 'val'), transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.test_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    model.eval()
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    criterion = nn.CrossEntropyLoss().cuda()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
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

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
    return top1.avg, top5.avg


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


print("Starting evaluating...")
# acc = test()
print("Skip evaluation. Aborted.")

print("Computing FLOPs...")
print("cfg: ", cfg)

# calculate FLOPs
flops = count_model_param_flops(new_model.cuda(), 224)
flops_unpruned = count_model_param_flops(model.cuda(), 224)
print("FLOPs after pruning: {}".format(flops))
print("FLOPs Unpruned: {}".format(flops_unpruned))
