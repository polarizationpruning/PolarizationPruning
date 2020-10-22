import argparse
import os
import shutil
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import models
import models.common
import utils
from models import ResNetExpand
from models.mobilenet import mobilenet_v2
from models import resnet50
from utils import common
from utils.common import adjust_learning_rate
from utils.evaluation import AverageMeter, accuracy
from vgg import slimmingvgg as vgg11

model_names = ['vgg11', 'resnet50', 'mobilenetv2']

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
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument("--debug", action="store_true",
                    help="enable debug mode")
parser.add_argument("--expand", action="store_true",
                    help="use expanded addition in shortcut connections")
parser.add_argument('--seed', type=int, metavar='S', default=666,
                    help='random seed (default: 666)')
parser.add_argument('--width-multiplier', default=1.0, type=float,
                    help="The width multiplier (only) for MobileNet v2. "
                         "Unavailable for other networks. (default 1.0)")
parser.add_argument('--no-bn-wd', action='store_true',
                    help='Do not apply weight decay on BatchNorm layers')
parser.add_argument('--scratch', action='store_true',
                    help='Train from scratch (do not load weight parameters). '
                         'Only available for MobileNet v2.')
parser.add_argument('--lr-strategy', choices=['cos', 'step'],
                    help='Learning rate decay strategy. \n'
                         '- cos: Cosine learning rate decay. In this case, '
                         '--lr should be only one value, and --decay-epoch will be ignored.\n'
                         '- step: Decay as --lr and --decay-step.')
parser.add_argument('--lighting', action='store_true',
                    help='[DEPRECATED] Use lighting in data augmentation.')
parser.add_argument('--warmup', action='store_true',
                    help='Use learning rate warmup in first five epochs. '
                         'Only available when --scratch is enabled.')

best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()

    if (args.arch != "mobilenetv2" or args.lr_strategy == 'step') and len(args.lr) != len(args.decay_epoch) + 1:
        # MobileNet v2 uses cosine learning rate schedule
        print("args.lr: {}".format(args.lr))
        print("args.decay-epoch: {}".format(args.decay_epoch))
        raise ValueError("inconsistent between lr-decay-gamma and decay-epoch")

    if args.width_multiplier != 1.0 and args.arch != "mobilenetv2":
        if args.arch == "resnet50":
            print("For ResNet-50 with --width-multiplier, no need to specific --width-multiplier in finetuning.")
        raise ValueError("--width-multiplier only works for MobileNet v2. \n"
                         f"got --width-multiplier {args.width_multiplier} for --arch {args.arch}")

    if args.arch == "mobilenetv2" and not args.lr_strategy == 'step':
        assert len(args.lr) == 1, "For MobileNet v2, learning rate only needs one value for" \
                                  "cosine learning rate schedule."
        print("WARNING: --decay-step is disabled.")

    if args.warmup and not args.scratch:
        raise ValueError("Finetuning should not use --warmup.")

    print(args)
    print(f"Current git hash: {common.get_git_id()}")

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

                if args.expand:
                    # there is no parameters in ChannelMask layers
                    # we need to restore it manually
                    if 'expand_idx' not in checkpoint:
                        # compatible to resprune-expand
                        expand_idx = []
                        bn3_masks = checkpoint["bn3_masks"]
                        # bottleneck_modules = list(filter(lambda m: isinstance(m[1], Bottleneck), model.named_modules()))
                        # assert len(bn3_masks) == len(bottleneck_modules)
                        for mask in bn3_masks:
                            idx = np.argwhere(mask[1].clone().cpu().numpy()).squeeze()
                            expand_idx.append(idx)
                    else:
                        # resprune-expand-gate pruning save expand_idx in checkpoint
                        expand_idx = checkpoint['expand_idx']
                else:
                    expand_idx = None

                model = ResNetExpand(cfg=checkpoint['cfg'], expand_idx=expand_idx,
                                     aux_fc=False, downsample_cfg=downsample_cfg, gate=False)
            else:
                raise NotImplementedError("Use --expand option.")
        elif args.arch == "mobilenetv2":
            if 'gate' in checkpoint and checkpoint['gate'] is True:
                input_mask = True
            else:
                input_mask = False
            model = mobilenet_v2(inverted_residual_setting=checkpoint['cfg'],
                                 width_mult=args.width_multiplier, input_mask=input_mask)
        else:
            raise NotImplementedError("{} is not supported".format(args.arch))

        if not args.scratch:
            # do not load weight parameters when retrain from scratch
            model.load_state_dict(checkpoint['state_dict'])

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

        if args.arch == "mobilenetv2":
            # restore the mask of the Expand layer
            expand_idx = checkpoint['expand_idx']
            if expand_idx is not None:
                for m_name, sub_module in model.named_modules():
                    if isinstance(sub_module, models.common.ChannelOperation):
                        sub_module.idx = expand_idx[m_name]
            else:
                print("Warning: expand_idx not set in checkpoint. Use default settings.")

        # the mask changes the content of tensors
        # weights less than threshold will be set as zero
        # the mask operation must be done before data parallel
        weight_masks = None

        if not args.distributed:
            if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
                model.features = torch.nn.DataParallel(model.features)
                model.cuda()
            else:
                model = torch.nn.DataParallel(model).cuda()
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        raise ValueError("--refine must be specified in finetuning!")

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    if args.no_bn_wd:
        no_wd_params = []
        for module_name, sub_module in model.named_modules():
            if isinstance(sub_module, nn.BatchNorm1d) or isinstance(sub_module, nn.BatchNorm2d):
                for param_name, param in sub_module.named_parameters():
                    no_wd_params.append(param)
                    print(f"No weight decay param: module {module_name} param {param_name}")
        no_wd_params_set = set(no_wd_params)
        wd_params = []
        for param_name, model_p in model.named_parameters():
            if model_p not in no_wd_params_set:
                wd_params.append(model_p)
                print(f"Weight decay param: parameter name {param_name}")

        optimizer = torch.optim.SGD([{'params': list(no_wd_params), 'weight_decay': 0.},
                                     {'params': list(wd_params), 'weight_decay': args.weight_decay}],
                                    args.lr[0],
                                    momentum=args.momentum)
    else:
        optimizer = torch.optim.SGD(model.parameters(), args.lr[0],
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            if isinstance(best_prec1, torch.Tensor):
                best_prec1 = best_prec1.item()

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

    training_transformers = [transforms.RandomResizedCrop(224)]
    if args.lighting:
        training_transformers.append(utils.common.Lighting(0.1))
    training_transformers += [transforms.RandomHorizontalFlip(),
                              transforms.ToTensor(),
                              normalize]

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose(training_transformers))

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
        summary = pruning_summary_resnet50(model, args.expand, args.width_multiplier)
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

        weights = bn_weights(model)
        for bn_name, bn_weight in weights:
            writer.add_histogram("bn/" + bn_name, bn_weight, global_step=epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, writer, mask=weight_masks)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, epoch, writer)

        if args.debug:
            # make sure the prec1 is large enough to test saving functions
            prec1 = epoch

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
            save_backup=(epoch - args.start_epoch) % 10 == 0,
            backup_path=args.backup_path,
            epoch=epoch)

    writer.close()
    print("Best prec@1: {}".format(best_prec1))


def pruning_summary_resnet50(model, expand=False, width_multiplier=1.):
    model_ref = resnet50(aux_fc=False, width_multiplier=1., gate=False) if not expand else ResNetExpand(gate=False, width_multiplier=width_multiplier)
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

    # TensorBoard text uses Markdown format
    # see https://stackoverflow.com/questions/45016458/tensorflow-tf-summary-text-and-linebreaks
    return "  \n".join(pruning_layers)


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


def train(train_loader, model, criterion, optimizer, epoch, writer=None, mask=None):
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
        adjust_learning_rate(optimizer, epoch,
                             train_loader_len=len(train_loader), iteration=i,
                             decay_strategy=args.lr_strategy, warmup=args.warmup,
                             total_epoch=args.epochs, lr=args.lr, decay_epoch=args.decay_epoch)

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

        # Mask finetuning style: do not actullay prune the network,
        # just simply disable the updating of the pruned layers
        if mask is not None:
            for name, p in model.named_parameters():
                if 'weight' in name:
                    p.grad.data = p.grad.data * mask[name]

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

        if args.debug and i >= 5:
            break

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


if __name__ == '__main__':
    main()
