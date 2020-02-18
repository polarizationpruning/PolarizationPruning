import argparse
from enum import Enum
from random import randint

import numpy as np
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.multiprocessing as mp

from tensorboardX import SummaryWriter

import models.resnet
from models import resnet50
from vgg import slimmingvgg as vgg11

model_names = ["vgg11", "resnet50"]


class LossType(Enum):
    ORIGINAL = 0
    L1_SPARSITY_REGULARIZATION = 1
    POLARIZATION = 4

    @staticmethod
    def from_string(desc):
        mapping = LossType.loss_name()
        return mapping[desc.lower()]

    @staticmethod
    def loss_name():
        return {"original": LossType.ORIGINAL,
                "sr": LossType.L1_SPARSITY_REGULARIZATION,
                "zol": LossType.POLARIZATION,
                }


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training with Polarization Regularization')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg11',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=25, type=int, metavar='N',
                    help='number of data loading workers (default: 25)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', type=float, nargs='*', default=[1e-1, 1e-2, 1e-3], metavar='LR',
                    help="the learning rate in each stage (default 1e-2, 1e-3)")
parser.add_argument('--decay-epoch', type=float, nargs='*', default=[30, 60],
                    help="the epoch to decay the learning rate (default None)")
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
# NOTE the definition of the world size there (the number of the NODE)
# is different from the world size in dist.init_process_group (the number of PROCESS)
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed NODE (not process)')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--lbd', type=float, default=0.0001,
                    help='scale sparse rate (default: 0.0001)')
parser.add_argument('--alpha', type=float, default=1.,
                    help='coefficient of mean term in zol loss (default: 1)')
parser.add_argument('--t', type=float, default=1.,
                    help='coefficient of L1 term in zol loss (default: 1)')
parser.add_argument('--save', default='./', type=str, metavar='PATH',
                    help='path to save model (default: current directory)')
parser.add_argument('--backup-path', default='.', type=str, metavar='PATH',
                    help='path to save model (default: current directory)')
parser.add_argument('-summary-freq', default=100, type=int)
parser.add_argument('--rank', default=-1, type=int,
                    help='node (not process) rank for distributed training')
parser.add_argument("--loss-type", "-loss", dest="loss", required=True,
                    choices=list(LossType.loss_name().keys()), help="the type of loss")
parser.add_argument("--debug", action="store_true",
                    help="enable debug mode")
parser.add_argument("-ddp", action="store_true",
                    help="use DistributedDataParallel mode instead of DataParallel")
parser.add_argument("--last-sparsity", action="store_true",
                    help="apply sparsity loss on the last bn in the block")
parser.add_argument("--bn-init-value", type=float, default=0.5,
                    help="The initial value of BatchNormnd weight")
parser.add_argument('--seed', type=int, metavar='S', default=666,
                    help='random seed (default: 666)')
parser.add_argument("--fc-sparsity", default="unified",
                    choices=["unified", "separate", "single"],
                    help='''Method of calculating average for vgg network. (default unified)
                    unified: default behaviour. use the global mean for all layers.
                    separate: only available for vgg11. use different mean for CNN layers and FC layers separately.
                    single: only available for vgg11. use global mean for CNN layers and different mean for each FC layers.
                    ''')

best_prec1 = 0


def main():
    # set environment variables
    os.environ["NCCL_DEBUG"] = "INFO"

    # parse args
    args = parser.parse_args()

    args.loss = LossType.from_string(args.loss)
    if args.last_sparsity and not (args.loss == LossType.POLARIZATION):
        print("WARNING: loss type {} does not support --last-sparsity!".format(args.loss))

    if len(args.lr) != len(args.decay_epoch) + 1:
        print("args.lr: {}".format(args.lr))
        print("args.decay-epoch: {}".format(args.decay_epoch))
        raise ValueError("inconsistent between lr-decay-gamma and decay-epoch")
    args.decay_epoch = [int(e) for e in args.decay_epoch]

    args.fc_sparsity = str.lower(args.fc_sparsity)
    if args.fc_sparsity != "unified" and args.loss != LossType.POLARIZATION:
        raise NotImplementedError(f"Option --fc-sparsity is conflict with loss {args.loss}")
    if args.fc_sparsity != "unified" and args.arch != "vgg11":
        raise NotImplementedError(f"Option --fc-sparsity only support VGG. Set to unified for {args.arch}")

    print(args)
    if args.debug:
        print("*****WARNING! DEBUG MODE IS ENABLED!******")
        print("******The model will NOT be trained!******")
        pass

    # reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # the number of gpu in current device
    ngpus_per_node = torch.cuda.device_count()
    # enable distributed mode if there are more than one gpu
    args.distributed = args.ddp and (args.world_size > 1 or ngpus_per_node > 1)

    # start process
    if args.distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        # (suppose the number of gpu in each node is SAME)
        args.world_size = ngpus_per_node * args.world_size
        print("actual args.world_size: {}".format(args.world_size))
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function (with all gpus)
        main_worker("", ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_prec1
    args.gpu = gpu

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    if args.rank == 0:
        if not os.path.exists(args.save):
            os.makedirs(args.save)
        if not os.path.exists(args.backup_path):
            os.makedirs(args.backup_path)

    if args.distributed:
        # For multiprocessing distributed training, rank needs to be the
        # global rank among all the processes
        args.rank = args.rank * ngpus_per_node + gpu
        print("Starting process rank {}".format(args.rank))
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    print("rank#{}: CUDA_VISIBLE_DEVICES: {}".format(args.rank, os.environ['CUDA_VISIBLE_DEVICES']))

    if args.arch == "vgg11":
        if args.resume and os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            if "cfg" in checkpoint:
                model = vgg11(config=checkpoint['cfg'])
            else:
                model = vgg11()
        else:
            model = vgg11()
    elif args.arch == "resnet50":
        model = resnet50(mask=False, bn_init_value=args.bn_init_value,
                         aux_fc=False, save_feature_map=False)
    else:
        raise NotImplementedError("model {} is not supported".format(args.arch))

    if not args.distributed:
        # DataParallel
        model.cuda()
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            # see discussion
            # https://discuss.pytorch.org/t/are-there-reasons-why-dataparallel-was-used-differently-on-alexnet-and-vgg-in-the-imagenet-example/19844
            model.features = torch.nn.DataParallel(model.features)
        else:
            model = torch.nn.DataParallel(model).cuda()
    else:
        # DistributedDataParallel
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr[0],
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.debug:
        # fake polarization to test pruning
        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
                module.weight.data.zero_()
                one_num = randint(3, 30)
                module.weight.data[:one_num] = 0.99

                print(f"{name} remains {one_num}")

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            raise ValueError("=> no checkpoint found at '{}'".format(args.resume))

    print("Model loading completed. Model Summary:")
    print(model)

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    print("rank #{}: loading the dataset...".format(args.rank))

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

    print("rank #{}: dataloader loaded!".format(args.rank))

    if args.evaluate:
        validate(val_loader, model, criterion, epoch=0, args=args, writer=None)
        return

    # only master process in each node write to disk
    writer = SummaryWriter(logdir=args.save, write_to_disk=args.rank % ngpus_per_node == 0)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        # the adjusting only work when epoch is at decay_epoch
        adjust_learning_rate(optimizer, epoch, lr=args.lr, decay_epoch=args.decay_epoch)

        # draw bn hist to tensorboard
        weights, bias = bn_weights(model)
        for bn_name, bn_weight in weights:
            writer.add_histogram("bn/" + bn_name, bn_weight, global_step=epoch)
        for bn_name, bn_bias in bias:
            writer.add_histogram("bn_bias/" + bn_name, bn_bias, global_step=epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch,
              args.lbd, args=args,
              is_debug=args.debug, writer=writer)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, epoch,
                         args=args, writer=writer)

        report_prune_result(model)  # do not really prune the model

        writer.add_scalar("train/lr", optimizer.param_groups[0]['lr'], epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        if args.rank % ngpus_per_node == 0:
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

        writer.flush()

    writer.close()
    print("Best prec@1: {}".format(best_prec1))


def updateBN(model, sparsity, sparsity_on_bn3):
    bn_modules = list(filter(lambda m: (("bn3" not in m[0] and "downsample" not in m[0]) or sparsity_on_bn3) and (
            isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.BatchNorm1d)), model.named_modules()))
    bn_modules = list(map(lambda m: m[1], bn_modules))  # remove module name
    for m in bn_modules:
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.weight.grad.data.add_(sparsity * torch.sign(m.weight.data))


def BN_grad_zero(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            mask = (m.weight.data != 0)
            mask = mask.float().cuda()
            m.weight.grad.data.mul_(mask)
            m.bias.grad.data.mul_(mask)


def bn_weights(model):
    weights = []
    bias = []
    for name, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            weights.append((name, m.weight.data))
            bias.append((name, m.bias.data))

    return weights, bias
    pass


def bn_sparsity(model, loss_type, sparsity, t, alpha, sparsity_on_bn3=True):
    """

    :type model: torch.nn.Module
    :type alpha: float
    :type t: float
    :type sparsity: float
    :type loss_type: LossType
    :type sparsity_on_bn3: bool
    """
    bn_modules = list(filter(lambda m: (("bn3" not in m[0] and "downsample" not in m[0]) or sparsity_on_bn3) and (
            isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.BatchNorm1d)), model.named_modules()))
    bn_modules = list(map(lambda m: m[1], bn_modules))  # remove module name

    mask_modules = list(filter(lambda m: isinstance(m, models.resnet.ChannelMask), model.modules()))
    bn_modules += mask_modules
    assert len(bn_modules) != 0, f"no bn modules available in the model {str(model)}"

    if loss_type == LossType.POLARIZATION:
        n_ = sum(map(lambda m: m.weight.data.shape[0], bn_modules))
        bn_weights_mean = torch.sum(torch.stack(list(map(lambda m: torch.sum(m.weight), bn_modules)))) / n_

        sparsity_loss = 0
        for m in bn_modules:
            if loss_type == LossType.POLARIZATION:
                sparsity_term = t * torch.sum(torch.abs(m.weight)) - torch.sum(
                    torch.abs(m.weight - alpha * bn_weights_mean))
            else:
                raise ValueError("Do not support loss {}".format(loss_type))
            sparsity_loss += sparsity * sparsity_term

        return sparsity_loss
    else:
        raise ValueError()

    pass


def clamp_bn(model):
    bn_modules = list(filter(lambda m: isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d), model.modules()))
    for m in bn_modules:
        m.weight.data.clamp_(0, 1)


def report_prune_result(model):
    print("*******PRUNING REPORT*******")
    for k, m in enumerate(model.modules()):
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            weight_copy = m.weight.data.abs().clone()
            thre = 0.01
            mask = weight_copy.gt(thre)
            mask = mask.float().cuda()
            print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                  format(k, mask.shape[0], int(torch.sum(mask))))
    print("****************************")


def train(train_loader, model, criterion, optimizer, epoch, sparsity, args, is_debug=False,
          writer=None):
    print("rank #{}: start training epoch {}".format(args.rank, epoch))

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    avg_sparsity_loss = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (image, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        image = image.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(image)
        if isinstance(output, tuple):
            output, extra_info = output
        loss = criterion(output, target)
        losses.update(loss.data.item(), image.size(0))

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        top1.update(prec1[0], image.size(0))
        top5.update(prec5[0], image.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        if args.loss in {LossType.POLARIZATION}:
            if args.fc_sparsity == "unified":
                # default behaviour
                # use the global mean for all layers
                sparsity_loss = bn_sparsity(model, args.loss, args.lbd, args.t, args.alpha,
                                            sparsity_on_bn3=args.last_sparsity)

            elif args.fc_sparsity == "separate":
                # use average value for CNN and FC separately
                # note: the separate option is only available for VGG-like network (CNN with more than one fc layers)

                # handle different cases for dp warpper
                feature_module = model.features if hasattr(model, "features") else model.module.features
                classifier_module = model.classifier if hasattr(model, "classifier") else model.module.classifier

                sparsity_loss_feature = bn_sparsity(feature_module,
                                                    args.loss, args.lbd, args.t, args.alpha,
                                                    sparsity_on_bn3=args.last_sparsity)
                sparsity_loss_classifier = bn_sparsity(classifier_module,
                                                       args.loss, args.lbd, args.t, args.alpha,
                                                       sparsity_on_bn3=args.last_sparsity)
                sparsity_loss = sparsity_loss_feature + sparsity_loss_classifier
            elif args.fc_sparsity == "single":
                # apply bn_sparsity for each FC layer

                # handle different cases for dp warpper
                feature_module = model.features if hasattr(model, "features") else model.module.features
                classifier_module = model.classifier if hasattr(model, "classifier") else model.module.classifier

                sparsity_loss_feature = bn_sparsity(feature_module,
                                                    args.loss, args.lbd, args.t, args.alpha,
                                                    sparsity_on_bn3=args.last_sparsity)
                sparsity_loss_classifier = 0.
                for name, submodule in classifier_module.named_modules():
                    if isinstance(submodule, nn.BatchNorm1d):
                        sparsity_loss_classifier += bn_sparsity(submodule,
                                                                args.loss, args.lbd, args.t, args.alpha,
                                                                sparsity_on_bn3=args.last_sparsity)
                sparsity_loss = sparsity_loss_feature + sparsity_loss_classifier
            else:
                raise NotImplementedError(f"do not support --fc-sparsity as {args.fc_sparsity}")
            loss += sparsity_loss
            avg_sparsity_loss.update(sparsity_loss.data.item(), image.size(0))

        loss.backward()
        if args.loss == LossType.L1_SPARSITY_REGULARIZATION:
            updateBN(model, sparsity, sparsity_on_bn3=args.last_sparsity)
        # BN_grad_zero(model)
        optimizer.step()
        if args.loss in {LossType.POLARIZATION}:
            clamp_bn(model)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and (args.rank == 0):
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Sparsity Loss {s_loss.val:.4f} ({s_loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, s_loss=avg_sparsity_loss,
                top1=top1, top5=top5))
        if is_debug and i >= 5:
            break

    if writer is not None:
        writer.add_scalar("train/cross_entropy", losses.avg, epoch)
        writer.add_scalar("train/sparsity_loss", avg_sparsity_loss.avg, epoch)
        writer.add_scalar("train/top1", top1.avg.item(), epoch)
        writer.add_scalar("train/top5", top5.avg.item(), epoch)


def validate(val_loader, model, criterion, epoch, args, writer=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (image, target) in enumerate(val_loader):
            image = image.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(image)
            if isinstance(output, tuple):
                output, out_aux = output
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.data.item(), image.size(0))
            top1.update(prec1[0], image.size(0))
            top5.update(prec5[0], image.size(0))

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


def adjust_learning_rate(optimizer, epoch, decay_epoch, lr):
    if epoch in decay_epoch:
        lr_idx = decay_epoch.index(epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr[lr_idx + 1]


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
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
