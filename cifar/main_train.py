from __future__ import print_function

import argparse
import typing

import numpy as np
import os
import random
import re
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torchvision import datasets, transforms

import common
import models
from common import LossType, compute_conv_flops
from models.common import SparseGate

#### TRAINING SETTINGS ####
parser = argparse.ArgumentParser(description='PyTorch CIFAR training')

parser.add_argument('--arch', default='resnet56', type=str,
                    help='architecture to use')
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'],
                    help='training dataset (default: cifar10)')
parser.add_argument("--loss-type", "-loss", dest="loss",
                    choices=list(LossType.loss_name().keys()), help="the type of loss")

# binary-search in Algorithm 1
parser.add_argument('--target-flops', type=float, default=None,
                    help='Stop when pruned model achieve the target FLOPs')
parser.add_argument('--lbd', type=float, default=0.0002,
                    help='scale sparse rate (i.e. lambda in eq.2) (default: 0.0002)')
parser.add_argument('--t', type=float, default=1.2,
                    help='coefficient of L1 term in polarization regularizer (default: 1.2)')
parser.add_argument('--delta', type=float, default=0.02,
                    help='allowable offset when searching for suitable lbd and t to achieve stable target-flops')

## DON'T CHANGE (specs are according to the paper)
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 200)')
parser.add_argument('--max-epoch', type=int, default=None, metavar='N',
                    help='the max number of epoch, default None')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')

## DON'T CHANGE (specs are according to the paper)
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--decay-epoch', type=float, nargs='*', default=[1, 60, 120, 160],
                    help="the epoch to decay the learning rate (default 1, 60, 120, 160)")
parser.add_argument('--gammas', type=float, nargs='+', default=[10, 0.2, 0.2, 0.2],
                    help='LR is multiplied by gamma on decay-epoch, number of gammas should be equal to decay-epoch')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')


parser.add_argument('--gate', action='store_true', default=False,
                    help='Add an extra scaling factor after the BatchNrom layers.')
parser.add_argument('--bn-wd', action='store_true',
                    help='Apply weight decay on BatchNorm layers')
## DON'T CHANGE
# parser.add_argument('--fix-gate', action='store_true',
#                     help='Do not update parameters of SparseGate while training.')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--clamp', default=1.0, type=float,
                    help='Upper bound of the bn scaling factors (only available at Polarization!) (default: 1.0)')
parser.add_argument('--bn-init-value', default=0.5, type=float,
                    help='initial value of bn weight (default: 0.5, following NetworkSlimming)')

# SAVE & LOG
parser.add_argument('--save', type=str, metavar='PATH', required=True,
                    help='path to save prune model')
parser.add_argument('--log', type=str, metavar='PATH', required=True,
                    help='path to tensorboard log ')
## DON'T CHANGE
parser.add_argument('--max-backup', type=int, default=None,
                    help='The max number of backup files')
parser.add_argument('--backup-path', default=None, type=str, metavar='PATH',
                    help='path to tensorboard log')
parser.add_argument('--backup-freq', default=10, type=float,
                    help='Backup checkpoint frequency')

# LOAD
## DON'T CHANGE
parser.add_argument('--retrain', type=str, default=None, metavar="PATH",
                    help="Pruned checkpoint for RETRAIN model.")
# parser.add_argument('--resume', default='', type=str, metavar='PATH',
#                     help='path to latest checkpoint (default: none)')

# MISC (DON'T CHANGE)
parser.add_argument('--input-mask', action='store_true',
                    help='If use input mask in ResNet models.')
parser.add_argument('--width-multiplier', default=1.0, type=float,
                    help="The width multiplier (only) for ResNet-56 and VGG16-linear. "
                         "Unavailable for other networks. (default 1.0)")
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, metavar='S', default=None,
                    help='random seed (default: a random int)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')

# DEPRECATED
parser.add_argument('--alpha', type=float, default=1.,
                    help='coefficient of mean term in polarization regularizer. deprecated (default: 1)')
# parser.add_argument('--flops-weighted', action='store_true',
#                     help='The polarization parameters will be weighted by FLOPs.')
parser.add_argument('--weight-max', type=float, default=None,
                    help='Maximum FLOPs weight. Only available when --flops-weighted is enabled.')
parser.add_argument('--weight-min', type=float, default=None,
                    help='Minimum FLOPs weight. Only available when --flops-weighted is enabled.')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.loss = LossType.from_string(args.loss)
args.decay_epoch = sorted([int(args.epochs * i if i < 1 else i) for i in args.decay_epoch])
if not args.seed:
    args.seed = random.randint(500, 1000)

# if args.retrain:
#     if not os.path.exists(args.retrain) or not os.path.isfile(args.retrain):
#         raise ValueError(f"Path error: {args.retrain}")

if args.clamp != 1.0 and args.loss == LossType.ORIGINAL:
    print("WARNING: Clamp only available at Polarization!")

# if args.fix_gate:
#     if not args.gate:
#         raise ValueError("--fix-gate should be with --gate.")

# if args.flops_weighted:
#     if args.arch not in {'resnet56', 'vgg16_linear'}:
#         raise ValueError(f"Unsupported architecture {args.arch}")

# if not args.flops_weighted and (args.weight_max is not None or args.weight_min is not None):
#     raise ValueError("When --flops-weighted is not set, do not specific --max-weight or --min-weight")

# if args.flops_weighted and (args.weight_max is None or args.weight_min is None):
#     raise ValueError("When --flops-weighted is set, do specific --max-weight or --min-weight")

# if args.max_backup is not None:
#     if args.max_backup <= 0:
#         raise ValueError("--max-backup is supposed to be greater than 0, got {}".format(args.max_backup))
#     pass

if args.target_flops and not args.gate:
    raise ValueError(f"Conflict option: --target-flops only available at --gate mode")

print(args)
print(f"Current git hash: {common.get_git_id()}")






#------- Reproducibility -------
torch.manual_seed(args.seed)
np.random.seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
#------- Working Dir -------
if not os.path.exists(args.save):
    os.makedirs(args.save)
# if args.backup_path is not None and not os.path.exists(args.backup_path):
#     os.makedirs(args.backup_path)
if not os.path.exists(args.log):
    os.makedirs(args.log)
#------- Dataset -------
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
else:
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('./data.cifar100', download=True, train=True,
                          transform=transforms.Compose([
                              transforms.Pad(4),
                              transforms.RandomCrop(32),
                              transforms.RandomHorizontalFlip(),
                              transforms.ToTensor(),
                              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                          ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('./data.cifar100', download=True, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

num_classes = 10 if args.dataset == 'cifar10' else 100

#------- Model -------
if not args.retrain:
    if re.match("resnet[0-9]+", args.arch):
        model = models.__dict__[args.arch](num_classes=num_classes,
                                           gate=args.gate,
                                           bn_init_value=args.bn_init_value, aux_fc=False,
                                           width_multiplier=args.width_multiplier,
                                           use_input_mask=args.input_mask)
    else:
        raise NotImplementedError("Do not support {}".format(args.arch))

else:  # initialize model for retraining with configs
    raise NotImplementedError("Do not support {} in this version, check the original code".format(args.arch))

training_flops = compute_conv_flops(model, cuda=True)
print(f"Training model. FLOPs: {training_flops:,}")

if args.cuda:
    model.cuda()

#------ Build Optim ------
if args.bn_wd:
    no_wd_type = [models.common.SparseGate]
else:
    # do not apply weight decay on bn layers
    no_wd_type = [models.common.SparseGate, nn.BatchNorm2d, nn.BatchNorm1d]

no_wd_params = []  # do not apply weight decay on these parameters
for module_name, sub_module in model.named_modules():
    for t in no_wd_type:
        if isinstance(sub_module, t):
            for param_name, param in sub_module.named_parameters():
                no_wd_params.append(param)
                print(f"No weight decay param: module {module_name} param {param_name}")

no_wd_params_set = set(no_wd_params)  # apply weight decay on the rest of parameters
wd_params = []
for param_name, model_p in model.named_parameters():
    if model_p not in no_wd_params_set:
        wd_params.append(model_p)
        print(f"Weight decay param: parameter name {param_name}")

optimizer = torch.optim.SGD([{'params': list(no_wd_params), 'weight_decay': 0.},
                             {'params': list(wd_params), 'weight_decay': args.weight_decay}],
                            args.lr,
                            momentum=args.momentum)













#-------- Running --------
history_score = np.zeros((args.epochs, 6))


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


def bn_sparsity(model: nn.Module,
                loss_type: LossType,
                lbd: float, t: float, alpha: float):
    """Compute sparsity loss"""
    bn_modules = model.get_sparse_layers()

    if loss_type == LossType.POLARIZATION:
        # compute global mean of all sparse vectors
        n_ = sum(map(lambda m: m.weight.data.shape[0], bn_modules))
        sparse_weights_mean = torch.sum(torch.stack(list(map(lambda m: torch.sum(m.weight), bn_modules)))) / n_

        sparsity_loss = 0.
        for m in bn_modules:
            if loss_type == LossType.POLARIZATION:
                sparsity_term = t * torch.sum(torch.abs(m.weight)) - torch.sum(
                        torch.abs(m.weight - alpha * sparse_weights_mean))
            else:
                raise ValueError(f"Unexpected loss type: {loss_type}")
            sparsity_loss += lbd * sparsity_term

        return sparsity_loss
    else:
        raise ValueError()

# additional subgradient descent on the sparsity-induced penalty term
def updateBN():
    if args.loss == LossType.L1_SPARSITY_REGULARIZATION:
        sparsity = args.lbd
        bn_modules = list(filter(lambda m: (isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.BatchNorm1d)),
                                 model.named_modules()))
        bn_modules = list(map(lambda m: m[1], bn_modules))  # remove module name
        for m in bn_modules:
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.grad.data.add_(sparsity * torch.sign(m.weight.data))
    else:
        raise NotImplementedError(f"Do not support loss: {args.loss}")


def clamp_bn(model, lower_bound=0, upper_bound=1):
    if model.gate:
        sparse_modules = list(filter(lambda m: isinstance(m, SparseGate), model.modules()))
    else:
        sparse_modules = list(
            filter(lambda m: isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d), model.modules()))

    for m in sparse_modules:
        m.weight.data.clamp_(lower_bound, upper_bound)


def train(epoch, curr_lbd, curr_t):
    """Model training procedure (for 1 epoch)"""
    model.train()
    global history_score, global_step
    avg_loss = 0.
    avg_sparsity_loss = 0.
    train_acc = 0.
    total_data = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        if isinstance(output, tuple):
            output, output_aux = output
        loss = F.cross_entropy(output, target)

        # logging
        avg_loss += loss.data.item()
        pred = output.data.max(1, keepdim=True)[1]
        train_acc += pred.eq(target.data.view_as(pred)).cpu().sum()
        total_data += target.data.shape[0]

        if args.loss in {LossType.POLARIZATION}:
            sparsity_loss = bn_sparsity(
                model, args.loss, curr_lbd, curr_t, args.alpha)
            loss += sparsity_loss
            avg_sparsity_loss += sparsity_loss.data.item()
        loss.backward()
        if args.loss in {LossType.L1_SPARSITY_REGULARIZATION}:
            updateBN()
        optimizer.step()
        if args.loss in {LossType.POLARIZATION}:
            clamp_bn(model, upper_bound=args.clamp)
        global_step += 1
        if batch_idx % args.log_interval == 0:
            print('Step: {} Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                global_step, epoch, batch_idx * len(data), len(train_loader.dataset),
                                    100. * batch_idx / len(train_loader), loss.data.item()))

    history_score[epoch][0] = avg_loss / len(train_loader)
    history_score[epoch][1] = float(train_acc) / float(total_data)
    history_score[epoch][3] = avg_sparsity_loss / len(train_loader)
    pass

def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
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
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * float(correct) / len(test_loader.dataset)))
    return float(correct) / float(len(test_loader.dataset))

def save_checkpoint(state, is_best, filepath, backup: bool, backup_path: str, epoch: int, max_backup: int):
    state['args'] = args

    torch.save(state, os.path.join(filepath, 'checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'), os.path.join(filepath, 'model_best.pth.tar'))
    if backup and backup_path is not None:
        shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'),
                        os.path.join(backup_path, 'checkpoint_{}.pth.tar'.format(epoch)))

        if max_backup is not None:
            while True:
                # remove redundant backup checkpoints to save space
                checkpoint_match = map(lambda f_name: re.fullmatch("checkpoint_([0-9]+).pth.tar", f_name),
                                       os.listdir(backup_path))
                checkpoint_match = filter(lambda m: m is not None, checkpoint_match)
                checkpoint_id: typing.List[int] = list(map(lambda m: int(m.group(1)), checkpoint_match))
                checkpoint_count = len(checkpoint_id)
                if checkpoint_count > max_backup:
                    min_checkpoint_epoch = min(checkpoint_id)
                    min_checkpoint_path = os.path.join(backup_path,
                                                       'checkpoint_{}.pth.tar'.format(min_checkpoint_epoch))
                    print(f"Too much checkpoints (Max {max_backup}, got {checkpoint_count}).")
                    print(f"Remove file: {min_checkpoint_path}")
                    os.remove(min_checkpoint_path)
                else:
                    break
    pass

def prune_while_training(model: nn.Module, arch: str, prune_mode: str, num_classes: int):
    if arch == "resnet56":
        from main_prune import prune_resnet
        from models.resnet_expand import resnet56 as resnet50_expand
        saved_model_grad = prune_resnet(sparse_model=model,
                                        pruning_strategy='grad',
                                        sanity_check=False,
                                        prune_mode=prune_mode,
                                        num_classes=num_classes)
        baseline_model = resnet50_expand(num_classes=num_classes, gate=False, aux_fc=False)
    else:
        raise NotImplementedError(f"do not support arch {arch}")

    saved_flops_grad = compute_conv_flops(saved_model_grad, cuda=True)
    baseline_flops = compute_conv_flops(baseline_model, cuda=True)

    return saved_flops_grad, baseline_flops

def freeze_sparse_gate(model: nn.Module):
    # do not update all SparseGate
    for sub_module in model.modules():
        if isinstance(sub_module, models.common.SparseGate):
            for p in sub_module.parameters():
                # do not update SparseGate
                p.requires_grad = False


best_prec1 = 0.
global_step = 0

writer = SummaryWriter(logdir=args.log)

# binary search
lbd_l, lbd_u = 0., args.lbd
curr_lbd = (lbd_l + lbd_u) / 2
curr_t = args.t
fix_lbd, fix_t = False, False
epoch_flops = []

for epoch in range(args.start_epoch, args.epochs):
    if args.max_epoch is not None and epoch >= args.max_epoch:
        break

    current_learning_rate = adjust_learning_rate(optimizer, epoch, args.gammas, args.decay_epoch)
    print("Start epoch {}/{} with learning rate {}...".format(epoch, args.epochs, current_learning_rate))

    train(epoch, curr_lbd, curr_t)

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
    }, is_best, filepath=args.save,
        backup_path=args.backup_path,
        backup=epoch % args.backup_freq == 0,
        epoch=epoch,
        max_backup=args.max_backup
    )

    # write the tensorboard
    writer.add_scalar("train/average_loss", history_score[epoch][0], epoch)
    writer.add_scalar("train/sparsity_loss", history_score[epoch][3], epoch)
    writer.add_scalar("train/train_acc", history_score[epoch][1], epoch)
    writer.add_scalar("train/lr", optimizer.param_groups[0]['lr'], epoch)
    writer.add_scalar("val/acc", prec1, epoch)
    writer.add_scalar("val/best_acc", best_prec1, epoch)

    # FLOPs
    if args.loss in {LossType.POLARIZATION}:
        flops_grad, baseline_flops = prune_while_training(model,
                                                          arch=args.arch,
                                                          prune_mode='default',
                                                          num_classes=num_classes)
        flops_grad_ratio = flops_grad / baseline_flops
        print(f" --> FLOPs in epoch (grad) {epoch}: {flops_grad:,}, ratio: {flops_grad_ratio}")
        writer.add_scalar("train/flops", flops_grad, epoch)
        writer.add_scalar("train/flops_grad_ratio", flops_grad_ratio, epoch)

        epoch_flops.append(flops_grad_ratio)
        flops_diff = flops_grad_ratio - args.target_flops

        if not fix_lbd and len(epoch_flops) >= 5 and np.std(epoch_flops[:-5]) <= 0.03: # F stable
            if flops_diff >= 2 * args.delta:
                lbd_u = curr_lbd
            elif flops_diff <= -2 * args.delta:
                lbd_l = curr_lbd
            else:
                fix_lbd = True

        if fix_lbd and not fix_t and len(epoch_flops) >= 5 and np.std(epoch_flops[:-5]) <= 0.03: # F stable
            if flops_diff >= args.delta:
                pass
            elif flops_diff <= -args.delta:
                pass
            else:
                fix_t = True

        if fix_lbd and fix_t and args.gate:
            print("The grad pruning FLOPs achieve the target FLOPs.")
            print(f"Current pruning ratio: {flops_grad_ratio}")
            print("Stop polarization from current epoch and continue training.")

            # take out polarization loss
            args.lbd = 0
            freeze_sparse_gate(model)
            if args.backup_freq > 20:
                args.backup_freq = 20

if args.loss == LossType.POLARIZATION and args.target_flops and abs(flops_diff) > args.delta and args.gate:
    print("WARNING: the FLOPs does not achieve the target FLOPs at the end of training.")
print("Best accuracy: " + str(best_prec1))
history_score[-1][0] = best_prec1
np.savetxt(os.path.join(args.save, 'record.txt'), history_score, fmt='%10.5f', delimiter=',')

writer.close()