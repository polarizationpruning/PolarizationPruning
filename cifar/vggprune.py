import argparse

import numpy as np
import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms

import models
from compute_flops import count_model_param_flops

# Prune settings
parser = argparse.ArgumentParser(description='PyTorch CIFAR prune')
parser.add_argument('--dataset', type=str, default='cifar100',
                    help='training dataset (default: cifar10)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--arch', default=None, type=str,
                    help='architecture to use')
parser.add_argument('--depth', type=int, default=None,
                    help='depth of the vgg')
parser.add_argument('--percent', type=float, default=None,
                    help='scale sparse rate (default: None)')
parser.add_argument('--model', default='', type=str, metavar='PATH',
                    help='path to the model (default: none)')
parser.add_argument('--save', default='', type=str, metavar='PATH',
                    help='path to save pruned model (default: none)')
parser.add_argument("--pruning-strategy", type=str,
                    choices=["percent", "fixed", "grad", "search"],
                    help="Pruning strategy", required=True)


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
                if hist_x[i + 1] > 0.1:
                    raise ValueError("TOO LARGE PRUNING THRESHOLD {}!".format(hist_x[i + 1]))
                return hist_x[i + 1]
    elif alg == "fixed":
        return hist_x[1]
    return 0


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if not os.path.exists(args.save):
    os.makedirs(args.save)

model = models.__dict__[args.arch](num_classes=10 if args.dataset == 'cifar10' else 100)

if args.model:
    if os.path.isfile(args.model):
        print("=> loading checkpoint '{}'".format(args.model))
        checkpoint = torch.load(args.model, map_location=lambda storage, loc: storage)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
              .format(args.model, checkpoint['epoch'], best_prec1))
    else:
        raise ValueError("=> no checkpoint found at '{}'".format(args.model))

print(model)

total = 0
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        total += m.weight.data.shape[0]

if args.pruning_strategy == 'percent':
    bn = torch.zeros(total)
    index = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            size = m.weight.data.shape[0]
            bn[index:(index + size)] = m.weight.data.abs().clone()
            index += size

    y, i = torch.sort(bn)
    thre_index = int(total * args.percent)
    thre_percent = y[thre_index]

pruned = 0
cfg = []
cfg_mask = []
for k, m in enumerate(model.modules()):
    if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        weight_copy = m.weight.data.abs().clone()
        thre = thre_percent if args.pruning_strategy == "percent" else __search_threshold(weight_copy,
                                                                                          args.pruning_strategy)

        mask = weight_copy.gt(thre).float()
        pruned = pruned + mask.shape[0] - torch.sum(mask)
        m.weight.data.mul_(mask)
        m.bias.data.mul_(mask)
        cfg.append(int(torch.sum(mask)))
        cfg_mask.append(mask.clone())
        print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
              format(k, mask.shape[0], int(torch.sum(mask))))
    elif isinstance(m, nn.MaxPool2d):
        cfg.append('M')

pruned_ratio = pruned / total

print('Pre-processing Successful!')


# simple test model after Pre-processing prune (simple set BN scales to zeros)
def test(model):
    model = model.cuda()
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    if args.dataset == 'cifar10':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    elif args.dataset == 'cifar100':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./data.cifar100', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    else:
        raise ValueError("No valid dataset is given.")
    model.eval()
    correct = 0
    for data, target in test_loader:
        with torch.no_grad():
            data, target = data.cuda(), target.cuda()
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    print('\nTest set: Accuracy: {}/{} ({:.1f}%)\n'.format(
        correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    return correct.item() / float(len(test_loader.dataset))


acc = test(model)

# Make real prune
print(cfg)
newmodel = models.__dict__[args.arch](num_classes=10 if args.dataset == 'cifar10' else 100, cfg=cfg)

num_parameters = sum([param.nelement() for param in newmodel.parameters()])
savepath = os.path.join(args.save, "prune.txt")
with open(savepath, "w") as fp:
    fp.write("Configuration: \n" + str(cfg) + "\n")
    fp.write("Number of parameters: \n" + str(num_parameters) + "\n")
    fp.write("Test accuracy: \n" + str(acc))

layer_id_in_cfg = 0
start_mask = torch.ones(3)
end_mask = cfg_mask[layer_id_in_cfg]
assert len(list(model.modules())) == len(list(newmodel.modules()))
for [m0, m1] in zip(model.modules(), newmodel.modules()):
    if isinstance(m0, nn.BatchNorm2d) or isinstance(m0, nn.BatchNorm1d):
        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
        if idx1.size == 1:
            idx1 = np.resize(idx1, (1,))
        m1.weight.data = m0.weight.data[idx1.tolist()].clone()
        m1.bias.data = m0.bias.data[idx1.tolist()].clone()
        m1.running_mean = m0.running_mean[idx1.tolist()].clone()
        m1.running_var = m0.running_var[idx1.tolist()].clone()
        layer_id_in_cfg += 1
        start_mask = end_mask.clone()
        if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
            end_mask = cfg_mask[layer_id_in_cfg]
    elif isinstance(m0, nn.Conv2d):
        idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
        print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
        if idx0.size == 1:
            idx0 = np.resize(idx0, (1,))
        if idx1.size == 1:
            idx1 = np.resize(idx1, (1,))
        w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
        w1 = w1[idx1.tolist(), :, :, :].clone()
        m1.weight.data = w1.clone()
    elif isinstance(m0, nn.Linear):
        idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
        if idx0.size == 1:
            idx0 = np.resize(idx0, (1,))
        if idx1.size == 1:
            idx1 = np.resize(idx1, (1,))
        w1 = m0.weight.data[:, idx0.tolist()].clone()
        if layer_id_in_cfg != len(cfg_mask):
            w1 = w1[idx1.tolist(), :].clone()
            bias1 = m0.bias.data[idx1.tolist()].clone()
        else:
            bias1 = m0.bias.data.clone()
        assert m1.weight.data.shape == w1.shape
        assert m1.bias.data.shape == bias1.shape
        m1.weight.data = w1.clone()
        m1.bias.data = bias1.clone()

torch.save({'cfg': cfg, 'state_dict': newmodel.state_dict()}, os.path.join(args.save, 'pruned.pth.tar'))

print(newmodel)
pruned_acc = test(newmodel)
print("Accuracy after pruning: {}".format(pruned_acc))

# calculate FLOPs
base_flops = count_model_param_flops(model, 32)
pruned_flops = count_model_param_flops(newmodel, 32)
flops_ratio = pruned_flops / base_flops
print("Pruning FLOPs: {}".format(flops_ratio))
