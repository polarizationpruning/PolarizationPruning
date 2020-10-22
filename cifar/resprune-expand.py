import argparse

import numpy as np
import os
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import datasets, transforms

from compute_flops import count_model_param_flops
from models.common import Identity
from models.resnet_expand import resnet56, BasicBlock
from common import LossType


def test(model, test_loader):
    # use GPU at testing
    test_model = model.cuda()
    test_model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.cuda()
            target = target.cuda()
            output, _ = test_model(data)  # ignore aux layer at testing
            test_loss += F.cross_entropy(output, target, size_average=False).data.item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * float(correct) / len(test_loader.dataset)))
    return float(correct) / float(len(test_loader.dataset))


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


def pruning_summary_resnet56(model, num_classes):
    model_ref = resnet56(num_classes)
    if hasattr(model, "module"):
        # remove parallel wrapper
        model = model.module

    pruning_layers = []
    for (name, m), (name_ref, m_ref) in zip(model.named_modules(), model_ref.named_modules()):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            assert len(m_ref.weight.shape) == 1
            assert len(m.weight.shape) == 1
            pruning_layers.append("{}: original shape: {}, pruned shape: {}"
                                  .format(name, m_ref.weight.shape[0], m.weight.shape[0]))
    return "\n".join(pruning_layers)


parser = argparse.ArgumentParser(description='PyTorch CIFAR prune')
parser.add_argument('--dataset', type=str, default='cifar100',
                    help='training dataset (default: cifar10)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--depth', type=int, default=164,
                    help='depth of the resnet')
parser.add_argument('--percent', type=float, default=0.5,
                    help='scale sparse rate (default: 0.5)')
parser.add_argument('--model', default='', type=str, metavar='PATH',
                    help='path to the model (default: none)')
parser.add_argument('--save', default='', type=str, metavar='PATH',
                    help='path to save pruned model (default: none)')
parser.add_argument("--pruning-strategy", type=str,
                    choices=["percent", "fixed", "grad", "search"],
                    help="Pruning strategy", required=True)
parser.add_argument("-ac", "--auxiliary-classifier", action="store_true",
                    help="enable auxiliary classifier (follow DCP)")

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
print(args)

output_name = "pruned_{}".format(args.pruning_strategy) if args.pruning_strategy != "percent" else "pruned_{}".format(
    args.percent)

if not os.path.exists(args.save):
    os.makedirs(args.save)

if str.lower(args.dataset) == "cifar100":
    num_classes = 100
elif str.lower(args.dataset) == "cifar10":
    num_classes = 10
else:
    raise NotImplementedError("do not support dataset {}".format(args.dataset))
model = resnet56(num_classes, aux_fc=False)

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
        raise ValueError("=> no checkpoint found at '{}'".format(args.resume))

if args.dataset == 'cifar10':
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])),
        batch_size=args.test_batch_size, shuffle=True)
else:
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('./data.cifar100', download=True, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])),
        batch_size=args.test_batch_size, shuffle=True)

load_acc = test(model, test_loader)
print("=> Loaded completed. Test acc: {}".format(load_acc))
model = model.cpu()

total = 0  # total dim nums of bn layers

for name, m in model.named_modules():
    if isinstance(m, nn.BatchNorm2d) and "downsample" not in name:
        total += m.weight.data.shape[0]

"""
compute all bn weights for percent pruning
"""
bn = torch.zeros(total)  # concat all bn weight
index = 0
for name, m in model.named_modules():
    if isinstance(m, torch.nn.BatchNorm2d) and "downsample" not in name:
        size = m.weight.data.shape[0]
        bn[index:(index + size)] = m.weight.data.abs().clone()
        index += size
y, i = torch.sort(bn)
thre_index = int(total * args.percent)
thre_precent = y[thre_index]

pruned = 0
cfg = []
cfg_mask = []
sub_modules = list(model.named_modules())
for k, (name, m) in enumerate(sub_modules):
    if isinstance(m, nn.BatchNorm2d) \
            and "layer" in name:
        # should not include the first bn layer
        weight_copy = m.weight.data.abs().clone()
        thre = thre_precent if args.pruning_strategy == "percent" else __search_threshold(weight_copy,
                                                                                          args.pruning_strategy)
        mask = weight_copy.gt(thre).float()  # mask REMAINS dimensions
        # assert np.all((m.bias.data.cpu().numpy() * ~(mask.cpu().numpy().astype(np.bool))) < 1e-6)
        pruned = pruned + mask.shape[0] - torch.sum(mask)
        m.weight.data.mul_(mask)
        m.bias.data.mul_(mask)
        cfg.append(int(torch.sum(mask)))
        cfg_mask.append((name, mask.clone()))
        print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
              format(k, mask.shape[0], int(torch.sum(mask))))
    # elif isinstance(m, nn.MaxPool2d):
    #     cfg.append('M')
pruned_ratio = pruned / total

print('Pre-processing Successful!')

print("Cfg:")
print(cfg)

newmodel = resnet56(num_classes=num_classes, cfg=cfg, aux_fc=False)

num_parameters = sum([param.nelement() for param in newmodel.parameters()])
savepath = os.path.join(args.save, "{}.txt".format(output_name))
with open(savepath, "w") as fp:
    fp.write("Configuration: \n" + str(cfg) + "\n")
    fp.write("Number of parameters: \n" + str(num_parameters) + "\n")

old_modules = list(model.named_modules())
new_modules = list(newmodel.named_modules())
assert len(old_modules) == len(new_modules)
layer_id_in_cfg = 0
mask = torch.ones(3)
conv_count = 0
# bn_count = 0  # debug variable

for layer_id in range(len(old_modules)):
    m0 = old_modules[layer_id][1]
    m1 = new_modules[layer_id][1]
    layer_name = old_modules[layer_id][0]
    assert old_modules[layer_id][0] == new_modules[layer_id][0]

    if "bn" in layer_name:
        mask = cfg_mask[layer_id_in_cfg]
        idx1 = np.squeeze(np.argwhere(np.asarray(mask[1].cpu().numpy())))
        if idx1.size == 1:
            idx1 = np.resize(idx1, (1,))

        if "layer" not in layer_name:
            # do not prune the first bn layer (bn does not in any block)
            m1.weight.data = m0.weight.data.clone()
            m1.bias.data = m0.bias.data.clone()
            m1.running_mean = m0.running_mean.clone()
            m1.running_var = m0.running_var.clone()
        else:
            # prune the bn parameters
            if not isinstance(m1, Identity):
                assert m1.weight.data.shape == m0.weight.data[idx1.tolist()].clone().shape
                m1.weight.data = m0.weight.data[idx1.tolist()].clone()
                m1.bias.data = m0.bias.data[idx1.tolist()].clone()
                m1.running_mean = m0.running_mean[idx1.tolist()].clone()
                m1.running_var = m0.running_var[idx1.tolist()].clone()

            layer_id_in_cfg += 1

    elif "conv" in layer_name:
        if "layer" not in layer_name:
            # do not prune the first conv layer
            m1.weight.data = m0.weight.data.clone()
            # conv_count += 1
            continue
        elif "downsample" not in layer_name:
            if not isinstance(m1, Identity):
                # This convers the convolutions in the residual block.
                # The convolutions are either after the channel selection layer or after the batch normalization layer.
                # exclude conv layers in downsample branches
                conv_count += 1
                idx0 = np.squeeze(np.argwhere(np.asarray(mask[1].cpu().numpy())))  # input
                idx1 = np.squeeze(np.argwhere(np.asarray(cfg_mask[layer_id_in_cfg][1].cpu().numpy())))  # output
                print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))

                # change the number of input channels
                # do not change the input channel of the first conv in each block
                if "conv1" not in layer_name:
                    w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
                else:
                    w1 = m0.weight.data.clone()

                w1 = w1[idx1.tolist(), :, :, :].clone()

                assert m1.weight.data.shape == w1.shape
                m1.weight.data = w1.clone()

                continue

    elif isinstance(m0, nn.Linear):
        # do not prune the last fc layer
        m1.weight.data = m0.weight.data.clone()
        m1.bias.data = m0.bias.data.clone()
    else:
        ValueError("Unexpected case for m0: {}".format(m0))

bn2_masks = list(filter(lambda m: "bn2" in m[0], cfg_mask))
block_modules = list(filter(lambda m: isinstance(m[1], BasicBlock), new_modules))
assert len(bn2_masks) == len(block_modules)
for i, (name, m) in enumerate(block_modules):
    if isinstance(m, BasicBlock):
        if isinstance(m.expand_layer, Identity):
            continue
        mask = bn2_masks[i]
        assert mask[1].shape[0] == m.expand_layer.idx.shape[0]
        m.expand_layer.idx = np.argwhere(mask[1].clone().cpu().numpy()).squeeze().reshape(-1)

torch.save({'cfg': cfg, 'state_dict': newmodel.state_dict(), "bn3_masks": bn2_masks},
           os.path.join(args.save, '{}.pth.tar'.format(output_name)))

model.enable_aux_fc = False
newmodel.enable_aux_fc = False
flops_ref = count_model_param_flops(model.cpu(), 32)
model = newmodel
flops = count_model_param_flops(model.cpu(), 32)

summary = pruning_summary_resnet56(model, num_classes=num_classes)
print(summary)

pruned_acc = test(model, test_loader)
print("=> Pruned completed. Test acc: {}".format(load_acc))

with open(savepath, "a") as fp:
    fp.write("FLOPs before pruning: {} \n".format(flops_ref))
    fp.write("FLOPs after pruning: {} \n".format(flops))
    fp.write("\n\n\n")
    fp.write("************MODEL SUMMARY************")
    fp.write(summary)
    fp.write("*************************************")

print("FLOPs before pruning: {}".format(flops_ref))
print("FLOPs after pruning: {}".format(flops))
print("FLOPs remains {}".format(flops / flops_ref))
