import argparse

import numpy as np
import os
import torch
from torch import nn

from compute_flops import count_model_param_flops
from models.resnet_expand import resnet50, ResNetExpand, Bottleneck
from utils.evaluation import test


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


def pruning_summary_resnet50(model, aux_fc):
    model_ref = resnet50(aux_fc=aux_fc)
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
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
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

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
print(args)

output_name = "pruned" if args.pruning_strategy != "percent" else "pruned_{}".format(args.percent)

if not os.path.exists(args.save):
    os.makedirs(args.save)

model = resnet50(aux_fc=False)
model = nn.DataParallel(model)

if args.cuda:
    model.cuda()
if args.model:
    if os.path.isfile(args.model):
        print("=> loading checkpoint '{}'".format(args.model))
        checkpoint = torch.load(args.model)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']

        for param_name, param in model.named_parameters():
            if param_name not in checkpoint['state_dict']:
                checkpoint['state_dict'][param_name] = param.data
                print("Missing parameter {}, do not load!".format(param_name))
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
              .format(args.model, checkpoint['epoch'], best_prec1))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
model = model.module
model = model.cuda()
total = 0  # total dim nums of bn layers

for name, m in model.named_modules():
    if isinstance(m, nn.BatchNorm2d) and "downsample" not in name:
        total += m.weight.data.shape[0]

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
thre_precent = thre_precent.cuda() if args.cuda else thre_precent.cpu()

pruned = 0
cfg = []
cfg_mask = []
sub_modules = list(model.named_modules())
for k, (name, m) in enumerate(sub_modules):
    if isinstance(m, nn.BatchNorm2d) \
            and "downsample" not in name:
        # should not include bn layers in downsample branches
        weight_copy = m.weight.data.abs().clone()
        thre = thre_precent if args.pruning_strategy == "percent" else __search_threshold(weight_copy,
                                                                                          args.pruning_strategy)
        mask = weight_copy.gt(thre).float().cuda()
        pruned = pruned + mask.shape[0] - torch.sum(mask)
        # m.weight.data.mul_(mask)
        # m.bias.data.mul_(mask)
        cfg.append(int(torch.sum(mask)))
        cfg_mask.append((name, mask.clone()))
        print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
              format(k, mask.shape[0], int(torch.sum(mask))))
    # elif isinstance(m, nn.MaxPool2d):
    #     cfg.append('M')
pruned_ratio = pruned / total

print('Pre-processing Successful!')

# keep the last dimension
# cfg.append(2048)
# keep the first dimension
cfg[0] = 64

print("Cfg:")
print(cfg)

newmodel = ResNetExpand(cfg=cfg, aux_fc=False)
if args.cuda:
    newmodel.cuda()

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

    if isinstance(m0, nn.BatchNorm2d):
        mask = cfg_mask[layer_id_in_cfg]
        idx1 = np.squeeze(np.argwhere(np.asarray(mask[1].cpu().numpy())))
        if idx1.size == 1:
            idx1 = np.resize(idx1, (1,))

        if "layer" not in layer_name or "downsample" in layer_name:
            # do not prune the first bn layer (bn does not in any block)
            # do not prune the bn layer in the downsample
            m1.weight.data = m0.weight.data.clone()
            m1.bias.data = m0.bias.data.clone()
            m1.running_mean = m0.running_mean.clone()
            m1.running_var = m0.running_var.clone()

            if "downsample" in layer_name:
                pass
                # there is no cfg for downsample bn layers
                # layer_id_in_cfg += 1
                # start_mask = end_mask.clone()
                # if layer_id_in_cfg < len(cfg_mask):
                #     end_mask = cfg_mask[layer_id_in_cfg]

                # bn_count += 1
            else:
                layer_id_in_cfg += 1
        else:
            # prune the bn parameters
            assert m1.weight.data.shape == m0.weight.data[idx1.tolist()].clone().shape
            m1.weight.data = m0.weight.data[idx1.tolist()].clone()
            m1.bias.data = m0.bias.data[idx1.tolist()].clone()
            m1.running_mean = m0.running_mean[idx1.tolist()].clone()
            m1.running_var = m0.running_var[idx1.tolist()].clone()

            layer_id_in_cfg += 1
            # start_mask = end_mask.clone()
            # if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
            #     end_mask = cfg_mask[layer_id_in_cfg]

    elif isinstance(m0, nn.Conv2d):
        if "layer" not in layer_name:
            # do not prune the first conv layer
            m1.weight.data = m0.weight.data.clone()
            conv_count += 1
            continue
        elif "downsample" not in layer_name:
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

            # We need to set the channel selection layer.
            # if isinstance(old_modules[layer_id - 1], channel_selection):
            #     m2 = new_modules[layer_id - 1]
            #     m2.indexes = np.zeros_like(m2.indexes)
            #     m2.indexes[idx1.tolist()] = 1.0
            continue

        # We need to consider the case where there are downsampling convolutions.
        # For these convolutions, we just copy the weights.
        m1.weight.data = m0.weight.data.clone()
    elif isinstance(m0, nn.Linear):
        # do not prune the last fc layer
        m1.weight.data = m0.weight.data.clone()
        m1.bias.data = m0.bias.data.clone()

bn3_masks = list(filter(lambda m: "bn3" in m[0], cfg_mask))
bottleneck_modules = list(filter(lambda m: isinstance(m[1], Bottleneck), new_modules))
assert len(bn3_masks) == len(bottleneck_modules)
for i, (name, m) in enumerate(bottleneck_modules):
    if isinstance(m, Bottleneck):
        mask = bn3_masks[i]
        assert mask[1].shape[0] == m.expand_layer.idx.shape[0]
        m.expand_layer.idx = np.argwhere(mask[1].clone().cpu().numpy()).squeeze()

torch.save({'cfg': cfg, 'state_dict': newmodel.state_dict(), "bn3_masks": bn3_masks},
           os.path.join(args.save, '{}.pth.tar'.format(output_name)))

# print(newmodel)
model = newmodel

flops = count_model_param_flops(model.cuda(), 224)
print("FLOPs after pruning: {}".format(flops))

summary = pruning_summary_resnet50(model, False)
print(summary)

# evaluate model
test(model, args)

with open(savepath, "a") as fp:
    fp.write("FLOPs after pruning: {} \n".format(flops))
    fp.write("\n\n\n")
    fp.write("************MODEL SUMMARY************")
    fp.write(summary)
    fp.write("*************************************")
