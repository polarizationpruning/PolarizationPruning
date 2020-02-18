from typing import List

import numpy as np
import torch
from random import randint
from torch import nn

from vgg import slimmingvgg as vgg11


def _search_threshold(weight, alg: str) -> float:
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


def _calculate_channel_mask(model: nn.Module, pruning_strategy: str, cuda=True) -> (List[int], List[torch.Tensor]):
    """calculate pruned channel and mask layer by layer"""
    total = 0
    pruned = 0
    cfg = []
    cfg_mask = []
    for k, m in enumerate(model.modules()):
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            weight_copy = m.weight.data.abs().clone()
            total += weight_copy.shape[0]
            thre = _search_threshold(weight_copy,
                                     pruning_strategy)
            mask = weight_copy.gt(thre)
            if cuda:
                mask = mask.float().cuda()
            else:
                mask = mask.float()
            pruned = pruned + mask.shape[0] - torch.sum(mask)
            m.weight.data.mul_(mask)
            m.bias.data.mul_(mask)
            cfg.append(int(torch.sum(mask)))
            cfg_mask.append(mask.clone())
            print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                  format(k, mask.shape[0], int(torch.sum(mask))))
        elif isinstance(m, nn.MaxPool2d):
            cfg.append('M')

    return cfg, cfg_mask


def assign_model(original_model: nn.Module, pruned_model: nn.Module, cfg_mask: list) -> nn.Module:
    """transfer parameters from old model to new model"""
    old_modules = list(original_model.named_modules())
    new_modules = list(pruned_model.named_modules())
    assert len(old_modules) == len(
        new_modules), f"expected equal module nums, got {len(old_modules)} v.s. {len(new_modules)}"

    first_linear = True
    bn_idx = 0  # the index of output bn mask for conv layers
    for i in range(len(old_modules)):
        old_name, old_module = old_modules[i]
        new_name, new_module = new_modules[i]

        assert old_name == new_name, f"Expected same module name, got {old_name} and {new_name}"

        if isinstance(old_module, nn.BatchNorm2d) or isinstance(old_module, nn.BatchNorm1d):
            idx = np.squeeze(np.argwhere(np.asarray(cfg_mask[bn_idx].cpu().numpy())))
            if idx.size == 1:
                idx = np.resize(idx, (1,))

            assert new_module.weight.data.shape == old_module.weight.data[idx.tolist()].clone().shape
            new_module.weight.data = old_module.weight.data[idx.tolist()].clone()
            new_module.bias.data = old_module.bias.data[idx.tolist()].clone()
            new_module.running_mean = old_module.running_mean[idx.tolist()].clone()
            new_module.running_var = old_module.running_var[idx.tolist()].clone()

            bn_idx += 1
            pass
        elif isinstance(old_module, nn.Conv2d):
            old_conv_weight = old_module.weight.clone()
            old_conv_bias = old_module.bias.clone()

            # prune input dim
            if bn_idx - 1 != -1:
                # -1 is the first layer of conv, do not prune the input dimension
                idx_input = np.squeeze(np.argwhere(np.asarray(cfg_mask[bn_idx - 1].cpu().numpy())))
                if idx_input.size == 1:
                    idx_input = np.resize(idx_input, (1,))
                old_conv_weight = old_conv_weight.data[:, idx_input.tolist(), :, :].clone()
            # prune output dim
            idx_output = np.squeeze(np.argwhere(np.asarray(cfg_mask[bn_idx].cpu().numpy())))
            if idx_output.size == 1:
                idx_output = np.resize(idx_output, (1,))
            old_conv_weight = old_conv_weight.data[idx_output.tolist(), :, :, :].clone()
            old_conv_bias = old_conv_bias.data[idx_output.tolist()].clone()

            assert old_conv_weight.shape == new_module.weight.shape, f"Expected same shape to assign, got {old_conv_weight.shape} and {new_module.weight.shape}"
            assert old_conv_bias.shape == new_module.bias.shape, f"Expected same shape to assigin, got {old_conv_bias.shape} and {new_module.bias.shape}"

            new_module.weight.data = old_conv_weight.clone()
            new_module.bias.data = old_conv_bias.clone()

        elif isinstance(old_module, nn.Linear):
            old_linear_weight = old_module.weight.clone()
            old_linear_bias = old_module.bias.clone()

            # prune the input dimension
            idx_input = np.squeeze(np.argwhere(np.asarray(cfg_mask[bn_idx - 1].cpu().numpy())))
            if idx_input.size == 1:
                idx_input = np.resize(idx_input, (1,))
            if first_linear:
                def gen_list(offset):
                    base_list = np.arange(7 * 7)
                    return base_list + offset * 49

                idx_input = [gen_list(x) for x in idx_input]
                idx_input = np.concatenate(idx_input)
                idx_input = np.sort(idx_input)
                first_linear = False
            old_linear_weight = old_linear_weight.data[:, idx_input.tolist()].clone()

            # prune output layer
            if bn_idx == len(cfg_mask):
                # do not prune the output layer
                idx_output = np.arange(old_linear_weight.shape[0])
            else:
                # prune output dim
                idx_output = np.squeeze(np.argwhere(np.asarray(cfg_mask[bn_idx].cpu().numpy())))
                if idx_output.size == 1:
                    idx_output = np.resize(idx_output, (1,))
            old_linear_weight = old_linear_weight.data[idx_output.tolist(), :].clone()
            old_linear_bias = old_linear_bias.data[idx_output.tolist()].clone()

            assert old_linear_weight.shape == new_module.weight.shape, f"Expected same shape to assign, got {old_conv_weight.shape} and {new_module.weight.shape}"
            assert old_linear_bias.shape == new_module.bias.shape, f"Expected same shape to assigin, got {old_conv_bias.shape} and {new_module.bias.shape}"

            new_module.weight.data = old_linear_weight.clone()
            new_module.bias.data = old_linear_bias.clone()


def prune_vgg(model, pruning_strategy, cuda=True, dataparallel=True):
    cfg, cfg_mask = _calculate_channel_mask(model, pruning_strategy, cuda=cuda)
    pruned_model = vgg11(config=cfg)
    if cuda:
        pruned_model.cuda()
    if dataparallel:
        pruned_model.features = torch.nn.DataParallel(pruned_model.features)
    assign_model(model, pruned_model, cfg_mask)

    return pruned_model, cfg
    pass


if __name__ == '__main__':
    test_model = vgg11().cuda()

    # fake polarization to test pruning
    for name, module in test_model.named_modules():
        if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
            module.weight.data.zero_()
            one_num = randint(3, 30)
            module.weight.data[:one_num] = 0.99

            print(f"{name} remains {one_num}")

    pruned_model, cfg = prune_vgg(test_model, pruning_strategy="fixed")

    demo_input = torch.rand(3, 3, 224, 224).cuda()

    original_output = test_model(demo_input)
    pruned_output = pruned_model(demo_input)
    pass
