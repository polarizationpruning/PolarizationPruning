import subprocess
from typing import List, Union, Tuple

import numpy as np
import torch
from PIL import Image
from torch.optim import Optimizer

import models
import utils
from models.common import SparseGate


def get_git_id() -> str:
    try:
        commit_id = subprocess.check_output(["git", "rev-parse", "HEAD"]).rstrip().strip().decode()
    except subprocess.CalledProcessError:
        # the current directory is not a git repository
        return ""
    return commit_id


def cos_lr(base_lr, epoch, iteration, num_iter, num_epoch, warmup=False):
    """
    cosine learning rate schedule
    from https://github.com/d-li14/mobilenetv2.pytorch
    :param epoch: current epoch
    :param iteration: current iteration
    :param num_iter: the number of the iteration in a epoch
    :param num_epoch: the number of the total epoch
    :param warmup: learning rate warm-up in first 5 epoch
    :return: learning rate
    """
    from math import cos, pi

    warmup_epoch = 5 if warmup else 0
    warmup_iter = warmup_epoch * num_iter
    current_iter = iteration + epoch * num_iter
    max_iter = num_epoch * num_iter

    lr = base_lr * (1 + cos(pi * (current_iter - warmup_iter) / (max_iter - warmup_iter))) / 2

    if epoch < warmup_epoch:
        lr = base_lr * current_iter / warmup_iter

    return lr


# lighting data augmentation
imagenet_pca = {
    'eigval': np.asarray([0.2175, 0.0188, 0.0045]),
    'eigvec': np.asarray([
        [-0.5675, 0.7192, 0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948, 0.4203],
    ])
}


class Lighting(object):
    """
    Lighting data augmentation
    """

    def __init__(self, alphastd,
                 eigval=imagenet_pca['eigval'],
                 eigvec=imagenet_pca['eigvec']):
        self.alphastd = alphastd
        assert eigval.shape == (3,)
        assert eigvec.shape == (3, 3)
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0.:
            return img
        rnd = np.random.randn(3) * self.alphastd
        rnd = rnd.astype('float32')
        v = rnd
        old_dtype = np.asarray(img).dtype
        v = v * self.eigval
        v = v.reshape((3, 1))
        inc = np.dot(self.eigvec, v).reshape((3,))
        img = np.add(img, inc)
        if old_dtype == np.uint8:
            img = np.clip(img, 0, 255)
        img = Image.fromarray(img.astype(old_dtype), 'RGB')
        return img

    def __repr__(self):
        return self.__class__.__name__ + '()'


def compute_conv_flops(model: torch.nn.Module, cuda=False):
    """
    compute the FLOPs for MobileNet v2 model
    NOTE: ONLY compute the FLOPs for Convolution layers

    if cuda mode is enabled, the model must be transferred to gpu!
    """

    list_conv = []

    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups)

        flops = kernel_ops * output_channels * output_height * output_width

        list_conv.append(flops)

    list_linear = []

    def linear_hook(self, input, output):
        weight_ops = self.weight.nelement()

        flops = weight_ops
        list_linear.append(flops)

    def add_hooks(net, hook_handles: list):
        """
        apply FLOPs handles to conv layers recursively
        """
        children = list(net.children())
        if not children:
            if isinstance(net, torch.nn.Conv2d):
                hook_handles.append(net.register_forward_hook(conv_hook))
            if isinstance(net, torch.nn.Linear):
                hook_handles.append(net.register_forward_hook(linear_hook))
            return
        for c in children:
            add_hooks(c, hook_handles)

    handles = []
    add_hooks(model, handles)
    demo_input = torch.rand(8, 3, 224, 224)
    if cuda:
        demo_input = demo_input.cuda()
        model = model.cuda()
    model(demo_input)

    total_flops = sum(list_conv) + sum(list_linear)

    # clear handles
    for h in handles:
        h.remove()
    return total_flops


def test():
    # test get_git_id()
    git_hash = get_git_id()
    print(f"Current git hash: {git_hash}")

    # test compute flops
    # Baseline MobileNet v2 model
    from models.mobilenet import mobilenet_v2
    mobilenet_flops = compute_conv_flops(mobilenet_v2())
    print(f"Baseline MobileNet v2 FLOPs: {mobilenet_flops:,}")

    # Baseline ResNet-50 model
    from models import resnet50
    resnet_flops = compute_conv_flops(resnet50(width_multiplier=1., gate=False, aux_fc=False))
    print(f"Baseline ResNet-50 FLOPs: {resnet_flops:,}")

    # Baseline ResNet-50 with expander
    from models.resnet_expand import resnet50 as resnet50_expand
    resnet_expand_flops = compute_conv_flops(resnet50_expand(width_multiplier=1., gate=False, aux_fc=False))
    print(f"Baseline ResNet-50 with ChannelExpand FLOPs: {resnet_expand_flops:,}")


class SparseLayerCollection:
    """
    The collection of single convolution layer, with batch norm and sparse layer.
    """

    def __init__(self, conv_layer: torch.nn.Conv2d,
                 bn_layer: torch.nn.BatchNorm2d,
                 sparse_layer: Union[torch.nn.BatchNorm2d, SparseGate],
                 layer_weight: float):
        self._conv_layer = conv_layer
        self._bn_layer = bn_layer
        self._sparse_layer = sparse_layer
        self._layer_flops_weight = layer_weight

    def get_sparse_layer(self, gate: bool, sparse1: bool, sparse2: bool, sparse3: bool, with_weight=False) -> \
            Union[list, Tuple[list, list]]:
        if with_weight:
            return [self._sparse_layer], [self._layer_flops_weight]
        else:
            return [self._sparse_layer]

    @property
    def module(self):
        return self


if __name__ == '__main__':
    test()


def adjust_learning_rate(optimizer: Optimizer, epoch: int, lr: List[float],
                         total_epoch: int, decay_epoch: List[int],
                         iteration: int, train_loader_len: int, decay_strategy: str, warmup: bool):
    """
    Adjust learning rate in each **step** (not epoch). That means
    this method should be called in every iteration.

    When the `decay_strategy` is `cos`, we use cosine
    learning rate decay schedule. In `step` case, the learning rate
    in each stage is specified by --decay_epoch and --lr.

    Argument `iteration` and `train_loader_len` only required in
    cosine learning rate decay schedule.

    :param train_loader_len: the number of steps in each epoch
    :param epoch: the current epoch
    :param iteration: the current iteration in current epoch
    :param optimizer: the optimizer for the network
    :param decay_strategy: learning rate decay strategy. Support
    `cos` and `step`.
    :param warmup: use warm-up in first five epochs
    """
    if decay_strategy == 'cos':
        # use cosine learning rate
        cur_lr = utils.common.cos_lr(lr[0], epoch=epoch, iteration=iteration,
                                     num_iter=train_loader_len, num_epoch=total_epoch,
                                     warmup=warmup)
        for param_group in optimizer.param_groups:
            param_group['lr'] = cur_lr

    elif decay_strategy == 'step':
        # the learning rate is determined by --lr and --decay-epoch
        if epoch in decay_epoch:
            lr_idx = decay_epoch.index(epoch)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr[lr_idx + 1]
    else:
        raise NotImplementedError(f"Do not support {decay_strategy}")


def freeze_gate(model):
    """do not update all SparseGate in the model"""
    for sub_module in model.modules():
        if isinstance(sub_module, models.common.SparseGate):
            for p in sub_module.parameters():
                # do not update SparseGate
                p.requires_grad = False