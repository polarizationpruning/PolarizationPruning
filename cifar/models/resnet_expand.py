'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].

The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.

Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:

name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m

which this implementation indeed has.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import typing
from abc import ABC
from typing import Callable, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
from torch.autograd import Variable

__all__ = ['ResNetExpand', 'BasicBlock', 'resnet56']

from models.common import SparseGate, prune_conv_layer, compute_conv_flops_weight, BuildingBlock, Identity


def _weights_init(m, bn_init_value):
    classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(bn_init_value)
        m.bias.data.zero_()


def _multiply_width(original_width: int, multiplier: float) -> int:
    if isinstance(original_width, bool):
        # bool type will not be changed
        return original_width
    width = max(2, int(original_width * multiplier))

    divisor = 2
    width = int(width + divisor / 2) // divisor * divisor

    return width


class ChannelOperation(nn.Module, ABC):
    def __init__(self, channel_num: int):
        super().__init__()
        self.set_channel_num(channel_num)

    def set_channel_num(self, channel_num: int):
        """
        reset the channel_num of the expander
        NOTE: the idx will also be reset
        """
        if channel_num <= 0:
            raise ValueError("channel_num should be positive")
        if not isinstance(channel_num, int):
            raise ValueError(f"channel_num should be int, got {type(channel_num)}")

        self.channel_num = channel_num
        self.idx = np.arange(channel_num)

    def __repr__(self):
        return f"{type(self).__name__}(channel_num={self.channel_num})"


class ChannelExpand(ChannelOperation):
    def forward(self, x):
        if len(self.idx) == self.channel_num:
            # no need to do expand
            return x
        data = torch.zeros(x.size()[0], self.channel_num, x.size()[2], x.size()[3], device=x.device)
        data[:, self.idx, :, :] = x

        return data


class ChannelSelect(ChannelOperation):
    def forward(self, x):
        """Select channels by channel index"""
        if len(self.idx) == self.channel_num:
            return x

        data = x[:, self.idx, :, :]
        return data


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(BuildingBlock):
    expansion = 1

    def __init__(self, in_planes, outplanes, cfg, stride=1, option='A', gate=False, use_input_mask=False):
        super(BasicBlock, self).__init__()
        self.gate = gate
        self.use_input_mask = use_input_mask
        conv_in = cfg[2]

        if len(cfg) != 3:
            raise ValueError("cfg len should be 3, got {}".format(cfg))

        self.is_empty_block = 0 in cfg

        if not self.is_empty_block:
            if self.use_input_mask:
                # input channel: in_planes, output_channel: conv_in
                self.input_channel_selector = ChannelSelect(in_planes)
                # input channel: conv_in, output_channel: conv_in
                self.input_mask = SparseGate(conv_in)
            else:
                self.input_channel_selector = None
                self.input_mask = None
            self.conv1 = nn.Conv2d(conv_in, cfg[0], kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(cfg[0])
            self.gate1 = SparseGate(cfg[0]) if self.gate else Identity()

            self.conv2 = nn.Conv2d(cfg[0], cfg[1], kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(cfg[1])
            self.gate2 = SparseGate(cfg[1]) if self.gate else Identity()

            self.expand_layer = ChannelExpand(outplanes)
        else:
            self.conv1 = Identity()
            self.bn1 = Identity()
            self.conv2 = Identity()
            self.bn2 = Identity()

            self.expand_layer = Identity()

        self.shortcut = nn.Sequential()  # do nothing
        if stride != 1 or in_planes != outplanes:
            if option == 'A':
                """For CIFAR10 ResNet paper uses option A."""
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (
                                                0, 0, 0, 0, (outplanes - in_planes) // 2, (outplanes - in_planes) // 2),
                                                  "constant",
                                                  0))
            elif option == 'B':
                raise NotImplementedError("Option B is not implemented")
                # self.shortcut = nn.Sequential(
                #     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                #     nn.BatchNorm2d(self.expansion * planes)
                # )

    def forward(self, x):
        if self.is_empty_block:
            out = self.shortcut(x)
            out = F.relu(out)
            return out
        else:
            if self.use_input_mask:
                out = self.input_channel_selector(x)
                out = self.input_mask(out)
            else:
                out = x

            # relu-gate and gate-relu is same
            out = F.relu(self.bn1(self.conv1(out)))
            out = self.gate1(out)

            out = self.bn2(self.conv2(out))
            out = self.gate2(out)

            out = self.expand_layer(out)

            out += self.shortcut(x)
            out = F.relu(out)
            return out

    def do_pruning(self, pruner: Callable[[np.ndarray], float], prune_mode: str,
                   in_channel_mask: np.ndarray = None, prune_on=None) -> None:
        """
        Prune the block in place.
        Note: There is not ChannelExpand layer at the end of the block. After pruning, the output dimension might be
        changed. There will be dimension conflict

        :param pruner: the method to determinate the pruning threshold.
        :param prune_mode: same as `models.common.prune_conv_layer`

        """
        if in_channel_mask is not None:
            raise ValueError("Do not set in_channel_mask")

        if self.is_empty_block:
            return

        # keep input dim and output dim unchanged
        in_channel_mask = np.ones(self.conv1.in_channels)

        # prune conv1

        in_channel_mask, conv1_input_channel_mask = prune_conv_layer(conv_layer=self.conv1,
                                                                     bn_layer=self.bn1,
                                                                     sparse_layer_in=self.input_mask,
                                                                     sparse_layer=self.gate1 if self.gate else self.bn1,
                                                                     in_channel_mask=None if self.use_input_mask else in_channel_mask,
                                                                     pruner=pruner,
                                                                     prune_output_mode="prune",
                                                                     prune_mode=prune_mode,
                                                                     prune_on=prune_on, )
        if not np.any(in_channel_mask) or not np.any(conv1_input_channel_mask):
            # prune the entire block
            self.is_empty_block = True
            return

        if self.use_input_mask:
            # prune the input dimension of the first conv layer (conv1)
            channel_select_idx = np.squeeze(np.argwhere(np.asarray(conv1_input_channel_mask)))
            if len(channel_select_idx.shape) == 0:
                # expand the single scalar to array
                channel_select_idx = np.expand_dims(channel_select_idx, 0)
            elif len(channel_select_idx.shape) == 1 and channel_select_idx.shape[0] == 0:
                # nothing left
                # this code should not be executed, if there is no channel left,
                # the identity will be set as True and return (see code above)
                raise NotImplementedError("No layer left in input channel")
            self.input_channel_selector.idx = channel_select_idx
            self.input_mask.do_pruning(conv1_input_channel_mask)

        # prune conv2
        out_channel_mask, _ = prune_conv_layer(conv_layer=self.conv2,
                                               bn_layer=self.bn2,
                                               sparse_layer=self.gate2 if self.gate else self.bn2,
                                               in_channel_mask=in_channel_mask,
                                               pruner=pruner,
                                               prune_output_mode="prune",
                                               prune_mode=prune_mode,
                                               prune_on=prune_on, )
        if not np.any(out_channel_mask):
            # prune the entire block
            self.is_empty_block = True
            return

        # do padding allowing adding with residual connection
        # the output dim is unchanged
        # note that the idx of the expander might be set in a pruned model
        original_expander_idx = self.expand_layer.idx
        assert len(original_expander_idx) == len(out_channel_mask), "the output channel should be consistent"
        pruned_expander_idx = original_expander_idx[out_channel_mask]
        idx = np.squeeze(pruned_expander_idx)
        if len(idx.shape) == 0:
            # expand 0-d idx
            idx = np.expand_dims(idx, 0)
            pass
        self.expand_layer.idx = idx

    def config(self) -> typing.Tuple[int, int, int]:
        if self.is_empty_block:
            return 0, 0, 0
        return self.conv1.out_channels, self.conv2.out_channels, self.conv1.in_channels

    @property
    def expand_idx(self):
        raise NotImplementedError()
        if self.is_empty_block:
            return None
        return self.expand_layer.idx

    @expand_idx.setter
    def expand_idx(self, value):
        if self.is_empty_block:
            if value is not None:
                raise ValueError(f"The expand_idx of the empty block is supposed to be None, got {value}")
            # do nothing for empty block
        self.expand_layer.idx = value

    def _compute_flops_weight(self, scaling):
        conv1_flops_weight = self.conv1.d_flops_out + self.conv2.d_flops_in
        conv2_flops_weight = self.conv2.d_flops_out

        def scale(raw_value):
            if raw_value is None:
                return None
            return (raw_value - self.raw_weight_min) / (self.raw_weight_max - self.raw_weight_min)

        def identity(raw_value):
            return raw_value

        if scaling:
            scaling_func = scale
        else:
            scaling_func = identity

        self.conv_flops_weight = (scaling_func(conv1_flops_weight),
                                  scaling_func(conv2_flops_weight))

    def get_conv_flops_weight(self, update: bool, scaling: bool):
        # force update
        # time of update flops weight is very cheap
        self._compute_flops_weight(scaling=scaling)

        assert self._conv_flops_weight is not None
        return self._conv_flops_weight

    @property
    def conv_flops_weight(self) -> typing.Tuple[float, float]:
        """This method is supposed to used in forward pass.
        To use more argument, call `get_conv_flops_weight`."""
        return self.get_conv_flops_weight(update=True, scaling=True)

    @conv_flops_weight.setter
    def conv_flops_weight(self, weight: typing.Tuple[float, float]):
        assert len(weight) == 2, f"The length convolution FLOPs weight should be 2, got {len(weight)}"
        self._conv_flops_weight = weight

    def get_sparse_modules(self):
        if self.gate:
            return self.gate1, self.gate2
        else:
            return self.bn1, self.bn2


class ResNetExpand(nn.Module):
    def __init__(self, block, num_blocks, bn_init_value, cfg=None, num_classes=10, aux_fc=False,
                 gate=False, expand_idx=None, width_multiplier=1.,
                 use_input_mask: bool = False):
        super(ResNetExpand, self).__init__()

        self.gate = gate
        self.flops_weight_computed = False
        self.in_planes = _multiply_width(16, width_multiplier)
        self._width_multiplier = width_multiplier
        self._use_input_mask = use_input_mask

        assert len(num_blocks) == 3, "only 3 layers, got {}".format(len(num_blocks))
        default_cfg = self._get_default_config(num_blocks, _multiply_width(16, width_multiplier))
        self._block_cfg_len = 3
        default_cfg: List[typing.Optional[int, bool]] = [item for sub_list in default_cfg for item in sub_list]
        if cfg is not None and width_multiplier != 1.:
            raise ValueError('do not specific width_multiplier when specific cfg.')
        if cfg is None:
            # Construct config variable.
            cfg = default_cfg
            for i in range(len(cfg)):
                cfg[i] = _multiply_width(cfg[i], width_multiplier)
        assert len(cfg) == len(default_cfg), "config length error!"

        self.conv1 = nn.Conv2d(3, _multiply_width(16, width_multiplier), kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(_multiply_width(16, width_multiplier))
        self.layer1 = self._make_layer(block, out_planes=16, num_blocks=num_blocks[0], stride=1,
                                       cfg=cfg[:self._block_cfg_len * num_blocks[0]])
        self.layer2 = self._make_layer(block, out_planes=32, num_blocks=num_blocks[1], stride=2,
                                       cfg=cfg[self._block_cfg_len * num_blocks[0]:self._block_cfg_len * sum(
                                           num_blocks[0:2])])
        self.layer3 = self._make_layer(block, out_planes=64, num_blocks=num_blocks[2], stride=2,
                                       cfg=cfg[self._block_cfg_len * sum(num_blocks[0:2]):self._block_cfg_len * sum(
                                           num_blocks[0:3])])
        self.linear = nn.Linear(self.in_planes, num_classes)

        self.enable_aux_fc = aux_fc
        if aux_fc:
            raise NotImplementedError("do not support aux fc")
            # self.aux_fc_layer = nn.Linear(32, num_classes)
        else:
            self.aux_fc_layer = None

        self._initialize_weights(bn_init_value)

        # init expand idx
        self.channel_index = expand_idx

        # init SparseGate parameter
        for m in self.modules():
            # avoid the conv be initialized as normal
            # this line should be after the conv initialization
            if isinstance(m, SparseGate):
                nn.init.constant_(m.weight, 1.)

    def _get_default_config(self, num_blocks: List[int], input_dim: int):
        default_cfg = [[16, 16] for _ in range(num_blocks[0])] + \
                      [[32, 32] for _ in range(num_blocks[1])] + \
                      [[64, 64] for _ in range(num_blocks[2])]

        for i in range(len(default_cfg)):
            block_config = default_cfg[i]
            output_dim = block_config[-1]
            block_config.append(input_dim)
            input_dim = output_dim

        return default_cfg

    def _make_layer(self, block, out_planes, num_blocks, stride, cfg):
        block: BasicBlock
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        out_planes = _multiply_width(out_planes, self._width_multiplier)
        for i, stride in enumerate(strides):
            layers.append(block(in_planes=self.in_planes, outplanes=out_planes,
                                stride=stride,
                                cfg=cfg[self._block_cfg_len * i:self._block_cfg_len * (i + 1)],
                                gate=self.gate, use_input_mask=self._use_input_mask))
            self.in_planes = out_planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)

        if self.enable_aux_fc:
            out_aux = F.adaptive_avg_pool2d(out, 1)
            out_aux = out_aux.view(out_aux.size(0), -1)
            out_aux = self.aux_fc_layer(out_aux)
        else:
            out_aux = None

        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out, out_aux

    def _initialize_weights(self, bn_init_value=1):
        for m in self.modules():
            _weights_init(m, bn_init_value)

    def prune_model(self, **kwargs):
        """
        Prune the ResNet model.

        Note:
             1. The input dimension and the output dimension of each blocks will be unchanged
             2. The downsample layer will not be pruned
             3. Only prune the conv layers in each building blocks. The first and the last conv
             layers will NOT be pruned.
        """
        for name, sub_module in self.named_modules():
            if isinstance(sub_module, BasicBlock):
                sub_module.do_pruning(**kwargs)

    def get_sparse_layers(self) -> List[nn.Module]:
        sparse_layers = []
        for m in self.modules():
            if isinstance(m, BasicBlock):
                m: BasicBlock
                if self.gate:
                    sparse_layers.append(m.gate1)
                    sparse_layers.append(m.gate2)
                else:
                    sparse_layers.append(m.bn1)
                    sparse_layers.append(m.bn2)

        return sparse_layers

    def config(self) -> List[int]:
        # a flatten config list
        config: List[int] = []
        for m in self.modules():
            if isinstance(m, BasicBlock):
                m: BasicBlock
                for config_item in m.config():
                    config.append(config_item)

        return config

    def compute_flops_weight(self, cuda=False) -> typing.List[typing.Tuple[int]]:
        return compute_conv_flops_weight(self, BasicBlock)
        pass

    @property
    def channel_index(self) -> typing.Dict[str, np.ndarray]:
        idx = {}
        for name, submodule in self.named_modules():
            if isinstance(submodule, ChannelOperation):
                idx[name] = submodule.idx

        return idx

    @channel_index.setter
    def channel_index(self, channel_idx: typing.Dict[str, np.ndarray]):
        if channel_idx is None:
            return
        for name, submodule in self.named_modules():
            if isinstance(submodule, ChannelOperation):
                submodule.idx = channel_idx[name]

    @property
    def building_block(self):
        return BasicBlock

    @property
    def use_input_mask(self) -> bool:
        return self._use_input_mask


def resnet20(num_classes):
    return ResNetExpand(BasicBlock, [3, 3, 3], num_classes)


def resnet32(num_classes):
    return ResNetExpand(BasicBlock, [5, 5, 5], num_classes)


def resnet44(num_classes):
    return ResNetExpand(BasicBlock, [7, 7, 7], num_classes)


def resnet56(num_classes, bn_init_value=1.0, cfg=None, aux_fc=False, gate=False, expand_idx=None, width_multiplier=1.,
             use_input_mask=False):
    return ResNetExpand(BasicBlock, [9, 9, 9],
                        cfg=cfg,
                        num_classes=num_classes,
                        bn_init_value=bn_init_value,
                        aux_fc=aux_fc,
                        gate=gate, expand_idx=expand_idx,
                        width_multiplier=width_multiplier,
                        use_input_mask=use_input_mask)


def resnet110(num_classes):
    return ResNetExpand(BasicBlock, [18, 18, 18], num_classes)


def resnet1202(num_classes):
    return ResNetExpand(BasicBlock, [200, 200, 200], num_classes)


def _check_models(model, ref_model, threshold=1e-5):
    """Assert two model is same"""
    ref_state_dict = ref_model.state_dict()
    # if gate is enabled, do not load gate parameter in the state_dict
    if model.gate:
        # there is no gate parameters in ref model
        # allow missing parameters in SparseGate
        for key in model.state_dict().keys():
            if key not in ref_state_dict:
                assert "_conv" in key, "only allow missing parameters of SparseGate"
                ref_state_dict[key] = model.state_dict()[key]

    # use same parameter
    model.load_state_dict(ref_state_dict)

    # do not update running_mean and running_var of the bn layers
    model.eval()
    ref_model.eval()

    with torch.no_grad():
        random_input = torch.rand(8, 3, 224, 224)
        torch_output, _ = ref_model(random_input)
        cur_output, extra_info = model(random_input)

        diff = torch_output - cur_output
        max_diff = torch.max(diff.abs().view(-1)).item()
        print(f"Max diff: {max_diff}")

    return max_diff < threshold


def test_gate():
    # test the forward pass
    ref_model = resnet56(num_classes=10, bn_init_value=0.5, aux_fc=False, gate=False)
    gate_model = resnet56(num_classes=10, bn_init_value=0.5, aux_fc=False, gate=True)

    assert _check_models(gate_model, ref_model, threshold=1e-5), f"The diff is too large"


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size()) > 1, net.parameters()))))


def test_flops_weight():
    model = resnet56(num_classes=10, bn_init_value=0.5, aux_fc=False, gate=True)
    flops_weight = compute_conv_flops_weight(model, BasicBlock)
    pass


def _test_width_multiplier():
    model = resnet56(10)
    model_multiplier = resnet56(10, width_multiplier=1.)

    model.eval()
    model_multiplier.eval()

    model.load_state_dict(model_multiplier.state_dict())

    rand_input = torch.rand(8, 3, 32, 32)

    model_out, _ = model(rand_input)
    model_multiplier_out, _ = model_multiplier(rand_input)

    max_diff = (model_out - model_multiplier_out).view(-1).abs().max().item()
    assert max_diff < 1e-5, f"Max diff between multiplier model and original model should < 1e-5, got {max_diff}"

    import math
    for multi in [0.5, 0.555555, math.pi, 1 / math.pi, 0.0001]:
        print(f"Testing width multiplier with multiplier {multi}")
        model = resnet56(10, width_multiplier=multi)
        model(rand_input)

    print("Width multiplier: Test pass!")
    print()


if __name__ == "__main__":
    _test_width_multiplier()
    test_flops_weight()
    test_gate()
