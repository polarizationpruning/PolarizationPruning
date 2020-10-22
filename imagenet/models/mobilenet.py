import typing
from typing import List

import numpy as np
import torch
from torch import nn

# from .utils import load_state_dict_from_url


__all__ = ['MobileNetV2', 'mobilenet_v2', 'get_sparse_layers', 'InvertedResidual']

from models import ChannelExpand
from models.common import SparseGate, prune_conv_layer, compute_conv_flops_weight, test_conv_flops_weight, Identity, \
    ChannelSelect, Pruner

model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


def get_sparse_layers(model: nn.Module, gate: bool, exclude_out: bool):
    """
    get all layers need sparsity
    if gate is not enabled, get all bn layers exclude dw layers from MobileNet v2 model
    if gate is enabled, return all SparseGate

    :param gate: return the SparseGate layers as sparse layers
    :param exclude_out: do not return the output layer of each building blocks
    """
    sparse_modules = []
    if gate:
        for m_name, sub_module in model.named_modules():
            if isinstance(sub_module, InvertedResidual):
                first_layer: ConvBNReLU = sub_module.conv[0]
                if isinstance(first_layer.sparse_layer, SparseGate):
                    # the first layer contains SparseGate, so this is a pixel-wise layer
                    sparse_modules.append(first_layer.sparse_layer)

                if not exclude_out:
                    last_gate_layer = sub_module.conv[-2]
                    assert isinstance(last_gate_layer, SparseGate), \
                        f"the penultimate layer of the block must be a SparseGate when --gate is enabled, " \
                        f"got {last_gate_layer}"
                    sparse_modules.append(last_gate_layer)
    else:
        for m_name, sub_module in model.named_modules():
            if isinstance(sub_module, InvertedResidual):
                for sub_m_name, subsub_module in sub_module.named_modules():
                    if isinstance(subsub_module, ConvBNReLU):
                        conv_layer = subsub_module[0]
                        bn_layer = subsub_module[1]
                        if conv_layer.groups == 1:
                            # do not apply on groups != 1 conv layers (dw convs)
                            sparse_modules.append(bn_layer)

                # the last bn in the inverted residual block
                assert isinstance(sub_module.conv[-3],
                                  nn.BatchNorm2d), "The InvertedResidual[-3] is supposed to be a bn layer"

                if not exclude_out:
                    # the last bn layer in the block
                    sparse_modules.append(sub_module.conv[-3])

    return sparse_modules


def get_output_gate(model: nn.Module):
    """get last SparseGate for each building blocks"""
    output_gates = []
    for m_name, sub_module in model.named_modules():
        if isinstance(sub_module, InvertedResidual):
            last_gate_layer = sub_module.conv[-2]
            assert isinstance(last_gate_layer, SparseGate), \
                f"the penultimate layer of the block must be a SparseGate when --gate is enabled, " \
                f"got {last_gate_layer}"
            output_gates.append(last_gate_layer)

    return output_gates


def flat_mobilenet_settings(settings: List[List[int]], width_mult=1.0, input_channel=32, round_nearest=8) -> List[
    List[int]]:
    """
    Expand MobileNet settings (in format t, c, n, s. See MobileNet v2 paper, Table 2.)
    The converted settings follow the protocol:
        1. Assure `n` is always 1.
        2. Use `hidden_dim` instead of the expand ratio `t`.
        3. Explicitly specify whether the block need shortcut connection.

    Format for single block:
        [conv_in, hidden_dim, output_channel, 1, stride, need_shortcut]

    Example:
        `[[6, 24, 2, 2]]`
        will be converted to
        ```
        [[conv_in, hidden_dim, 24, 1, 2, need_shortcut, pw],
         [conv_in, hidden_dim, 24, 1, 1, need_shortcut, pw]]
        ```
        where
        `conv_in`: the input_channel of the convolutional layer (after channel selection)
        `hidden_dim = int(round(input_channel * t))`
        `need_shortcut` will be True only if the stride is 1 and
        the number of input channel is equal to the number of the
        output channel. Since the output channel will be pruned,
        we can not determine whether the block need shortcut
        connection or not. So we need to specify `need_shortcut`
        explicitly.
        `pw`: a bool value, indicate if the block has a pixel-wise
        convolution layer
    """
    expand_settings = []
    for expand_ratio, output_channel, repeat_time, stride in settings:  # i.e. for t, c, n, s in settings:
        output_channel = _make_divisible(output_channel * width_mult, round_nearest)
        for i in range(repeat_time):
            # input_channel: the input dimension of the current block
            # when use the default config (unpruned), the ChannelSelect layer select all input layers
            # so the conv_in is equal to the input_channel
            hidden_dim = int(round(input_channel * expand_ratio))
            need_shortcut = (stride == 1 or i != 0) and input_channel == output_channel
            pw = input_channel != hidden_dim  # if there is a pixel-wise conv in the block
            if i == 0:
                expand_settings.append([input_channel, hidden_dim, output_channel, 1, stride, need_shortcut, pw])
            else:
                expand_settings.append([input_channel, hidden_dim, output_channel, 1, 1, need_shortcut, pw])
            input_channel = output_channel

    return expand_settings


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, gate: bool, kernel_size=3, stride=1, groups=1):
        """
        A sequence of modules
            - Conv
            - BN
            - Gate (optional, if gate is True)
            - ReLU
        """
        padding = (kernel_size - 1) // 2

        layers = [
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
        ]
        if gate:
            layers.append(SparseGate(out_planes))
        layers.append(nn.ReLU6(inplace=True))
        super(ConvBNReLU, self).__init__(*layers)

    @property
    def sparse_layer(self):
        # if there is SparseGate then return the gate, else return the BatchNorm2d
        return self[-2]


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, conv_in, hidden_dim, use_shortcut_connection: bool,
                 use_gate: bool, input_mask: bool, pw: bool):
        """
        :param inp: the number of input channel of the block
        :param oup: the number of output channel of the block
        :param stride: the stride of the deep-wise conv layer
        :param conv_in: the input dimension of the conv layer
        :param hidden_dim: the inner dimension of the conv layers
        :param use_shortcut_connection: if use shortcut connect or not
        :param use_gate: if use SparseGate layers between conv layers

        :param input_mask: if use a mask at the beginning of the model.
        The mask is supposed to replace the first gate before the input.
        """
        super(InvertedResidual, self).__init__()

        self.stride = stride
        self.output_channel = oup
        self.hidden_dim = hidden_dim

        self._gate = use_gate
        self._conv_flops_weight: typing.Optional[typing.Tuple[float, float]] = None
        assert stride in [1, 2]

        if hidden_dim != 0:

            # the ChannelSelect layer is supposed to select conv_in channels from inp channels
            self.select = ChannelSelect(inp)
            # this gate will not be affected by gate option
            # the gate should be kept in finetuning stage
            self.input_gate = None

            # this part is moved to flat_mobilenet_settings method, so comment it
            # hidden_dim = int(round(inp * expand_ratio))
            # self.use_res_connect = self.stride == 1 and inp == oup
            self.use_res_connect = use_shortcut_connection

            layers = []
            self.pw = pw  # if there is a pixel-wise conv in the block
            # if hidden_dim != inp: # this part is moved to the config generation method
            if self.pw:
                if use_gate or input_mask:
                    self.input_gate = SparseGate(conv_in)

                # pw (use this bn to prune the hidden_dim)
                layers.append(ConvBNReLU(conv_in, hidden_dim, kernel_size=1, gate=use_gate))
                self.pw = True
            layers.extend([
                # dw (do not apply sparsity on this bn)
                ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, gate=False),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                # use this bn to prune the output dim
                nn.BatchNorm2d(oup),
            ])
            if use_gate:
                layers.append(SparseGate(oup))
            else:
                layers.append(Identity())
            layers.append(ChannelExpand(oup))
            self.conv = nn.Sequential(*layers)
        else:
            self.select = None
            self.input_gate = None
            self.conv = None
            self.use_res_connect = True

    def forward(self, x):
        original_input = x

        if self.conv is not None:
            # Select channels from input
            x = self.select(x)
            if self.input_gate is not None:
                x = self.input_gate(x)

        if self.use_res_connect:
            if self.conv is None:
                # the whole layer is pruned
                return original_input
            return original_input + self.conv(x)
        else:
            return self.conv(x)

    def _prune_whole_layer(self):
        """set the layer as a identity mapping"""
        if not self.use_res_connect:
            raise ValueError("The network will unrelated to the input if prune the whole block without the shortcut.")
        self.conv = None
        self.hidden_dim = 0
        self.pw = False
        pass

    def do_pruning(self, in_channel_mask: np.ndarray, pruner: Pruner):
        """
        Prune the block in place
        :param in_channel_mask: a 0-1 vector indicates whether the corresponding channel should be pruned (0) or not (1)
        :param pruner: the method to determinate the pruning threshold.
        the pruner accepts a torch.Tensor as input and return a threshold
        """
        # prune the pixel-wise conv layer
        if self.pw:
            pw_layer = self.conv[0]
            in_channel_mask, input_gate_mask = prune_conv_layer(conv_layer=pw_layer[0],
                                                                bn_layer=pw_layer[1],
                                                                sparse_layer_in=self.input_gate if self.has_input_mask else None,
                                                                sparse_layer_out=pw_layer.sparse_layer,
                                                                in_channel_mask=None if self.has_input_mask else in_channel_mask,
                                                                pruner=pruner,
                                                                prune_output_mode="prune",
                                                                prune_mode='default')
            if not np.any(in_channel_mask) or not np.any(input_gate_mask):
                # no channel left
                self._prune_whole_layer()
                return self.output_channel

            channel_select_idx = np.squeeze(np.argwhere(np.asarray(input_gate_mask)))
            if len(channel_select_idx.shape) == 0:
                # expand the single scalar to array
                channel_select_idx = np.expand_dims(channel_select_idx, 0)
            elif len(channel_select_idx.shape) == 1 and channel_select_idx.shape[0] == 0:
                # nothing left
                raise NotImplementedError("No layer left in input channel")
            self.select.idx = channel_select_idx
            if self.has_input_mask:
                self.input_gate.do_pruning(input_gate_mask)

        # update the hidden dim
        self.hidden_dim = int(in_channel_mask.astype(np.int).sum())

        # prune the output of the dw layer
        # this in_channel_mask is supposed unchanged
        dw_layer = self.conv[-5]
        in_channel_mask, _ = prune_conv_layer(conv_layer=dw_layer[0],
                                              bn_layer=dw_layer[1],
                                              sparse_layer_in=None,
                                              sparse_layer_out=dw_layer.sparse_layer,
                                              in_channel_mask=in_channel_mask,
                                              pruner=pruner,
                                              prune_output_mode="same",
                                              prune_mode='default')

        # prune input of the dw-linear layer (the last layer)
        out_channel_mask, _ = prune_conv_layer(conv_layer=self.conv[-4],
                                               bn_layer=self.conv[-3],
                                               sparse_layer_in=None,
                                               sparse_layer_out=self.conv[-2] if isinstance(self.conv[-2],
                                                                                            SparseGate) else self.conv[
                                                   -3],
                                               in_channel_mask=in_channel_mask,
                                               pruner=pruner,
                                               prune_output_mode="prune",
                                               prune_mode='default')

        # update output_channel
        self.output_channel = int(out_channel_mask.astype(np.int).sum())

        # if self.use_res_connect:
        # do padding allowing adding with residual connection
        # the output dim is unchanged
        expander: ChannelExpand = self.conv[-1]
        # note that the idx of the expander might be set in a pruned model
        original_expander_idx = expander.idx
        assert len(original_expander_idx) == len(out_channel_mask), "the output channel should be consistent"
        pruned_expander_idx = original_expander_idx[out_channel_mask]
        idx = np.squeeze(pruned_expander_idx)
        expander.idx = idx
        pass

        # return the output dim
        # the output dim is kept unchanged
        return expander.channel_num

    def get_config(self):
        # format
        # conv_in, hidden_dim, output_channel, repeat_num, stride, use_shortcut
        if self.conv is not None:
            if self.pw:
                conv_in = self.pw_layer[0].in_channels
            else:
                conv_in = self.hidden_dim
        else:
            conv_in = 0

        return [conv_in, self.hidden_dim, self.output_channel, 1, self.stride, self.use_res_connect, self.pw]

    def _compute_flops_weight(self, scaling: bool):
        """
        compute the FLOPs weight for each layer according to `d_flops_in` and `d_flops_out`
        """
        # before compute the flops weight, need to
        # 1. set self.raw_weight_max and self.raw_weight_min
        # 2. compute d_flops_out and d_flops_in

        if self.pw:
            pw_flops_weight = self.conv[0][0].d_flops_out + self.conv[1][0].d_flops_in + self.conv[2].d_flops_in
            linear_flops_weight = self.conv[2].d_flops_in
        else:
            # there is no pixel-wise layer
            pw_flops_weight = None
            linear_flops_weight = self.conv[1].d_flops_in

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

        self.conv_flops_weight = (scaling_func(pw_flops_weight),
                                  scaling_func(linear_flops_weight))

    def get_conv_flops_weight(self, update: bool, scaling: bool) -> typing.Tuple[float, float]:
        # force to update the weight, because _compute_flops_weight is very fast
        # if update:
        #     self._compute_flops_weight(scaling=scaling)
        self._compute_flops_weight(scaling=scaling)

        assert self._conv_flops_weight is not None
        return self._conv_flops_weight

    @property
    def conv_flops_weight(self) -> typing.Tuple[float, float]:
        """This method is supposed to used in forward pass.
        To use more argument, call `get_conv_flops_weight`."""
        return self.get_conv_flops_weight(update=True, scaling=True)

    @conv_flops_weight.setter
    def conv_flops_weight(self, weight: typing.Tuple[int, int, int]):
        assert len(weight) == 2, f"The length convolution FLOPs weight should be 2, got {len(weight)}"
        self._conv_flops_weight = weight

    @property
    def pw_layer(self) -> typing.Tuple[nn.Conv2d, nn.BatchNorm2d, typing.Optional[SparseGate]]:
        """get the pixel-wise layer (conv, bn, gate)"""
        if not self.pw:
            return [None, None, None]
        if self.conv is None:
            return [None, None, None]
        pw_conv_bn_relu = self.conv[0]
        pw_conv = pw_conv_bn_relu[0]
        pw_bn = pw_conv_bn_relu[1]
        if self._gate:
            pw_gate = pw_conv_bn_relu[2]
        else:
            pw_gate = None

        return pw_conv, pw_bn, pw_gate

    @property
    def has_input_mask(self) -> bool:
        return self.input_gate is not None

    @property
    def linear_layer(self) -> typing.Tuple[nn.Conv2d, nn.BatchNorm2d, typing.Optional[SparseGate]]:
        """get the linear layer (conv, bn, gate)"""
        if self.pw:
            linear_conv_idx = 2
        else:
            linear_conv_idx = 1

        linear_conv = self.conv[linear_conv_idx]
        linear_bn = self.conv[linear_conv_idx + 1]
        if self._gate:
            linear_gate = self.conv[linear_conv_idx + 2]
        else:
            linear_gate = None

        return linear_conv, linear_bn, linear_gate


class MobileNetV2(nn.Module):
    def __init__(self,
                 use_gate=False,
                 input_mask=False,
                 num_classes=1000,
                 width_mult=1.0,
                 inverted_residual_setting=None,
                 round_nearest=8,
                 block=None):
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet

        """
        super(MobileNetV2, self).__init__()

        if block is None:
            block = InvertedResidual

        self.gate: bool = use_gate

        self.flops_weight_computed = False  # if the conv_flops_weight has been computed
        # the input_size to compute the conv_flops_weight
        # if the input size is changed, the conv_flops_weight should be computed again
        self.input_size: typing.Optional[typing.Tuple[int, int]] = None

        self.input_channel = 32  # the input channel for the whole network
        self.last_channel = 1280

        self.input_channel = _make_divisible(self.input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(self.last_channel * max(1.0, width_mult), round_nearest)

        # the default mobilenet settings (MobileNet v2 paper, Table 2)
        default_inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        default_inverted_residual_setting = flat_mobilenet_settings(default_inverted_residual_setting,
                                                                    input_channel=self.input_channel,
                                                                    width_mult=width_mult)

        if inverted_residual_setting is None:
            inverted_residual_setting = default_inverted_residual_setting
            self._pruned = False
        else:
            # use custom network setting
            self._pruned = True

        self._settings = inverted_residual_setting
        print("MobileNet v2 created. Settings: ")
        print(self._settings)
        print()

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 7:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 7-element list, got {}".format(inverted_residual_setting))

        input_channel = self.input_channel  # the input channel for each block
        # building first layer
        # do not apply sparsity on the first layer, only apply on each residual blocks
        features = [ConvBNReLU(3, input_channel, stride=2, gate=False)]
        # building inverted residual blocks
        for setting_idx, (conv_in, hidden_dim, c, n, s, use_shortcut, pw) in enumerate(inverted_residual_setting):
            if n != 1:
                raise ValueError("do not accept n != 1 settings")

            if not self._pruned:
                # default settings
                output_channel = _make_divisible(c, round_nearest)
                hidden_dim = _make_divisible(hidden_dim, round_nearest)
            else:
                # custom settings
                output_channel = c
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(inp=input_channel, oup=output_channel, stride=stride,
                                      conv_in=conv_in,
                                      hidden_dim=hidden_dim,
                                      use_shortcut_connection=use_shortcut,
                                      use_gate=use_gate,
                                      input_mask=input_mask,
                                      pw=pw, ))
                if not self._pruned:
                    input_channel = output_channel
                else:
                    default_output_dim = default_inverted_residual_setting[setting_idx][2]
                    # expand the output to original dimension
                    if features[-1].conv is not None:
                        features[-1].conv[-1].set_channel_num(default_output_dim)
                    # the input channel of each block will be unchanged
                    input_channel = default_output_dim

        # building last several layers
        # do not apply any sparsity on the last conv layer
        features.append(ConvBNReLU(input_channel, self.last_channel,
                                   kernel_size=1,
                                   gate=False))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

        for m in self.modules():
            # avoid the conv be initialized as normal
            # this line should be after the conv initialization
            if isinstance(m, SparseGate):
                nn.init.constant_(m.weight, 1.)

    def _forward_impl(self, x):
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1 => must use reshape with x.shape[0]
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)

    def get_config(self):
        config = []
        for name, sub_module in self.named_modules():
            if isinstance(sub_module, InvertedResidual):
                config.append(sub_module.get_config())

        return config

    def get_sparse_layer(self, gate: bool, pw_layer, linear_layer, with_weight=False) -> \
            typing.Union[List[nn.Module], typing.Tuple[list, list]]:
        assert gate == self.gate

        sparse_modules = []
        sparse_weights = []

        for m_name, sub_module in self.named_modules():
            if isinstance(sub_module, InvertedResidual):
                sub_module: InvertedResidual
                if with_weight:
                    sub_module_weight = sub_module.get_conv_flops_weight(update=True, scaling=True)
                else:
                    sub_module_weight = [None] * 2

                # the index of (conv, bn, gate)
                if gate:
                    sparse_idx = 2
                else:
                    sparse_idx = 1

                if pw_layer:
                    sparse_modules.append(sub_module.pw_layer[sparse_idx])
                    sparse_weights.append(sub_module_weight[0])
                if linear_layer:
                    sparse_modules.append(sub_module.linear_layer[sparse_idx])
                    sparse_weights.append(sub_module_weight[1])

        assert len(sparse_modules) != 0, "Nothing to return"

        if with_weight:
            return sparse_modules, sparse_weights
        else:
            return sparse_modules

    def get_sparse_weight(self, pw_layer=True, linear_layer=True) -> List[nn.Module]:

        sparse_modules = []

        for m_name, sub_module in self.named_modules():
            if isinstance(sub_module, InvertedResidual):
                sub_module: InvertedResidual

                # in order of the conv, bn, gate
                # only choose conv layers
                sparse_idx = 0

                if pw_layer:
                    sparse_modules.append(sub_module.pw_layer[sparse_idx])
                if linear_layer:
                    sparse_modules.append(sub_module.linear_layer[sparse_idx])

        assert len(sparse_modules) != 0, "Nothing to return"

        return sparse_modules

    def get_conv_flops_weight(self,
                              input_size: typing.Tuple[int, int] = (224, 224)) -> List[typing.Tuple[int, int, int]]:
        """
        :param input_size: the input spatial size of the network
        """
        return compute_conv_flops_weight(self, building_block=InvertedResidual, input_size=input_size)


def mobilenet_v2(pretrained=False, progress=True, **kwargs):
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MobileNetV2(**kwargs)
    if pretrained:
        raise NotImplementedError("MobileNet v2 do not support load pretrain model.")
        # state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
        #                                       progress=progress)
        # model.load_state_dict(state_dict)
    return model


def _test_model(model, model_ref, threshold=1e-5, cuda=False) -> bool:
    """
    test if two model is equivalent by given same input

    :param cuda: if test in GPU
    """

    if cuda:
        model.cuda()
        model_ref.cuda()

    # make sure parameters are same
    # avoid random in initialization
    ref_state_dict = model_ref.state_dict()

    # do not require strict parameter loading
    # the correctness is checked by output the sanity check
    model.load_state_dict(ref_state_dict, strict=False)

    # avoid random in dropout
    model_ref.eval()
    model.eval()

    with torch.no_grad():
        random_input = torch.rand(8, 3, 224, 224)
        if cuda:
            random_input = random_input.cuda()

        torch_output = model_ref(random_input)
        cur_output = model(random_input)

        diff = torch_output - cur_output
        max_diff = torch.max(diff.abs().view(-1)).item()
        print(f"Max diff: {max_diff}")

    return max_diff < threshold


def test():
    """Do unit testing"""

    from torchvision.models import mobilenet_v2 as torch_mobilenet_func
    import math
    # compare this modified mobilenet and torchvision mobilenet

    assert _test_model(mobilenet_v2(), torch_mobilenet_func(), cuda=True), "baseline MobileNet test failed"
    print()
    for use_gate in [True, False]:
        for multi in [0.5, 1.0, 1.5, 1.379, 0.9, 0.33333333333, math.pi]:
            print(f"Testing MobileNet v2 with multiplier {multi}")
            model = mobilenet_v2(width_mult=multi, use_gate=use_gate)

            # if not use_gate, there will not be gate in the model
            if not use_gate:
                for name, sub_module in model.named_modules():
                    assert not isinstance(sub_module, SparseGate), \
                        f"There should not exist SparseGate in the model, got {name}"

            assert _test_model(model,
                               torch_mobilenet_func(width_mult=multi),
                               cuda=True), f"width_mult {multi} test failed"
            print(f"Testing passed: MobileNet v2 with multiplier {multi}")
            print()

            print("Testing get_config function")
            model_default_config = model._settings
            model_config = model.get_config()

            for default_block, model_conf_block in zip(model_default_config, model_config):
                for i, j in zip(default_block, model_conf_block):
                    assert i == j, f"config inconsistent, default: {i}, generated: {j}. \n" \
                                   f"Whole config: \n" \
                                   f"default: \n {model_default_config} \n" \
                                   f"generated: \n {model_config}"

            print("Testing passed: get_config is same as default settings")
            print()

    print("Test passed. The custom MobileNet v2 implementation is as same as Torchvision.")

    # test conv flops weight
    model = mobilenet_v2(use_gate=True)
    flops_weight = model.get_conv_flops_weight()
    test_conv_flops_weight(flops_weight=flops_weight)


if __name__ == '__main__':
    test()
