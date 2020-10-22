from __future__ import absolute_import

import math
import numpy as np
import torch
import torch.nn as nn
import typing
from typing import List

__all__ = ['ResNetExpand', 'Bottleneck', 'resnet50']

import models
from models.common import SparseGate, ChannelExpand, ChannelSelect, Identity, Pruner, compute_conv_flops_weight
from utils.common import SparseLayerCollection


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes: int, outplanes: int, cfg: List[int], gate: bool, stride=1, downsample=None,
                 expand=False):
        """

        :param inplanes: the input dimension of the block
        :param outplanes: the output dimension of the block
        :param cfg: the output dimension of each convolution layer
            config format:
            [conv1_out, conv2_out, conv3_out, conv1_in]
        :param gate: if use gate between conv layers
        :param stride: the stride of the first convolution layer
        :param downsample: if use the downsample convolution in the residual connection
        :param expand: if use ChannelExpand layer in the block
        """
        super(Bottleneck, self).__init__()

        conv_in = cfg[3]
        self.use_gate = gate
        self._identity = False
        if 0 in cfg or conv_in == 0:
            # the whole block is pruned
            self.identity = True
        else:
            # the main body of the block

            # add a SparseGate before the first conv
            # to enable the pruning of the input dimension for further reducing computational complexity
            self.select = ChannelSelect(inplanes)  # after select, the channel number of feature map is conv_in
            self.input_gate = SparseGate(conv_in)

            self.conv1 = nn.Conv2d(conv_in, cfg[0], kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm2d(cfg[0])
            self.gate1 = SparseGate(cfg[0]) if gate else Identity()

            self.conv2 = nn.Conv2d(cfg[0], cfg[1], kernel_size=3, stride=stride,
                                   padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(cfg[1])
            self.gate2 = SparseGate(cfg[1]) if gate else Identity()

            self.conv3 = nn.Conv2d(cfg[1], cfg[2], kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm2d(cfg[2])
            self.gate3 = SparseGate(cfg[2]) if gate else Identity()

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.expand = expand
        self.expand_layer = ChannelExpand(outplanes) if expand else None

    def forward(self, x):
        residual = x

        if not self.identity:
            out = self.select(x)
            out = self.input_gate(out)

            out = self.conv1(out)
            out = self.bn1(out)
            out = self.gate1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.gate2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)
            out = self.gate3(out)
        else:
            # the whole layer is pruned
            out = x

        if self.downsample:
            residual = self.downsample(x)

        if self.expand_layer:
            out = self.expand_layer(out)

        out += residual
        out = self.relu(out)

        return out

    def do_pruning(self, pruner: Pruner, prune_mode: str) -> None:
        """
        Prune the block in place.
        Note: There is not ChannelExpand layer at the end of the block. After pruning, the output dimension might be
        changed. There will be dimension conflict

        :param pruner: the method to determinate the pruning threshold.
        :param prune_mode: same as `models.common.prune_conv_layer`

        """

        # keep input dim and output dim unchanged
        # prune conv1
        in_channel_mask, input_gate_mask = models.common.prune_conv_layer(conv_layer=self.conv1,
                                                                          bn_layer=self.bn1,
                                                                          # prune the input gate
                                                                          sparse_layer_in=self.input_gate,
                                                                          sparse_layer_out=self.gate1 if self.use_gate else self.bn1,
                                                                          in_channel_mask=None,
                                                                          pruner=pruner,
                                                                          prune_output_mode="prune",
                                                                          prune_mode=prune_mode)
        # this layer has no channel left. the whole block is pruned
        if not np.any(in_channel_mask) or not np.any(input_gate_mask):
            self.identity = True
            return

        # prune the input dimension of the first conv layer (conv1)
        channel_select_idx = np.squeeze(np.argwhere(np.asarray(input_gate_mask)))
        if len(channel_select_idx.shape) == 0:
            # expand the single scalar to array
            channel_select_idx = np.expand_dims(channel_select_idx, 0)
        elif len(channel_select_idx.shape) == 1 and channel_select_idx.shape[0] == 0:
            # nothing left
            # this code should not be executed, if there is no channel left,
            # the identity will be set as True and return (see code above)
            raise NotImplementedError("No layer left in input channel")
        self.select.idx = channel_select_idx
        self.input_gate.do_pruning(input_gate_mask)

        # prune conv2
        in_channel_mask, _ = models.common.prune_conv_layer(conv_layer=self.conv2,
                                                            bn_layer=self.bn2,
                                                            sparse_layer_in=None,
                                                            sparse_layer_out=self.gate2 if self.use_gate else self.bn2,
                                                            in_channel_mask=in_channel_mask,
                                                            pruner=pruner,
                                                            prune_output_mode="prune",
                                                            prune_mode=prune_mode)

        remain_channel_num = np.sum(in_channel_mask == 1)
        if remain_channel_num == 0:
            # this layer has no channel left. the whole block is pruned
            self.identity = True
            return

            # prune conv3
        out_channel_mask, _ = models.common.prune_conv_layer(conv_layer=self.conv3,
                                                             bn_layer=self.bn3,
                                                             sparse_layer_in=None,
                                                             sparse_layer_out=self.gate3 if self.use_gate else self.bn3,
                                                             in_channel_mask=in_channel_mask,
                                                             pruner=pruner,
                                                             prune_output_mode="prune",
                                                             prune_mode=prune_mode)

        remain_channel_num = np.sum(out_channel_mask == 1)
        if remain_channel_num == 0:
            # this layer has no channel left. the whole block is pruned
            self.identity = True
            return

        # do not prune downsample layers
        # if need pruning downsample layers (especially with gate), remember to add gate to downsample layer

        # if self.use_res_connect:
        # do padding allowing adding with residual connection
        # the output dim is unchanged
        # note that the idx of the expander might be set in a pruned model
        original_expander_idx = self.expand_layer.idx
        assert len(original_expander_idx) == len(out_channel_mask), "the output channel should be consistent"
        pruned_expander_idx = original_expander_idx[out_channel_mask]
        idx = np.squeeze(pruned_expander_idx)
        self.expand_layer.idx = idx

    def config(self):
        if self.identity:
            return [0] * 4
        else:
            return [self.conv1.out_channels, self.conv2.out_channels, self.conv3.out_channels, self.conv1.in_channels]

    @property
    def identity(self):
        """
        If the block as a identity block.
        Note: When there is a downsample module in the block, the downsample will NOT
        be pruned. In this case, the block will NOT be a identity mapping.
        """
        return self._identity

    @identity.setter
    def identity(self, value):
        """
        Set the block as a identity block. Equivalent to the whole block is pruned.
        Note: When there is a downsample module in the block, the downsample will NOT
        be pruned. In this case, the block will NOT be a identity mapping.
        """
        self._identity = value

    def _compute_flops_weight(self, scaling: bool):
        # check if flops is computed
        # the checking might be time-consuming, if the flops is not computed, there will be error
        # for i in range(1, 4):
        #     conv_layer: nn.Conv2d = getattr(self, f"conv{i}")
        #     if not hasattr(conv_layer, "d_flops_in") or not hasattr(conv_layer, "d_flops_out"):
        #         raise AssertionError("Need compute FLOPs for each conv layer first!")

        conv1_flops_weight = self.conv1.d_flops_out + self.conv2.d_flops_in
        conv2_flops_weight = self.conv2.d_flops_out + self.conv3.d_flops_in
        conv3_flops_weight = self.conv3.d_flops_out

        def scale(raw_value):
            return (raw_value - self.raw_weight_min) / (self.raw_weight_max - self.raw_weight_min)

        def identity(raw_value):
            return raw_value

        if scaling:
            scaling_func = scale
        else:
            scaling_func = identity

        self.conv_flops_weight = (scaling_func(conv1_flops_weight),
                                  scaling_func(conv2_flops_weight),
                                  scaling_func(conv3_flops_weight))

    def get_conv_flops_weight(self, update: bool, scaling: bool) -> typing.Tuple[float, float, float]:
        if update:
            self._compute_flops_weight(scaling=scaling)

        assert self._conv_flops_weight is not None
        return self._conv_flops_weight

    @property
    def conv_flops_weight(self) -> typing.Tuple[float, float, float]:
        """This method is supposed to used in forward pass.
        To use more argument, call `get_conv_flops_weight`."""
        return self.get_conv_flops_weight(update=True, scaling=True)

    @conv_flops_weight.setter
    def conv_flops_weight(self, weight: typing.Tuple[int, int, int]):
        assert len(weight) == 3, f"The length convolution FLOPs weight should be 3, got {len(weight)}"
        self._conv_flops_weight = weight

    def layer_wise_collection(self) -> List[SparseLayerCollection]:
        layer_weight: typing.Tuple[float, float, float] = self.get_conv_flops_weight(update=True, scaling=True)
        collection: List[SparseLayerCollection] = [SparseLayerCollection(conv_layer=self.conv1,
                                                                         bn_layer=self.bn1,
                                                                         sparse_layer=self.gate1 if self.use_gate else self.bn1,
                                                                         layer_weight=layer_weight[0]),
                                                   SparseLayerCollection(conv_layer=self.conv2,
                                                                         bn_layer=self.bn2,
                                                                         sparse_layer=self.gate2 if self.use_gate else self.bn2,
                                                                         layer_weight=layer_weight[1]),
                                                   SparseLayerCollection(conv_layer=self.conv3,
                                                                         bn_layer=self.bn3,
                                                                         sparse_layer=self.gate3 if self.use_gate else self.bn3,
                                                                         layer_weight=layer_weight[2])]
        return collection
        pass


class ResNetExpand(nn.Module):
    def __init__(self, gate: bool, layers=None, cfg=None, downsample_cfg=None, mask=False, aux_fc=False,
                 width_multiplier=1.0, expand_idx=None):
        super(ResNetExpand, self).__init__()

        if mask:
            raise NotImplementedError("do not support mask layer")

        if layers is None:
            # resnet 50
            layers = [3, 4, 6, 3]
        if len(layers) != 4:
            raise ValueError("resnet should be 4 blocks")

        # if width_multiplier != 1.0 and cfg is not None:
        #     raise ValueError("custom cfg is conflict with width_multiplier")
        self.width_multiplier = width_multiplier

        self.flops_weight_computed = False  # if the conv_flops_weight has been computed

        self.use_gate = gate

        block = Bottleneck

        # config format:
        # the config of each bottleneck is a list with length = 4
        # [conv1_out, conv2_out, conv3_out, conv1_in]
        self._cfg_len = 4
        default_cfg = self._default_config(layers)
        for i in range(len(default_cfg)):
            # multiply the width_multiplier
            for j in range(len(default_cfg[i])):
                default_cfg[i][j] = int(default_cfg[i][j] * width_multiplier)

            if len(default_cfg[i]) != 1:  # skip the first layer (model.conv1)
                # check the config length
                if len(default_cfg[i]) != self._cfg_len:
                    raise ValueError(f"Each block should have {self._cfg_len} layer, got {len(default_cfg[i])}")

        # flatten the config
        default_cfg = [item for sub_list in default_cfg for item in sub_list]

        default_downsample_cfg = [256, 512, 1024, 2048]
        if not downsample_cfg:
            downsample_cfg = default_downsample_cfg
        for i in range(len(downsample_cfg)):
            downsample_cfg[i] = int(downsample_cfg[i] * width_multiplier)
        assert len(downsample_cfg) == len(default_downsample_cfg)

        if cfg is None:
            # Construct config variable.
            cfg = default_cfg
        assert len(cfg) == len(default_cfg), f"Config length error! Expected {len(default_cfg)}, got {len(cfg)}"

        # dimension of residuals
        # self.planes = [256, 512, 1024, 2048]
        # assert len(self.planes) == 4

        self.conv1 = nn.Conv2d(3, cfg[0], kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(cfg[0])
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block,
                                       inplanes=cfg[0],
                                       outplanes=int(256 * width_multiplier),
                                       blocks=layers[0],
                                       downsample_cfg=downsample_cfg[0],
                                       cfg=cfg[1:self._cfg_len * layers[0] + 1],
                                       downsample_out=int(256 * width_multiplier))
        self.layer2 = self._make_layer(block,
                                       inplanes=int(256 * width_multiplier),
                                       outplanes=int(512 * width_multiplier),
                                       blocks=layers[1],
                                       downsample_cfg=downsample_cfg[1],
                                       cfg=cfg[self._cfg_len * layers[0] + 1:self._cfg_len * sum(layers[0:2]) + 1],
                                       stride=2,
                                       downsample_out=int(512 * width_multiplier))
        self.layer3 = self._make_layer(block,
                                       inplanes=int(512 * width_multiplier),
                                       outplanes=int(1024 * width_multiplier),
                                       blocks=layers[2],
                                       downsample_cfg=downsample_cfg[2],
                                       cfg=cfg[
                                           self._cfg_len * sum(layers[0:2]) + 1:self._cfg_len * sum(layers[0:3]) + 1],
                                       stride=2,
                                       downsample_out=int(1024 * width_multiplier))
        self.layer4 = self._make_layer(block,
                                       inplanes=int(1024 * width_multiplier),
                                       outplanes=int(2048 * width_multiplier),
                                       blocks=layers[3],
                                       downsample_cfg=downsample_cfg[3],
                                       cfg=cfg[
                                           self._cfg_len * sum(layers[0:3]) + 1:self._cfg_len * sum(layers[0:4]) + 1],
                                       stride=2,
                                       downsample_out=int(2048 * width_multiplier))
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(int(2048 * width_multiplier), 1000)

        self.enable_aux_fc = aux_fc
        if aux_fc:
            self.aux_fc_layer = nn.Linear(1024, 1000)
            raise DeprecationWarning("The aux_fc is deprecated. Do not use.")
        else:
            self.aux_fc_layer = None

        if expand_idx:
            # set channel expand index
            if expand_idx is not None:
                for m_name, sub_module in self.named_modules():
                    if isinstance(sub_module, models.common.ChannelOperation):
                        sub_module.idx = expand_idx[m_name]

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()

        # init SparseGate parameter
        for m in self.modules():
            # avoid the conv be initialized as normal
            # this line should be after the conv initialization
            if isinstance(m, SparseGate):
                nn.init.constant_(m.weight, 1)

    def _make_layer(self, block, inplanes, outplanes, blocks, cfg, downsample_cfg, downsample_out, stride=1):
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, downsample_cfg,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(downsample_cfg),
            ChannelExpand(downsample_out),
        )

        layers = []
        layers.append(block(inplanes=inplanes, outplanes=outplanes,
                            cfg=cfg[:self._cfg_len], stride=stride, downsample=downsample,
                            expand=True, gate=self.use_gate))
        for i in range(1, blocks):
            layers.append(block(inplanes=downsample_out, outplanes=outplanes,
                                cfg=cfg[self._cfg_len * i:self._cfg_len * (i + 1)], expand=True, gate=self.use_gate))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8

        if self.enable_aux_fc:
            x_aux = self.avgpool(x)
            x_aux = x_aux.view(x_aux.size(0), -1)
            x_aux = self.aux_fc_layer(x_aux)
        else:
            x_aux = None

        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x, x_aux

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
            if isinstance(sub_module, Bottleneck):
                sub_module.do_pruning(**kwargs)

    def config(self, flatten=True):
        cfg = [[self.conv1.out_channels]]
        for submodule in self.modules():
            if isinstance(submodule, Bottleneck):
                cfg.append(submodule.config())

        if flatten:
            flatten_cfg = []
            for sublist in cfg:
                for item in sublist:
                    flatten_cfg.append(item)

            return flatten_cfg

        return cfg

    def expand_idx(self) -> typing.Dict[str, np.ndarray]:
        """get the idx dict of ChannelExpand layers"""
        expand_idx: typing.Dict[str, np.ndarray] = {}
        for m_name, sub_module in self.named_modules():
            if isinstance(sub_module, models.common.ChannelOperation):
                expand_idx[m_name] = sub_module.idx

        return expand_idx

    def get_sparse_layer(self, gate: bool, sparse1: bool, sparse2: bool, sparse3: bool, with_weight=False) -> \
            typing.Union[List[nn.Module], typing.Tuple[list, list]]:
        """
        get all sparse layers
        :param gate: if gate is enabled.
        :param with_weight: return the corresponding weight of each layer. Note: The weight is rescaled to [0, 1]
        """
        assert gate == self.use_gate

        sparse_modules = []
        sparse_weights = []
        for m_name, sub_module in self.named_modules():
            # only support bottleneck
            # do not support ChannelMask!
            # do not apply sparsity on downsample layers
            if isinstance(sub_module, Bottleneck):
                sub_module: Bottleneck
                if with_weight:
                    sub_module_weight = sub_module.get_conv_flops_weight(update=True, scaling=True)
                else:
                    sub_module_weight = [None] * 3

                if gate:
                    if sparse1:
                        sparse_modules.append(sub_module.gate1)
                        sparse_weights.append(sub_module_weight[0])
                    if sparse2:
                        sparse_modules.append(sub_module.gate2)
                        sparse_weights.append(sub_module_weight[1])
                    if sparse3:
                        sparse_modules.append(sub_module.gate3)
                        sparse_weights.append(sub_module_weight[2])

                    # assert gate is enabled
                    assert sub_module.use_gate
                    assert isinstance(sub_module.gate1, SparseGate)
                    assert isinstance(sub_module.gate2, SparseGate)
                    assert isinstance(sub_module.gate3, SparseGate)
                else:
                    if sparse1:
                        sparse_modules.append(sub_module.bn1)
                        sparse_weights.append(sub_module_weight[0])
                    if sparse2:
                        sparse_modules.append(sub_module.bn2)
                        sparse_weights.append(sub_module_weight[1])
                    if sparse3:
                        sparse_modules.append(sub_module.bn3)
                        sparse_weights.append(sub_module_weight[2])

        assert len(sparse_modules) != 0, "Nothing to return"

        if with_weight:
            return sparse_modules, sparse_weights
        else:
            return sparse_modules

    def get_sparse_weight(self, sparse1: bool = True, sparse2: bool = True, sparse3: bool = True):
        """get the sparse kernels (all conv modules) of ResNet"""
        sparse_modules = []
        for m_name, sub_module in self.named_modules():
            # only support bottleneck
            # do not support ChannelMask!
            # do not apply sparsity on downsample layers
            if isinstance(sub_module, Bottleneck):
                sub_module: Bottleneck
                if sparse1:
                    sparse_modules.append(sub_module.conv1)
                if sparse2:
                    sparse_modules.append(sub_module.conv2)
                if sparse3:
                    sparse_modules.append(sub_module.conv3)

        return sparse_modules

    def get_output_gate(self) -> List[SparseGate]:
        """
        get output SparseGate layers, do not update the output gate gradient in keep_out mode
        :return:
        """
        if not self.gate:
            return []

        sparse_modules = []
        for m_name, sub_module in self.named_modules():
            # only support bottleneck
            if isinstance(sub_module, Bottleneck):
                sub_module: Bottleneck
                sparse_modules.append(sub_module.gate3)

                # assert gate is enabled
                assert sub_module.use_gate
                assert isinstance(sub_module.gate3, SparseGate)

        return sparse_modules

    def get_conv_flops_weight(self,
                              input_size: typing.Tuple[int, int] = (224, 224)) -> List[typing.Tuple[int, int, int]]:
        """
        :param input_size: the input spatial size of the network
        """
        return compute_conv_flops_weight(self, building_block=Bottleneck, input_size=input_size)

    def set_conv_flops_weight(self, weight: List[typing.Tuple[int, int, int]],
                              input_size: typing.Tuple[int, int] = (224, 224)):
        for submodule in self.modules():
            if isinstance(submodule, Bottleneck):
                submodule: Bottleneck
                submodule.conv_flops_weight = weight.pop(0)

        self.input_size = input_size

    def layer_wise_collection(self) -> List[SparseLayerCollection]:
        collection = []
        for submodule in self.modules():
            if isinstance(submodule, Bottleneck):
                submodule: Bottleneck
                collection += submodule.layer_wise_collection()

        return collection

    def _default_config(self, layers: List[int]) -> List[List[int]]:
        # the output dimension of the conv1 in the network (NOT the block)
        conv1_output = 64

        default_cfg = [[64, 64, 256] for _ in range(layers[0])] + \
                      [[128, 128, 512] for _ in range(layers[1])] + \
                      [[256, 256, 1024] for _ in range(layers[2])] + \
                      [[512, 512, 2048] for _ in range(layers[3])]

        input_dim = conv1_output
        for block_cfg in default_cfg:
            # the conv_in is as same as the input dim by default (ChannelSelection selects all channels)
            block_cfg.append(input_dim)
            input_dim = block_cfg[2]

        default_cfg = [[conv1_output]] + default_cfg

        return default_cfg


def resnet50(aux_fc, width_multiplier, gate: bool):
    if aux_fc is True:
        raise ValueError("Auxiliary fully connected layer is deprecated.")
    model = ResNetExpand(width_multiplier=width_multiplier,
                         layers=[3, 4, 6, 3], aux_fc=aux_fc,
                         gate=gate)
    return model


def _check_models(model, ref_model, threshold=1e-5, hook=False, cuda=False):
    """Assert two model is same"""
    ref_state_dict = ref_model.state_dict()
    # if gate is enabled, do not load gate parameter in the state_dict
    if model.use_gate:
        # there is no gate parameters in ref model
        # allow missing parameters in SparseGate
        for key in model.state_dict().keys():
            if key not in ref_state_dict:
                assert "_conv" in key, f"only allow missing parameters of SparseGate, got {key}"
                ref_state_dict[key] = model.state_dict()[key]

    # use same parameter
    model.load_state_dict(ref_state_dict, strict=False)

    # do not update running_mean and running_var of the bn layers
    model.eval()
    ref_model.eval()

    with torch.no_grad():
        random_input = torch.rand(8, 3, 224, 224)

        if cuda:
            model.cuda()
            ref_model.cuda()
            random_input = random_input.cuda()

        if hook:
            ref_model_intermediate_variables = []
            model_intermediate_variables = []

            def conv_hook(module, conv_input, conv_output, output_list: list):
                """
                save all intermediate values for debug
                only enable it when the testing failed
                """
                output_list.append((module, conv_input, conv_output))

            for sub_module in ref_model.modules():
                sub_module.register_forward_hook(
                    lambda module, conv_input, conv_output:
                    conv_hook(module, conv_input, conv_output, ref_model_intermediate_variables))

            for sub_module in model.modules():
                sub_module.register_forward_hook(
                    lambda module, conv_input, conv_output:
                    conv_hook(module, conv_input, conv_output, model_intermediate_variables))

        torch_output = ref_model(random_input)
        cur_output, extra_info = model(random_input)

        diff = torch_output - cur_output
        max_diff = torch.max(diff.abs().view(-1)).item()
        print(f"Max diff: {max_diff}")

    return max_diff < threshold


def _test():
    """unit test of ResNet-50 model"""

    from torchvision.models import resnet50 as torch_resnet50

    print("########## Unit test of the ResNet-50 with ChannelExpand ##########")

    # test default config
    print("Testing default config")
    model = resnet50(width_multiplier=1.0, gate=False, aux_fc=False)
    assert _check_models(model, torch_resnet50(), cuda=True), "default config testing failed."
    print()

    # test default config with gate enabled
    print("Testing default config with gate enabled")
    model = resnet50(width_multiplier=1.0, gate=True, aux_fc=False)
    assert _check_models(model, torch_resnet50(), hook=False, cuda=True), "gate enabled testing failed"

    print("Test passed!")


if __name__ == '__main__':
    _test()
