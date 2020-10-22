"""the common component of different networks"""
import typing
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import torch
from torch import nn


class SparseGate(nn.Module):
    def __init__(self, channel_num: int):
        super().__init__()
        self._conv = nn.Conv2d(in_channels=channel_num,
                               out_channels=channel_num,
                               kernel_size=1,
                               groups=channel_num,
                               bias=False)

        # initialize as ones
        nn.init.constant_(self.weight, 1)

        # since the conv layer is already initialized, simply change this variable will change nothing
        # the channel_num should never be changed
        self._channel_num = channel_num

    def forward(self, x):
        return self._conv(x)

    @property
    def weight(self) -> torch.nn.Parameter:
        """
        The weight of the gate.

        NOTE: the method does not promise the size of the weight. Make sure do .view(-1)
        before using it as a scaling factor.
        """
        return self._conv.weight

    def clamp(self, upper_bound=1., lower_bound=0.):
        self.weight.data.clamp_(lower_bound, upper_bound)

    def prune(self, idx: np.ndarray) -> None:
        assert len(idx.shape) == 1, f"channel index is supposed to be a 1-d vector, got shape {idx.shape}"

        self.weight.data = self.weight.data[idx.tolist(), :, :, :].clone()
        self._conv.in_channels = len(idx)
        self._conv.out_channels = len(idx)
        self._conv.groups = len(idx)
        pass

    def set_ones(self):
        """
        Set the weight to all-one vector.
        This operation actually disable the SparseGate layer.
        After calling the method, this layer is actually an identity mapping.
        """
        self._conv.weight.data = torch.ones_like(self._conv.weight)

    def do_pruning(self, mask: np.ndarray):
        idx = np.squeeze(np.argwhere(np.asarray(mask)))
        if len(idx.shape) == 0:
            # expand the single scalar to array
            idx = np.expand_dims(idx, 0)
        elif len(idx.shape) == 1 and idx.shape[0] == 0:
            # nothing left
            raise NotImplementedError("No layer left in input channel")

        self._channel_num = len(idx)

        # prune the conv layer
        self._conv.weight.data = self._conv.weight.data.clone()[idx.tolist(), :, :, :]
        self._conv.in_channels = self._channel_num
        self._conv.out_channels = self._channel_num
        self._conv.groups = self._channel_num

    def __repr__(self):
        return f"SparseGate(channel_num={self._channel_num})"
        pass


class Identity(torch.nn.Module):
    r"""
    Copy from PyTorch nn.Identity, to replace the nn.Identity in
    PyTorch <= 1.1

    A placeholder identity operator that is argument-insensitive.

    Args:
        args: any argument (unused)
        kwargs: any keyword argument (unused)

    Examples::

        >>> m = nn.Identity(54, unused_argument1=0.1, unused_argument2=False)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 20])

    """

    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input


class Pruner(ABC):
    @abstractmethod
    def __call__(self, weight: np.ndarray, **kwargs):
        """Given a sparse weight, return a mask"""
        pass


def prune_conv_layer(conv_layer: nn.Conv2d,
                     bn_layer: nn.BatchNorm2d,
                     sparse_layer_out: typing.Optional[Union[nn.BatchNorm2d, SparseGate]],
                     in_channel_mask: typing.Optional[np.ndarray],
                     prune_output_mode: str,
                     pruner: Pruner,
                     prune_mode: str,
                     sparse_layer_in: typing.Optional[Union[nn.BatchNorm2d, SparseGate]], ) -> typing.Tuple[
    np.ndarray, np.ndarray]:
    """
    Note: if the sparse_layer is SparseGate, the gate will be replaced by BatchNorm
    scaling factor. The value of the gate will be set to all ones.

    :param prune_output_mode: how to handle the output channel (case insensitive)
        "keep": keep the output channel intact
        "same": keep the output channel as same as input channel
        "prune": prune the output according to bn_layer
    :param pruner: the method to determinate the pruning mask.
    :param in_channel_mask: a 0-1 vector indicates whether the corresponding channel should be pruned (0) or not (1)
    :param bn_layer: the BatchNorm layer after the convolution layer
    :param conv_layer: the convolution layer to be pruned

    :param sparse_layer_out: the layer to determine the output sparsity. Support BatchNorm2d and SparseGate.
    :param sparse_layer_in: the layer to determine the input sparsity. Support BatchNorm2d and SparseGate.
        When the `sparse_layer_in` is None, there is no input sparse layers,
        the input channel will be determined by the `in_channel_mask`
        Note: `in_channel_mask` is CONFLICT with `sparse_layer_in`!

    :param prune_mode: pruning mode (`str`):
        - `"multiply"`: pruning threshold is determined by the multiplication of `sparse_layer` and `bn_layer`
            only available when `sparse_layer` is `SparseGate`
        - `None` or `"default"`: default behaviour. The pruning threshold is determined by `sparse_layer`
    :return out_channel_mask
    """
    assert isinstance(conv_layer, nn.Conv2d), f"conv_layer got {conv_layer}"

    assert isinstance(sparse_layer_out, nn.BatchNorm2d) or isinstance(sparse_layer_out,
                                                                      SparseGate), f"sparse_layer got {sparse_layer_out}"
    if in_channel_mask is not None and sparse_layer_in is not None:
        raise ValueError("Conflict option: in_channel_mask and sparse_layer_in")

    prune_mode = prune_mode.lower()
    prune_output_mode = str.lower(prune_output_mode)

    if prune_mode == 'multiply':
        if not isinstance(sparse_layer_out, SparseGate):
            raise ValueError(f"Do not support prune_mode {prune_mode} when the sparse_layer is {sparse_layer_out}")

    with torch.no_grad():
        conv_weight: torch.Tensor = conv_layer.weight.data.clone()

        # prune the input channel of the conv layer
        # if sparse_layer_in and in_channel_mask are both None, the input dim will NOT be pruned
        if sparse_layer_in is not None:
            if in_channel_mask is not None:
                raise ValueError("")
            sparse_weight_in: np.ndarray = sparse_layer_in.weight.view(-1).data.cpu().numpy()
            # the in_channel_mask will be overwrote
            in_channel_mask = pruner(sparse_weight_in)

        if in_channel_mask is not None:
            # prune the input channel according to the in_channel_mask
            # convert mask to channel indexes
            idx_in = np.squeeze(np.argwhere(np.asarray(in_channel_mask)))
            if len(idx_in.shape) == 0:
                # expand the single scalar to array
                idx_in = np.expand_dims(idx_in, 0)
            elif len(idx_in.shape) == 1 and idx_in.shape[0] == 0:
                # nothing left, prune the whole block
                out_channel_mask = np.full(conv_layer.out_channels, False)
                return in_channel_mask, out_channel_mask

        # prune the input of the conv layer
        if conv_layer.groups == 1:
            conv_weight = conv_weight[:, idx_in.tolist(), :, :]
        else:
            assert conv_weight.shape[1] == 1, "only works for groups == num_channels"

        # prune the output channel of the conv layer
        if prune_output_mode == "prune":
            # the sparse_layer.weight need to be flatten, because the weight of SparseGate is not 1d
            sparse_weight_out: np.ndarray = sparse_layer_out.weight.view(-1).data.cpu().numpy()
            if prune_mode == 'multiply':
                bn_weight = bn_layer.weight.data.cpu().numpy()
                sparse_weight_out = sparse_weight_out * bn_weight  # element-wise multiplication
            elif prune_mode != 'default':
                raise ValueError(f"Do not support prune_mode {prune_mode}")

            # prune and get the pruned mask
            out_channel_mask = pruner(sparse_weight_out)
        elif prune_output_mode == "keep":
            # do not prune the output
            out_channel_mask = np.ones(conv_layer.out_channels)
        elif prune_output_mode == "same":
            # prune the output channel with the input mask
            # keep the conv layer in_channel == out_channel
            out_channel_mask = in_channel_mask
        else:
            raise ValueError(f"invalid prune_output_mode: {prune_output_mode}")

        idx_out: np.ndarray = np.squeeze(np.argwhere(np.asarray(out_channel_mask)))
        if len(idx_out.shape) == 0:
            # expand the single scalar to array
            idx_out = np.expand_dims(idx_out, 0)
        elif len(idx_out.shape) == 1 and idx_out.shape[0] == 0:
            # no channel left
            # return mask directly
            # the block is supposed to be set as a identity mapping
            return out_channel_mask, in_channel_mask
        conv_weight = conv_weight[idx_out.tolist(), :, :, :]

        # change the property of the conv layer
        conv_layer.in_channels = len(idx_in)
        conv_layer.out_channels = len(idx_out)
        conv_layer.weight.data = conv_weight
        if conv_layer.groups != 1:
            # set the new groups for dw layer
            conv_layer.groups = conv_layer.in_channels
            pass

        # prune the bn layer
        bn_layer.weight.data = bn_layer.weight.data[idx_out.tolist()].clone()
        bn_layer.bias.data = bn_layer.bias.data[idx_out.tolist()].clone()
        bn_layer.running_mean = bn_layer.running_mean[idx_out.tolist()].clone()
        bn_layer.running_var = bn_layer.running_var[idx_out.tolist()].clone()

        # set bn properties
        bn_layer.num_features = len(idx_out)

        # prune the gate
        if isinstance(sparse_layer_out, SparseGate):
            sparse_layer_out.prune(idx_out)
            # multiply the bn weight and SparseGate weight
            sparse_weight_out: torch.Tensor = sparse_layer_out.weight.view(-1)
            bn_layer.weight.data = (bn_layer.weight.data * sparse_weight_out).clone()
            bn_layer.bias.data = (bn_layer.bias.data * sparse_weight_out).clone()
            # the function of the SparseGate is now replaced by bn layers
            # the SparseGate should be disabled
            sparse_layer_out.set_ones()

    return out_channel_mask, in_channel_mask


class ThresholdPruner(Pruner):
    @staticmethod
    def _search_threshold(weight: np.ndarray, alg: str):
        if alg not in ["fixed", "grad", "search"]:
            raise NotImplementedError()

        hist_y, hist_x = np.histogram(weight, bins=100, range=(0, 1))
        if alg == "search":
            raise ValueError(f"Deprecated pruning algorithm: {alg}")
        elif alg == "grad":
            hist_y_diff = np.diff(hist_y)
            for i in range(len(hist_y_diff) - 1):
                if hist_y_diff[i] <= 0 <= hist_y_diff[i + 1]:
                    threshold = hist_x[i + 1]
                    if threshold > 0.2:
                        print(f"WARNING: threshold might be too large: {threshold}")
                    return threshold
        elif alg == "fixed":
            return hist_x[1]

    def __init__(self, alg: str, **kwargs):
        self._alg = alg

        if alg == 'percent':
            if 'threshold' not in kwargs:
                raise ValueError("For percent pruning, the threshold must be specified.")
            # the threshold is fixed for all layers
            self._global_threshold = kwargs['threshold']

    def __call__(self, weight: np.ndarray, **kwargs):
        if self._alg == 'percent':
            threshold = self._global_threshold
        else:
            threshold = self._search_threshold(weight, self._alg)
        mask = weight > threshold
        return mask


class RandomPruner(Pruner):
    def __init__(self, ratio: float):
        """
        :param ratio: pruned channel / total channel
        """
        if ratio > 1 or ratio <= 0:
            raise ValueError("Expected 0 < ratio <= 1, got {}".format(ratio))
        self._ratio = ratio
        pass

    def __call__(self, weight: np.ndarray, **kwargs):
        mask_len = len(weight)
        pruned_channel_num = int(mask_len * self._ratio)

        # generate a random mask
        mask = np.full((mask_len,), True)
        mask[:pruned_channel_num] = False
        np.random.shuffle(mask)

        return mask


def _conv_raw_weight_hook(conv: nn.Conv2d, input_data: torch.Tensor, output_data: torch.Tensor):
    """
    a hook to set `d_flops_in` and `d_flops_out` for each convolution layers

    `d_flops_in`: the FLOPs drops when the input channel drops by 1

    `d_flops_out`: the FLOPs drops when the output channel drops by 1
    """
    if conv.groups != 1:
        # for SparseGate and deep-wise layer in MobileNet v2
        # note the `d_flops_in` and `d_flops_out` of SparseGate should NOT be used

        # in MobileNet v2, the groups will change according to the input channel and output channel
        assert conv.groups == conv.in_channels and conv.groups == conv.out_channels

    output_channels, output_height, output_width = output_data[0].size()

    if conv.groups == 1:
        new_conv_groups = conv.groups
    else:
        # the conv_groups will change according to the input channel and output channel
        new_conv_groups = conv.groups - 1

    kernel_ops = conv.kernel_size[0] * conv.kernel_size[1] * (conv.in_channels / new_conv_groups)
    d_kernel_ops_in = conv.kernel_size[0] * conv.kernel_size[1] * (1 / new_conv_groups)

    # flops = kernel_ops * output_channels * output_height * output_width
    if conv.groups == 1:
        # normal conv layer
        conv.d_flops_in = d_kernel_ops_in * output_channels * output_height * output_width
        conv.d_flops_out = kernel_ops * 1 * output_height * output_width
    else:
        # for deepwise layer
        # this layer will not be pruned, so do not set d_flops_out
        conv.d_flops_in = d_kernel_ops_in * (output_channels - 1) * output_height * output_width


def compute_raw_weight(model, input_size: typing.Tuple[int, int]):
    """
    compute d_flops_in and d_flops_out for every convolutional layers in the model

    Note: this method needs to do forward pass, which time-consuming
    """
    model._flops_weight_computed = True

    # register hooks
    hook_handles = []
    for submodule in model.modules():
        if isinstance(submodule, nn.Conv2d):
            hook_handles.append(submodule.register_forward_hook(_conv_raw_weight_hook))

    # do forward pass to compute the input spatial size for each layer
    random_input = torch.rand(8, 3, *input_size)
    model(random_input)

    # remove hooks
    for h in hook_handles:
        h.remove()


def compute_conv_flops_weight(model, building_block, input_size: typing.Tuple[int, int] = (224, 224),
                              ) -> typing.List[
    typing.Tuple[int, int, int]]:
    """
    compute the conv_flops_weight for the model

    :param building_block: the basic building block for CNN.
     Use Bottleneck for ResNet-50, InvertedResidual for MobileNet v2.
    :param input_size: the input spatial size of the network
    """
    if not model.flops_weight_computed or model.input_size is None or model.input_size != input_size:
        # initialization

        # set flag
        model._flops_weight_computed = True
        model.input_size = input_size

        # update when
        # 1. the flops is never be computed
        # 2. the input size is changed
        compute_raw_weight(model, input_size)  # compute d_flops_in and d_flops_out

        # the weight is raw weight (without scaling!)
        # now compute the weight min and max for rescaling
        conv_flops_weight_raw: typing.List[typing.Tuple[int, int, int]] = []
        for submodule in model.modules():
            if isinstance(submodule, building_block):
                submodule: building_block
                block_raw_weight = submodule.get_conv_flops_weight(update=True, scaling=False)
                if block_raw_weight[0] is None:
                    # In MobileNet v2, there might be no pw layer, in this case, its weight is None
                    assert len(block_raw_weight) == 2, "Only MobileNet v2 has None weight"
                    block_raw_weight = block_raw_weight[1:]
                conv_flops_weight_raw.append(block_raw_weight)
        # scale weight to [0, 1]
        # set weight_max and weight_min for each building blocks
        weights = []
        for block in conv_flops_weight_raw:
            for i in block:
                weights.append(i)

        max_weights = max(weights)
        min_weights = min(weights)

        # compute the min_weights and max_weights for rescaling
        for submodule in model.modules():
            if isinstance(submodule, building_block):
                submodule: building_block
                submodule.raw_weight_min = min_weights
                submodule.raw_weight_max = max_weights

    conv_flops_weight: typing.List[typing.Tuple[int, int, int]] = []
    for submodule in model.modules():
        if isinstance(submodule, building_block):
            submodule: building_block
            conv_flops_weight.append(submodule.conv_flops_weight)

    return conv_flops_weight


def test_conv_flops_weight(flops_weight):
    # analysis the conv flops weight
    # for block in flops_weight:
    #     print(",".join([str(b) for b in block]))
    # do normalization
    weights = []
    for block in flops_weight:
        for i in block:
            weights.append(i)

    # None weights will be filtered
    max_weights = max(filter(lambda w: w is not None, weights))
    min_weights = min(filter(lambda w: w is not None, weights))

    print()
    for block in flops_weight:
        print(",".join(
            [str((b - min_weights) / (max_weights - min_weights)) if b is not None else "None" for b in block]))


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
