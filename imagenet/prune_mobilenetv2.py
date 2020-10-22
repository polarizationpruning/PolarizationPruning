import argparse
import copy
import os
from typing import Dict, Any

import numpy as np
import torch
from torch.nn import Module

import models
import models.common
import utils.checkpoints
from models.common import ThresholdPruner, Pruner, RandomPruner
from models.mobilenet import mobilenet_v2, InvertedResidual, MobileNetV2
from utils import common
from utils.common import compute_conv_flops


def _get_parser():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR prune')
    parser.add_argument('--model', default='', type=str, metavar='PATH',
                        help='path to the model (default: none)')
    parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                        help='input batch size for testing (default: 256)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--percent', type=float, default=1,
                        help='scale sparse rate (default: 1.)')
    parser.add_argument('--save', default=None, type=str, metavar='PATH',
                        help='path to save pruned model (default: none)')
    parser.add_argument("--pruning-strategy", type=str,
                        choices=["percent", "fixed", "grad", "search", "random"],
                        help="Pruning strategy", required=True)
    parser.add_argument('--width-multiplier', default=1.0, type=float,
                        help="The width multiplier for MobileNet v2. "
                             "Unavailable for other networks. (default 1.0) "
                             "Must same with sparse training.")
    parser.add_argument('--same', action='store_true',
                        help='The model before pruning and after pruning is required to be exactly the same')
    parser.add_argument('--gate', action='store_true',
                        help='Add gate after the BatchNrom layers. Only available for MobileNet v2!')
    parser.add_argument('--ratio', default=None, type=float,
                        help="Uniformly prune the model by channel ratio. Only available in `random` mode")

    return parser


def _prune_mobilenetv2_inplace(sparse_model: torch.nn.Module, pruner: Pruner):
    in_channel = sparse_model.input_channel
    for module_name, sub_module in sparse_model.named_modules():
        if isinstance(sub_module, InvertedResidual):
            in_mask = np.ones(in_channel)
            in_channel = sub_module.do_pruning(in_channel_mask=in_mask,
                                               pruner=pruner)


def compute_global_threshold(model, percent: float) -> int:
    if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module

    model: MobileNetV2
    sparse_layers = model.get_sparse_layer(gate=model.gate,
                                           pw_layer=True,
                                           linear_layer=True,
                                           with_weight=False)
    # flatten all weights and concat them
    sparse_layers = filter(lambda l: l is not None, sparse_layers)
    sparse_weights: np.ndarray = np.concatenate(
        list(map(lambda l: l.weight.view(-1).data.cpu().numpy(), sparse_layers)))
    sparse_weights = np.sort(sparse_weights)

    threshold_index = int(len(sparse_weights) * percent)
    threshold = sparse_weights[threshold_index]

    return threshold


def prune_mobilenet(sparse_model: Module, pruning_strategy: str,
                    sanity_check: bool, force_same: bool, width_multiplier: float, ratio=None, percent:float=None):
    """

    :param sparse_model: the large model with sparsity
    :param pruning_strategy: same as `models.common.search_threshold`
    :param sanity_check: if do sanity check after pruning
    :param force_same: force the pruned model and the sparse model same
    :param width_multiplier: the width_multiplier of the model
    :param ratio: Pruning ratio when pruning uniformly. Only used in `pruning_strategy` is `"random"`
    :param percent: Pruning percent when use global threshold pruning

    :return: the model to be saved for fine-tuning

    """
    if isinstance(sparse_model, torch.nn.DataParallel) or isinstance(sparse_model,
                                                                     torch.nn.parallel.DistributedDataParallel):
        sparse_model: models.mobilenet.MobileNetV2 = sparse_model.module
    gate = sparse_model.gate

    pruned_model = copy.deepcopy(sparse_model)

    # For percent pruning (prune globally), we need to compute the globally threshold
    if pruning_strategy == 'percent':
        global_threshold = compute_global_threshold(sparse_model, percent)
        pruner = ThresholdPruner(pruning_strategy, threshold=global_threshold)
    elif pruning_strategy == 'random':
        pruner = RandomPruner(ratio=ratio)
    else:
        pruner = ThresholdPruner(pruning_strategy)

    _prune_mobilenetv2_inplace(pruned_model,
                               pruner=pruner)
    pruned_model.eval()

    # save pruned model
    # extract idx of ChannelExpand and ChannelSelection layers
    expand_idx: Dict[str, np.ndarray] = {}
    for m_name, sub_module in pruned_model.named_modules():
        if isinstance(sub_module, models.common.ChannelOperation):
            expand_idx[m_name] = sub_module.idx

    # remove the SparseGate from state_dict
    pruned_state_dict = pruned_model.state_dict()

    # Test loading the config to a new model
    # test if the saved model could be loaded correctly
    # new model is built without SparseGate
    saved_model = build_new_model(pruned_state_dict, pruned_model.get_config(), expand_idx,
                                  width_multiplier, gate=gate)

    if sanity_check:
        # check the network is unchanged
        demo_input = torch.rand(8, 3, 224, 224)
        original_out = sparse_model(demo_input)
        pruned_out = pruned_model(demo_input)
        diff = torch.max((original_out - pruned_out).abs().view(-1)).item()
        if force_same:
            assert diff < 1e-5, f"difference should be less than 1e-5, got {diff}"
            print("Test passed. The output of pruned model and the original model is same.")
        else:
            print(f"Max diff between unpruned model and pruned model: {diff}")

        # Test if new model is same as pruned model
        saved_model.eval()  # avoid random in dropout
        new_model_out = saved_model(demo_input)
        diff = torch.max((new_model_out - pruned_out).abs().view(-1)).item()
        print(f"Diff between pruned model and re-loaded model is {diff}")

        # since the SparseGate is multiplied on the bn weight and bias, the
        # gate should be moved without any side effect
        assert diff < 1e-5, f"difference between pruned model and loaded model should be less than 1e-5, got {diff}"

        # check: the saved_model should not contains any SparseGate
        for name, sub_module in saved_model.named_modules():
            if isinstance(sub_module, models.common.SparseGate):
                if 'input_gate' in name:
                    # The input gate should be kept in fine-tuning stage
                    continue
                raise AssertionError(f"saved_model should not contains any SparseGate modules, got {name}")

    return saved_model, pruned_model, expand_idx


def build_new_model(state_dict, cfg, expand_idx, width_multiplier, gate=False):
    # use input mask in each block to prune the input channel for each block
    model = mobilenet_v2(inverted_residual_setting=cfg,
                         width_mult=width_multiplier, input_mask=gate)
    # make sure forward pass will not change the network parameters, e.g., bn mean and var
    model.eval()

    # allow unexpected keys of SparesGate
    if gate:
        state_dict_wo_sparse_gate = {}
        for key in model.state_dict().keys():
            if key in state_dict:
                state_dict_wo_sparse_gate[key] = state_dict[key]
        state_dict = state_dict_wo_sparse_gate

    model.load_state_dict(state_dict)
    # restore the mask of the Expand layer
    for m_name, sub_module in model.named_modules():
        if isinstance(sub_module, models.common.ChannelOperation):
            sub_module.idx = expand_idx[m_name]

    # TEST: do forward pass
    random_input = torch.rand(8, 3, 224, 224)
    model(random_input)

    return model
    pass


def main():
    # reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    parser = _get_parser()
    args = parser.parse_args()

    if args.save is not None and not os.path.exists(args.save):
        os.makedirs(args.save)

    print(args)
    print(f"Current git hash: {common.get_git_id()}")

    if not os.path.isfile(args.model):
        raise ValueError("=> no checkpoint found at '{}'".format(args.model))

    checkpoint: Dict[str, Any] = torch.load(args.model)
    print(f"=> Loading the model...\n=> Epoch: {checkpoint['epoch']}, Acc.: {checkpoint['best_prec1']}")
    utils.checkpoints.remove_module(checkpoint)  # remove the module wrapper
    if 'cfg' not in checkpoint:
        # Compatible with previous versions, which does not contains cfg field in checkpoint
        checkpoint['cfg'] = None

    # build the sparse model
    sparse_model = mobilenet_v2(width_mult=args.width_multiplier,
                                use_gate=args.gate,
                                inverted_residual_setting=checkpoint['cfg'])
    sparse_model.load_state_dict(checkpoint['state_dict'])

    # avoid the random in dropout
    # avoid the mean and var change in bn
    sparse_model.eval()

    # set ChannelExpand index
    if 'expand_idx' in checkpoint and checkpoint['expand_idx'] is not None:
        expand_idx = checkpoint['expand_idx']
        for m_name, sub_module in sparse_model.named_modules():
            if isinstance(sub_module, models.common.ChannelExpand):
                sub_module.idx = expand_idx[m_name]
    else:
        assert checkpoint['cfg'] is None, "expand_idx is required when given a pruned model"

    saved_model, pruned_model, expand_idx = prune_mobilenet(sparse_model=sparse_model,
                                                            pruning_strategy=args.pruning_strategy,
                                                            sanity_check=True, force_same=args.same,
                                                            width_multiplier=args.width_multiplier,
                                                            ratio=args.ratio, percent=args.percent)

    # compute the flops
    # when --gate is enabled, the pruned_model will contains SparseGates, which gives more FLOPs
    original_flops = compute_conv_flops(sparse_model, cuda=True)
    saved_flops = compute_conv_flops(saved_model, cuda=True)

    if args.width_multiplier != 1.0:
        print(f"WARNING: the FLOPs is not the baseline FLOPs with width multiplier {args.width_multiplier}")
    print(f"Unpruned FLOPs: {original_flops:,}")
    print(f"Saved FLOPs: {saved_flops:,}")
    print(f"FLOPs ratio: {saved_flops / original_flops:,}")

    if args.save is not None:
        # the state_dict for fine-tuning
        torch.save({'state_dict': saved_model.state_dict(),
                    'cfg': pruned_model.get_config(),
                    'gate': args.gate,  # if apply sparsity on gates
                    "expand_idx": expand_idx},
                   os.path.join(args.save, f'pruned_{args.pruning_strategy}.pth.tar'))

        # save the state_dict with SparseGate parameters.
        # this model could be used for multi-run pruning.
        multi_run_state_dict = {'state_dict': pruned_model.state_dict(),
                                'cfg': pruned_model.get_config(),
                                "expand_idx": expand_idx}
        for key in checkpoint.keys():
            # the model need more keys (e.g., optimizers, best acc, ...)
            # copy all other fields from the original checkpoint
            if key not in multi_run_state_dict:
                multi_run_state_dict[key] = checkpoint[key]
        torch.save(multi_run_state_dict,
                   os.path.join(args.save, f'pruned_{args.pruning_strategy}_gate.pth.tar'))


if __name__ == '__main__':
    main()
