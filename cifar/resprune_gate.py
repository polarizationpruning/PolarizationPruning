import argparse
import copy
import os
import numpy as np
from typing import Any, Dict

import torch

import common
from models.common import search_threshold, l1_norm_threshold
from models.resnet_expand import resnet56 as resnet50_expand, ResNetExpand


def _get_parser():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR prune')
    # parser.add_argument('data', metavar='DIR',
    #                     help='path to dataset')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'],
                        help='training dataset (default: cifar10)')
    parser.add_argument('--model', default='', type=str, metavar='PATH',
                        help='path to the model (default: none)')
    parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                        help='input batch size for testing (default: 256)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--percent', type=float, default=0.5,
                        help='scale sparse rate (default: 0.5)')
    parser.add_argument('--save', default='', type=str, metavar='PATH',
                        help='path to save pruned model (default: none)')
    parser.add_argument("--pruning-strategy", type=str,
                        choices=["percent", "fixed", "grad", "search"],
                        help="Pruning strategy", required=True)
    parser.add_argument('--same', action='store_true',
                        help='The model before pruning and after pruning is required to be exactly the same')
    parser.add_argument('--gate', action='store_true',
                        help='Add gate after the BatchNrom layers. Only available for MobileNet v2!')
    parser.add_argument("--prune-mode", type=str, default='default',
                        choices=["multiply", 'default'],
                        help="Pruning mode. Same as `models.common.prune_conv_layer`", )
    parser.add_argument('--prune-type', type=str, default='polarization',
                        choices=['polarization', 'ns', 'l1-norm'],
                        help='Pruning method.')
    parser.add_argument('--l1-norm-ratio', type=float, default=None,
                        help='Pruning ratio of the L1-Norm')
    parser.add_argument('--input-mask', action='store_true',
                        help='If use input mask in ResNet models.')
    return parser


def _check_model_same(model1: torch.nn.Module, model2: torch.nn.Module) -> float:
    """
    check if the output is same by same input.
    """
    model1.eval()
    model2.eval()

    rand_input = torch.rand((8, 3, 32, 32))  # the same input size as CIFAR
    out1, _ = model1(rand_input)  # ignore aux output for resnet
    out2, _ = model2(rand_input)

    diff = out1 - out2
    max_diff = torch.max(diff.abs().view(-1)).item()

    return max_diff


def prune_resnet(num_classes: int, sparse_model: torch.nn.Module, pruning_strategy: str, sanity_check: bool,
                 prune_mode: str, prune_type: str = 'polarization', l1_norm_ratio=None):
    """
    :param sparse_model: The model trained with sparsity regularization
    :param pruning_strategy: same as `models.common.search_threshold`
    :param sanity_check: whether do sanity check
    :param prune_mode: same as `models.common.prune_conv_layer`
    :return:
    """
    if isinstance(sparse_model, torch.nn.DataParallel) or isinstance(sparse_model,
                                                                     torch.nn.parallel.DistributedDataParallel):
        sparse_model = sparse_model.module

    # note that pruned model could not do forward pass.
    # need to set channel expand.
    pruned_model = copy.deepcopy(sparse_model)
    pruned_model.cpu()

    if prune_type == 'polarization':
        pruner = lambda weight: search_threshold(weight, pruning_strategy)
        prune_on = 'factor'
    elif prune_type == 'l1-norm':
        pruner = lambda weight: l1_norm_threshold(weight, ratio=l1_norm_ratio)
        prune_on = 'weight'
    elif prune_type == 'ns':
        # find the threshold
        sparse_layers = pruned_model.get_sparse_layers()
        sparse_weight_concat = np.concatenate([l.weight.data.clone().view(-1).cpu().numpy() for l in sparse_layers])
        sparse_weight_concat = np.abs(sparse_weight_concat)
        sparse_weight_concat = np.sort(sparse_weight_concat)
        thre_index = int(len(sparse_weight_concat) * l1_norm_ratio)
        threshold = sparse_weight_concat[thre_index]
        pruner = lambda weight: threshold
        prune_on = 'factor'
    else:
        raise ValueError(f"Unsupport prune type: {prune_type}")

    pruned_model.prune_model(pruner=pruner,
                             prune_mode=prune_mode,
                             prune_on=prune_on)
    print("Pruning finished. cfg:")
    print(pruned_model.config())

    if sanity_check:
        # sanity check: check if pruned model is as same as sparse model
        print("Sanity check: checking if pruned model is as same as sparse model")
        max_diff = _check_model_same(sparse_model, pruned_model)
        print(f"Max diff between Sparse model and Pruned model: {max_diff}\n")

    # load weight to finetuning model
    saved_model = resnet50_expand(aux_fc=False,
                                  num_classes=num_classes,
                                  gate=False,
                                  cfg=pruned_model.config(),
                                  expand_idx=pruned_model.channel_index,
                                  use_input_mask=pruned_model.use_input_mask)

    pruned_state_dict = {}
    # remove gate param from model
    for param_name, param in pruned_model.state_dict().items():
        if param_name in saved_model.state_dict():
            pruned_state_dict[param_name] = param
        else:
            if "_conv" not in param_name:
                # when the entire block is pruned, the conv parameter will miss, which is expected
                print(f"[WARNING] missing parameter: {param_name}")

    saved_model.load_state_dict(pruned_state_dict)

    if sanity_check:
        print("Sanity check: checking if pruned model is as same as saved model")
        max_diff = _check_model_same(saved_model, pruned_model)
        print(f"Max diff between Saved model and Pruned model: {max_diff}\n")
        assert max_diff < 1e-5, f"Test failed: Max diff should be less than 1e-5, got {max_diff}"

    return saved_model


def main():
    # reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    parser = _get_parser()
    args = parser.parse_args()

    if args.dataset == "cifar10":
        num_classes = 10
    elif args.dataset == 'cifar100':
        num_classes = 100
    else:
        raise ValueError(f"Unrecognized dataset {args.dataset}")

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    print(args)
    print(f"Current git hash: {common.get_git_id()}")

    if not os.path.isfile(args.model):
        raise ValueError("=> no checkpoint found at '{}'".format(args.model))

    checkpoint: Dict[str, Any] = torch.load(args.model)
    print(f"=> Loading the model...\n=> Epoch: {checkpoint['epoch']}, Acc.: {checkpoint['best_prec1']}")

    # build the sparse model
    sparse_model: ResNetExpand = resnet50_expand(num_classes=num_classes, aux_fc=False,
                                                 gate=args.gate, use_input_mask=args.input_mask)
    sparse_model.load_state_dict(checkpoint['state_dict'])

    saved_model = prune_resnet(num_classes=num_classes,
                               sparse_model=sparse_model,
                               pruning_strategy=args.pruning_strategy,
                               sanity_check=True, prune_mode=args.prune_mode,
                               prune_type=args.prune_type,
                               l1_norm_ratio=args.l1_norm_ratio)

    # compute FLOPs
    baseline_flops = common.compute_conv_flops(
        resnet50_expand(num_classes=num_classes, gate=False, aux_fc=False), cuda=True)
    saved_flops = common.compute_conv_flops(saved_model, cuda=True)

    print(f"Unpruned FLOPs: {baseline_flops:,}")
    print(f"Saved FLOPs: {saved_flops:,}")
    print(f"FLOPs ratio: {saved_flops / baseline_flops:,}")

    # save state_dict
    torch.save({'state_dict': saved_model.state_dict(),
                'cfg': saved_model.config(),
                "expand_idx": saved_model.channel_index},
               os.path.join(args.save, f'pruned_{args.pruning_strategy}.pth.tar'))


if __name__ == '__main__':
    main()
