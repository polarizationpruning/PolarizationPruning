import argparse
import copy
import os
from typing import Any, Dict

import torch

import common
from models import vgg16_linear
from models.common import search_threshold
from models.vgg import VGG


def _get_parser():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR prune')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'],
                        help='training dataset (default: cifar10)')
    parser.add_argument('--model', default='', type=str, metavar='PATH',
                        help='path to the model (default: none)')
    parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                        help='input batch size for testing (default: 256)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
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

    return parser


def _check_model_same(model1: torch.nn.Module, model2: torch.nn.Module) -> float:
    """
    check if the output is same by same input.
    """
    model1.eval()
    model2.eval()

    rand_input = torch.rand((8, 3, 32, 32))  # the same input size as CIFAR
    out1 = model1(rand_input)
    out2 = model2(rand_input)

    diff = out1 - out2
    max_diff = torch.max(diff.abs().view(-1)).item()

    return max_diff


def prune_vgg(num_classes: int, sparse_model: torch.nn.Module, pruning_strategy: str, sanity_check: bool,
              prune_mode: str):
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
    pruned_model.prune_model(pruner=lambda weight: search_threshold(weight, pruning_strategy),
                             prune_mode=prune_mode)
    print("Pruning finished. cfg:")
    print(pruned_model.config())

    if sanity_check:
        # sanity check: check if pruned model is as same as sparse model
        print("Sanity check: checking if pruned model is as same as sparse model")
        max_diff = _check_model_same(sparse_model, pruned_model)
        print(f"Max diff between Sparse model and Pruned model: {max_diff}\n")

    # load weight to finetuning model
    saved_model = vgg16_linear(num_classes=num_classes,
                               gate=False,
                               cfg=pruned_model.config(), )

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
    sparse_model: VGG = vgg16_linear(num_classes=num_classes, gate=args.gate)
    sparse_model.load_state_dict(checkpoint['state_dict'])

    saved_model = prune_vgg(num_classes=num_classes,
                            sparse_model=sparse_model,
                            pruning_strategy=args.pruning_strategy,
                            sanity_check=True, prune_mode=args.prune_mode)

    # compute FLOPs
    baseline_flops = common.compute_conv_flops(
        vgg16_linear(num_classes=num_classes, gate=False))
    saved_flops = common.compute_conv_flops(saved_model)

    print(f"Unpruned FLOPs: {baseline_flops:,}")
    print(f"Saved FLOPs: {saved_flops:,}")
    print(f"FLOPs ratio: {saved_flops / baseline_flops:,}")

    # save state_dict
    torch.save({'state_dict': saved_model.state_dict(),
                'cfg': saved_model.config()},
               os.path.join(args.save, f'pruned_{args.pruning_strategy}.pth.tar'))


if __name__ == '__main__':
    main()
