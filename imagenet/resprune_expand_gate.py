import argparse
import copy
import os
from typing import Any, Dict

import numpy as np
import torch
import torch.nn

import utils
import utils.checkpoints
from models.common import ThresholdPruner, RandomPruner
from models.resnet_expand import resnet50 as resnet50_expand, ResNetExpand
from utils import common
from utils.common import compute_conv_flops


def _get_parser():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet prune ResNet-50')
    parser.add_argument('--model', default='', type=str, metavar='PATH',
                        help='path to the model (default: none)')
    parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                        help='input batch size for testing (default: 256)')
    parser.add_argument('--save', default=None, type=str, metavar='PATH',
                        help='path to save pruned model.'
                             'If not specified, the pruned model will not be saved. (default: None)')
    parser.add_argument("--pruning-strategy", type=str,
                        choices=["percent", "fixed", "grad", "search", "random"],
                        help="Pruning strategy. \n "
                             "'random': Randomly select which channel will be pruned.", required=True)
    parser.add_argument('--width-multiplier', default=1.0, type=float,
                        help="The width multiplier for MobileNet v2. "
                             "Unavailable for other networks. (default 1.0) "
                             "Must same with sparse training.")
    parser.add_argument('--same', action='store_true',
                        help='The model before pruning and after pruning is required to be exactly the same')
    parser.add_argument('--gate', action='store_true',
                        help='Add gate after the BatchNrom layers. Only available for MobileNet v2!')
    parser.add_argument("--prune-mode", type=str, default='default',
                        choices=["multiply", 'default'],
                        help="Pruning mode. Same as `models.common.prune_conv_layer`", )
    parser.add_argument('--ratio', default=None, type=float,
                        help="Uniformly prune the model by channel ratio. Only available in `random` mode")
    parser.add_argument('--percent', type=float, default=1,
                        help='scale sparse rate (default: 1.)')


    return parser


def _compute_global_threshold(model, percent: float) -> int:
    if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module

    model: ResNetExpand
    sparse_layers = model.get_sparse_layer(gate=model.use_gate,
                                           sparse1=True,
                                           sparse2=True,
                                           sparse3=True)
    # flatten all weights and concat them
    sparse_layers = filter(lambda l: l is not None, sparse_layers)
    sparse_weights: np.ndarray = np.concatenate(
        list(map(lambda l: l.weight.view(-1).data.cpu().numpy(), sparse_layers)))
    sparse_weights = np.sort(sparse_weights)

    threshold_index = int(len(sparse_weights) * percent)
    threshold = sparse_weights[threshold_index]

    return threshold


def _check_model_same(model1: torch.nn.Module, model2: torch.nn.Module, cuda=False) -> float:
    """
    check if the output is same by same input.
    """
    model1.eval()
    model2.eval()

    rand_input = torch.rand((8, 3, 224, 224))  # the same input size as ImageNet

    if cuda:
        rand_input = rand_input.cuda()
        model1.cuda()
        model2.cuda()

    out1, _ = model1(rand_input)  # ignore aux output for resnet
    out2, _ = model2(rand_input)

    diff = out1 - out2
    max_diff = torch.max(diff.abs().view(-1)).item()

    return max_diff


def prune_resnet(sparse_model: torch.nn.Module, pruning_strategy: str, sanity_check: bool, prune_mode: str,
                 ratio: float = None, percent: float = None):
    """
    :param sparse_model: The model trained with sparsity regularization
    :param pruning_strategy: same as `models.common.search_threshold`
    :param sanity_check: whether do sanity check
    :param prune_mode: same as `models.common.prune_conv_layer`
    :param ratio: Pruning ratio when pruning uniformly. Only used in `pruning_strategy` is `"random"`
    :param percent: Pruning percent when use global threshold pruning


    :return:
    """
    if isinstance(sparse_model, torch.nn.DataParallel) or isinstance(sparse_model,
                                                                     torch.nn.parallel.DistributedDataParallel):
        sparse_model = sparse_model.module

    # note that pruned model could not do forward pass.
    # need to set channel expand.
    pruned_model = copy.deepcopy(sparse_model)
    pruned_model.cpu()

    if pruning_strategy == 'percent':
        global_threshold = _compute_global_threshold(sparse_model, percent)
        pruner = ThresholdPruner(pruning_strategy, threshold=global_threshold)
    elif pruning_strategy == 'random':
        pruner = RandomPruner(ratio=ratio)
    else:
        pruner = ThresholdPruner(pruning_strategy)

    pruned_model.prune_model(pruner=pruner,
                             prune_mode=prune_mode)
    print("Pruning finished. cfg:")
    print(pruned_model.config())

    if sanity_check:
        # sanity check: check if pruned model is as same as sparse model
        print("Sanity check: checking if pruned model is as same as sparse model")
        max_diff = _check_model_same(sparse_model, pruned_model, cuda=True)
        print(f"Max diff between Sparse model and Pruned model: {max_diff}\n")

    # load weight to finetuning model
    saved_model = ResNetExpand(cfg=pruned_model.config(), expand_idx=pruned_model.expand_idx(),
                               aux_fc=False, gate=False, width_multiplier=sparse_model.width_multiplier)

    # allow missing keys in pruned model
    # the sameness between saved_model and the pruned_model will be assured by the sanity check
    saved_model.load_state_dict(pruned_model.state_dict(), strict=False)

    if sanity_check:
        print("Sanity check: checking if pruned model is as same as saved model")
        max_diff = _check_model_same(saved_model, pruned_model, cuda=True)
        print(f"Max diff between Saved model and Pruned model: {max_diff}\n")
        assert max_diff < 1e-5, f"Test failed: Max diff should be less than 1e-5, got {max_diff}"

    return saved_model


def main():
    # reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    parser = _get_parser()
    args = parser.parse_args()

    if args.ratio is not None and args.pruning_strategy != 'random':
        raise ValueError("--ratio is only available in random pruning")
    if args.ratio is None and args.pruning_strategy == 'random':
        raise ValueError("--ratio must be specified in random pruning mode.")

    if args.save is not None and not os.path.exists(args.save):
        os.makedirs(args.save)

    print(args)
    print(f"Current git hash: {common.get_git_id()}")

    if args.model == 'torchvision':
        import torchvision
        torchvision_state_dict = torchvision.models.resnet50(pretrained=True, ).state_dict()
        checkpoint = {'state_dict': torchvision_state_dict}
    else:
        if not os.path.isfile(args.model):
            raise ValueError("=> no checkpoint found at '{}'".format(args.model))

        checkpoint: Dict[str, Any] = torch.load(args.model)
        print(f"=> Loading the model...\n=> Epoch: {checkpoint['epoch']}, Acc.: {checkpoint['best_prec1']}")
        utils.checkpoints.remove_module(checkpoint)  # remove the module wrapper

    # build the sparse model
    sparse_model: ResNetExpand = resnet50_expand(aux_fc=False, width_multiplier=args.width_multiplier, gate=args.gate)
    # strict=False: compatible with models without input_mask
    sparse_model.load_state_dict(checkpoint['state_dict'], strict=False)

    saved_model = prune_resnet(sparse_model,
                               pruning_strategy=args.pruning_strategy,
                               sanity_check=True, prune_mode=args.prune_mode,
                               ratio=args.ratio, percent=args.percent)

    # compute FLOPs
    baseline_flops = compute_conv_flops(
        resnet50_expand(width_multiplier=args.width_multiplier, gate=False, aux_fc=False), cuda=True)
    saved_flops = compute_conv_flops(saved_model, cuda=True)

    if args.width_multiplier != 1.0:
        print(f"WARNING: the baseline FLOPs is the FLOPs with width multiplier {args.width_multiplier}")
    print(f"Unpruned FLOPs: {baseline_flops:,}")
    print(f"Saved FLOPs: {saved_flops:,}")
    print(f"FLOPs ratio: {saved_flops / baseline_flops:,}")

    # save state_dict
    if args.save is not None:
        torch.save({'state_dict': saved_model.state_dict(),
                    'cfg': saved_model.config(),
                    "expand_idx": saved_model.expand_idx()},
                   os.path.join(args.save, f'pruned_{args.pruning_strategy}.pth.tar'))


if __name__ == '__main__':
    main()
