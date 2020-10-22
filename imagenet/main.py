import argparse
import os
import shutil
import time
from enum import Enum
import random
from random import randint

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import typing
from tensorboardX import SummaryWriter
from typing import Dict

import models
import models.common
from models import resnet50
from models.mobilenet import get_sparse_layers as get_mobilenet_sparse_layers
from models.mobilenet import mobilenet_v2
from utils import common
from utils.common import adjust_learning_rate, compute_conv_flops, freeze_gate
from utils.evaluation import AverageMeter, accuracy
from vgg import slimmingvgg as vgg11

model_names = ["resnet50", "mobilenetv2"]


class LossType(Enum):
    ORIGINAL = 0
    L1_SPARSITY_REGULARIZATION = 1
    POLARIZATION = 4
    L2_POLARIZATION = 6
    POLARIZATION_GRAD = 7  # in this mode, the gradient does not propagate through the mean term

    @staticmethod
    def from_string(desc):
        mapping = LossType.loss_name()
        return mapping[desc.lower()]

    @staticmethod
    def loss_name():
        return {"original": LossType.ORIGINAL,
                "sr": LossType.L1_SPARSITY_REGULARIZATION,
                "zol": LossType.POLARIZATION,
                "zol2": LossType.L2_POLARIZATION,
                "pol_grad": LossType.POLARIZATION_GRAD,
                }


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training with Polarization Regularization')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default=None,
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=25, type=int, metavar='N',
                    help='number of data loading workers (default: 25)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', type=float, nargs='*', default=[1e-1, 1e-2, 1e-3], metavar='LR',
                    help="the learning rate in each stage (default 1e-2, 1e-3)")
parser.add_argument('--decay-epoch', type=float, nargs='*', default=[30, 60],
                    help="the epoch to decay the learning rate (default None)")
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
# NOTE the definition of the world size there (the number of the NODE)
# is different from the world size in dist.init_process_group (the number of PROCESS)
# The ddp training is no longer supported
parser.add_argument('--world-size', default=1, type=int,
                    help='[DEPRECATED] number of distributed NODE (not process)')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='[DEPRECATED] url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='[DEPRECATED] distributed backend')
parser.add_argument('--lbd', type=float, default=0.0001,
                    help='scale sparse rate (i.e. lambda in eq.2) (default: 0.0001)')
parser.add_argument('--alpha', type=float, default=1.,
                    help='[DEPRECATED] coefficient of mean term in polarization regularizer. (default: 1)')
parser.add_argument('--t', type=float, default=1.,
                    help='coefficient of L1 term in polarization regularizer (default: 1)')
parser.add_argument('--save', default='./', type=str, metavar='PATH',
                    help='path to save model (default: current directory)')
parser.add_argument('--backup-path', default='.', type=str, metavar='PATH',
                    help='path to save model (default: current directory)')
parser.add_argument('--backup-freq', default=5, type=int,
                    help="The frequency of backup checkpoint.")
parser.add_argument('--rank', default=0, type=int,
                    help='node (not process) rank for distributed training')
parser.add_argument("--loss-type", "-loss", dest="loss", required=True,
                    choices=list(LossType.loss_name().keys()), help="the type of loss")
parser.add_argument("--debug", action="store_true",
                    help="enable debug mode")
parser.add_argument("-ddp", action="store_true",
                    help="[DEPRECATED] use DistributedDataParallel mode instead of DataParallel")
parser.add_argument("--last-sparsity", action="store_true",
                    help="apply sparsity loss on the last bn in the block")
parser.add_argument("--bn-init-value", type=float, default=0.5,
                    help="The initial value of BatchNormnd weight")
parser.add_argument('--seed', type=int, metavar='S', default=666,
                    help='random seed (default: 666)')
parser.add_argument("--fc-sparsity", default="unified",
                    choices=["unified", "separate", "single"],
                    help='''Method of calculating average for vgg network. (default unified)
                    unified: default behaviour. use the global mean for all layers.
                    separate: only available for vgg11. use different mean for CNN layers and FC layers separately.
                    single: only available for vgg11. use global mean for CNN layers and different mean for each FC layers.
                    ''')
parser.add_argument("--bn3-sparsity", default="unified",
                    choices=["unified", "separate"],
                    help='''[DEPRECATED] Method of calculating average for vgg network. (default unified)
                    unified: default behaviour. use the global mean for all layers.
                    separate: only available for ResNet-50. use different mean for bn3 layers.
                    ''')
parser.add_argument('--clamp', default=1.0, type=float,
                    help='Upper bound of the bn scaling factors (only available at Polarization!) (default: 1.0)')
parser.add_argument('--no-bn-wd', action='store_true',
                    help='Do not apply weight decay on BatchNorm layers')
parser.add_argument('--width-multiplier', default=1.0, type=float,
                    help="The width multiplier (only) for ResNet-50 and MobileNet v2. "
                         "Unavailable for other networks. (default 1.0)")
parser.add_argument('--gate', action='store_true',
                        help='Add an extra scaling factor after the BatchNrom layers.')
parser.add_argument('--refine', default=None, type=str, metavar='PATH',
                    help='Path to the pruned model. Only support MobileNet v2.')
parser.add_argument('--keep-out', action='store_true',
                    help='Keep output dimension unpruned for each building blocks. Only support MobileNet v2.')
parser.add_argument('--warmup', action='store_true',
                    help='Use learning rate warmup in first five epoch. '
                         'Only support cosine learning rate schedule.')
parser.add_argument('--lr-strategy', type=str, required=True, choices=['cos', 'step'],
                    help='Learning rate decay strategy. \n'
                         '- cos: Cosine learning rate decay. In this case, '
                         '--lr should be only one value, and --decay-epoch will be ignored.\n'
                         '- step: Decay as --lr and --decay-step.'
                    )
parser.add_argument('--fix-gate', action='store_true',
                    help='Do not update SparseGate. Used for early resume to fix the sparsity of the network.')
parser.add_argument("--prune-mode", type=str, default='default',
                    choices=["multiply", 'default'],
                    help="Pruning mode. Same as `models.common.prune_conv_layer`", )
# the parameters on the last bn layer in the residual block
parser.add_argument('--bn3-lbd', type=float, default=None,
                    help='[DEPRECATED] scale sparse rate for the last bn layer for ResNet-50 (i.e. lambda in eq.2) '
                         '(default: same as --lbd)')
parser.add_argument('--bn3-alpha', type=float, default=None,
                    help='[DEPRECATED] coefficient of mean term for the last bn layer for ResNet-50 in polarization regularizer. '
                         'deprecated. (default: same as --alpha)')
parser.add_argument('--bn3-t', type=float, default=None,
                    help='[DEPRECATED] coefficient of L1 term for ResNet-50 in polarization regularizer '
                         '(default: same as --t)')

parser.add_argument('--flops-weighted', action='store_true',
                    help='Use weighted polarization parameters. The weight is determined by dFLOPs for each layer.'
                         'Only available for ResNet-50 at polarization.')
parser.add_argument('--weight-max', type=float, default=None,
                    help='Maximum FLOPs weight. Only available when --flops-weighted is enabled.')
parser.add_argument('--weight-min', type=float, default=None,
                    help='Minimum FLOPs weight. Only available when --flops-weighted is enabled.')
parser.add_argument('--weighted-mean', action='store_true',
                    help='The mean in the polarization will be weighted by FLOPs weight. Only for ResNet-50.')

parser.add_argument('--layerwise', action='store_true',
                    help='[DEPRECATED] The polarization is applied layer-wised. The mean is computed for each convolution layer. '
                         'Only available for ResNet-50 at Polarization loss.')
parser.add_argument('--load-param-only', action='store_true',
                    help="Only load parameter when resume. "
                         "The training will start from epoch 0. Optimizers will not be loaded.")
parser.add_argument('--pretrain', default=None, type=str, metavar='PATH',
                    help='Path to pretrain checkpoint (default: None)')
parser.add_argument('--target-flops', type=float, default=None,
                    help='Stop when pruned model archive the target FLOPs')

best_prec1 = 0


def fix_seed(seed):
    """
    reproducibility: fix the random seed
    make sure the seed is fixed each worker in DataLoader (use worker_init_fnƒ)
    see:
    - https://pytorch.org/docs/stable/notes/randomness.html
    - https://discuss.pytorch.org/t/is-there-a-way-to-fix-the-random-seed-of-every-workers-in-dataloader/21687
    """
    # Python random lib
    random.seed(seed)

    # PyTorch random seed
    torch.manual_seed(seed)

    # Numpy random seed
    np.random.seed(seed)

    # CUDA random seed
    torch.cuda.manual_seed(seed)

    # CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id):
    """
    fix numpy random seed in each dataloader workers
    from https://github.com/kenshohara/3D-ResNets-PyTorch/blob/8e6a026d57eda8eb54db45090e315c310750762f/utils.py
    """
    torch_seed = torch.initial_seed()

    random.seed(torch_seed + worker_id)

    if torch_seed >= 2 ** 32:
        torch_seed = torch_seed % 2 ** 32

    fix_seed(torch_seed + worker_id)


def main():
    # set environment variables
    os.environ["NCCL_DEBUG"] = "INFO"

    # parse args
    args = parser.parse_args()

    args.loss = LossType.from_string(args.loss)
    if args.last_sparsity and not (
            args.loss == LossType.POLARIZATION or args.loss == LossType.POLARIZATION_GRAD or args.loss == LossType.L2_POLARIZATION):
        print("WARNING: loss type {} does not support --last-sparsity!".format(args.loss))

    if args.lr_strategy == "step" and len(args.lr) != len(args.decay_epoch) + 1:
        print("args.lr: {}".format(args.lr))
        print("args.decay-epoch: {}".format(args.decay_epoch))
        raise ValueError("inconsistent between lr-decay-gamma and decay-epoch")
    args.decay_epoch = [int(e) for e in args.decay_epoch]

    if args.lr_strategy != 'cos' and args.warmup:
        raise ValueError(f"Only cosine learning rate support warmup. Got {args.lr_strategy}.")

    if args.lr_strategy == 'cos':
        assert len(args.lr) == 1, "Only allow one learning rate for cosine learning rate"

    args.fc_sparsity = str.lower(args.fc_sparsity)
    if args.fc_sparsity != "unified" and args.loss not in {LossType.POLARIZATION, LossType.L2_POLARIZATION,
                                                           LossType.POLARIZATION_GRAD}:
        raise NotImplementedError(f"Option --fc-sparsity is conflict with loss {args.loss}")
    if args.fc_sparsity != "unified" and args.arch != "vgg11":
        raise NotImplementedError(f"Option --fc-sparsity only support VGG. Set to unified for {args.arch}")

    if args.width_multiplier != 1.0 and (args.arch != "resnet50" and args.arch != "mobilenetv2"):
        raise ValueError("--width-multiplier only support ResNet-50 and MobileNet v2,"
                         "got: {} for {}".format(args.width_multiplier, args.arch))
    if args.arch == "mobilenetv2":
        print("MobileNet Warning:")
        print("1. The learning rate arguments (--deacy-epoch) is disabled."
              " Use cosine learning rate decay schedule.")

        assert len(args.lr) == 1, "cosine schedule only need one initial learning rate."

        if args.batch_size != 256:
            print("WARNING: MobileNet v2 default batch size is 256, got {}".format(args.batch_size))
        if args.weight_decay != 0.00004:
            print("WARNING: MobileNet v2 default weight decay is 0.00004, got {}".format(args.weight_decay))

    if args.gate and args.arch not in {'mobilenetv2', 'resnet50'}:
        raise ValueError(f"--gate option only works for MobileNet v2 and ResNet-50, got {args.arch}")

    if args.gate and args.loss not in {LossType.POLARIZATION, LossType.L1_SPARSITY_REGULARIZATION}:
        raise ValueError(f"--gate does not compatible with loss: {args.loss}")

    if args.keep_out and args.arch != 'mobilenetv2':
        raise ValueError(f"--keep-out option only works for MobileNet v2, got {args.arch}")

    if args.last_sparsity and args.arch != 'resnet50':
        raise ValueError(f"--last-sparsity only support ResNet-50, got {args.arch}.")

    if args.gate and args.clamp != 1:
        print(f"WARNING! Gate is enabled, got --clamp {args.clamp} (default 1.0)!")

    if args.target_flops and args.loss != LossType.POLARIZATION:
        raise ValueError(f"Conflict option: --loss {args.loss} --target-flops {args.target_flops}")

    if args.target_flops and not args.gate:
        raise ValueError(f"Conflict option: --target-flops only available at --gate mode")

    if args.target_flops and args.target_flops > 1:
        raise ValueError("The target flops should be less than 1")

    # bn3 arguments
    for param in ['t', 'lbd', 'alpha']:
        argument_name = f'bn3_{param}'
        argument_value = getattr(args, argument_name)
        if argument_value is None:
            default_value = getattr(args, param)
            setattr(args, argument_name, default_value)
        else:
            if args.arch != 'resnet50':
                raise ValueError(f"Do not set {argument_name} in arch {args.arch}")
            if args.loss != LossType.POLARIZATION:
                raise ValueError(f"Do not support loss: {args.loss} when set {argument_name}")
            if not args.last_sparsity:
                raise ValueError(f"Do not specific {argument_name} when not --last-sparsity.")

    if args.bn3_sparsity != "unified" and args.arch != 'resnet50':
        raise ValueError(f"--bn3-sparsity {args.bn3_sparsity} only support ResNet-50, got {args.arch}")

    if args.bn3_sparsity != 'unified' and not args.last_sparsity:
        raise ValueError(f"Conflict option: --bn3-sparsity {args.bn3_sparsity} and not --last-sparsity.")

    if args.flops_weighted and args.arch not in {'resnet50', 'mobilenetv2'}:
        raise ValueError(f"Expected ResNet-50 or MobileNet v2, got {args.arch}")

    if args.flops_weighted and args.loss != LossType.POLARIZATION:
        raise ValueError(f"Expected polarization loss, got {args.loss}")

    if not args.flops_weighted and (args.weight_max is not None or args.weight_min is not None):
        raise ValueError("When --flops-weighted is not set, do not specific --max-weight or --min-weight")

    if args.flops_weighted and (args.weight_max is None or args.weight_min is None):
        raise ValueError("When --flops-weighted is set, do specific --max-weight or --min-weight")

    if args.weighted_mean and not args.flops_weighted:
        raise ValueError("--weighted-mean only available at --flops-weighted mode.")

    if args.weighted_mean and args.loss != LossType.POLARIZATION:
        raise ValueError("--weighted-mean only available at Polarization loss.")

    if args.layerwise:
        if args.loss != LossType.POLARIZATION:
            raise ValueError(f"Conflict option: --layerwise and loss {args.loss}")
        if args.arch != 'resnet50':
            raise ValueError(f"Conflict option: --layerwise and --arch {args.arch}")
        if args.bn3_sparsity != "unified":
            raise ValueError(f"Conflict option: --layerwise and --bn3-sparsity {args.bn3_sparsity}")
    print(args)
    print(f"Current git hash: {common.get_git_id()}")
    if args.debug:
        print("*****WARNING! DEBUG MODE IS ENABLED!******")
        print("******The model will NOT be trained!******")
        pass

    # reproducibility
    fix_seed(args.seed)

    # the number of gpu in current device
    ngpus_per_node = torch.cuda.device_count()
    # enable distributed mode if there are more than one gpu
    args.distributed = args.ddp and (args.world_size > 1 or ngpus_per_node > 1)

    # start process
    if args.distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        # (suppose the number of gpu in each node is SAME)
        args.world_size = ngpus_per_node * args.world_size
        print("actual args.world_size: {}".format(args.world_size))
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function (with all gpus)
        main_worker("", ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_prec1
    args.gpu = gpu

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    if args.rank == 0:
        if not os.path.exists(args.save):
            os.makedirs(args.save)
        if not os.path.exists(args.backup_path):
            os.makedirs(args.backup_path)

    if args.distributed:
        # For multiprocessing distributed training, rank needs to be the
        # global rank among all the processes
        args.rank = args.rank * ngpus_per_node + gpu
        print("Starting process rank {}".format(args.rank))
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    print("rank#{}: CUDA_VISIBLE_DEVICES: {}".format(args.rank, os.environ['CUDA_VISIBLE_DEVICES']))

    # Initialize the model, build the model structure
    # If the refine is enabled, the model should be initialized by the refine config else
    # the model is initialized by the default config.
    # Note: the optimizer will NOT be restored, because the network parameters were pruned,
    # but the optimizer parameters (e.g. momentum) were not pruned. This behavior is supposed
    # not to affect accuracy. The optimizer parameters would be learned very fast.
    refine_checkpoint = None
    if args.refine:
        if not os.path.exists(args.refine):
            raise ValueError(f"Not existed path: {args.refine}")
        if not os.path.isfile(args.refine):
            raise ValueError(f"Expect a file path, got {args.refine}")
        refine_checkpoint = torch.load(args.refine)

        print("=> Loading the refine model. cfg: ")
        print(refine_checkpoint['cfg'])

        if args.arch == "mobilenetv2":
            model = mobilenet_v2(inverted_residual_setting=refine_checkpoint['cfg'],
                                 width_mult=args.width_multiplier,
                                 use_gate=args.gate, input_mask=args.gate)

            # set ChannelExpand index
            expand_idx = refine_checkpoint['expand_idx']
            for m_name, sub_module in model.named_modules():
                if isinstance(sub_module, models.common.ChannelExpand):
                    sub_module.idx = expand_idx[m_name]
        else:
            raise NotImplementedError(f"--refine only support MobileNet v2 model. Got {args.arch}")

        # note the refine checkpoint does not contain DataParallel wrapper
        model.load_state_dict(refine_checkpoint['state_dict'])

    else:
        if args.arch == "vgg11":
            if args.resume and os.path.isfile(args.resume):
                checkpoint = torch.load(args.resume)
                if "cfg" in checkpoint:
                    model = vgg11(config=checkpoint['cfg'])
                else:
                    model = vgg11()
            else:
                model = vgg11()
        elif args.arch == "resnet50":
            model = resnet50(aux_fc=False,
                             width_multiplier=args.width_multiplier,
                             gate=args.gate)
        elif args.arch == "mobilenetv2":
            model = mobilenet_v2(width_mult=args.width_multiplier,
                                 use_gate=args.gate)
        else:
            raise NotImplementedError("model {} is not supported".format(args.arch))

    # compute flops weight at flops weighted mode
    if args.flops_weighted:
        print("Computing the FLOPs weight...")
        flops_weight = model.get_conv_flops_weight()
        flops_weight_string_builder: typing.List[str] = []
        for fw in flops_weight:
            flops_weight_string_builder.append(",".join(str(w) for w in fw))
        flops_weight_string = "\n".join(flops_weight_string_builder)
        print("FLOPs weight:")
        print(flops_weight_string)
        print()

    if not args.distributed:
        # DataParallel
        model.cuda()
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            # see discussion
            # https://discuss.pytorch.org/t/are-there-reasons-why-dataparallel-was-used-differently-on-alexnet-and-vgg-in-the-imagenet-example/19844
            model.features = torch.nn.DataParallel(model.features)
        else:
            model = torch.nn.DataParallel(model).cuda()
    else:
        # DistributedDataParallel
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    # do not update the last SparseGate for each block
    # keepout option is for MobileNet v2
    # last-sparsity option is for ResNet
    if args.gate and (
            (args.keep_out and args.arch == 'mobilenetv2') or (not args.last_sparsity and args.arch == 'resnet50')):
        # if args.keep_out and not args.gate, bn should be updated as normal
        assert args.arch in {'mobilenetv2', 'resnet50'}, "--keep-out only works for MobileNet v2 and ResNet-50"
        if args.arch == 'mobilenetv2':
            output_gates = models.mobilenet.get_output_gate(model)
        elif args.arch == 'resnet50':
            output_gates = model.module.get_output_gate()
        else:
            raise NotImplementedError(f"Do not support {args.arch}")

        for gate in output_gates:
            assert isinstance(gate, models.common.SparseGate), f"Expected a SparseGate, got {gate}"
            for p in gate.parameters():
                # do not update this gate
                p.requires_grad = False

    if args.fix_gate:
        freeze_gate(model)

    # create the optimizer
    if args.no_bn_wd:
        no_wd_params = []
        for module_name, sub_module in model.named_modules():
            if isinstance(sub_module, nn.BatchNorm1d) or \
                    isinstance(sub_module, nn.BatchNorm2d) or \
                    isinstance(sub_module, models.common.SparseGate):  # never apply weight decay on SparseGate
                for param_name, param in sub_module.named_parameters():
                    no_wd_params.append(param)
                    print(f"No weight decay param: module {module_name} param {param_name}")
    else:
        no_wd_params = []
        for module_name, sub_module in model.named_modules():
            if isinstance(sub_module, models.common.SparseGate):  # never apply weight decay on SparseGate
                for param_name, param in sub_module.named_parameters():
                    no_wd_params.append(param)
                    print(f"No weight decay param: module {module_name} param {param_name}")

    no_wd_params_set = set(no_wd_params)
    wd_params = []
    for param_name, model_p in model.named_parameters():
        if model_p not in no_wd_params_set:
            wd_params.append(model_p)
            print(f"Weight decay param: parameter name {param_name}")

    optimizer = torch.optim.SGD([{'params': list(no_wd_params), 'weight_decay': 0.},
                                 {'params': list(wd_params), 'weight_decay': args.weight_decay}],
                                args.lr[0],
                                momentum=args.momentum)

    if args.debug:
        # fake polarization to test pruning
        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm1d) or \
                    isinstance(module, nn.BatchNorm2d) or \
                    isinstance(module, models.common.SparseGate):
                module.weight.data.zero_()
                one_num = randint(3, 30)
                module.weight.data[:one_num] = 1.

                print(f"{name} remains {one_num}")

    if args.pretrain:
        if os.path.isfile(args.pretrain):
            print("=> loading pre-train checkpoint '{}'".format(args.pretrain))
            checkpoint = torch.load(args.pretrain)
            pretrain_state_dict = {}
            for key, value in model.state_dict().items():
                if key in checkpoint['state_dict']:
                    pretrain_state_dict[key] = checkpoint['state_dict'][key]
                else:
                    print(f"\tMissing parameter in pre-train model: {key}")
            model.load_state_dict(pretrain_state_dict)

            print("=> Pre-train checkpoint loaded.")
        else:
            raise ValueError("=> no checkpoint found at '{}'".format(os.path.abspath(args.pretrain)))

    # load the refine epoch, continue training from the refine point
    if args.refine:
        args.start_epoch = refine_checkpoint['epoch']
        best_prec1 = refine_checkpoint['best_prec1']

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            if not args.load_param_only:
                args.start_epoch = checkpoint['epoch']
                best_prec1 = checkpoint['best_prec1']
                optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            raise ValueError("=> no checkpoint found at '{}'".format(args.resume))

    print("Model loading completed. Model Summary:")
    print(model)

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    print("rank #{}: loading the dataset...".format(args.rank))

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler,
        worker_init_fn=worker_init_fn)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    print("rank #{}: dataloader loaded!".format(args.rank))

    if args.evaluate:
        validate(val_loader, model, criterion, epoch=0, args=args, writer=None)
        return

    # restore the learning rate
    for epoch in range(args.start_epoch):
        for iter_step in range(len(train_loader)):
            adjust_learning_rate(optimizer, epoch, lr=args.lr, decay_epoch=args.decay_epoch,
                                 total_epoch=args.epochs,
                                 train_loader_len=len(train_loader), iteration=0,
                                 warmup=args.warmup, decay_strategy=args.lr_strategy)

    # only master process in each node write to disk
    writer = SummaryWriter(logdir=args.save, write_to_disk=args.rank % ngpus_per_node == 0)

    if args.flops_weighted:
        writer.add_text("train/conv_flops_weight", flops_weight_string, global_step=0)

    if args.loss in {LossType.POLARIZATION, LossType.POLARIZATION_GRAD, LossType.L2_POLARIZATION} and \
            args.arch in {"resnet50", "mobilenetv2"} and \
            args.target_flops and args.gate and args.resume:
        # if the flops less than the target flops after resuming, then stop polarization
        cur_flops, baseline_flops = prune_while_training(model, args.arch, prune_mode=args.prune_mode,
                                                         width_multiplier=args.width_multiplier)
        flops_ratio = cur_flops / baseline_flops
        if flops_ratio <= args.target_flops:
            # do not apply polarization loss on the model to keep the pruned flops unchanged
            args.lbd = 0
            args.bn3_lbd = 0
            freeze_gate(model)
            # no need to use small backup freq for saving storage space
            if args.backup_freq > 20:
                args.backup_freq = 20

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # draw bn hist to tensorboard
        weights, bias = bn_weights(model)
        for bn_name, bn_weight in weights:
            if torch.sum(torch.abs(bn_weight)) > 0:
                writer.add_histogram("bn/" + bn_name, bn_weight, global_step=epoch)
        for bn_name, bn_bias in bias:
            if torch.sum(torch.abs(bn_bias)) > 0:
                writer.add_histogram("bn_bias/" + bn_name, bn_bias, global_step=epoch)
        for name, sub_module in model.named_modules():
            if isinstance(sub_module, nn.Conv2d):
                # visualize conv kernels
                if torch.sum(torch.abs(sub_module.weight)) > 0:
                    writer.add_histogram("conv_kernels/" + name, sub_module.weight, global_step=epoch)
            elif isinstance(sub_module, models.common.SparseGate):
                # visualize sparse gates
                writer.add_histogram("sparse_gate/" + name, sub_module.weight, global_step=epoch)

        # visualize all sparsity layers for MobileNet
        if args.loss in {LossType.POLARIZATION, LossType.L1_SPARSITY_REGULARIZATION}:
            if args.arch == 'mobilenetv2':
                sparse_layers = get_mobilenet_sparse_layers(model, gate=args.gate, exclude_out=args.keep_out)
            elif args.arch == 'resnet50':
                sparse_layers = model.module.get_sparse_layer(gate=args.gate,
                                                              sparse1=True,
                                                              sparse2=True,
                                                              sparse3=args.last_sparsity)
            else:
                sparse_layers = None

            if sparse_layers is not None:
                sparse_weights = list(map(lambda layer: layer.weight.view(-1), sparse_layers))
                sparse_weights = torch.cat(sparse_weights).view(-1)
                weight_mean = torch.mean(sparse_weights)

                writer.add_histogram("whole_sparse_layer/concat_all_sparse_weight", sparse_weights, global_step=epoch)
                writer.add_scalar("train/sparse_weight_mean", weight_mean, global_step=epoch)

            # special visualization for ResNet-50
            if args.arch == "resnet50":
                sparse12_layers = model.module.get_sparse_layer(gate=args.gate,
                                                                # exclude sparse3
                                                                sparse1=True,
                                                                sparse2=True,
                                                                sparse3=False,
                                                                with_weight=args.flops_weighted)
                sparse3_layers = model.module.get_sparse_layer(gate=args.gate,
                                                               sparse1=False,
                                                               sparse2=False,
                                                               sparse3=True,
                                                               with_weight=args.flops_weighted)
            elif args.arch == 'mobilenetv2':
                sparse12_layers = model.module.get_sparse_layer(gate=args.gate,
                                                                pw_layer=True,
                                                                linear_layer=False,
                                                                with_weight=args.flops_weighted)
                sparse3_layers = model.module.get_sparse_layer(gate=args.gate,
                                                               pw_layer=False,
                                                               linear_layer=True,
                                                               with_weight=args.flops_weighted)
            else:
                raise NotImplementedError(f"do not support arch {args.arch}")

            if args.flops_weighted:
                # unpack the flops weights
                sparse12_layers, sparse12_flops_weights = sparse12_layers
                sparse3_layers, sparse3_flops_weights = sparse3_layers
            else:
                sparse12_flops_weights = None
                sparse3_flops_weights = None

            # remove None items for MobileNet v2
            # (some layers do not have pw layer, in that case, there will be None)
            if args.arch == 'mobilenetv2':
                sparse12_layers = list(filter(lambda x: x is not None, sparse12_layers))
                sparse3_layers = list(filter(lambda x: x is not None, sparse3_layers))
                if args.flops_weighted:
                    # flops_weights is None when not args.flops_weighted
                    sparse12_flops_weights = list(filter(lambda x: x is not None, sparse12_flops_weights))
                    sparse3_flops_weights = list(filter(lambda x: x is not None, sparse3_flops_weights))

            visualization_group = {"sparse12": {"layer": sparse12_layers,
                                                "weight": sparse12_flops_weights},
                                   "sparse3": {"layer": sparse3_layers,
                                               "weight": sparse3_flops_weights}}

            for key in visualization_group.keys():
                sparse_weights = list(map(lambda layer: layer.weight.view(-1), visualization_group[key]["layer"]))
                sparse_weights = torch.cat(sparse_weights).view(-1)
                # compute the mean of layers
                weight_mean = _sparse_mean(sparse_modules=visualization_group[key]["layer"],
                                           sparse_weights=visualization_group[key]["weight"],
                                           disable_grad=True,
                                           weighted_mean=args.weighted_mean,
                                           weight_min=args.weight_min,
                                           weight_max=args.weight_max)

                writer.add_histogram(f"whole_sparse_layer/{key}", sparse_weights, global_step=epoch)
                writer.add_scalar(f"train/sparse_weight_mean_{key}", weight_mean, global_step=epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch,
              args.lbd, args=args,
              is_debug=args.debug, writer=writer)

        # save the trained checkpoint for debug
        save_checkpoint({
            # if the model is built by the refine model, the config need to be stored for pruning
            # pruning need the model cfg to load the state_dict
            'cfg': refine_checkpoint['cfg'] if args.refine else None,
            'expand_idx': refine_checkpoint['expand_idx'] if args.refine else None,
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,  # note that the best_prec1 is the prec1 of the PREVIOUS epoch
            'optimizer': optimizer.state_dict(),
        }, is_best=False, filepath=args.save,
            save_backup=False,
            backup_path=None,
            epoch=epoch,
            name="checkpoint.pth.tar.tmp"
        )

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, epoch,
                         args=args, writer=writer)

        report_prune_result(model)  # do not really prune the model

        writer.add_scalar("train/lr", optimizer.param_groups[0]['lr'], epoch)

        # prune the network and record FLOPs at each epoch
        if args.loss in {LossType.POLARIZATION, LossType.POLARIZATION_GRAD, LossType.L2_POLARIZATION}:
            if args.arch in {"resnet50", "mobilenetv2"}:
                print('Evaluating FLOPs ...')
                start_time = time.time()
                cur_flops, baseline_flops = prune_while_training(model, args.arch, prune_mode=args.prune_mode,
                                                                 width_multiplier=args.width_multiplier)
                end_time = time.time()
                print(
                    f"Epoch [{epoch}] FLOPs {cur_flops} (ratio: {cur_flops / baseline_flops:.4f}), "
                    f"evaluate cost: {end_time - start_time} seconds.")
                writer.add_scalar("train/flops", cur_flops, epoch)
                writer.add_scalar("train/flops_ratio", cur_flops / baseline_flops, epoch)

                if args.target_flops is not None and cur_flops / baseline_flops <= args.target_flops and args.gate:
                    print("The grad pruning FLOPs archieve the target FLOPs.")
                    print(f"Current pruning ratio: {cur_flops / baseline_flops}")
                    print("Stop polarization from current epoch and continue training.")

                    # do not apply polarization loss on the model to keep the pruned flops unchanged
                    args.lbd = 0
                    args.bn3_lbd = 0
                    freeze_gate(model)
                    # no need to use small backup freq for saving storage space
                    if args.backup_freq > 20:
                        args.backup_freq = 20

        # save checkpoint in debug mode
        if args.debug:
            prec1 = float(epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        if args.rank % ngpus_per_node == 0:
            save_checkpoint({
                # if the model is built by the refine model, the config need to be stored for pruning
                # pruning need the model cfg to load the state_dict
                'cfg': refine_checkpoint['cfg'] if args.refine else None,
                'expand_idx': refine_checkpoint['expand_idx'] if args.refine else None,
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, is_best, args.save,
                save_backup=epoch % args.backup_freq == 0,
                backup_path=args.backup_path,
                epoch=epoch)

        writer.flush()

    writer.close()
    print("Best prec@1: {}".format(best_prec1))


def updateBN(model, sparsity, sparsity_on_bn3, gate: bool, exclude_out: bool, is_mobilenet=False):
    """Apply L1-Norm on sparse layers"""
    if not is_mobilenet:
        if gate:
            raise NotImplementedError("Do not support gate sparsity for L1 sparsity")
        bn_modules = list(filter(lambda m: (("bn3" not in m[0] and "downsample" not in m[0]) or sparsity_on_bn3) and (
                isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.BatchNorm1d)), model.named_modules()))
        bn_modules = list(map(lambda m: m[1], bn_modules))  # remove module name
    else:
        bn_modules = model.module.get_sparse_layer(gate=gate,
                                                   pw_layer=True,  # always prune the pw layer
                                                   # if exclude out, do not apply sparsity on linear layer
                                                   linear_layer=not exclude_out,
                                                   with_weight=False)

    for m in bn_modules:
        if m is not None:
            m.weight.grad.data.add_(sparsity * torch.sign(m.weight.data))


def BN_grad_zero(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            mask = (m.weight.data != 0)
            mask = mask.float().cuda()
            m.weight.grad.data.mul_(mask)
            m.bias.grad.data.mul_(mask)


def bn_weights(model):
    weights = []
    bias = []
    for name, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            weights.append((name, m.weight.data))
            bias.append((name, m.bias.data))

    return weights, bias
    pass


def _sparse_mean(sparse_modules: typing.Iterable, sparse_weights,
                 disable_grad: bool, weighted_mean: bool,
                 weight_min, weight_max):
    """
    compute the global mean of all sparse layers
    :param sparse_modules: The collection of sparse layers (required)
    :param sparse_weights: the FLOPs weight of each sparse layers. Only required when
    :param disable_grad: do not do backward pass in computing mean
    :param weighted_mean: if compute weighted arithmetic mean with sparse_weights.
    :param weight_min: the maximum value of the FLOPs weight
    :param weight_max: the minimum value of the FLOPs weight
    :return: the global mean of all sparse layers
    """

    # there might be None layers and weights in MobileNet v2
    sparse_modules = list(filter(lambda x: x is not None, sparse_modules))
    if sparse_weights is not None:
        sparse_weights = list(filter(lambda x: x is not None, sparse_weights))

    if disable_grad:
        # do not need gradient for mean
        torch.set_grad_enabled(False)
    if weighted_mean:
        weigted_sum = 0.  # the sum of the parameters of sparse layers
        weight_sum = 0.  # the sum of scaled flops weights
        for sparse_m, sparse_w in zip(sparse_modules, sparse_weights):
            # scale the weight to [weight_min, weight_max]
            sparse_w = weight_min + (weight_max - weight_min) * sparse_w
            assert sparse_w != 0
            weight_vector = sparse_m.weight.view(-1)
            weigted_sum += weight_vector.sum() * sparse_w
            weight_sum += weight_vector.shape[0] * sparse_w

        # weighted arithmetic mean
        sparse_weights_mean = weigted_sum / weight_sum
    else:
        # arithmetic mean
        # note: sparse_layer_weight_concat is the parameter of sparse layers
        sparse_layer_weight_concat = torch.cat(list(map(lambda m: m.weight.view(-1), sparse_modules)))
        assert len(sparse_layer_weight_concat.shape) == 1, "sparse_weight_concat is expected as a vector"
        sparse_weights_mean = sparse_layer_weight_concat.mean()
    torch.set_grad_enabled(True)

    return sparse_weights_mean


def bn_sparsity(model, loss_type, sparsity, t, alpha, gate, keep_out, arch,
                bn3_argument: typing.Optional[Dict[str, float]],
                sparsity_on_bn3=True, bn3_only=False,
                flops_weighted: bool = False,
                weight_max: typing.Optional[float] = None,
                weight_min: typing.Optional[float] = None,
                weighted_mean: bool = False,
                layerwise: bool = False):
    """

    :type model: torch.nn.Module
    :type alpha: float
    :type t: float
    :type sparsity: float
    :type loss_type: LossType
    :type sparsity_on_bn3: bool
    :type gate: bool
    :type arch: str

    :param keep_out: keep output layer unpruned for each block
    :param bn3_argument: use different lbd, t and alpha on bn3 layers of ResNet-50
    :param bn3_only: only apply polarization on bn3 layers, only available for ResNet-50
    :param flops_weighted: the polarization parameter will be tuned according to the FLOPs weight
    :param weight_max: limit the maximum value of conv flops weight in `flops_weighted` mode
    :param weight_min: limit the minimum value of conv flops weight in `flops_weighted` mode
    :param layerwise: Each layer use separate mean to compute polarization

    Note: for MobileNet v2, we do not apply any sparsity on deep-wise conv layers
    """
    if (weight_min is not None or weight_max is not None) and not flops_weighted:
        raise ValueError("Conflict option: lambda_max (lambda_min) and flops_weighted")

    sparse_weights = None  # default value
    if arch == "resnet50":
        if not sparsity_on_bn3 and bn3_only:
            raise ValueError("Conflict option: not sparsity_on_bn3 and bn3_only.")

        sparse_modules = model.module.get_sparse_layer(gate=gate,
                                                       sparse1=not bn3_only,
                                                       sparse2=not bn3_only,
                                                       sparse3=sparsity_on_bn3,
                                                       with_weight=weighted_mean)

    elif arch == "mobilenetv2":
        # MobieNet v2
        # if keep_out, the output layer will NOT be pruned, the linear_layer will not be polarized
        sparse_modules = model.module.get_sparse_layer(gate=gate,
                                                       pw_layer=True,
                                                       linear_layer=not keep_out,
                                                       with_weight=weighted_mean)

    else:
        raise NotImplementedError("This part need to be updated for VGG-11")

    # unpack the weight
    if weighted_mean:
        # if weighted_mean is True, with_weight is True, the get_sparse_layer method will return both layers and weights
        # NOTE: the sparse_weights is scaled to [0, 1]
        sparse_modules, sparse_weights = sparse_modules

    if len(sparse_modules) == 0:
        raise ValueError(f"No sparse modules available in the model {str(model)}")

    if loss_type in {LossType.POLARIZATION, LossType.L2_POLARIZATION, LossType.POLARIZATION_GRAD}:
        # compute mean of all sparse layers
        # if layerwise is True, this global mean is useless
        if not layerwise:
            # global mean
            sparse_weights_mean = _sparse_mean(sparse_modules=sparse_modules,
                                               sparse_weights=sparse_weights,
                                               disable_grad=loss_type == LossType.POLARIZATION_GRAD,
                                               weighted_mean=weighted_mean,
                                               weight_min=weight_min, weight_max=weight_max)
        else:
            # compute the mean layer by layer
            sparse_weights_mean = None

        # compute sparsity loss
        if arch != 'resnet50':

            if flops_weighted and arch == 'mobilenetv2':
                # MobileNet v2 flops weighted
                sparsity_loss = 0.
                for submodule in model.modules():
                    if isinstance(submodule, models.mobilenet.InvertedResidual):
                        submodule: models.mobilenet.InvertedResidual
                        flops_weight = submodule.conv_flops_weight
                        if gate:
                            # the index of the sparse layer, (conv, bn, gate)
                            sparse_layers_idx = 2
                        else:
                            sparse_layers_idx = 1

                        sparse_layers = [submodule.pw_layer[sparse_layers_idx],
                                         submodule.linear_layer[sparse_layers_idx]]

                        if keep_out:
                            # keep the output dimension, i.e., do not prune the linear layer
                            flops_weight: tuple = flops_weight[:1]
                            sparse_layers: list = sparse_layers[:1]

                        # weighted polarization
                        for layer, weight in zip(sparse_layers, flops_weight):
                            if layer is None and weight is None:
                                # there is no pw layer, skip
                                continue

                            # linear rescale the weight from [0, 1] to [lambda_min, lambda_max]
                            weight = weight_min + (weight_max - weight_min) * weight

                            if layerwise:
                                # compute layer-wise mean
                                layerwise_mean = layer.weight.view(-1).mean()
                                sparse_weights_mean = layerwise_mean

                            sparsity_loss += _compute_polarization_sparsity([layer],
                                                                            lbd=sparsity * weight, t=t,
                                                                            alpha=alpha,
                                                                            bn_weights_mean=sparse_weights_mean,
                                                                            loss_type=loss_type)

            else:
                # default behaviour (for MobileNet v2 and VGG-11)
                if arch == 'mobilenetv2':
                    # there might be None sparse layer (the first pw layer is None)
                    sparse_module_original_len = len(sparse_modules)
                    sparse_modules = list(filter(lambda m: m is not None, sparse_modules))
                    sparse_module_not_none_len = len(sparse_modules)
                    assert (sparse_module_original_len - sparse_module_not_none_len) == 1, \
                        f"there must only one None pw layer, got {sparse_module_original_len - sparse_module_not_none_len}"
                sparsity_loss = _compute_polarization_sparsity(sparse_modules,
                                                               lbd=sparsity, t=t, alpha=alpha,
                                                               bn_weights_mean=sparse_weights_mean,
                                                               loss_type=loss_type)
        else:
            # only for ResNet-50
            # use different lbd, t and alpha on bn3 (or gate3) for ResNet-50
            # different layers share the same mean value
            if flops_weighted:
                sparsity_loss = 0.
                for submodule in model.modules():
                    # use different parameter for different layers
                    if isinstance(submodule, models.Bottleneck):
                        submodule: models.Bottleneck
                        flops_weight = submodule.conv_flops_weight
                        # get sparse layers in the building block
                        if gate:
                            sparse_layers = [submodule.gate1, submodule.gate2, submodule.gate3]
                        else:
                            sparse_layers = [submodule.bn1, submodule.bn2, submodule.bn3]

                        # layer filter
                        if bn3_only:
                            # only sparse3 layer
                            sparse_layers = sparse_layers[-1]
                            flops_weight = flops_weight[-1]
                            sparse3_idx = 0
                        elif not sparsity_on_bn3:
                            # sparse1,2 layer
                            sparse_layers = sparse_layers[:2]
                            flops_weight = flops_weight[:2]
                            sparse3_idx = None  # no sparse3 layer
                        else:
                            # apply polarization to all layers
                            # the last layer is the sparse3 layer
                            sparse3_idx = 2

                        # weighted polarization
                        for layer_idx, (sparse_layer, weight) in enumerate(zip(sparse_layers, flops_weight)):
                            if sparse3_idx == layer_idx:
                                # use sparse3 index instead of default parameter
                                cur_lambda = bn3_argument['lbd']
                                cur_t = bn3_argument['t']
                                cur_alpha = bn3_argument['alpha']
                            else:
                                # use default parameter
                                cur_lambda = sparsity
                                cur_t = t
                                cur_alpha = alpha

                            # linear rescale the weight from [0, 1] to [lambda_min, lambda_max]
                            weight = weight_min + (weight_max - weight_min) * weight

                            if layerwise:
                                # compute layer-wise mean
                                layerwise_mean = sparse_layer.weight.view(-1).mean()
                                sparse_weights_mean = layerwise_mean
                            sparsity_loss += _compute_polarization_sparsity([sparse_layer],
                                                                            lbd=cur_lambda * weight, t=cur_t,
                                                                            alpha=cur_alpha,
                                                                            bn_weights_mean=sparse_weights_mean,
                                                                            loss_type=loss_type)
                            pass
                pass
            else:
                # the sparse loss of bn1 and bn2 layers
                # set keep_out as True to exclude bn3 layers
                sparsity_loss = 0.
                if not bn3_only:
                    # sparsity on bn1 and bn2 layers
                    sparse_layers_exclude_bn3 = model.module.get_sparse_layer(gate=gate,
                                                                              sparse1=True,
                                                                              sparse2=True,
                                                                              sparse3=False, )
                    sparsity_loss += _compute_polarization_sparsity(sparse_layers_exclude_bn3,
                                                                    lbd=sparsity, t=t, alpha=alpha,
                                                                    bn_weights_mean=sparse_weights_mean,
                                                                    loss_type=loss_type)

                # the sparsity loss of bn3 layers
                if sparsity_on_bn3:
                    # if not sparsity_on_bn3, do not prune the layer in each building blocks
                    # only apply sparsity loss on bn3 when sparsity_on_bn3
                    sparse_layers_bn3 = model.module.get_sparse_layer(gate=gate,
                                                                      sparse1=False,
                                                                      sparse2=False,
                                                                      sparse3=True, )
                    sparsity_loss += _compute_polarization_sparsity(sparse_layers_bn3,
                                                                    lbd=bn3_argument['lbd'],
                                                                    t=bn3_argument['t'],
                                                                    alpha=bn3_argument['alpha'],
                                                                    bn_weights_mean=sparse_weights_mean,
                                                                    loss_type=loss_type)

        return sparsity_loss
    else:
        raise NotImplementedError(f"Unsupported loss: {loss_type}")


def _compute_polarization_sparsity(sparse_modules: list, lbd, t, alpha, bn_weights_mean, loss_type):
    sparsity_loss = 0
    for m in sparse_modules:
        if loss_type in {LossType.POLARIZATION,
                         LossType.POLARIZATION_GRAD}:
            sparsity_term = t * torch.sum(torch.abs(m.weight)) - torch.sum(
                torch.abs(m.weight - alpha * bn_weights_mean))
        elif loss_type == LossType.L2_POLARIZATION:
            sparsity_term = t * torch.sum(torch.abs(m.weight)) - torch.sum(
                (m.weight - alpha * bn_weights_mean) ** 2)
        else:
            raise ValueError("Do not support loss {}".format(loss_type))
        sparsity_loss += lbd * sparsity_term

    return sparsity_loss


def clamp_bn(model, gate, lower_bound=0, upper_bound=1):
    # allow gate value > 1 (but be careful!)
    # if gate and (lower_bound != 0 or upper_bound != 1):
    #     raise ValueError(f"SparseGate is supposed to clamp to [0, 1], got [{lower_bound}, {upper_bound}] ")

    if gate:
        clamp_modules = list(filter(lambda m: isinstance(m, models.common.SparseGate), model.modules()))
    else:
        clamp_modules = list(
            filter(lambda m: isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d), model.modules()))

    for m in clamp_modules:
        m.weight.data.clamp_(lower_bound, upper_bound)


def report_prune_result(model):
    print("*******PRUNING REPORT*******")
    for k, m in enumerate(model.modules()):
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            weight_copy = m.weight.data.abs().clone()
            thre = 0.01
            mask = weight_copy.gt(thre)
            mask = mask.float().cuda()
            print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                  format(k, mask.shape[0], int(torch.sum(mask))))
    print("****************************")


def prune_while_training(model: nn.Module, arch: int, prune_mode: str, width_multiplier=1.):
    if isinstance(model, nn.DataParallel) or isinstance(model, nn.parallel.DistributedDataParallel):
        model = model.module

    if arch == "resnet50":
        from resprune_expand_gate import prune_resnet
        saved_model = prune_resnet(model, pruning_strategy='grad',
                                   sanity_check=False, prune_mode=prune_mode)
        baseline_model = resnet50(width_multiplier=1., gate=False, aux_fc=False)
    elif arch == 'mobilenetv2':
        from prune_mobilenetv2 import prune_mobilenet
        saved_model, _, _ = prune_mobilenet(model, pruning_strategy='grad',
                                            sanity_check=False, force_same=False,
                                            width_multiplier=width_multiplier)
        baseline_model = mobilenet_v2(inverted_residual_setting=None,
                                      width_mult=1., use_gate=False)
    else:
        # not available
        raise NotImplementedError(f"do not support arch {arch}")

    saved_flops = compute_conv_flops(saved_model, cuda=True)
    baseline_flops = compute_conv_flops(baseline_model, cuda=True)

    return saved_flops, baseline_flops


def train(train_loader, model, criterion, optimizer, epoch, sparsity, args, is_debug=False,
          writer=None):
    print("rank #{}: start training epoch {}".format(args.rank, epoch))

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    avg_sparsity_loss = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (image, target) in enumerate(train_loader):
        # the adjusting only work when epoch is at decay_epoch
        adjust_learning_rate(optimizer, epoch, lr=args.lr, decay_epoch=args.decay_epoch,
                             total_epoch=args.epochs,
                             train_loader_len=len(train_loader), iteration=i,
                             warmup=args.warmup, decay_strategy=args.lr_strategy)

        # measure data loading time
        data_time.update(time.time() - end)

        image = image.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(image)
        if isinstance(output, tuple):
            output, extra_info = output
        loss = criterion(output, target)
        losses.update(loss.data.item(), image.size(0))

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        top1.update(prec1[0], image.size(0))
        top5.update(prec5[0], image.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        if args.loss in {LossType.POLARIZATION,
                         LossType.POLARIZATION_GRAD,
                         LossType.L2_POLARIZATION}:
            if args.fc_sparsity == "unified" and args.bn3_sparsity == 'unified':
                # default behaviour
                sparsity_loss = bn_sparsity(model, args.loss, args.lbd, args.t, args.alpha,
                                            sparsity_on_bn3=args.last_sparsity,
                                            arch=args.arch,
                                            gate=args.gate,
                                            keep_out=args.keep_out,
                                            flops_weighted=args.flops_weighted,
                                            weight_min=args.weight_min,
                                            weight_max=args.weight_max,
                                            weighted_mean=args.weighted_mean,
                                            bn3_argument={'lbd': args.bn3_lbd,
                                                          't': args.bn3_t,
                                                          'alpha': args.bn3_alpha},
                                            layerwise=args.layerwise)

            elif args.fc_sparsity == "separate":
                # use average value for CNN and FC separately
                # note: the separate option is only available for VGG-like network (CNN with more than one fc layers)

                # handle different cases for dp warpper
                feature_module = model.features if hasattr(model, "features") else model.module.features
                classifier_module = model.classifier if hasattr(model, "classifier") else model.module.classifier

                sparsity_loss_feature = bn_sparsity(feature_module,
                                                    args.loss, args.lbd, args.t, args.alpha,
                                                    sparsity_on_bn3=args.last_sparsity,
                                                    arch=args.arch,
                                                    gate=args.gate,
                                                    keep_out=args.keep_out,
                                                    bn3_argument=None)
                sparsity_loss_classifier = bn_sparsity(classifier_module,
                                                       args.loss, args.lbd, args.t, args.alpha,
                                                       sparsity_on_bn3=args.last_sparsity,
                                                       arch=args.arch,
                                                       gate=args.gate,
                                                       keep_out=args.keep_out,
                                                       bn3_argument=None)
                sparsity_loss = sparsity_loss_feature + sparsity_loss_classifier
            elif args.fc_sparsity == "single":
                # apply bn_sparsity for each FC layer

                # handle different cases for dp warpper
                feature_module = model.features if hasattr(model, "features") else model.module.features
                classifier_module = model.classifier if hasattr(model, "classifier") else model.module.classifier

                sparsity_loss_feature = bn_sparsity(feature_module,
                                                    args.loss, args.lbd, args.t, args.alpha,
                                                    sparsity_on_bn3=args.last_sparsity,
                                                    arch=args.arch,
                                                    gate=args.gate,
                                                    keep_out=args.keep_out,
                                                    bn3_argument=None)
                sparsity_loss_classifier = 0.
                for name, submodule in classifier_module.named_modules():
                    if isinstance(submodule, nn.BatchNorm1d):
                        sparsity_loss_classifier += bn_sparsity(submodule,
                                                                args.loss, args.lbd, args.t, args.alpha,
                                                                sparsity_on_bn3=args.last_sparsity,
                                                                arch=args.arch,
                                                                gate=args.gate,
                                                                keep_out=args.keep_out,
                                                                bn3_argument=None)
                sparsity_loss = sparsity_loss_feature + sparsity_loss_classifier
            elif args.bn3_sparsity == 'separate':
                # use different mean for bn3 layers
                # only for ResNet-50

                # sparsity on bn1 and bn2 layers
                sparsity_loss = bn_sparsity(model, args.loss, args.lbd, args.t, args.alpha,
                                            sparsity_on_bn3=False,
                                            arch=args.arch,
                                            gate=args.gate,
                                            keep_out=args.keep_out,
                                            bn3_only=False,
                                            bn3_argument=None,
                                            flops_weighted=args.flops_weighted,
                                            weighted_mean=args.weighted_mean,
                                            weight_min=args.weight_min,
                                            weight_max=args.weight_max, )
                # sparsity on bn3 layers
                sparsity_loss += bn_sparsity(model, args.loss,
                                             # this 3 arguments will be ignored
                                             sparsity=0., t=args.t, alpha=args.alpha,
                                             sparsity_on_bn3=True,
                                             bn3_only=True,
                                             arch=args.arch,
                                             gate=args.gate,
                                             keep_out=args.keep_out,
                                             flops_weighted=args.flops_weighted,
                                             weighted_mean=args.weighted_mean,
                                             weight_min=args.weight_min,
                                             weight_max=args.weight_max,
                                             bn3_argument={'lbd': args.bn3_lbd,
                                                           't': args.bn3_t,
                                                           'alpha': args.bn3_alpha})
            else:
                raise NotImplementedError(f"do not support --fc-sparsity as {args.fc_sparsity}")
            loss += sparsity_loss
            avg_sparsity_loss.update(sparsity_loss.data.item(), image.size(0))

        loss.backward()
        if args.loss == LossType.L1_SPARSITY_REGULARIZATION:
            updateBN(model, sparsity,
                     sparsity_on_bn3=args.last_sparsity,
                     is_mobilenet=args.arch == "mobilenetv2",
                     gate=args.gate,
                     exclude_out=args.keep_out)
        # BN_grad_zero(model)
        optimizer.step()
        if args.loss in {LossType.POLARIZATION,
                         LossType.POLARIZATION_GRAD,
                         LossType.L2_POLARIZATION} or \
                (args.loss == LossType.L1_SPARSITY_REGULARIZATION and args.gate):
            # if enable gate, do not clamp bn, clamp gate to [0, 1]
            clamp_bn(model, gate=args.gate, upper_bound=args.clamp)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and (args.rank == 0):
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Sparsity Loss {s_loss.val:.4f} ({s_loss.avg:.4f})\t'
                  'Learning rate {lr}\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, s_loss=avg_sparsity_loss,
                top1=top1, top5=top5, lr=optimizer.param_groups[0]['lr']))
        if is_debug and i >= 5:
            break

    if writer is not None:
        writer.add_scalar("train/cross_entropy", losses.avg, epoch)
        writer.add_scalar("train/sparsity_loss", avg_sparsity_loss.avg, epoch)
        writer.add_scalar("train/top1", top1.avg.item(), epoch)
        writer.add_scalar("train/top5", top5.avg.item(), epoch)


def validate(val_loader, model, criterion, epoch, args, writer=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (image, target) in enumerate(val_loader):
            image = image.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(image)
            if isinstance(output, tuple):
                output, out_aux = output
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.data.item(), image.size(0))
            top1.update(prec1[0], image.size(0))
            top5.update(prec5[0], image.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))
            if args.debug and i >= 5:
                break

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    if writer is not None:
        writer.add_scalar("val/cross_entropy", losses.avg, epoch)
        writer.add_scalar("val/top1", top1.avg.item(), epoch)
    return top1.avg


def save_checkpoint(state, is_best, filepath, save_backup, backup_path, epoch, name='checkpoint.pth.tar'):
    torch.save(state, os.path.join(filepath, name))
    if is_best:
        shutil.copyfile(os.path.join(filepath, name), os.path.join(filepath, 'model_best.pth.tar'))
    if save_backup:
        shutil.copyfile(os.path.join(filepath, name),
                        os.path.join(backup_path, 'checkpoint_{}.pth.tar'.format(epoch)))


if __name__ == '__main__':
    main()
