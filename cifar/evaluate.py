from __future__ import print_function

import argparse

import os
from typing import Dict
import numpy as np
import random
from typing import Any, Dict

import torch
from torch.cuda import random
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

import common
from common import SnippetTimer, compute_conv_flops
from models.resnet_expand import resnet56 as resnet50_expand, ResNetExpand

def calculate_flops(current_model):
    if args.arch == "resnet56":
        model_ref = resnet50_expand(num_classes=num_classes)
    else:
        raise NotImplementedError()

    current_flops = compute_conv_flops(current_model.cpu())
    ref_flops = compute_conv_flops(model_ref.cpu())
    flops_ratio = current_flops / ref_flops

    print("FLOPs remains {}".format(flops_ratio))

parser = argparse.ArgumentParser(description='Pytorch CIFAR evaluation')


parser.add_argument('--arch', default='resnet56', type=str,
                    help='architecture to use')
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'],
                    help='training dataset (default: cifar10)')
parser.add_argument('--repeat', type=int, default=5,
                    help='test repeat time')
# CONFIG
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--gate', action='store_true', default=False,
                    help='Add an extra scaling factor after the BatchNrom layers.')

# LOAD
parser.add_argument('--original', type=str,
                    help='path to the orignal model ckpt')
parser.add_argument('--fine-tuned', type=str,
                    help='path to the fine-tuned model ckpt')

# NO NEED TO CHANGE
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--seed', type=int, metavar='S', default=123,
                    help='random seed (default: a random int)')
parser.add_argument('--input-mask', action='store_true',
                    help='If use input mask in ResNet models.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
if not args.seed:
    args.seed = random.randint(500, 1000)

print(args)
print(f"Current git hash: {common.get_git_id()}")

#------- Reproducibility -------
torch.manual_seed(args.seed)
np.random.seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#------- Dataset -------
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
if args.dataset == 'cifar10':
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data.cifar10', download=True, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
else:
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('./data.cifar100', download=True, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

num_classes = 10 if args.dataset == 'cifar10' else 100

def test(mdl):
    mdl.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            output = mdl(data)
            if isinstance(output, tuple):
                output, output_aux = output
            test_loss += F.cross_entropy(output, target, size_average=False).data.item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * float(correct) / len(test_loader.dataset)))
    return float(correct) / float(len(test_loader.dataset))

#------ Load & Test ------
if args.original and os.path.isfile(args.original):
    ## Original
    ori_ckpt: Dict[str, Any] = torch.load(args.original)
    print(f"=> Loading the original model...\n=> Epoch: {ori_ckpt['epoch']}, Acc.: {ori_ckpt['best_prec1']}")

    ori_model: ResNetExpand = resnet50_expand(num_classes=num_classes,
                                            aux_fc=False,
                                            gate=args.gate,
                                            use_input_mask=args.input_mask)
    ori_model.load_state_dict(ori_ckpt['state_dict'])

    with SnippetTimer("Original Model", args.repeat):
        for _ in range(args.repeat):
            test(ori_model)

if args.fine_tuned and os.path.isfile(args.fine_tuned):
    ## Fine-tuned
    tuned_ckpt: Dict[str, Any] = torch.load(args.fine_tuned)
    print(f"=> Loading the fine-tuned model...\n=> Epoch: {tuned_ckpt['epoch']}, Acc.: {tuned_ckpt['best_prec1']}")

    tuned_model: ResNetExpand = resnet50_expand(num_classes=num_classes,
                                                aux_fc=False,
                                                gate=False,
                                                cfg=tuned_ckpt['cfg'],
                                                expand_idx=tuned_ckpt['expand_idx'],
                                                use_input_mask=args.input_mask)
    tuned_model.load_state_dict(tuned_ckpt['state_dict'])

    calculate_flops(tuned_model)

    with SnippetTimer("Fine-tuned Model", args.repeat):
        for _ in range(args.repeat):
            test(tuned_model)
