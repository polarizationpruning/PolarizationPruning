# Neuron-level Structured Pruning using Polarization Regularizer

NeurIPS 2020

## Introduction

Pipeline:

1. Sparsity Training
2. Pruning
3. Fine-tuning

##  Running

We test our code on Python 3.6. Our code is *incompatible* with Python 2.x.

Install packages:

```bash
pip install -r requirements.txt
```

> We recommend to run the code on PyTorch 1.2 and CUDA 10.0. The project is *incompatible* with PyTorch <= 1.0.

See README in `./imagenet` or `./cifar` for guidelines on running experiments on ImageNet ([ILSVRC-12](http://image-net.org/challenges/LSVRC/2012/)) or CIFAR10/100 datasets.

We upload the the pruned checkpoints on [OneDrive](https://1drv.ms/u/s!AkN_Fy35WZXGmMRwSXPvC2hseXTtYQ?e=3Dygzl).

## Note

### Pruning strategy

We introduce a novel pruning method in our paper (Fig. 2). We have implemented multiple pruning methods in our code (option `--pruning-strategy`).

- `grad`: The method introduced in our paper (Section 3.3).
- `fixed`: Use a global pruning threshold for all layers (0.01 as default).
- `percent`: Determine the threshold by a global pruning percent (as Network Slimming).
- `search`: **Deprecated**. Not recommend to use.

### Loss Type

- `original`: There is no any sparse regularization on the loss function, i.e., baseline model.
- `sr`: Apply L1 regularization on the scaling factors, i.e., [Network Slimming](https://arxiv.org/abs/1708.06519).
- `zol`: Polarization regularization. See equation 2 in the paper.


## Acknowledgement

We build our code based on [rethinking-network-pruning](https://github.com/Eric-mingjie/rethinking-network-pruning). We'd like to thank their contribution to the research on structured pruning.
