# Neuron-level Structured Pruning using Polarization Regularizer

ICML 2020 Anonymous Submission #2062

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

See `./imagenet` or `./cifar` for detail guides running on ImageNet ([ILSVRC-12](http://image-net.org/challenges/LSVRC/2012/)) or CIFAR10/100.

## Checkpoints

~~We release our checkpoints, training logs and TensorBoard events on [Google Drive](#). Hope them help you better understand our work.~~

We have uploaded our training logs and Tensorboard events on CMT. Limited by the upload file size, we can't upload checkpoints. We will release checkpoints as soon as possible after review.

## Note

### Implementation of the FLOPs computing

We compute the FLOPs for all layers (Conv, Linear, BN, ReLU, ...), instead of only computing Conv layers.

### Pruning strategy

We introduce a novel pruning method in our paper (Fig. 2). We have implemented multiple pruning methods in our code (option `--pruning-strategy`).

- `grad`: The method introduced in our paper.
- `fixed`: Use a global pruning threshold for all layers (0.01 as default).
- `percent`: Determine the threshold by a global pruning percent (as Network Slimming).
- `search`: **Deprecated**. Not recommend to use.

In our practice, `grad` and `fixed` always give similar results at pruning polarized distributions.

### Loss Type

- `original`: There is no any sparse regularization on the loss function, i.e., baseline model.
- `sr`: Apply L1 regularization on the loss function, i.e., [Network Slimming](https://arxiv.org/abs/1708.06519).
- `zol`: Polarization regularization. We implement the polarization regularizer as <img src="https://latex.codecogs.com/svg.latex?R_s(\bm{\gamma})&space;=&space;t&space;\lVert&space;\bm{\gamma}&space;\rVert_1&space;-&space;\lVert&space;\bm{\gamma}&space;-&space;\alpha&space;\bar&space;\gamma&space;\bm{1}_n\rVert_1" title="R_s(\bm{\gamma}) = t \lVert \bm{\gamma} \rVert_1 - \lVert \bm{\gamma} - \alpha \bar \gamma \bm{1}_n\rVert_1" />, and always set α as 1. The implementation is equivalent to the equation 2 in the paper. The argument `--lbd` is the coefficient of the sparsity regularization term (i.e., λ), as shown in the equation 1 in the paper.



## Acknowledgement

We build our code based on [rethinking-network-pruning](https://github.com/Eric-mingjie/rethinking-network-pruning). We'd like to thank their contribution to the pruning community.

