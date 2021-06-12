# Pytorch Pruning API vs Neuron-level Structured Pruning using Polarization Regularizer

NeurIPS 2020 [**[Paper]**](https://github.com/polarizationpruning/PolarizationPruning/blob/master/NIPS2020_PolarizationPruning.pdf)
Code Reference: fork from [**[Source]**](https://github.com/polarizationpruning/PolarizationPruning)

## Pipeline:

1. Sparsity Training
2. Pruning
3. Fine-tuning

## Running
We test our code on Python 3.6. Our code is *incompatible* with Python 2.x.

Install packages:

```bash
pip install -r requirements.txt
```

> We recommend to run the code on PyTorch 1.2 and CUDA 10.0. The project is *incompatible* with PyTorch <= 1.0.

## Note

### Goal
Compare post-pruning model performance between "Pytorch pruning API" and "Polarization Regularizer" introduced in [**[Paper]**](https://github.com/polarizationpruning/PolarizationPruning/blob/master/NIPS2020_PolarizationPruning.pdf) in both **workstation** and **embedding systems** with limited resource.

### Architecture & Dataset
Due to the limitation of computational resource, we only perform experiments with CIFAR10/100 on ResNet56.

> See README in `./cifar` for more details.
