# Train on CIFAR

> To train on CIFAR-100, use `--dataset cifar100`.

## CIFAR10/100

### Naive L1 Structured (w/o L1 sparsity regularizer in training)
1. Sparsity Train
    ```bash
    python -u main_train.py \
    --dataset cifar10 --loss-type original \
    --bn-wd \
    --save ./cifar10/naive/ckpts/ --log ./cifar10/naive/events/
    ```
#### 80%
2. Pruning
    ```bash
    python -u main_prune.py \
    --dataset cifar10 \
    --model ./cifar10/naive/ckpts/model_best.pth.tar \
    --prune-type l1-norm --pruning-strategy fixed --l1-norm-ratio 0.2 \
    --save ./cifar10/p80/naive/ckpts/
    ```
3. Fine-tuning
    ```bash
    python -u main_finetune.py \
    --dataset cifar10 \
    --bn-wd --expand \
    --refine ./cifar10/p80/naive/ckpts/pruned_fixed.pth.tar \
    --save ./cifar10/p80/naive/ckpts/ --log ./cifar10/p80/naive/events/
    ```

#### 60%
2. Pruning
    ```bash
    python -u main_prune.py \
    --dataset cifar10 \
    --model ./cifar10/naive/ckpts/model_best.pth.tar \
    --prune-type l1-norm --pruning-strategy fixed --l1-norm-ratio ? \
    --save ./cifar10/p60/naive/ckpts/
    ```
3. Fine-tuning
    ```bash
    python -u main_finetune.py \
    --dataset cifar10 \
    --bn-wd --expand \
    --refine ./cifar10/p60/naive/ckpts/pruned_fixed.pth.tar \
    --save ./cifar10/p60/naive/ckpts/ --log ./cifar10/p60/naive/events/
    ```

#### 40%
2. Pruning
    ```bash
    python -u main_prune.py \
    --dataset cifar10 \
    --model ./cifar10/naive/ckpts/model_best.pth.tar \
    --prune-type l1-norm --pruning-strategy fixed --l1-norm-ratio ? \
    --save ./cifar10/p40/naive/ckpts/
    ```
3. Fine-tuning
    ```bash
    python -u main_finetune.py \
    --dataset cifar10 \
    --bn-wd --expand \
    --refine ./cifar10/p40/naive/ckpts/pruned_fixed.pth.tar \
    --save ./cifar10/p40/naive/ckpts/ --log ./cifar10/p40/naive/events/
    ```

### L1 Structured (w/ L1 sparsity regularizer in training)
1. Sparsity Train
    ```bash
    python -u main_train.py \
    --dataset cifar10 --loss-type l1-sr \
    --bn-wd \
    --save ./cifar10/l1-sr/ckpts/ --log ./cifar10/l1-sr/events/
    ```

#### 80%
2. Pruning
    ```bash
    python -u main_prune.py \
    --dataset cifar10 \
    --model ./cifar10/l1-sr/ckpts/model_best.pth.tar \
    --prune-type l1-norm --pruning-strategy fixed --l1-norm-ratio 0.2 \
    --save ./cifar10/p80/l1-sr/ckpts/
    ```
3. Fine-tuning
    ```bash
    python -u main_finetune.py \
    --dataset cifar10 \
    --bn-wd --expand \
    --refine ./cifar10/p80/l1-sr/ckpts/pruned_fixed.pth.tar \
    --save ./cifar10/p80/l1-sr/ckpts/ --log ./cifar10/p80/l1-sr/events/
    ```

#### 60%
2. Pruning
    ```bash
    python -u main_prune.py \
    --dataset cifar10 \
    --model ./cifar10/l1-sr/ckpts/model_best.pth.tar \
    --prune-type l1-norm --pruning-strategy fixed --l1-norm-ratio 0.2 \
    --save ./cifar10/p60/l1-sr/ckpts/
    ```
3. Fine-tuning
    ```bash
    python -u main_finetune.py \
    --dataset cifar10 \
    --bn-wd --expand \
    --refine ./cifar10/p60/l1-sr/ckpts/pruned_fixed.pth.tar \
    --save ./cifar10/p60/l1-sr/ckpts/ --log ./cifar10/p60/l1-sr/events/
    ```

#### 40%
2. Pruning
    ```bash
    python -u main_prune.py \
    --dataset cifar10 \
    --model ./cifar10/l1-sr/ckpts/model_best.pth.tar \
    --prune-type l1-norm --pruning-strategy fixed --l1-norm-ratio 0.2 \
    --save ./cifar10/p40/l1-sr/ckpts/
    ```
3. Fine-tuning
    ```bash
    python -u main_finetune.py \
    --dataset cifar10 \
    --bn-wd --expand \
    --refine ./cifar10/p40/l1-sr/ckpts/pruned_fixed.pth.tar \
    --save ./cifar10/p40/l1-sr/ckpts/ --log ./cifar10/p40/l1-sr/events/
    ```

### L1 Polarization (need to try)
#### 80%
1. Sparsity Train
    ```bash
    python -u main_train.py \
    --dataset cifar10 --loss-type polar \
    --target-flops 0.8 \
    --gate --bn-wd \
    --save ./cifar10/p80/polar/ckpts/ --log ./cifar10/p80/polar/events/
    ```
2. Pruning
    ```bash
    python -u main_prune.py \
    --dataset cifar10 \
    --model ./cifar10/p80/polar/ckpts/model_best.pth.tar \
    --prune-type polarization --pruning-strategy grad \
    --gate \
    --save ./cifar10/p80/polar/ckpts/
    ```
3. Fine-tuning
    ```bash
    python -u main_finetune.py \
    --dataset cifar10 \
    --bn-wd --expand \
    --refine ./cifar10/p80/polar/ckpts/pruned_grad.pth.tar \
    --save ./cifar10/p80/polar/ckpts/ --log ./cifar10/p80/polar/events/
    ```
#### 60%
1. Sparsity Train
    ```bash
    python -u main_train.py \
    --dataset cifar10 --loss-type polar \
    --target-flops 0.6 \
    --gate --bn-wd \
    --save ./cifar10/p60/polar/ckpts/ --log ./cifar10/p60/polar/events/
    ```
2. Pruning
    ```bash
    python -u main_prune.py \
    --dataset cifar10 \
    --model ./cifar10/p60/polar/ckpts/model_best.pth.tar \
    --prune-type polarization --pruning-strategy grad \
    --gate \
    --save ./cifar10/p60/polar/ckpts/
    ```
3. Fine-tuning
    ```bash
    python -u main_finetune.py \
    --dataset cifar10 \
    --bn-wd --expand \
    --refine ./cifar10/p60/polar/ckpts/pruned_grad.pth.tar \
    --save ./cifar10/p60/polar/ckpts/ --log ./cifar10/p60/polar/events/
    ```

#### 40%
1. Sparsity Train
    ```bash
    python -u main_train.py \
    --dataset cifar10 --loss-type polar \
    --target-flops 0.6 \
    --gate --bn-wd \
    --save ./cifar10/p40/polar/ckpts/ --log ./cifar10/p40/polar/events/
    ```
2. Pruning
    ```bash
    python -u main_prune.py \
    --dataset cifar10 \
    --model ./cifar10/p40/polar/ckpts/model_best.pth.tar \
    --prune-type polarization --pruning-strategy grad \
    --gate \
    --save ./cifar10/p40/polar/ckpts/
    ```
3. Fine-tuning
    ```bash
    python -u main_finetune.py \
    --dataset cifar10 \
    --bn-wd --expand \
    --refine ./cifar10/p40/polar/ckpts/pruned_grad.pth.tar \
    --save ./cifar10/p40/polar/ckpts/ --log ./cifar10/p40/polar/events/
    ```