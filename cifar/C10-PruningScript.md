# Train on CIFAR

> To train on CIFAR-100, use `--dataset cifar100`.

## CIFAR10

### Naive L1 Structured (w/o L1 sparsity regularizer in training)
1. Sparsity Train (check 0.9408)
    ```bash
    CUDA_VISIBLE_DEVICES=0 python -u main_train.py \
    --dataset cifar10 --loss-type original \
    --bn-wd \
    --save ./cifar10/naive/ckpts/ --log ./cifar10/naive/events/
    ```
#### 80% (done)
2. Pruning (check 0.8199186622832295)
    ```bash
    CUDA_VISIBLE_DEVICES=0 python -u main_prune.py \
    --dataset cifar10 \
    --model ./cifar10/naive/ckpts/model_best.pth.tar \
    --prune-type l1-norm --pruning-strategy fixed --l1-norm-ratio 0.12 \
    --save ./cifar10/p80/naive/ckpts/
    ```
3. Fine-tuning (check 0.9409)
    ```bash
    CUDA_VISIBLE_DEVICES=0 python -u main_finetune.py \
    --dataset cifar10 \
    --bn-wd --expand \
    --refine ./cifar10/p80/naive/ckpts/pruned_fixed.pth.tar \
    --save ./cifar10/p80/naive/ckpts/ --log ./cifar10/p80/naive/events/
    ```

#### 60% ()
2. Pruning (check 0.6044012538289623)
    ```bash
    CUDA_VISIBLE_DEVICES=0 python -u main_prune.py \
    --dataset cifar10 \
    --model ./cifar10/naive/ckpts/model_best.pth.tar \
    --prune-type l1-norm --pruning-strategy fixed --l1-norm-ratio 0.28 \
    --save ./cifar10/p60/naive/ckpts/
    ```
3. Fine-tuning (check 0.939)
    ```bash
    CUDA_VISIBLE_DEVICES=0 python -u main_finetune.py \
    --dataset cifar10 \
    --bn-wd --expand \
    --refine ./cifar10/p60/naive/ckpts/pruned_fixed.pth.tar \
    --save ./cifar10/p60/naive/ckpts/ --log ./cifar10/p60/naive/events/
    ```

#### 40%
2. Pruning (check 0.40109827338408355)
    ```bash
    CUDA_VISIBLE_DEVICES=0 python -u main_prune.py \
    --dataset cifar10 \
    --model ./cifar10/naive/ckpts/model_best.pth.tar \
    --prune-type l1-norm --pruning-strategy fixed --l1-norm-ratio 0.45 \
    --save ./cifar10/p40/naive/ckpts/
    ```
3. Fine-tuning (check 0.9314)
    ```bash
    CUDA_VISIBLE_DEVICES=2 python -u main_finetune.py \
    --dataset cifar10 \
    --bn-wd --expand \
    --refine ./cifar10/p40/naive/ckpts/pruned_fixed.pth.tar \
    --save ./cifar10/p40/naive/ckpts/ --log ./cifar10/p40/naive/events/
    ```

### L1 Structured (w/ L1 sparsity regularizer in training)
1. Sparsity Train (check 0.9219)
    ```bash
    CUDA_VISIBLE_DEVICES=1 python -u main_train.py \
    --dataset cifar10 --loss-type l1-sr \
    --bn-wd \
    --save ./cifar10/l1-sr/ckpts/ --log ./cifar10/l1-sr/events/
    ```

#### 80% (done)
2. Pruning (check 0.8199186622832295)
    ```bash
    CUDA_VISIBLE_DEVICES=1 python -u main_prune.py \
    --dataset cifar10 \
    --model ./cifar10/l1-sr/ckpts/model_best.pth.tar \
    --prune-type l1-norm --pruning-strategy fixed --l1-norm-ratio 0.12 \
    --save ./cifar10/p80/l1-sr/ckpts/
    ```
3. Fine-tuning (check 0.9234)
    ```bash
    CUDA_VISIBLE_DEVICES=1 python -u main_finetune.py \
    --dataset cifar10 \
    --bn-wd --expand \
    --refine ./cifar10/p80/l1-sr/ckpts/pruned_fixed.pth.tar \
    --save ./cifar10/p80/l1-sr/ckpts/ --log ./cifar10/p80/l1-sr/events/
    ```

#### 60% ()
2. Pruning (check 0.6044012538289623)
    ```bash
    CUDA_VISIBLE_DEVICES=1 python -u main_prune.py \
    --dataset cifar10 \
    --model ./cifar10/l1-sr/ckpts/model_best.pth.tar \
    --prune-type l1-norm --pruning-strategy fixed --l1-norm-ratio 0.28 \
    --save ./cifar10/p60/l1-sr/ckpts/
    ```
3. Fine-tuning (check 0.9207)
    ```bash
    CUDA_VISIBLE_DEVICES=1 python -u main_finetune.py \
    --dataset cifar10 \
    --bn-wd --expand \
    --refine ./cifar10/p60/l1-sr/ckpts/pruned_fixed.pth.tar \
    --save ./cifar10/p60/l1-sr/ckpts/ --log ./cifar10/p60/l1-sr/events/
    ```

#### 40%
2. Pruning (check 0.40109827338408355)
    ```bash
    CUDA_VISIBLE_DEVICES=1 python -u main_prune.py \
    --dataset cifar10 \
    --model ./cifar10/l1-sr/ckpts/model_best.pth.tar \
    --prune-type l1-norm --pruning-strategy fixed --l1-norm-ratio 0.45 \
    --save ./cifar10/p40/l1-sr/ckpts/
    ```
3. Fine-tuning (check 0.9137)
    ```bash
    CUDA_VISIBLE_DEVICES=2 python -u main_finetune.py \
    --dataset cifar10 \
    --bn-wd --expand \
    --refine ./cifar10/p40/l1-sr/ckpts/pruned_fixed.pth.tar \
    --save ./cifar10/p40/l1-sr/ckpts/ --log ./cifar10/p40/l1-sr/events/
    ```

### L1 Polarization
#### 80%
1. Sparsity Train (check 0.9352)
    ```bash
    CUDA_VISIBLE_DEVICES=0 python -u main_train.py \
    --dataset cifar10 --loss-type polar \
    --target-flops 0.8 \
    --gate --bn-wd \
    --save ./cifar10/p80/polar/ckpts/ --log ./cifar10/p80/polar/events/ \
    --lbd 0.0001
    ```
2. Pruning (check 0.7849002965246333)
    ```bash
    CUDA_VISIBLE_DEVICES=0 python -u main_prune.py \
    --dataset cifar10 \
    --model ./cifar10/p80/polar/ckpts/model_best.pth.tar \
    --prune-type polarization --pruning-strategy grad \
    --gate \
    --save ./cifar10/p80/polar/ckpts/
    ```
3. Fine-tuning (check 0.9387)
    ```bash
    CUDA_VISIBLE_DEVICES=0 python -u main_finetune.py \
    --dataset cifar10 \
    --bn-wd --expand \
    --refine ./cifar10/p80/polar/ckpts/pruned_grad.pth.tar \
    --save ./cifar10/p80/polar/ckpts/ --log ./cifar10/p80/polar/events/
    ```
#### 60%
1. Sparsity Train (check 0.9305)
    ```bash
    CUDA_VISIBLE_DEVICES=1 python -u main_train.py \
    --dataset cifar10 --loss-type polar \
    --target-flops 0.6 \
    --gate --bn-wd \
    --save ./cifar10/p60/polar/ckpts/ --log ./cifar10/p60/polar/events/
    ```
2. Pruning (check 0.5999625646575686)
    ```bash
    CUDA_VISIBLE_DEVICES=1 python -u main_prune.py \
    --dataset cifar10 \
    --model ./cifar10/p60/polar/ckpts/model_best.pth.tar \
    --prune-type polarization --pruning-strategy grad \
    --gate \
    --save ./cifar10/p60/polar/ckpts/
    ```
3. Fine-tuning (check 0.9335)
    ```bash
    CUDA_VISIBLE_DEVICES=1 python -u main_finetune.py \
    --dataset cifar10 \
    --bn-wd --expand \
    --refine ./cifar10/p60/polar/ckpts/pruned_grad.pth.tar \
    --save ./cifar10/p60/polar/ckpts/ --log ./cifar10/p60/polar/events/
    ```

#### 40% (check)
1. Sparsity Train (check 0.9265)
    ```bash
    CUDA_VISIBLE_DEVICES=2 python -u main_train.py \
    --dataset cifar10 --loss-type polar \
    --target-flops 0.4 \
    --gate --bn-wd \
    --save ./cifar10/p40/polar/ckpts/ --log ./cifar10/p40/polar/events/
    ```
2. Pruning (check 0.4176366364497831)
    ```bash
    CUDA_VISIBLE_DEVICES=2 python -u main_prune.py \
    --dataset cifar10 \
    --model ./cifar10/p40/polar/ckpts/model_best.pth.tar \
    --prune-type polarization --pruning-strategy grad \
    --gate \
    --save ./cifar10/p40/polar/ckpts/
    ```
3. Fine-tuning (check 0.9272)
    ```bash
    CUDA_VISIBLE_DEVICES=2 python -u main_finetune.py \
    --dataset cifar10 \
    --bn-wd --expand \
    --refine ./cifar10/p40/polar/ckpts/pruned_grad.pth.tar \
    --save ./cifar10/p40/polar/ckpts/ --log ./cifar10/p40/polar/events/
    ```