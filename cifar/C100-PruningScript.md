# Train on CIFAR

> To train on CIFAR-100, use `--dataset cifar100`.

## CIFAR100

### Naive L1 Structured (w/o L1 sparsity regularizer in training)
1. Sparsity Train (Check 0.7253)
    ```bash
    CUDA_VISIBLE_DEVICES=0 python -u main_train.py \
    --dataset cifar100 --loss-type original \
    --bn-wd \
    --save ./cifar100/naive/ckpts/ --log ./cifar100/naive/events/
    ```
#### 80%
2. Pruning (check 0.8199269279336435)
    ```bash
    CUDA_VISIBLE_DEVICES=0 python -u main_prune.py \
    --dataset cifar100 \
    --model ./cifar100/naive/ckpts/model_best.pth.tar \
    --prune-type l1-norm --pruning-strategy fixed --l1-norm-ratio 0.12 \
    --save ./cifar100/p80/naive/ckpts/
    ```
3. Fine-tuning (check 0.7141)
    ```bash
    CUDA_VISIBLE_DEVICES=0 python -u main_finetune.py \
    --dataset cifar100 \
    --bn-wd --expand \
    --refine ./cifar100/p80/naive/ckpts/pruned_fixed.pth.tar \
    --save ./cifar100/p80/naive/ckpts/ --log ./cifar100/p80/naive/events/
    ```

#### 60%
2. Pruning (check 0.6044194116291073)
    ```bash
    CUDA_VISIBLE_DEVICES=0 python -u main_prune.py \
    --dataset cifar100 \
    --model ./cifar100/naive/ckpts/model_best.pth.tar \
    --prune-type l1-norm --pruning-strategy fixed --l1-norm-ratio 0.28 \
    --save ./cifar100/p60/naive/ckpts/
    ```
3. Fine-tuning (check 0.7022)
    ```bash
    CUDA_VISIBLE_DEVICES=0 python -u main_finetune.py \
    --dataset cifar100 \
    --bn-wd --expand \
    --refine ./cifar100/p60/naive/ckpts/pruned_fixed.pth.tar \
    --save ./cifar100/p60/naive/ckpts/ --log ./cifar100/p60/naive/events/
    ```

#### 40%
2. Pruning (check 0.4011257626973425)
    ```bash
    CUDA_VISIBLE_DEVICES=0 python -u main_prune.py \
    --dataset cifar100 \
    --model ./cifar100/naive/ckpts/model_best.pth.tar \
    --prune-type l1-norm --pruning-strategy fixed --l1-norm-ratio 0.45 \
    --save ./cifar100/p40/naive/ckpts/
    ```
3. Fine-tuning (check 0.6848)
    ```bash
    CUDA_VISIBLE_DEVICES=0 python -u main_finetune.py \
    --dataset cifar100 \
    --bn-wd --expand \
    --refine ./cifar100/p40/naive/ckpts/pruned_fixed.pth.tar \
    --save ./cifar100/p40/naive/ckpts/ --log ./cifar100/p40/naive/events/
    ```

### L1 Structured (w/ L1 sparsity regularizer in training)
1. Sparsity Train (check 0.7092)
    ```bash
    CUDA_VISIBLE_DEVICES=1 python -u main_train.py \
    --dataset cifar100 --loss-type l1-sr \
    --bn-wd \
    --save ./cifar100/l1-sr/ckpts/ --log ./cifar100/l1-sr/events/
    ```

#### 80%
2. Pruning (check 0.8199269279336435)
    ```bash
    CUDA_VISIBLE_DEVICES=1 python -u main_prune.py \
    --dataset cifar100 \
    --model ./cifar100/l1-sr/ckpts/model_best.pth.tar \
    --prune-type l1-norm --pruning-strategy fixed --l1-norm-ratio 0.12 \
    --save ./cifar100/p80/l1-sr/ckpts/
    ```
3. Fine-tuning (check 0.6992)
    ```bash
    CUDA_VISIBLE_DEVICES=1 python -u main_finetune.py \
    --dataset cifar100 \
    --bn-wd --expand \
    --refine ./cifar100/p80/l1-sr/ckpts/pruned_fixed.pth.tar \
    --save ./cifar100/p80/l1-sr/ckpts/ --log ./cifar100/p80/l1-sr/events/
    ```

#### 60%
2. Pruning (check 0.6044194116291073)
    ```bash
    CUDA_VISIBLE_DEVICES=1 python -u main_prune.py \
    --dataset cifar100 \
    --model ./cifar100/l1-sr/ckpts/model_best.pth.tar \
    --prune-type l1-norm --pruning-strategy fixed --l1-norm-ratio 0.28 \
    --save ./cifar100/p60/l1-sr/ckpts/
    ```
3. Fine-tuning (check 0.6946)
    ```bash
    CUDA_VISIBLE_DEVICES=1 python -u main_finetune.py \
    --dataset cifar100 \
    --bn-wd --expand \
    --refine ./cifar100/p60/l1-sr/ckpts/pruned_fixed.pth.tar \
    --save ./cifar100/p60/l1-sr/ckpts/ --log ./cifar100/p60/l1-sr/events/
    ```

#### 40%
2. Pruning (check 0.4011257626973425)
    ```bash
    CUDA_VISIBLE_DEVICES=1 python -u main_prune.py \
    --dataset cifar100 \
    --model ./cifar100/l1-sr/ckpts/model_best.pth.tar \
    --prune-type l1-norm --pruning-strategy fixed --l1-norm-ratio 0.45 \
    --save ./cifar100/p40/l1-sr/ckpts/
    ```
3. Fine-tuning (check 0.6841)
    ```bash
    CUDA_VISIBLE_DEVICES=1 python -u main_finetune.py \
    --dataset cifar100 \
    --bn-wd --expand \
    --refine ./cifar100/p40/l1-sr/ckpts/pruned_fixed.pth.tar \
    --save ./cifar100/p40/l1-sr/ckpts/ --log ./cifar100/p40/l1-sr/events/
    ```

### L1 Polarization
#### 80%
1. Sparsity Train (check 0.7177)
    ```bash
    CUDA_VISIBLE_DEVICES=0 python -u main_train.py \
    --dataset cifar100 --loss-type polar \
    --target-flops 0.8 \
    --gate --bn-wd \
    --save ./cifar100/p80/polar/ckpts/ --log ./cifar100/p80/polar/events/ \
    --lbd 0.0001
    ```
2. Pruning (check 0.8040961768743842)
    ```bash
    CUDA_VISIBLE_DEVICES=0 python -u main_prune.py \
    --dataset cifar100 \
    --model ./cifar100/p80/polar/ckpts/model_best.pth.tar \
    --prune-type polarization --pruning-strategy grad \
    --gate \
    --save ./cifar100/p80/polar/ckpts/
    ```
3. Fine-tuning (check 0.7162)
    ```bash
    CUDA_VISIBLE_DEVICES=0 python -u main_finetune.py \
    --dataset cifar100 \
    --bn-wd --expand \
    --refine ./cifar100/p80/polar/ckpts/pruned_grad.pth.tar \
    --save ./cifar100/p80/polar/ckpts/ --log ./cifar100/p80/polar/events/
    ```
#### 60%
1. Sparsity Train (check 0.71)
    ```bash
    CUDA_VISIBLE_DEVICES=1 python -u main_train.py \
    --dataset cifar100 --loss-type polar \
    --target-flops 0.6 \
    --gate --bn-wd \
    --save ./cifar100/p60/polar/ckpts/ --log ./cifar100/p60/polar/events/
    ```
2. Pruning (check 0.6069301164216311)
    ```bash
    CUDA_VISIBLE_DEVICES=1 python -u main_prune.py \
    --dataset cifar100 \
    --model ./cifar100/p60/polar/ckpts/model_best.pth.tar \
    --prune-type polarization --pruning-strategy grad \
    --gate \
    --save ./cifar100/p60/polar/ckpts/
    ```
3. Fine-tuning (check 0.7036)
    ```bash
    CUDA_VISIBLE_DEVICES=1 python -u main_finetune.py \
    --dataset cifar100 \
    --bn-wd --expand \
    --refine ./cifar100/p60/polar/ckpts/pruned_grad.pth.tar \
    --save ./cifar100/p60/polar/ckpts/ --log ./cifar100/p60/polar/events/
    ```

#### 40% (
1. Sparsity Train (check 0.7018)
    ```bash
    CUDA_VISIBLE_DEVICES=2 python -u main_train.py \
    --dataset cifar100 --loss-type polar \
    --target-flops 0.4 \
    --gate --bn-wd \
    --save ./cifar100/p40/polar/ckpts/ --log ./cifar100/p40/polar/events/
    ```
2. Pruning (check 0.4170666726506066)
    ```bash
    CUDA_VISIBLE_DEVICES=0 python -u main_prune.py \
    --dataset cifar100 \
    --model ./cifar100/p40/polar/ckpts/model_best.pth.tar \
    --prune-type polarization --pruning-strategy grad \
    --gate \
    --save ./cifar100/p40/polar/ckpts/
    ```
3. Fine-tuning (check 0.6969)
    ```bash
    CUDA_VISIBLE_DEVICES=0 python -u main_finetune.py \
    --dataset cifar100 \
    --bn-wd --expand \
    --refine ./cifar100/p40/polar/ckpts/pruned_grad.pth.tar \
    --save ./cifar100/p40/polar/ckpts/ --log ./cifar100/p40/polar/events/
    ```