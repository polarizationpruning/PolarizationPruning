# Train on CIFAR

> To train on CIFAR-100, use `--dataset cifar100`.

## CIFAR10/100

### Random Unstructured
### Random Structured
### L1 Unstructured
### L1 Structured
### Global Unstructured
### L1 Polarization
0. Create ckpt directory
    ```bash
    mkdir -p ./polar-80/ckpts/
    mkdir -p ./polar-80/events/
    ```
1. Sparsity Train
    ```bash
    python -u main_train.py \
    --dataset cifar10 --loss-type polar \
    --target-flops 0.8 \
    --gate --bn-wd \
    --save ./polar-80/ckpts/ --log ./polar-80/events/
    ```
2. Pruning
    ```bash
    python -u main_prune.py \
    --dataset cifar10 --pruning-strategy grad \
    --gate \
    --model ./polar-80/ckpts/model_best.pth.tar \
    --save ./polar-80/ckpts/
    ```
3. Fine-tuning
    ```bash
    python -u main_finetune.py \
    --dataset cifar10 \
    --bn-wd --expand \
    --refine ./polar-80/ckpts/pruned_grad.pth.tar \
    --save ./polar-80/ckpts/ --log ./polar-80/events/
    ```