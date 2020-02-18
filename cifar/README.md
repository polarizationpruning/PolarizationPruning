# Train on CIFAR

1. Sparsity Train

    ```bash
    python -u main.py --dataset cifar10 --epochs 200 --lr 0.01 --gammas 10 0.2 0.2 0.2 --decay-epoch 1 60 120 160 --arch resnet56 --loss-type zol --lbd 5e-5 --t 1.4 --test-batch-size 128 --weight-decay 5e-4 --save ./logs/${CURRENT_NAME}/ --log ./log/${CURRENT_NAME}
    ```

2. Pruning

    ```bash
    python -u resprune-expand.py --model ./logs/model_best.pth.tar --save ./logs/ --pruning-strategy grad --no-cuda --dataset cifar10
    ```

3. Fine-tuning

    ```bash
    python -u main_finetune.py --dataset cifar10 --epochs 200 --lr 1e-3 --gammas 0.5 0.5 0.4 --decay-epoch 30 80 150 --arch resnet56 --refine ./logs/pruned_grad.pth.tar --test-batch-size 128 --weight-decay 5e-4 --seed 123 --save ./logs/ --log ./log/ --expand
    ```

## Evaluate

For example, to evaluate the CIFAR ResNet checkpoint:

```bash
# pruning
python -u resprune-expand.py --model ./CIFAR10-resnet56_zol_5e-5_t1.4/model_best.pth.tar --save ./ --pruning-strategy fixed --no-cuda --dataset cifar10

# evaluate fine-tuned model
python -u main_finetune.py --dataset cifar10 --arch resnet56 --refine ./pruned_fixed.pth.tar --test-batch-size 128 --seed 123 --expand --resume ./finetune_CIFAR10-resnet56_zol_5e-5_t1.4/model_best.pth.tar -e
```

