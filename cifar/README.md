# Train on CIFAR

> To train on CIFAR-100, use `--dataset cifar100`.

1. Create checkpoint directory
    ```bash
    mkdir ./checkpoints/  # checkpoints and tensorboard events
    mkdir ./events/       # TensorBoard events
    ```

    > Note: Checkpoints will be named as `checkpoint.pth.tar` and `model_best.pth.tar`. The program will overwrite existing checkpoints in the checkpoint directory. Make sure to use a different checkpoint directory name in different experiments. We recommend using the experiment name as directory, e.g., `--save ./resnet56-cifar`.


## ResNet-56

1. Sparsity Train

    ```bash
    python -u main.py --dataset cifar10 --epochs 200 --lr 0.01 --gammas 10 0.2 0.2 0.2 --decay-epoch 1 60 120 160 --arch resnet56 --loss-type zol --lbd 5e-5 --t 1.4 --bn-wd --test-batch-size 128 --weight-decay 5e-4 --save ./checkpoints/ --log ./events/
    ```

2. Pruning

    ```bash
    python -u resprune_gate.py --model ./checkpoints/model_best.pth.tar --save ./checkpoints/ --pruning-strategy grad --dataset cifar10
    ```

3. Fine-tuning

    ```bash
    python -u main_finetune.py --dataset cifar10 --epochs 200 --lr 1e-3 --gammas 0.5 0.5 0.4 --decay-epoch 30 80 150 --arch resnet56 --refine ./checkpoints/pruned_grad.pth.tar --bn-wd --test-batch-size 128 --weight-decay 5e-4 --seed 123 --save ./checkpoints/ --log ./events/ --expand
    ```

## VGG-16

1. Sparsity Train
    ```bash
    python -u main.py --dataset cifar10 --epochs 200 --lr 0.01 --gammas 10 0.2 0.2 0.2 --decay-epoch 1 60 120 160 --arch vgg16_linear --loss-type zol --lbd 3e-5 --t 1.5 --bn-wd --test-batch-size 128 --weight-decay 5e-4 --save ./checkpoints --log ./events
    ```

2. Pruning

    ```bash
    python vggprune_gate.py --dataset cifar10 --model ./checkpoints/model_best.pth.tar --save ./checkpoints/ --pruning-strategy grad
    ```

3. Fine-tuning
    ```bash
    python -u main_finetune.py --dataset cifar10 --epochs 200 --lr 5e-4 --gammas 0.5 0.4 --decay-epoch 60 150 --arch vgg16_linear --refine ./checkpoints/pruned_grad.pth.tar --bn-wd --test-batch-size 128 --weight-decay 5e-4 --save ./checkpoints/ --log ./events/
    ```