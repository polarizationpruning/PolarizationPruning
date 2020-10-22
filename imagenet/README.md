# Train on ImageNet

Dataset:
1. Download the ImageNet dataset
2. Add a softlink (optional)

    ```bash
    ln -s path/to/imagenet ./ImageNet
    ```
3. Create checkpoint directory
    ```bash
    mkdir ./checkpoints/  # checkpoints and tensorboard events
    mkdir ./backup_ckpt/  # backup checkpoints every five epochs
    ```

    > Note: Checkpoints will be named as `checkpoint.pth.tar` and `model_best.pth.tar`. The program will overwrite existing checkpoints in the checkpoint directory. Make sure to use a different checkpoint directory name in different experiments. We recommend using the experiment name as directory, e.g., `--save ./resnet50-imagenet`.

## Training

### ResNet-50

1. Sparsity training
    ```bash
    # train sparse model
    python -u main.py ./ImageNet -loss zol -b 512 --lbd 8e-5 --t 1.2 --lr-strategy step --lr 1e-1 1e-2 1e-3 --decay-epoch 40 80 --epochs 120 --arch resnet50 --workers 25 --world-size 1 --dist-url tcp://localhost:23456 --dist-backend gloo --rank 0 --save ./checkpoints/resnet/ --backup-path ./backup_ckpt/resnet/
    ```

2. Prune the sparse model
    ```bash
    # prune
    python resprune_expand_gate.py --pruning-strategy grad --model ./checkpoints/resnet/model_best.pth.tar --save ./checkpoints/resnet/
    ```
3. Fine-tune the pruned model

    ```bash
    # fine-tune
    python -u main_finetune.py ./ImageNet --arch resnet50 --epoch 128 --lr 0.05 --lr-strategy cos --refine ./checkpoints/resnet/pruned_grad.pth.tar --expand -b 512 --workers 25 --world-size 1 --dist-url tcp://localhost:23456 --dist-backend gloo --rank 0 --save ./checkpoints/resnet/finetune/ --backup-path ./backup_ckpt/resnet/finetune
    ```

### MobileNet v2

1. Sparsity training

    ```bash
    # sparsity training
    python -u main.py ./ImageNet -loss zol --target-flops 0.7 --warmup --gate -b 1024 --lbd 2.5e-5 --t 1 --lr-strategy cos --lr 0.4 --epochs 256 --wd 0.00004 --no-bn-wd --arch mobilenetv2 --workers 25 --world-size 1 --dist-url tcp://localhost:23456 --dist-backend gloo --rank 0 --save ./checkpoints/mobilenetv2/ --backup-path ./backup_ckpt/mobilenetv2/
    ```

2. Pruning

    ```bash
    # pruning
   python prune_mobilenetv2.py --pruning-strategy grad --gate --model ./checkpoints/mobilenetv2/model_best.pth.tar --save ./checkpoints/mobilenetv2/

    ```

3. Fine-tuning
    ```bash
    # fine-tuning
    python -u main_finetune.py ./ImageNet --arch mobilenetv2 --epoch 256  --wd 0.00004 --no-bn-wd --lr 5e-2 --lr-strategy cos --refine ./checkpoints/mobilenetv2/pruned_grad.pth.tar -b 1024 --workers 32 --world-size 1 --dist-url tcp://localhost:23456 --dist-backend gloo --rank 0 --save ./checkpoints/mobilenetv2/finetune/ --backup-path ./backup_ckpt/mobilenetv2/finetune/
    ```


## Documentation

Use `-h` for the documentation of arguments, e.g., `python main.py -h`.


