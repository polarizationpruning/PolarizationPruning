# Train on ImageNet

> Note: We recommend to use 4x GTX 1080Ti (or higher) to train ImageNet. Training takes about 1.5 days.

Dataset:
1. Download the ImageNet dataset
2. Add a softlink (optional)

    ```bash
    ln -s path/to/imagenet ./ImageNet
    ```

## Training

Our pruning pipeline contains three steps:

1. Sparsity training
    ```bash
    # train sparse model
    python -u main.py ./ImageNet -loss zol -b 256 --lbd 8e-5 --t 1.4 --lr 1e-1 1e-2 1e-3 --decay-epoch 30 60 --arch resnet50 --workers 25 --world-size 1 --dist-url tcp://localhost:23456 --dist-backend gloo --rank 0 --save ./logs/ --backup-path ./logs/imagenet_ckpt/
    ```

2. Prune the sparse model
    ```bash
    # prune
    python resprune-expand.py ./ImageNet --pruning-strategy grad --no-cuda --model ./logs/model_best.pth.tar --save ./logs/
    ```
3. Fine-tune the pruned model

    ```bash
    # fine-tune
    python -u main_finetune.py ./ImageNet --arch resnet50 --epoch 100 --lr 1e-2 1e-3 1e-4 3e-5 --decay-epoch 30 60 80 --refine ./logs/pruned.pth.tar --expand -b 256 --workers 25 --world-size 1 --dist-url tcp://localhost:23456 --dist-backend gloo --rank 0 --save ./logs/ --backup-path ./logs/imagenet_ckpt/
    ```

## Evaluation

To evaluate ResNet checkpoint:

```bash
# prune the sparse network
python resprune-expand.py ./ImageNet --pruning-strategy grad --no-cuda --model ./imagenet_resnet_polarized_model_best.pth.tar --save ./

# evaluate
python main_finetune.py ./ImageNet --arch resnet50 --expand --refine ./pruned.pth.tar --resume ./imagenet_resnet_pruned_model_best.pth.tar -e
```

## Note

### Learning rate

The learning rate is determined by two arguments: 

- `--lr`: a list of learning rates in different stages.
- `--decay-epoch`: the epoch to decay learning rate.

