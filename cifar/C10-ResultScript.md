## CIFAR10

### Naive L1 Structured (w/o L1 sparsity regularizer in training)

1. Baseline
    ```bash
    python evaluate.py --no-cuda --original ./cifar10/naive/ckpts/model_best.pth.tar
    ```
2. 80%
    ```bash
    python evaluate.py --no-cuda --fine-tuned ./cifar10/p80/naive/ckpts/ft_model_best.pth.tar
    ```
3. 60%
    ```bash
    python evaluate.py --no-cuda --fine-tuned ./cifar10/p60/naive/ckpts/ft_model_best.pth.tar
    ```
4. 40%
    ```bash
    python evaluate.py --no-cuda --fine-tuned ./cifar10/p40/naive/ckpts/ft_model_best.pth.tar
    ```

### L1 Structured (w/ L1 sparsity regularizer in training)

1. Baseline
    ```bash
    python evaluate.py --no-cuda --original ./cifar10/l1-sr/ckpts/model_best.pth.tar
    ```
2. 80%
    ```bash
    python evaluate.py --no-cuda --fine-tuned ./cifar10/p80/l1-sr/ckpts/ft_model_best.pth.tar
    ```
3. 60%
    ```bash
    python evaluate.py --no-cuda --fine-tuned ./cifar10/p60/l1-sr/ckpts/ft_model_best.pth.tar
    ```
4. 40%
    ```bash
    python evaluate.py --no-cuda --fine-tuned ./cifar10/p40/l1-sr/ckpts/ft_model_best.pth.tar
    ```

### L1 Polarization

1. 80%
    ```bash
    python evaluate.py --no-cuda --gate --original ./cifar10/p80/polar/ckpts/model_best.pth.tar --fine-tuned ./cifar10/p80/polar/ckpts/ft_model_best.pth.tar
    ```
2. 60%
    ```bash
    python evaluate.py --no-cuda --gate --original ./cifar10/p60/polar/ckpts/model_best.pth.tar --fine-tuned ./cifar10/p60/polar/ckpts/ft_model_best.pth.tar
    ```
3. 40%
    ```bash
    python evaluate.py --no-cuda --gate --original ./cifar10/p40/polar/ckpts/model_best.pth.tar --fine-tuned ./cifar10/p40/polar/ckpts/ft_model_best.pth.tar
    ```