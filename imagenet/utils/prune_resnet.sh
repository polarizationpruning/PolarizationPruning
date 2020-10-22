set -e

# activate python environment
if [[ $HOSTNAME == "london" ]]
then
    source ~/research/hyh/pruning/pruning/bin/activate
else
    source /home/common/anaconda3/etc/profile.d/conda.sh
    conda activate pytorch-1.1-py36-cuda9
fi

NAME=$1
EXTRA_PARAMS=$2

python resprune-expand.py ~/rethinking-network-pruning_polar/network_pruning/imagenet/network-slimming/ImageNet --pruning-strategy grad --no-cuda --width-multiplier ${EXTRA_PARAMS} --model "./ckpt/${NAME}/checkpoint.pth.tar" --save "./ckpt/${NAME}/"
