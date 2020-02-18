'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].

The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.

Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:

name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m

which this implementation indeed has.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
from torch.autograd import Variable

__all__ = ['ResNetExpand', 'BasicBlock', 'resnet56']


def _weights_init(m, bn_init_value):
    classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(bn_init_value)
        m.bias.data.zero_()


class ChannelExpand(nn.Module):
    def __init__(self, channel_num):
        """
        :param channel_num: the number of output channels
        """
        super().__init__()
        self.channel_num = channel_num
        self.idx = np.arange(channel_num)

    def forward(self, x):
        data = torch.zeros(x.size()[0], self.channel_num, x.size()[2], x.size()[3], device=x.device)
        data[:, self.idx, :, :] = x

        return data


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, outplanes, cfg, stride=1, option='A'):
        super(BasicBlock, self).__init__()

        if len(cfg) != 2:
            raise ValueError("cfg len should be 2, got {}".format(cfg))

        self.is_empty_block = 0 in cfg
        if not self.is_empty_block:
            self.conv1 = nn.Conv2d(in_planes, cfg[0], kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(cfg[0])
            self.conv2 = nn.Conv2d(cfg[0], cfg[1], kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(cfg[1])

            self.expand_layer = ChannelExpand(outplanes)
        else:
            self.conv1 = nn.Identity()
            self.bn1 = nn.Identity()
            self.conv2 = nn.Identity()
            self.bn2 = nn.Identity()

            self.expand_layer = nn.Identity()

        self.shortcut = nn.Sequential()  # do nothing
        if stride != 1 or in_planes != outplanes:
            if option == 'A':
                """For CIFAR10 ResNet paper uses option A."""
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, outplanes // 4, outplanes // 4),
                                                  "constant",
                                                  0))
            elif option == 'B':
                raise NotImplementedError("Option B is not implemented")
                # self.shortcut = nn.Sequential(
                #     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                #     nn.BatchNorm2d(self.expansion * planes)
                # )

    def forward(self, x):
        if self.is_empty_block:
            out = self.shortcut(x)
            out = F.relu(out)
            return out
        else:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out = self.expand_layer(out)

            out += self.shortcut(x)
            out = F.relu(out)
            return out


class ResNetExpand(nn.Module):
    def __init__(self, block, num_blocks, bn_init_value, cfg=None, num_classes=10, aux_fc=False):
        super(ResNetExpand, self).__init__()
        self.in_planes = 16

        assert len(num_blocks) == 3, "only 3 layers, got {}".format(len(num_blocks))
        default_cfg = [[16, 16] * num_blocks[0],
                       [32, 32] * num_blocks[1],
                       [64, 64] * num_blocks[2]]
        default_cfg = [item for sub_list in default_cfg for item in sub_list]
        if cfg is None:
            # Construct config variable.
            cfg = default_cfg
        assert len(cfg) == len(default_cfg), "config length error!"

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1,
                                       cfg=cfg[:2 * num_blocks[0]])
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2,
                                       cfg=cfg[2 * num_blocks[0]:2 * sum(num_blocks[0:2])])
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2,
                                       cfg=cfg[2 * sum(num_blocks[0:2]):2 * sum(num_blocks[0:3])])
        self.linear = nn.Linear(64, num_classes)

        self.enable_aux_fc = aux_fc
        if aux_fc:
            self.aux_fc_layer = nn.Linear(32, num_classes)
        else:
            self.aux_fc_layer = None

        self._initialize_weights(bn_init_value)

    def _make_layer(self, block, planes, num_blocks, stride, cfg):
        block: BasicBlock
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i, stride in enumerate(strides):
            layers.append(block(in_planes=self.in_planes, outplanes=planes,
                                stride=stride,
                                cfg=cfg[2 * i:2 * (i + 1)]))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)

        if self.enable_aux_fc:
            out_aux = F.adaptive_avg_pool2d(out, 1)
            out_aux = out_aux.view(out_aux.size(0), -1)
            out_aux = self.aux_fc_layer(out_aux)
        else:
            out_aux = None

        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out, out_aux

    def _initialize_weights(self, bn_init_value=1):
        for m in self.modules():
            _weights_init(m, bn_init_value)


def resnet20(num_classes):
    return ResNetExpand(BasicBlock, [3, 3, 3], num_classes)


def resnet32(num_classes):
    return ResNetExpand(BasicBlock, [5, 5, 5], num_classes)


def resnet44(num_classes):
    return ResNetExpand(BasicBlock, [7, 7, 7], num_classes)


def resnet56(num_classes, bn_init_value=1.0, cfg=None, aux_fc=False):
    return ResNetExpand(BasicBlock, [9, 9, 9],
                        cfg=cfg,
                        num_classes=num_classes,
                        bn_init_value=bn_init_value,
                        aux_fc=aux_fc)


def resnet110(num_classes):
    return ResNetExpand(BasicBlock, [18, 18, 18], num_classes)


def resnet1202(num_classes):
    return ResNetExpand(BasicBlock, [200, 200, 200], num_classes)


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size()) > 1, net.parameters()))))


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()
