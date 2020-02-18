from __future__ import absolute_import
import math

import torch
import torch.nn as nn
import numpy as np

__all__ = ['ResNetExpand', 'Bottleneck', 'resnet50']

"""
preactivation resnet with bottleneck design.
"""


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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, outplanes, cfg, stride=1, downsample=None,
                 expand=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, cfg[0], kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cfg[0])
        self.conv2 = nn.Conv2d(cfg[0], cfg[1], kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cfg[1])
        self.conv3 = nn.Conv2d(cfg[1], cfg[2], kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(cfg[2])
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.expand = expand
        self.expand_layer = ChannelExpand(outplanes) if expand else None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample:
            residual = self.downsample(x)

        if self.expand_layer:
            out = self.expand_layer(out)

        out += residual
        out = self.relu(out)

        return out


class ResNetExpand(nn.Module):
    def __init__(self, layers=None, cfg=None, downsample_cfg=None, mask=False, aux_fc=False):
        super(ResNetExpand, self).__init__()

        if mask:
            raise NotImplementedError("do not support mask layer")

        if layers is None:
            # resnet 50
            layers = [3, 4, 6, 3]
        assert len(layers) == 4, "resnet should be 4 blocks"

        block = Bottleneck

        default_cfg = [[64],
                       [64, 64, 256] * layers[0],
                       [128, 128, 512] * layers[1],
                       [256, 256, 1024] * layers[2],
                       [512, 512, 2048] * layers[3]]
        default_cfg = [item for sub_list in default_cfg for item in sub_list]

        default_downsample_cfg = [256, 512, 1024, 2048]
        if not downsample_cfg:
            downsample_cfg = default_downsample_cfg
        assert len(downsample_cfg) == len(default_downsample_cfg)

        if cfg is None:
            # Construct config variable.
            cfg = default_cfg
        assert len(cfg) == len(default_cfg), "config length error!"

        # dimension of residuals
        # self.planes = [256, 512, 1024, 2048]
        # assert len(self.planes) == 4

        self.conv1 = nn.Conv2d(3, cfg[0], kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(cfg[0])
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block,
                                       inplanes=cfg[0],
                                       outplanes=256,
                                       blocks=layers[0],
                                       downsample_cfg=downsample_cfg[0],
                                       cfg=cfg[1:3 * layers[0] + 1],
                                       downsample_out=256)
        self.layer2 = self._make_layer(block,
                                       inplanes=256,
                                       outplanes=512,
                                       blocks=layers[1],
                                       downsample_cfg=downsample_cfg[1],
                                       cfg=cfg[3 * layers[0] + 1:3 * sum(layers[0:2]) + 1],
                                       stride=2,
                                       downsample_out=512)
        self.layer3 = self._make_layer(block,
                                       inplanes=512,
                                       outplanes=1024,
                                       blocks=layers[2],
                                       downsample_cfg=downsample_cfg[2],
                                       cfg=cfg[3 * sum(layers[0:2]) + 1:3 * sum(layers[0:3]) + 1],
                                       stride=2,
                                       downsample_out=1024)
        self.layer4 = self._make_layer(block,
                                       inplanes=1024,
                                       outplanes=2048,
                                       blocks=layers[3],
                                       downsample_cfg=downsample_cfg[3],
                                       cfg=cfg[3 * sum(layers[0:3]) + 1:3 * sum(layers[0:4]) + 1],
                                       stride=2,
                                       downsample_out=2048)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(2048, 1000)

        self.enable_aux_fc = aux_fc
        if aux_fc:
            self.aux_fc_layer = nn.Linear(1024, 1000)
        else:
            self.aux_fc_layer = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()

    def _make_layer(self, block, inplanes, outplanes, blocks, cfg, downsample_cfg, downsample_out, stride=1):
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, downsample_cfg,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(downsample_cfg),
            ChannelExpand(downsample_out),
        )

        layers = []
        layers.append(block(inplanes, outplanes, cfg[:3], stride, downsample, expand=True))
        for i in range(1, blocks):
            layers.append(block(downsample_out, outplanes, cfg[3 * i:3 * (i + 1)], expand=True))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8

        if self.enable_aux_fc:
            x_aux = self.avgpool(x)
            x_aux = x_aux.view(x_aux.size(0), -1)
            x_aux = self.aux_fc_layer(x_aux)
        else:
            x_aux = None

        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x, x_aux


def resnet50(aux_fc):
    model = ResNetExpand(layers=[3, 4, 6, 3], aux_fc=aux_fc)
    return model


if __name__ == '__main__':
    model = resnet50(aux_fc=True)
    print(model)
