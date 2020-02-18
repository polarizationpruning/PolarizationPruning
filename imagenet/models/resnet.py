from __future__ import absolute_import
import math

import torch
import torch.nn as nn

import models

__all__ = ['resnet50']

"""
preactivation resnet with bottleneck design.
"""


class ChannelMask(nn.Module):
    def __init__(self, channel_num):
        super().__init__()
        mask = torch.ones(channel_num)
        self.weight = nn.Parameter(mask.view(-1, 1, 1))

    def forward(self, x):
        # the shape of x: NCHW
        return x * self.weight


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, outplanes, cfg, stride=1, downsample=None, mask=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, cfg[1], kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cfg[1])
        self.conv2 = nn.Conv2d(cfg[1], cfg[2], kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cfg[2])
        self.conv3 = nn.Conv2d(cfg[2], outplanes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        if self.downsample is None or mask is None:
            self.select = None
        else:
            self.select = models.channel_selection(self.downsample[0].out_channels)

        if mask is not None:
            self.mask = mask
        else:
            self.mask = None

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

        if self.downsample is not None:
            residual = self.downsample(x)
            if self.select is not None:
                residual = self.select(residual)
        if self.mask is not None:
            out = self.mask(out)

        out += residual
        out = self.relu(out)

        return out


class resnet(nn.Module):
    def __init__(self, bn_init_value, layers=None, cfg=None, outplanes=None, mask=False,
                 aux_fc=False, save_feature_map=False):
        super(resnet, self).__init__()

        self.save_feature_map = save_feature_map

        if layers is None:
            # resnet 50
            layers = [3, 4, 6, 3]
        assert len(layers) == 4, "resnet should be 4 blocks"

        block = Bottleneck

        default_cfg = [[64, 64, 64], [256, 64, 64] * (layers[0] - 1),
                       [256, 128, 128], [512, 128, 128] * (layers[1] - 1),
                       [512, 256, 256], [1024, 256, 256] * (layers[2] - 1),
                       [1024, 512, 512], [2048, 512, 512] * (layers[3] - 1),
                       [2048]]
        default_cfg = [item for sub_list in default_cfg for item in sub_list]
        if cfg is None:
            # Construct config variable.
            cfg = default_cfg
        assert len(cfg) == len(default_cfg), "config length error!"

        if outplanes is None:
            outplanes = [256, 512, 1024, 2048]
        self.planes = outplanes
        assert len(self.planes) == 4
        # self.inplanes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block,
                                       inplanes=64,
                                       outplanes=self.planes[0],
                                       blocks=layers[0],
                                       cfg=cfg[0:3 * layers[0]],
                                       if_mask=mask,
                                       downsample_out=256)
        self.layer2 = self._make_layer(block,
                                       inplanes=self.planes[0],
                                       outplanes=self.planes[1],
                                       blocks=layers[1],
                                       cfg=cfg[3 * layers[0]:3 * sum(layers[0:2])],
                                       stride=2,
                                       if_mask=mask,
                                       downsample_out=512)
        self.layer3 = self._make_layer(block,
                                       inplanes=self.planes[1],
                                       outplanes=self.planes[2],
                                       blocks=layers[2],
                                       cfg=cfg[3 * sum(layers[0:2]):3 * sum(layers[0:3])],
                                       stride=2,
                                       if_mask=mask,
                                       downsample_out=1024)
        self.layer4 = self._make_layer(block,
                                       inplanes=self.planes[2],
                                       outplanes=self.planes[3],
                                       blocks=layers[3],
                                       cfg=cfg[3 * sum(layers[0:3]):3 * sum(layers[0:4])],
                                       stride=2,
                                       if_mask=mask,
                                       downsample_out=2048)

        self.enable_aux_fc = aux_fc
        if aux_fc:
            self.aux_fc_layer = nn.Linear(self.planes[2], 1000)
        else:
            self.aux_fc_layer = None

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.planes[3], 1000)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(bn_init_value)
                m.bias.data.zero_()

    def _make_layer(self, block, inplanes, outplanes, blocks, cfg, if_mask, downsample_out, stride=1):
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, downsample_out,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(downsample_out),
        )

        if if_mask:
            mask = ChannelMask(channel_num=outplanes)
        else:
            mask = None

        layers = []
        layers.append(block(inplanes, outplanes, cfg[:3], stride, downsample, mask=mask))
        for i in range(1, blocks):
            layers.append(block(outplanes, outplanes, cfg[3 * i:3 * (i + 1)], mask=mask))

        return nn.Sequential(*layers)

    def forward(self, x):
        extra_info = {}

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)  # 32x32
        if self.save_feature_map:
            extra_info['out1'] = x
        x = self.layer2(x)  # 16x16
        if self.save_feature_map:
            extra_info['out2'] = x
        x = self.layer3(x)  # 8x8
        if self.save_feature_map:
            extra_info['out3'] = x

        if self.enable_aux_fc:
            x_aux = self.avgpool(x)
            x_aux = x_aux.view(x_aux.size(0), -1)
            x_aux = self.aux_fc_layer(x_aux)
            extra_info['x_aux'] = x_aux

        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x, extra_info


def resnet50(mask=False, bn_init_value=0.5, aux_fc=False, save_feature_map=False):
    model = resnet(layers=[3, 4, 6, 3], mask=mask, bn_init_value=bn_init_value,
                   aux_fc=aux_fc, save_feature_map=save_feature_map)
    return model


if __name__ == '__main__':
    model = resnet50()
    print(model)
