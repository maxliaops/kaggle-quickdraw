import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common import Flatten
from models.se_blocks import ChannelSEBlock


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation=1):
        super().__init__()
        self.delegate = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation),
            ChannelSEBlock(out_channels),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.delegate(x)


class HcFcCnn(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.bn0 = nn.BatchNorm2d(6)

        self.conv1 = ConvBlock(6, 64, kernel_size=3, padding=1)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = ConvBlock(64, 128, kernel_size=3, padding=1)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = ConvBlock(128, 256, kernel_size=3, padding=1)
        self.max_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.hc_conv = ConvBlock(6 + 64 + 128 + 256, 512, kernel_size=3, padding=1)

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Conv2d(512, num_classes, kernel_size=1)
        self.flatten = Flatten()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, x):
        hc_inputs = []

        x = self.bn0(x)
        hc_inputs.append(F.interpolate(x, scale_factor=1. / 8, mode="area"))

        x = self.conv1(x)
        hc_inputs.append(F.interpolate(x, scale_factor=1. / 8, mode="area"))
        x = self.max_pool1(x)

        x = self.conv2(x)
        hc_inputs.append(F.interpolate(x, scale_factor=1. / 4, mode="area"))
        x = self.max_pool2(x)

        x = self.conv3(x)
        hc_inputs.append(F.interpolate(x, scale_factor=1. / 2, mode="area"))
        x = self.max_pool3(x)

        x = self.hc_conv(torch.cat(hc_inputs, dim=1))

        x = self.avg_pool(x)
        x = self.fc(x)
        x = self.flatten(x)

        return x
