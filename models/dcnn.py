import math

import torch.nn as nn

from .common import Flatten


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation=1):
        super().__init__()
        self.delegate = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.delegate(x)


class SimpleDilatedCnn(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()

        last_layer_size = input_size

        self.delegate = nn.Sequential(
            nn.BatchNorm2d(1),
            ConvBlock(1, 32, kernel_size=5, padding=2),
            ConvBlock(32, 64, kernel_size=5, padding=2, dilation=2),
            ConvBlock(64, 128, kernel_size=3, padding=1, dilation=4),
            nn.AvgPool2d(last_layer_size),
            nn.Linear(128, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.4),
            nn.Linear(1024, num_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, x):
        return self.delegate(x)
