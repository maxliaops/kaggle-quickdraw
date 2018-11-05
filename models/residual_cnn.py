import math

import torch.nn as nn

from models.se_blocks import ChannelSEBlock
from .common import Flatten


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation=1):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)

        self.residual_adapter = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = ChannelSEBlock(out_channels)

    def forward(self, x):
        residual = self.residual_adapter(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.se(x)
        x += residual
        x = self.relu(x)

        return x


class ResidualCnn(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.delegate = nn.Sequential(
            nn.BatchNorm2d(6),
            ConvBlock(6, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(64, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(128, 256, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(256, 512, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d(output_size=1),
            Flatten(),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            # nn.Dropout(0.2),
            nn.Linear(1024, num_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, x):
        return self.delegate(x)
