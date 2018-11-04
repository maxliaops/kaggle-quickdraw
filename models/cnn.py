import math

import torch.nn as nn
from torchvision.models.resnet import BasicBlock

from models.se_blocks import ChannelSEBlock
from .common import Flatten


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


class SimpleCnn(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()

        last_layer_size = input_size
        for _ in range(3):
            last_layer_size //= 2

        self.delegate = nn.Sequential(
            nn.BatchNorm2d(1),
            BasicBlock(1, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            BasicBlock(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            BasicBlock(128, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            BasicBlock(256, 512),
            nn.AvgPool2d(kernel_size=last_layer_size),
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
