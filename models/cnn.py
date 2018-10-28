import torch.nn as nn

from .common import Flatten


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.delegate = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.delegate(x)


class SimpleCnn(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()

        last_layer_size = (input_size // 2) // 2

        self.delegate = nn.Sequential(
            nn.BatchNorm2d(1),
            ConvBlock(1, 32, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(32, 64, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Flatten(),
            nn.Linear(64 * last_layer_size ** 2, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.4),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        return self.delegate(x)
