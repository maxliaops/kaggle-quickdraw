import torch.nn as nn
from torchvision.models import resnet34

from utils import with_he_normal_weights


class ResNet34(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet34(pretrained=True)

    def forward(self, x):
        return self.resnet(x)


class SimpleCnn(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()

        last_layer_size = (input_size // 2) // 2

        self.delegate = nn.Sequential(
            with_he_normal_weights(nn.Conv2d(1, 32, kernel_size=5, padding=2)),
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            with_he_normal_weights(nn.Conv2d(32, 64, kernel_size=5, padding=2)),
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Flatten(),
            nn.Linear(64 * last_layer_size ** 2, 1024),
            nn.ReLU(inplace=True),
            # nn.BatchNorm1d(1024),
            nn.Dropout(0.4),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        return self.delegate(x)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
