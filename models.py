import torch.nn as nn
from torchvision.models import resnet34


class ResNet34(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet34(pretrained=True)

    def forward(self, x):
        return self.resnet(x)


class SimpleCnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.delegate = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Flatten(),
            nn.Linear(64 * 16 * 16, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.4),
            nn.Linear(1024, 345)
        )

    def forward(self, x):
        return self.delegate(x)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
