import torch.nn as nn
from torchvision.models import resnet34
from .common import ExpandChannels2d


class ResNet34(nn.Module):
    def __init__(self):
        super().__init__()
        self.expand_channels = ExpandChannels2d(3)
        self.resnet = resnet34(pretrained=True)

    def forward(self, x):
        return self.resnet(self.expand_channels(x))
