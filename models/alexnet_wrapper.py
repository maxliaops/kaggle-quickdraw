import torch.nn as nn

from models.alexnet import alexnet
from .common import ExpandChannels2d


class AlexNetWrapper(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.expand_channels = ExpandChannels2d(3)
        self.bn = nn.BatchNorm2d(3)

        self.alexnet = alexnet(pretrained=True)

        classifier = list(self.alexnet.classifier.children())[:-1]
        print(list(self.alexnet.classifier.children())[-1:])
        classifier.append(nn.Linear(4096, num_classes))
        self.alexnet.classifier = nn.Sequential(*classifier)

    def forward(self, x):
        x = self.expand_channels(x)
        x = self.bn(x)

        x = self.alexnet(x)

        return x
