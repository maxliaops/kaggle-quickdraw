import torch.nn as nn
from torchvision.models import resnet34

from .common import ExpandChannels2d


class ResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.bn = nn.BatchNorm2d(1)
        self.expand_channels = ExpandChannels2d(3)

        self.resnet = resnet34(pretrained=True)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.bn(x)
        x = self.expand_channels(x)

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
