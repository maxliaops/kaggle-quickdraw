import torch.nn as nn
from torchvision.models import resnet34
from .common import ExpandChannels2d


class ResNet34(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()

        last_layer_size = input_size
        for _ in range(4):
            last_layer_size //= 2

        self.bn = nn.BatchNorm2d(1)
        self.expand_channels = ExpandChannels2d(3)

        self.resnet = resnet34(pretrained=True)

        self.avgpool = nn.AvgPool2d(last_layer_size, stride=1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.bn(x)
        x = self.expand_channels(x)

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        # x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
