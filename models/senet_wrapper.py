from collections import OrderedDict

from torch import nn

from models.senet import se_resnext50_32x4d, senet154
from .common import ExpandChannels2d


class SeNet(nn.Module):
    def __init__(self, type, input_size, num_classes):
        super().__init__()

        last_layer_size = input_size
        for _ in range(5):
            last_layer_size //= 2

        self.bn = nn.BatchNorm2d(1)
        self.expand_channels = ExpandChannels2d(3)

        if type == "seresnext":
            self.senet = se_resnext50_32x4d(pretrained="imagenet")

            layer0_modules = [
                ('conv1', self.senet.layer0.conv1),
                ('bn1', self.senet.layer0.bn1),
                ('relu1', self.senet.layer0.relu1),
            ]

            self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        elif type == "senet":
            self.senet = senet154(pretrained="imagenet")
            self.layer0 = self.senet.layer0
        else:
            raise Exception("Unsupported senet model type: '{}".format(type))

        self.avg_pool = nn.AvgPool2d(last_layer_size, stride=1)
        self.dropout = nn.Dropout(0.2)
        self.last_linear = nn.Linear(2048, num_classes)

    def features(self, x):
        x = self.layer0(x)
        x = self.senet.layer1(x)
        x = self.senet.layer2(x)
        x = self.senet.layer3(x)
        x = self.senet.layer4(x)
        return x

    def logits(self, x):
        x = self.avg_pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, x):
        x = self.bn(x)
        x = self.expand_channels(x)
        x = self.features(x)
        x = self.logits(x)
        return x
