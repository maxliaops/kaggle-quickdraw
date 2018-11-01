from torch import nn

from models.drn import drn_d_22
from .common import ExpandChannels2d


class Drn(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()

        last_layer_size = input_size
        for _ in range(3):
            last_layer_size //= 2

        self.bn = nn.BatchNorm2d(1)
        self.expand_channels = ExpandChannels2d(3)

        self.drn = drn_d_22(pretrained=True)

        self.avgpool = nn.AvgPool2d(last_layer_size)
        self.fc = nn.Conv2d(self.drn.out_dim, num_classes, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        x = self.bn(x)
        x = self.expand_channels(x)

        x = self.drn.layer0(x)
        x = self.drn.layer1(x)
        x = self.drn.layer2(x)
        x = self.drn.layer3(x)
        x = self.drn.layer4(x)
        x = self.drn.layer5(x)
        x = self.drn.layer6(x)
        x = self.drn.layer7(x)
        x = self.drn.layer8(x)

        x = self.avgpool(x)
        x = self.fc(x)
        x = x.view(x.size(0), -1)

        return x
