import torch
from torch import nn

from models.senet import se_resnet50, se_resnet101, se_resnet152, se_resnext50_32x4d, se_resnext101_32x4d, senet154
from .common import ExpandChannels2d


class SeNet(nn.Module):
    def __init__(self, type, num_classes):
        super().__init__()

        if type == "seresnext50":
            self.senet = se_resnext50_32x4d(pretrained="imagenet")

            # layer0_modules = [
            #     ('conv1', self.senet.layer0.conv1),
            #     ('bn1', self.senet.layer0.bn1),
            #     ('relu1', self.senet.layer0.relu1),
            # ]
            # self.layer0 = nn.Sequential(OrderedDict(layer0_modules))

            self.layer0 = self.senet.layer0
        elif type == "seresnext101":
            self.senet = se_resnext101_32x4d(pretrained="imagenet")
            self.layer0 = self.senet.layer0
        elif type == "seresnet50":
            self.senet = se_resnet50(pretrained="imagenet")
            self.layer0 = self.senet.layer0
        elif type == "seresnet101":
            self.senet = se_resnet101(pretrained="imagenet")
            self.layer0 = self.senet.layer0
        elif type == "seresnet152":
            self.senet = se_resnet152(pretrained="imagenet")
            self.layer0 = self.senet.layer0
        elif type == "senet154":
            self.senet = senet154(pretrained="imagenet")
            self.layer0 = self.senet.layer0
        else:
            raise Exception("Unsupported senet model type: '{}".format(type))

        self.expand_channels = ExpandChannels2d(3)
        self.bn = nn.BatchNorm2d(3)

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
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
        x = self.expand_channels(x)
        x = self.bn(x)
        x = self.features(x)
        x = self.logits(x)
        return x


class SeResNext50Cs(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.parent = nn.DataParallel(SeNet("seresnext50", 340))

        self.parent.load_state_dict(
            torch.load("/storage/models/quickdraw/seresnext50/model.pth",
                       map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))

        self.parent.last_linear = nn.Linear(2048, num_classes)

    def forward(self, x):
        return self.parent(x)
