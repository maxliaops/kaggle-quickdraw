import torch.nn as nn
from torchvision.models import resnet34


class ResNet34(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet34(pretrained=True)

    def forward(self, x):
        return self.resnet(x)
