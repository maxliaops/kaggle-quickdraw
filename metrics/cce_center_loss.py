import torch.nn as nn

from .center_loss import CenterLoss


class CceCenterLoss(nn.Module):
    def __init__(self, num_classes, alpha):
        super().__init__()
        self.alpha = alpha
        self.cce = nn.CrossEntropyLoss()
        self.center = CenterLoss(num_classes=num_classes, feat_dim=num_classes)

    def forward(self, logits, labels):
        return self.cce(logits, labels) + self.alpha * self.center(logits, labels)
