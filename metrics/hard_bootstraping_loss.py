import torch
import torch.nn as nn
import torch.nn.functional as F


class HardBootstrapingLoss(nn.Module):
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.xent = nn.CrossEntropyLoss()

    def forward(self, logits, target):
        _, predicted_target = torch.max(F.softmax(logits, dim=1), dim=1)
        return self.beta * self.xent(logits, target) + (1. - self.beta) * self.xent(logits, predicted_target)
