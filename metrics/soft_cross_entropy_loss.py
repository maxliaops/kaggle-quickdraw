import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, target, target_one_hot):
        nll = -F.log_softmax(logits, dim=1)
        return torch.sum(nll * target_one_hot, dim=1).mean()
