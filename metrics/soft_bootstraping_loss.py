import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftBootstrapingLoss(nn.Module):
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def forward(self, logits, target):
        probabilities = F.softmax(logits, dim=1)
        nll = -F.log_softmax(logits, dim=1)
        bootstrap_target = self.beta * target + (1. - self.beta) * probabilities
        return torch.sum(nll * bootstrap_target, dim=1).mean()
