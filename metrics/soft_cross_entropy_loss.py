import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, target):
        log_likelihood = -F.log_softmax(logits, dim=1)
        print("foo")
        print(logits.shape)
        print(target.shape)
        print(log_likelihood.shape)
        print("bar", flush=True)
        return torch.sum(log_likelihood * target, dim=1).mean()
