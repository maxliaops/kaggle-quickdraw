import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, target):
        nll = -F.log_softmax(logits, dim=1)
        print("foo")
        print(logits.shape)
        print(target.shape)
        print(nll.shape)
        print("bar", flush=True)
        return torch.sum(nll * target, dim=1).mean()
