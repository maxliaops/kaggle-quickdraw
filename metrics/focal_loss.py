import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=1.2):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, labels):
        eps = 1e-7

        # loss =  - np.power(1 - p, gamma) * np.log(p))
        probs = F.softmax(logits)
        probs = probs.gather(dim=1, index=labels.view(-1, 1)).view(-1)
        probs = torch.clamp(probs, min=eps, max=1 - eps)

        loss = -torch.pow(1 - probs, self.gamma) * torch.log(probs)
        loss = loss.mean() * self.alpha

        return loss
