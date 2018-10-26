import torch
from .average_precision import mapk


def accuracy(prediction_logits, categories, topk=3):
    _, predicted_categories = torch.topk(prediction_logits, topk, dim=1, sorted=True)
    return mapk(categories.unsqueeze(1).cpu().data.numpy(), predicted_categories.cpu().data.numpy(), k=topk)
