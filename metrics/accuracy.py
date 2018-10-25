import torch
from .average_precision import mapk


def accuracy(prediction_logits, categories, topk=3):
    _, predicted_categories = torch.topk(prediction_logits, topk, dim=1)
    return mapk(categories, predicted_categories, k=topk)
