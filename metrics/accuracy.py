import torch
from .average_precision import mapk


def accuracy(prediction_logits, categories, topk=3):
    _, predicted_categories = torch.topk(prediction_logits, topk, dim=1)
    p = mapk(categories.unsqueeze(1).cpu().data.numpy(), predicted_categories.cpu().data.numpy(), k=topk)
    print("{} vs. {}".format(p, accuracy2(prediction_logits, categories, topk)))
    return p

def accuracy2(prediction_logits, categories, topk=3):
    _, predicted_categories = torch.topk(prediction_logits, topk, dim=1)
    return (predicted_categories == categories.unsqueeze(1).expand(-1, topk)).sum().float() / categories.size(0)
