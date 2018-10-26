import torch
import torch.nn.functional as F
from .average_precision import mapk


def accuracy(prediction_logits, categories, topk=3):
    predictions = F.softmax(prediction_logits, dim=1)
    _, predicted_categories = torch.topk(predictions, topk, dim=1, sorted=True)
    return mapk(categories.unsqueeze(1).cpu().data.numpy(), predicted_categories.cpu().data.numpy(), k=topk)
