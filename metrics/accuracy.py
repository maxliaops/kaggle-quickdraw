import torch
import torch.nn.functional as F
from .average_precision import mapk


def accuracy(prediction_logits, categories, topk=3):
    return 0.
