import torch


def accuracy(prediction_logits, categories):
    _, predicted_categories = torch.max(prediction_logits, 1)
    return (predicted_categories == categories).sum().float() / categories.size(0)
