import torch


def accuracy(predictions, categories):
    _, predicted_categories = torch.max(predictions, 1)
    return (predicted_categories == categories).sum().float() / categories.size(0)
