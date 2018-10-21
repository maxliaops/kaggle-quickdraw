import torch


def accuracy(predictions, categories):
    _, predicted_categories = torch.max(predictions, 1)
    return (predicted_categories == categories).sum() / categories.size(0)
