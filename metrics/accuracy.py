import torch
import torch.nn.functional as F
from .average_precision import mapk


# TODO: avoid redefinition
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def accuracy(prediction_logits, categories, topk=3):
    predictions = F.softmax(prediction_logits, dim=1)
    _, predicted_categories = torch.topk(predictions, topk, dim=1, sorted=True)

    apk = torch.tensor(0.0).float().to(device, non_blocking=True)
    for k in range(topk):
        matches = torch.eq(categories, predicted_categories[:, k])
        apk += matches.float().sum() / (k + 1)

    return apk
