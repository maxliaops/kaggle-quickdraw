import torch
import torch.nn.functional as F


def accuracy(prediction_logits, categories, topk=3):
    predictions = F.softmax(prediction_logits, dim=1)
    _, predicted_categories = predictions.topk(topk, dim=1, sorted=True)

    apk_v = torch.eq(categories, predicted_categories[:, 0]).float()
    for k in range(1, topk):
        apk_v += torch.eq(categories, predicted_categories[:, k]).float()

    return apk_v.mean()
