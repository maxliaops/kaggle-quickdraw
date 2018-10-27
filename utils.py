import argparse

import cv2
import numpy as np
from torch import nn


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def with_he_normal_weights(layer):
    nn.init.kaiming_normal_(layer.weight, a=0, mode="fan_in")
    return layer


def freeze(model):
    for param in model.parameters():
        param.requires_grad = False


def unfreeze(model):
    for param in model.parameters():
        param.requires_grad = True


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def assemble_strokes(x, y, lens):
    strokes = []
    offset = 0
    for i, l in enumerate(lens):
        strokes.append([x[offset:offset + l], y[offset:offset + l]])
        offset += l
    return strokes


def draw_strokes(strokes, size=256, line_width=7, padding=3):
    max_size = 255
    scale_factor = (max_size - 2 * padding) / max_size

    image = np.full((max_size, max_size), 255, dtype=np.uint8)

    for stroke in strokes:
        for i in range(len(stroke[0]) - 1):
            x0 = int(scale_factor * stroke[0][i]) + padding
            y0 = int(scale_factor * stroke[1][i]) + padding
            x1 = int(scale_factor * stroke[0][i + 1]) + padding
            y1 = int(scale_factor * stroke[1][i + 1]) + padding
            cv2.line(image, (x0, y0), (x1, y1), 0, line_width)

    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)

    return image
