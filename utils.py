import argparse

import numpy as np
from torch import nn
from PIL import Image, ImageDraw


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


def draw_it(strokes, size=256, line_width=6, padding=3):
    max_value = 255
    scale_factor = (size - 2 * padding - 1) / max_value

    image = Image.new("P", (size, size), color=255)
    image_draw = ImageDraw.Draw(image)

    for stroke in strokes:
        for i in range(len(stroke[0]) - 1):
            image_draw.line(
                [
                    int(scale_factor * stroke[0][i]) + padding,
                    int(scale_factor * stroke[1][i]) + padding,
                    int(scale_factor * stroke[0][i + 1]) + padding,
                    int(scale_factor * stroke[1][i + 1]) + padding
                ],
                fill=0,
                width=line_width)

    # image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_LINEAR)

    return np.array(image)
