import argparse
import math

import cv2
import numpy as np
from torch import nn

from sklearn.model_selection import StratifiedKFold



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


def kfold_split(n_splits, values, classes):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_value_indexes, test_value_indexes in skf.split(values, classes):
        train_values = [values[i] for i in train_value_indexes]
        test_values = [values[i] for i in test_value_indexes]
        yield train_values, test_values


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def read_lines(file_path):
    with open(file_path) as categories_file:
        return [l.rstrip("\n") for l in categories_file.readlines()]


def flatten_strokes(drawing, axis):
    stroke = []
    for s in drawing:
        stroke.extend(s[axis])
    return stroke


def flatten_stroke_lens(drawing):
    return [len(s[0]) for s in drawing]


def calculate_mean_stroke_len(drawing):
    return np.mean(flatten_stroke_lens(drawing))


def encode_stroke_start(drawing):
    start = []
    for s in drawing:
        s_start = [0] * len(s[0])
        s_start[0] = 1
        start.extend(s_start)
    return start


def assemble_strokes(x, y, lens):
    strokes = []
    offset = 0
    for i, l in enumerate(lens):
        strokes.append([x[offset:offset + l], y[offset:offset + l]])
        offset += l
    return strokes


def partition_strokes(strokes, num_partitions):
    total_num_points = sum([len(s[0]) for s in strokes])
    partition_num_points = math.ceil(total_num_points / num_partitions)

    partitions = []

    current_partition = []
    current_partition_points = 0
    for s, stroke in enumerate(strokes):
        current_partition.append(stroke)
        current_partition_points += len(stroke[0])
        if current_partition_points >= partition_num_points or s == len(strokes) - 1:
            partitions.append(current_partition)
            current_partition = []
            current_partition_points = 0

    for _ in range(len(partitions), num_partitions):
        partitions.append([])

    return partitions


def draw_strokes(strokes, size=256, line_width=7, padding=3, fliplr=False):
    draw_size = 256
    scale_factor = (draw_size - 2 * padding) / draw_size

    image = np.full((draw_size, draw_size), 255, dtype=np.uint8)

    stroke_colors = range(0, 240, 40)

    for s, stroke in enumerate(strokes):
        stroke_color = stroke_colors[s % len(stroke_colors)]
        for i in range(len(stroke[0]) - 1):
            x0 = int(scale_factor * stroke[0][i]) + padding
            y0 = int(scale_factor * stroke[1][i]) + padding
            x1 = int(scale_factor * stroke[0][i + 1]) + padding
            y1 = int(scale_factor * stroke[1][i + 1]) + padding
            if fliplr:
                x0 = draw_size - x0
                x1 = draw_size - x1
            cv2.line(image, (x0, y0), (x1, y1), stroke_color, line_width)

    if draw_size != size:
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)

    return image


def merge_stroke_drawings(drawings):
    merged_drawing = np.full(drawings[0].shape, 255, dtype=np.uint8)
    for drawing in drawings:
        merged_drawing[drawing != 255] = drawing[drawing != 255]
    return merged_drawing


def draw_temporal_strokes(strokes, size=256, line_width=7, padding=3, fliplr=False, extended_channels=True):
    draw_size = 256
    scale_factor = (draw_size - 2 * padding) / draw_size

    stroke_colors = range(0, 240, 40)

    partition_images = []

    stroke_partitions = partition_strokes(strokes, 3)
    stroke_color_index = 0
    for stroke_partition in stroke_partitions:
        image = np.full((draw_size, draw_size), 255, dtype=np.uint8)
        partition_images.append(image)

        for stroke in stroke_partition:
            stroke_color = stroke_colors[stroke_color_index % len(stroke_colors)]
            stroke_color_index += 1
            for i in range(len(stroke[0]) - 1):
                x0 = int(scale_factor * stroke[0][i]) + padding
                y0 = int(scale_factor * stroke[1][i]) + padding
                x1 = int(scale_factor * stroke[0][i + 1]) + padding
                y1 = int(scale_factor * stroke[1][i + 1]) + padding
                if fliplr:
                    x0 = draw_size - x0
                    x1 = draw_size - x1
                cv2.line(image, (x0, y0), (x1, y1), stroke_color, line_width)

    if draw_size != size:
        partition_images = [cv2.resize(i, (size, size), interpolation=cv2.INTER_AREA) for i in partition_images]

    final_images = []
    if extended_channels:
        final_images.extend(partition_images)
        final_images.append(merge_stroke_drawings([partition_images[0], partition_images[1]]))
        final_images.append(merge_stroke_drawings([partition_images[1], partition_images[2]]))
        final_images.append(merge_stroke_drawings([partition_images[0], partition_images[1], partition_images[2]]))
    else:
        final_images.append(partition_images[0])
        final_images.append(merge_stroke_drawings([partition_images[0], partition_images[1]]))
        final_images.append(merge_stroke_drawings([partition_images[0], partition_images[1], partition_images[2]]))

    return np.array(final_images)


def pack_confusion_sets(confusion_bitmap, max_size):
    s = np.array(confusion_bitmap)
    b = s.copy()
    n = s.shape[0]
    L = max_size

    r = [[i] for i in range(n)]

    while True:
        c = np.zeros((n, n), dtype=np.int32)
        for i in range(n):
            for j in range(n):
                if i != j and b[i].sum() != 0 and b[j].sum() != 0:
                    m = b[i] | b[j]
                    if m.sum() <= L:
                        c[i, j] = m.sum()

        if np.max(c) == 0:
            break

        o = np.argwhere(c == np.max(c))
        i, j = o[np.random.randint(len(o))]

        b[i] = b[i] | b[j]
        b[j] = False

        r[i] = r[i] + r[j]
        r[j] = []

    q = [x for x in b if x.sum() != 0]
    m = [sorted(y) for x, y in zip(b, r) if x.sum() != 0]

    return q, m


def save_confusion_set(file_path, confusion_set, categories):
    with open(file_path, "w") as file:
        for category in sorted(np.array(categories)[confusion_set]):
            file.write(category + "\n")


def read_confusion_set(file_path):
    with open(file_path) as categories_file:
        return [l.rstrip("\n") for l in categories_file.readlines()]
