import glob
import os
import shutil

import h5py
import numpy as np
import pandas as pd

from utils import draw_it


def read_categories():
    with open("/storage/kaggle/quickdraw/categories.txt") as categories_file:
        return [l.rstrip("\n") for l in categories_file.readlines()]


def flatten_strokes(drawing, axis):
    stroke = []
    for s in drawing:
        stroke.extend(s[axis])
    return stroke


def prepare_strokes():
    categories = read_categories()

    with h5py.File("quickdraw_train.hdf5", "w", libver="latest") as data_file:
        for csv_file in glob.glob("/storage/kaggle/quickdraw/train_simplified_shard_0/*.csv"):
            print("processing file '{}'".format(csv_file), flush=True)

            df = pd.read_csv(
                csv_file,
                index_col="key_id",
                converters={"drawing": lambda drawing: eval(drawing)})

            category = os.path.basename(csv_file)[:-6]

            group = data_file.create_group(category)
            # TODO: use enum type
            group["category"] = [categories.index(word) for word in df.word]
            # TODO: use vlen type
            group["stroke_x"] = np.stack([flatten_strokes(d, 0) for d in df.drawing])
            group["stroke_y"] = np.stack([flatten_strokes(d, 1) for d in df.drawing])
            group["stroke_len"] = [len(d[0]) for d in df.drawing]

    shutil.move("quickdraw_train.hdf5", "/storage/kaggle/quickdraw/")


def prepare_thumbnails():
    categories = read_categories()

    with h5py.File("quickdraw_train_thumbnails.hdf5", "w", libver="latest") as data_file:
        for csv_file in glob.glob("/storage/kaggle/quickdraw/train_simplified_shard_0/*.csv"):
            print("processing file '{}'".format(csv_file), flush=True)

            df = pd.read_csv(
                csv_file,
                index_col="key_id",
                converters={"drawing": lambda drawing: draw_it(eval(drawing), size=32)})

            category = os.path.basename(csv_file)[:-6]
            thumbnail = np.stack(df.drawing.values)

            group = data_file.create_group(category)
            group["thumbnail"] = thumbnail
            group["category"] = [categories.index(word) for word in df.word]

    shutil.move("quickdraw_train_thumbnails.hdf5", "/storage/kaggle/quickdraw/")


if __name__ == "__main__":
    prepare_strokes()
