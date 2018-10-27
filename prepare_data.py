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


def prepare_strokes_pandas():
    categories = read_categories()

    for category in categories:
        csv_file = "/storage/kaggle/quickdraw/train_simplified_shard_0/{}-0.csv".format(category)
        if not os.path.isfile(csv_file):
            print("skipping category '{}' for which no CSV file exists".format(category), flush=True)
            continue

        print("processing file '{}'".format(csv_file), flush=True)

        df = pd.read_csv(
            csv_file,
            index_col="key_id",
            usecols=["key_id", "drawing", "word"],
            converters={"drawing": lambda drawing: eval(drawing)})

        df.to_hdf("quickdraw_train_pd.hdf5", key=category)

    shutil.move("quickdraw_train_pd.hdf5", "/storage/kaggle/quickdraw/")


def prepare_strokes():
    categories = read_categories()

    with h5py.File("quickdraw_train.hdf5", "w", libver="latest") as data_file:
        for category in categories:
            csv_file = "/storage/kaggle/quickdraw/train_simplified_shard_0/{}-0.csv".format(category)
            if not os.path.isfile(csv_file):
                print("skipping category '{}' for which no CSV file exists".format(category), flush=True)
                continue

            print("processing file '{}'".format(csv_file), flush=True)

            df = pd.read_csv(
                csv_file,
                index_col="key_id",
                usecols=["key_id", "drawing", "word"],
                converters={"drawing": lambda drawing: eval(drawing)})

            group = data_file.create_group(category)

            key_id_ds = group.create_dataset("key_id", (len(df),), dtype=np.int32)
            category_ds = group.create_dataset("category", (len(df),), dtype=np.int16)
            stroke_x_ds = group.create_dataset("stroke_x", (len(df),), dtype=h5py.special_dtype(vlen=np.uint8))
            stroke_y_ds = group.create_dataset("stroke_y", (len(df),), dtype=h5py.special_dtype(vlen=np.uint8))
            stroke_len_ds = group.create_dataset("stroke_len", (len(df),), dtype=np.int32)

            key_id_ds[:] = df.index.values
            category_ds[:] = [categories.index(word) for word in df.word]
            stroke_x_ds[:] = [flatten_strokes(d, 0) for d in df.drawing]
            stroke_y_ds[:] = [flatten_strokes(d, 1) for d in df.drawing]
            stroke_len_ds[:] = [len(d[0]) for d in df.drawing]

    shutil.move("quickdraw_train.hdf5", "/storage/kaggle/quickdraw/")


def prepare_thumbnails():
    categories = read_categories()

    with h5py.File("quickdraw_train_thumbnails.hdf5", "w", libver="latest") as data_file:
        for category in categories:
            csv_file = "/storage/kaggle/quickdraw/train_simplified_shard_0/{}-0.csv".format(category)
            if not os.path.isfile(csv_file):
                print("skipping category '{}' for which no CSV file exists".format(category), flush=True)
                continue

            print("processing file '{}'".format(csv_file), flush=True)

            df = pd.read_csv(
                csv_file,
                index_col="key_id",
                usecols=["key_id", "drawing", "word"],
                converters={"drawing": lambda drawing: draw_it(eval(drawing), size=32)})

            thumbnail = np.stack(df.drawing.values)

            group = data_file.create_group(category)
            group["thumbnail"] = thumbnail
            group["category"] = [categories.index(word) for word in df.word]

    shutil.move("quickdraw_train_thumbnails.hdf5", "/storage/kaggle/quickdraw/")


if __name__ == "__main__":
    prepare_strokes()
