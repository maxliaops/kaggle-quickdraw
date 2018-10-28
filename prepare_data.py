import math
import os
import shutil

import h5py
import numpy as np
import pandas as pd

from utils import draw_strokes, read_categories, flatten_strokes, flatten_stroke_lens


def calculate_total_data_size():
    categories = read_categories("/storage/kaggle/quickdraw/categories.txt")

    size = 0
    for category in categories:
        csv_file_name = "/storage/kaggle/quickdraw/train_simplified_shard_0/{}-0.csv".format(category)

        with open(csv_file_name) as csv_file:
            size += sum(1 for _ in csv_file) - 1

    print("total data size: {}".format(size))
    return size


def prepare_strokes_pandas():
    categories = read_categories("/storage/kaggle/quickdraw/categories.txt")

    for category in categories:
        csv_file_name = "/storage/kaggle/quickdraw/train_simplified_shard_0/{}-0.csv".format(category)

        print("processing file '{}'".format(csv_file_name), flush=True)

        df = pd.read_csv(
            csv_file_name,
            index_col="key_id",
            usecols=["key_id", "drawing", "word"],
            converters={
                "word": lambda word: categories.index(word),
                "drawing": lambda drawing: eval(drawing)
            })

        df = df.rename(columns={"word": "category"})

        df.to_hdf("quickdraw_train_pd.hdf5", key=category)

    shutil.move("quickdraw_train_pd.hdf5", "/storage/kaggle/quickdraw/quickdraw_train_pd.hdf5")


def prepare_strokes():
    categories = read_categories("/storage/kaggle/quickdraw/categories.txt")

    with h5py.File("quickdraw_train.hdf5", "w", libver="latest") as data_file:
        data_size = calculate_total_data_size()

        key_id_ds = data_file.create_dataset("key_id", (data_size,), dtype=np.int64)
        category_ds = data_file.create_dataset("category", (data_size,), dtype=np.int16)
        stroke_x_ds = data_file.create_dataset("stroke_x", (data_size,), dtype=h5py.special_dtype(vlen=np.uint8))
        stroke_y_ds = data_file.create_dataset("stroke_y", (data_size,), dtype=h5py.special_dtype(vlen=np.uint8))
        stroke_len_ds = data_file.create_dataset("stroke_len", (data_size,), dtype=h5py.special_dtype(vlen=np.uint32))

        offset = 0

        for category in categories:
            csv_file_name = "/storage/kaggle/quickdraw/train_simplified_shard_0/{}-0.csv".format(category)

            print("processing file '{}'".format(csv_file_name), flush=True)

            df = pd.read_csv(
                csv_file_name,
                index_col="key_id",
                usecols=["key_id", "drawing", "word"],
                converters={"drawing": lambda drawing: eval(drawing)})

            key_id_ds[offset:offset + len(df)] = df.index.values
            category_ds[offset:offset + len(df)] = [categories.index(word) for word in df.word]
            stroke_x_ds[offset:offset + len(df)] = [flatten_strokes(d, 0) for d in df.drawing]
            stroke_y_ds[offset:offset + len(df)] = [flatten_strokes(d, 1) for d in df.drawing]
            stroke_len_ds[offset:offset + len(df)] = [flatten_stroke_lens(d) for d in df.drawing]

            offset += len(df)

        print("wrote {} data elements".format(offset))

    shutil.move("quickdraw_train.hdf5", "/storage/kaggle/quickdraw/quickdraw_train.hdf5")


def prepare_thumbnails():
    categories = read_categories("/storage/kaggle/quickdraw/categories.txt")

    with h5py.File("quickdraw_train_thumbnails.hdf5", "w", libver="latest") as data_file:
        for category in categories:
            csv_file_name = "/storage/kaggle/quickdraw/train_simplified_shard_0/{}-0.csv".format(category)

            print("processing file '{}'".format(csv_file_name), flush=True)

            df = pd.read_csv(
                csv_file_name,
                index_col="key_id",
                usecols=["key_id", "drawing", "word"],
                converters={"drawing": lambda drawing: draw_strokes(eval(drawing), size=32)})

            thumbnail = np.stack(df.drawing.values)

            group = data_file.create_group(category)
            group["thumbnail"] = thumbnail
            group["category"] = [categories.index(word) for word in df.word]

    shutil.move("quickdraw_train_thumbnails.hdf5", "/storage/kaggle/quickdraw/quickdraw_train_thumbnails.hdf5")


def prepare_shards():
    num_shards = 50

    if os.path.isdir("/storage/kaggle/quickdraw/train_simplified_shards"):
        os.rmdir("/storage/kaggle/quickdraw/train_simplified_shards")
    os.makedirs("/storage/kaggle/quickdraw/train_simplified_shards")

    categories = read_categories("/storage/kaggle/quickdraw/categories.txt")

    for category in categories[0:2]:
        csv_file_name = "/storage/kaggle/quickdraw/train_simplified/{}.csv".format(category)

        print("processing file '{}'".format(csv_file_name), flush=True)

        df = pd.read_csv(
            csv_file_name,
            index_col="key_id",
            usecols=["key_id", "drawing", "word"],
            converters={
                "word": lambda word: categories.index(word),
                "drawing": lambda drawing: eval(drawing)
            })

        df = df.rename(columns={"word": "category"})

        shard_size = math.ceil(len(df) / num_shards)
        indexes = df.index.values
        np.random.shuffle(indexes)

        for s in range(num_shards):
            start = s * shard_size
            end = min(start + shard_size, len(df))
            shard_df = df[df.index.isin(indexes[start:end])]
            with open("/storage/kaggle/quickdraw/train_simplified_shards/shard-{}.csv", "a") as shard_file:
                shard_df.to_csv(shard_file)


if __name__ == "__main__":
    prepare_shards()
