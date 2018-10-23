import glob
import os

import h5py
import numpy as np
import pandas as pd

from utils import draw_it

with open("/storage/kaggle/quickdraw/categories.txt") as categories_file:
    categories = [l.rstrip("\n") for l in categories_file.readlines()]

with h5py.File("/storage/kaggle/quickdraw/quickdraw_train.hdf5", "w", libver="latest") as data_file:
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
