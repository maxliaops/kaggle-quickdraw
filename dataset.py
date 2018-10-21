import glob
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision.transforms.functional import normalize

from utils import draw_it


class TrainData:
    def __init__(self, data_dir):
        data_files = glob.glob("{}/train_simplified_shard_0/*.csv".format(data_dir))

        with Pool(8) as pool:
            df = pd.concat([c for c in pool.map(self.load_data, data_files)])

        with open("{}/categories.txt".format(data_dir)) as categories_file:
            categories = [l.rstrip("\n") for l in categories_file.readlines()]

        df["category"] = [categories.index(word) for word in df.word]

        train_set_ids, val_set_ids = train_test_split(
            df.index,
            test_size=0.2,
            stratify=df.word
        )

        train_set_df = df[df.index.isin(train_set_ids)].copy()
        val_set_df = df[df.index.isin(val_set_ids)].copy()

        self.train_set_df = train_set_df
        self.val_set_df = val_set_df
        self.categories = categories

    def load_data(self, data_file):
        print("reading the data file '{}'".format(data_file), flush=True)
        return pd.read_csv(
            data_file,
            index_col="key_id",
            converters={"drawing": lambda drawing: eval(drawing)})


class TrainDataset(Dataset):
    def __init__(self, df, image_size):
        super().__init__()
        self.df = df
        self.image_size = image_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        print("getting item {}".format(index), flush=True)

        image = draw_it(self.df.drawing.values[index], size=self.image_size)
        category = self.df.category.values[index]

        image = image_to_tensor(image)
        category = category_to_tensor(category)

        image_mean = 0.0
        image_stdev = 1.0

        image = normalize(image, (image_mean, image_mean, image_mean), (image_stdev, image_stdev, image_stdev))

        return image, category


def calculate_coverage_class(mask):
    coverage = mask.sum() / mask.size
    for i in range(0, 11):
        if coverage * 10 <= i:
            return i


def image_to_tensor(image):
    image = np.expand_dims(image, 0)
    image = np.repeat(image, 3, 0)
    return torch.from_numpy((image / 255.)).float()


def category_to_tensor(category):
    return torch.tensor(category.item()).long()
