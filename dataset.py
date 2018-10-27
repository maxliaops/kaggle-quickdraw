import shutil

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision.transforms.functional import normalize

from utils import draw_strokes, assemble_strokes


class DataFrame:
    def __init__(self, data_file, locs):
        self.data_file = data_file
        self.locs = locs

    def __len__(self):
        return len(self.locs)

    def category(self, index):
        return self.data_file["category"][self.locs[index]].item()

    def strokes(self, index):
        print("{} -> {}".format(index, self.locs[index]))
        stroke_x = self.data_file["stroke_x"][self.locs[index]]
        stroke_y = self.data_file["stroke_y"][self.locs[index]]
        stroke_len = self.data_file["stroke_len"][self.locs[index]]
        return assemble_strokes(stroke_x, stroke_y, stroke_len)


class TrainData:
    def __init__(self, data_dir, samples_per_category, num_loaders):
        with open("{}/categories.txt".format(data_dir)) as categories_file:
            categories = [l.rstrip("\n") for l in categories_file.readlines()]

        shutil.copy("{}/quickdraw_train.hdf5".format(data_dir), ".")
        data_file = h5py.File("quickdraw_train.hdf5", "r", libver="latest")

        num_samples = len(data_file["category"])
        print("Loaded {} samples".format(num_samples))

        train_set_ids, val_set_ids = train_test_split(
            range(num_samples),
            test_size=0.06,
            stratify=data_file["category"].value,
            random_state=42
        )

        self.train_set_df = DataFrame(data_file, train_set_ids)
        self.val_set_df = DataFrame(data_file, val_set_ids)
        self.categories = categories

    @staticmethod
    def load_data(category, samples_per_category):
        print("reading the data for category '{}'".format(category), flush=True)
        return pd.read_hdf(
            "/storage/kaggle/quickdraw/quickdraw_train_pd.hdf5",
            key=category,
            start=0,
            stop=samples_per_category if samples_per_category > 0 else None)


class TrainDataset(Dataset):
    def __init__(self, df, image_size):
        super().__init__()
        self.df = df
        self.image_size = image_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image = draw_strokes(self.df.strokes(index))
        category = self.df.category(index)

        image = self.image_to_tensor(image)
        category = self.category_to_tensor(category)

        image_mean = 0.0
        image_stdev = 1.0

        image = normalize(image, (image_mean, image_mean, image_mean), (image_stdev, image_stdev, image_stdev))

        return image, category

    def image_to_tensor(self, image):
        image = np.expand_dims(image, 0)
        return torch.from_numpy((image / 255.)).float()

    def category_to_tensor(self, category):
        return torch.tensor(category).long()
