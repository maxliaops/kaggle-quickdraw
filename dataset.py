import datetime
import multiprocessing as mp
import time

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision.transforms.functional import normalize

from utils import read_categories, draw_strokes


class TrainDataProvider:
    def __init__(self, data_dir, num_shards, num_shard_preload, num_workers, test_size):
        self.data_dir = data_dir
        self.test_size = test_size

        self.shards = list(range(num_shards))
        np.random.shuffle(self.shards)

        self.pool = mp.Pool(processes=num_workers)
        self.requests = []

        self.next_shard_index = 0
        for _ in range(num_shard_preload):
            self.request_data()

    def get_next(self):
        start_time = time.time()

        self.request_data()
        data = self.requests.pop(0).get()

        end_time = time.time()
        print("[{}] Time to provide data of shard {}: {}".format(
            mp.current_process().name,
            data.shard,
            str(datetime.timedelta(seconds=end_time - start_time))),
            flush=True)

        return data

    def request_data(self):
        next_shard = self.shards[self.next_shard_index]
        print("[{}] Placing request for shard {}".format(mp.current_process().name, next_shard), flush=True)
        self.requests.append(self.pool.apply_async(TrainDataProvider.load_data, (self.data_dir, next_shard, self.test_size)))
        self.next_shard_index = (self.next_shard_index + 1) % len(self.shards)

    @staticmethod
    def load_data(data_dir, shard, test_size):
        print("[{}] Loading data for shard {}".format(mp.current_process().name, shard), flush=True)
        return TrainData(data_dir, shard, test_size)


class TrainData:
    def __init__(self, data_dir, shard, test_size):
        self.shard = shard

        start_time = time.time()

        categories = read_categories("{}/categories.txt".format(data_dir))

        data_file_name = "{}/train_simplified_shards/shard-{}.npz".format(data_dir, shard)
        print("Reading data file '{}'".format(data_file_name), flush=True)

        with np.load(data_file_name) as data_file:
            data_category = data_file["category"]
            data_drawing = data_file["drawing"]

        print("Loaded {} samples".format(len(data_category)))

        train_categories, val_categories, train_drawing, val_drawing = train_test_split(
            data_category,
            data_drawing,
            test_size=test_size,
            stratify=data_category,
            random_state=42
        )

        self.train_set_df = {"category": train_categories, "drawing": train_drawing}
        self.val_set_df = {"category": val_categories, "drawing": val_drawing}
        self.categories = categories

        end_time = time.time()
        print("Time to load data of shard {}: {}".format(shard, str(datetime.timedelta(seconds=end_time - start_time))),
              flush=True)


class TrainDataset(Dataset):
    def __init__(self, df, image_size):
        super().__init__()
        self.df = df
        self.image_size = image_size

    def __len__(self):
        return len(self.df["drawing"])

    def __getitem__(self, index):
        drawing = self.df["drawing"][index]
        category = self.df["category"][index]

        image = self.df["image"][index] if "image" in self.df else draw_strokes(drawing, size=self.image_size)

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
        return torch.tensor(category.item()).long()
