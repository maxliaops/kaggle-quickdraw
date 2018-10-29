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
    def __init__(self, data_dir, num_shards, num_shard_preload, num_workers):
        self.data_dir = data_dir
        self.shards = list(range(num_shards))
        np.random.shuffle(self.shards)

        self.pool = mp.Pool(processes=num_workers)
        self.requests = []

        self.next_shard_index = 0
        for _ in range(num_shard_preload):
            self.request_data()

    def get_next(self):
        start_time = time.time()

        if len(self.requests) == 0:
            self.request_data()
        # data = self.requests.pop(0).get()
        data = self.requests[0]

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
        # self.requests.append(self.pool.apply_async(TrainDataProvider.load_data, (self.data_dir, next_shard)))
        self.requests.append(TrainDataProvider.load_data(self.data_dir, next_shard))
        self.next_shard_index = (self.next_shard_index + 1) % len(self.shards)

    @staticmethod
    def load_data(data_dir, shard):
        print("[{}] Loading data for shard {}".format(mp.current_process().name, shard), flush=True)
        return TrainData(data_dir, shard)


class TrainData:
    def __init__(self, data_dir, shard):
        self.shard = shard

        start_time = time.time()

        categories = read_categories("{}/categories.txt".format(data_dir))

        data_file_name = "{}/train_simplified_shards/shard-{}.npz".format(data_dir, shard)
        print("Reading data file '{}'".format(data_file_name), flush=True)

        data = np.load(data_file_name)
        data_category = data["category"]
        data_drawing = data["drawing"]

        print("Loaded {} samples".format(len(data_category)))

        train_categories, val_categories, train_drawing, val_drawing = train_test_split(
            data_category,
            data_drawing,
            test_size=0.3,
            stratify=data_category,
            random_state=42
        )

        self.train_set_df = {"category": train_categories, "drawing": train_drawing}
        self.val_set_df = {"category": val_categories, "drawing": val_drawing}
        self.categories = categories

        data.close()

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
