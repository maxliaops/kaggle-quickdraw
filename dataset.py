from queue import Queue
from threading import Thread

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision.transforms.functional import normalize

from utils import read_categories, draw_strokes


class TrainDataProvider:
    def __init__(self, data_dir, num_shards, num_shard_preload=2, num_threads=1):
        self.data_dir = data_dir
        self.num_shards = num_shards

        self.request_queue = Queue()
        self.data_queue = Queue()

        self.next_shard = 0
        for _ in range(num_shard_preload):
            self.request_data()

        for _ in range(num_threads):
            loader_thread = Thread(target=self.process_data_requests)
            loader_thread.daemon = True
            loader_thread.start()

    def get(self):
        data = self.data_queue.get()
        self.data_queue.task_done()
        self.request_data()
        return data

    def request_data(self):
        self.request_queue.put(self.next_shard)
        self.next_shard = (self.next_shard + 1) % self.num_shards

    def process_data_requests(self):
        while True:
            shard = self.request_queue.get()
            data = TrainData(self.data_dir, shard)
            self.data_queue.put(data)
            self.request_queue.task_done()


class TrainData:
    def __init__(self, data_dir, shard):
        categories = read_categories("{}/categories.txt".format(data_dir))

        data_file_name = "{}/train_simplified_shards/shard-{}.csv".format(data_dir, shard)
        print("Reading data file '{}'".format(data_file_name))

        df = pd.read_csv(data_file_name, converters={"drawing": lambda drawing: eval(drawing)})

        print("Loaded {} samples".format(len(df)))

        train_set_ids, val_set_ids = train_test_split(
            df.index,
            test_size=0.1,
            stratify=df.category,
            random_state=42
        )

        train_set_df = df[df.index.isin(train_set_ids)]
        val_set_df = df[df.index.isin(val_set_ids)]

        self.train_set_df = train_set_df
        self.val_set_df = val_set_df
        self.categories = categories


class TrainDataset(Dataset):
    def __init__(self, df, image_size):
        super().__init__()
        self.df = df
        self.image_size = image_size

    def __len__(self):
        return len(self.df["drawing"])

    def __getitem__(self, index):
        image = draw_strokes(self.df.iloc[index].drawing, size=self.image_size)
        category = self.df.iloc[index].category

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
