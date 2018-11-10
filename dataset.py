import datetime
import multiprocessing as mp
import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from utils import read_categories, draw_temporal_strokes, read_confusion_set


class TrainDataProvider:
    def __init__(
            self,
            data_dir,
            num_shards,
            num_shard_preload,
            num_workers,
            test_size,
            train_on_unrecognized,
            confusion_set,
            num_category_shards,
            category_shard,
            exclude_categories):
        self.data_dir = data_dir
        self.test_size = test_size
        self.train_on_unrecognized = train_on_unrecognized
        self.confusion_set = confusion_set
        self.num_category_shards = num_category_shards
        self.category_shard = category_shard
        self.exclude_categories = exclude_categories

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
        self.requests.append(self.pool.apply_async(
            TrainDataProvider.load_data,
            (
                self.data_dir,
                next_shard,
                self.test_size,
                self.train_on_unrecognized,
                self.confusion_set,
                self.num_category_shards,
                self.category_shard,
                self.exclude_categories
            )))
        self.next_shard_index = (self.next_shard_index + 1) % len(self.shards)

    @staticmethod
    def load_data(
            data_dir,
            shard,
            test_size,
            train_on_unrecognized,
            confusion_set,
            num_category_shards,
            category_shard,
            exclude_categories):
        print("[{}] Loading data for shard {}".format(mp.current_process().name, shard), flush=True)
        return TrainData(
            data_dir,
            shard,
            test_size,
            train_on_unrecognized,
            confusion_set,
            num_category_shards,
            category_shard,
            exclude_categories)


class TrainData:
    def __init__(
            self,
            data_dir,
            shard,
            test_size,
            train_on_unrecognized,
            confusion_set,
            num_category_shards,
            category_shard,
            exclude_categories):
        self.shard = shard

        start_time = time.time()

        data_file_name = "{}/train_simplified_shards/shard-{}.npz".format(data_dir, shard)
        print("Reading data file '{}'".format(data_file_name), flush=True)

        with np.load(data_file_name) as data_file:
            data_category = data_file["category"]
            data_drawing = data_file["drawing"]
            data_recognized = data_file["recognized"]

        print("Loaded {} samples".format(len(data_drawing)))

        categories = read_categories("{}/categories.txt".format(data_dir))
        if num_category_shards != 1:
            category_shard_size = len(categories) // num_category_shards
            min_category = category_shard * category_shard_size
            max_category = min(min_category + category_shard_size, len(categories))
            categories = categories[min_category:max_category]
            print("Using the category range [{},{})".format(min_category, max_category))

            category_filter = (data_category >= min_category) & (data_category < max_category)
            data_category = data_category[category_filter] - min_category
            data_drawing = data_drawing[category_filter]
            data_recognized = data_recognized[category_filter]

        train_categories, val_categories, train_drawing, val_drawing, train_recognized, _ = \
            train_test_split(
                data_category,
                data_drawing,
                data_recognized,
                test_size=test_size,
                stratify=data_category,
                random_state=42
            )

        if True:
            confusion_set_categories = ['angel', 'arm', 'bat', 'bathtub', 'bottlecap', 'hospital', 'police car', 'spider', 'sun', 'tent', 'triangle', 'windmill']

            categories_mask = np.array([c in confusion_set_categories for c in categories])

            train_category_filter = np.array([categories_mask[dc] for dc in train_categories])
            train_categories = train_categories[train_category_filter]
            train_drawing = train_drawing[train_category_filter]
            train_recognized = train_recognized[train_category_filter]

            val_category_filter = np.array([categories_mask[dc] for dc in val_categories])
            val_categories = val_categories[val_category_filter]
            val_drawing = val_drawing[val_category_filter]

        if not train_on_unrecognized:
            train_categories = train_categories[train_recognized]
            train_drawing = train_drawing[train_recognized]

        self.train_set_df = {"category": train_categories, "drawing": train_drawing}
        self.val_set_df = {"category": val_categories, "drawing": val_drawing}
        self.categories = categories

        end_time = time.time()
        print("Time to load data of shard {}: {}".format(shard, str(datetime.timedelta(seconds=end_time - start_time))),
              flush=True)


class TrainDataset(Dataset):
    def __init__(self, df, image_size, use_extended_stroke_channels, augment, use_dummy_image):
        super().__init__()
        self.df = df
        self.image_size = image_size
        self.use_extended_stroke_channels = use_extended_stroke_channels
        self.augment = augment
        self.use_dummy_image = use_dummy_image

    def __len__(self):
        return len(self.df["drawing"])

    def __getitem__(self, index):
        drawing = self.df["drawing"][index]
        category = self.df["category"][index]

        if self.use_dummy_image:
            image = np.zeros((self.image_size, self.image_size))
        elif "image" in self.df:
            image = self.df["image"][index]
        else:
            fliplr = False
            padding = 3

            if self.augment:
                if np.random.rand() < 0.5:
                    fliplr = True
                if np.random.rand() < 0.2:
                    padding += np.random.randint(5, 50)

            image = draw_temporal_strokes(
                drawing,
                size=self.image_size,
                padding=padding,
                fliplr=fliplr,
                extended_channels=self.use_extended_stroke_channels)

        image = image_to_tensor(image)
        category = category_to_tensor(category)

        # image_mean = 0.0
        # image_stdev = 1.0
        # image = normalize(image, (image_mean, image_mean, image_mean), (image_stdev, image_stdev, image_stdev))

        return image, category


class TestData:
    def __init__(self, data_dir):
        self.df = pd.read_csv(
            "{}/test_simplified.csv".format(data_dir),
            index_col="key_id",
            converters={"drawing": lambda drawing: eval(drawing)})


class TestDataset(Dataset):
    def __init__(self, df, image_size, use_extended_stroke_channels):
        super().__init__()
        self.df = df
        self.image_size = image_size
        self.use_extended_stroke_channels = use_extended_stroke_channels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        drawing = self.df.iloc[index].drawing

        image = draw_temporal_strokes(
            drawing,
            size=self.image_size,
            padding=3,
            extended_channels=self.use_extended_stroke_channels)

        image = image_to_tensor(image)

        return (image,)


def image_to_tensor(image):
    if len(image.shape) == 2:
        image = np.expand_dims(image, 0)
    return torch.from_numpy(image / 255.).float()


def category_to_tensor(category):
    return torch.tensor(category.item()).long()
