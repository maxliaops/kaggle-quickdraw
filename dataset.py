import datetime
import math
import multiprocessing as mp
import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from torch.utils.data import Dataset
from torch.utils.data import Sampler

from utils import read_lines, draw_temporal_strokes, read_confusion_set, kfold_split


class TrainDataProvider:
    def __init__(
            self,
            data_dir,
            num_shards,
            num_shard_preload,
            num_workers,
            test_size,
            fold,
            train_on_unrecognized,
            confusion_set,
            num_category_shards,
            category_shard):
        self.data_dir = data_dir
        self.test_size = test_size
        self.fold = fold
        self.train_on_unrecognized = train_on_unrecognized
        self.confusion_set = confusion_set
        self.num_category_shards = num_category_shards
        self.category_shard = category_shard

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
                self.fold,
                self.train_on_unrecognized,
                self.confusion_set,
                self.num_category_shards,
                self.category_shard
            )))
        self.next_shard_index = (self.next_shard_index + 1) % len(self.shards)

    @staticmethod
    def load_data(
            data_dir,
            shard,
            test_size,
            fold,
            train_on_unrecognized,
            confusion_set,
            num_category_shards,
            category_shard):
        print("[{}] Loading data for shard {}".format(mp.current_process().name, shard), flush=True)
        return TrainData(
            data_dir,
            shard,
            test_size,
            fold,
            train_on_unrecognized,
            confusion_set,
            num_category_shards,
            category_shard)


class TrainData:
    def __init__(
            self,
            data_dir,
            shard,
            test_size,
            fold,
            train_on_unrecognized,
            confusion_set,
            num_category_shards,
            category_shard):
        self.shard = shard

        start_time = time.time()

        data_file_name = "{}/train_simplified_shards/shard-{}.npz".format(data_dir, shard)
        print("Reading data file '{}'".format(data_file_name), flush=True)

        with np.load(data_file_name) as data_file:
            data_category = data_file["category"]
            data_drawing = data_file["drawing"]
            data_recognized = data_file["recognized"]
            data_countrycode = data_file["countrycode"]

        print("Loaded {} samples".format(len(data_drawing)))

        categories = read_lines("{}/categories.txt".format(data_dir))

        countries = read_lines("{}/countries.txt".format(data_dir))
        country_index_map = {c: countries.index(c) for c in countries}

        data_country = np.array([country_index_map[c] if isinstance(c, str) else 255 for c in data_countrycode], dtype=np.uint8)

        # m = {0: 44, 1: 4, 2: 7, 3: 33, 4: 33, 5: 43, 6: 45, 7: 37, 8: 5, 9: 31, 10: 10, 11: 27, 12: 56, 13: 1, 14: 17, 15: 0, 16: 24, 17: 38, 18: 0, 19: 33, 20: 16, 21: 74, 22: 59, 23: 8, 24: 58, 25: 25, 26: 1, 27: 58, 28: 26, 29: 6, 30: 33, 31: 2, 32: 9, 33: 37, 34: 36, 35: 56, 36: 12, 37: 44, 38: 28, 39: 64, 40: 8, 41: 74, 42: 66, 43: 39, 44: 32, 45: 65, 46: 41, 47: 3, 48: 33, 49: 19, 50: 2, 51: 51, 52: 51, 53: 14, 54: 30, 55: 64, 56: 30, 57: 24, 58: 69, 59: 56, 60: 7, 61: 31, 62: 74, 63: 76, 64: 30, 65: 72, 66: 60, 67: 58, 68: 30, 69: 15, 70: 37, 71: 69, 72: 4, 73: 3, 74: 27, 75: 4, 76: 20, 77: 37, 78: 30, 79: 58, 80: 14, 81: 43, 82: 24, 83: 33, 84: 74, 85: 50, 86: 27, 87: 45, 88: 30, 89: 34, 90: 14, 91: 44, 92: 28, 93: 47, 94: 33, 95: 30, 96: 46, 97: 32, 98: 33, 99: 6, 100: 37, 101: 5, 102: 14, 103: 36, 104: 24, 105: 1, 106: 6, 107: 48, 108: 30, 109: 21, 110: 18, 111: 62, 112: 19, 113: 30, 114: 7, 115: 44, 116: 33, 117: 19, 118: 8, 119: 19, 120: 30, 121: 11, 122: 62, 123: 39, 124: 59, 125: 46, 126: 74, 127: 49, 128: 33, 129: 8, 130: 13, 131: 9, 132: 18, 133: 72, 134: 40, 135: 10, 136: 62, 137: 18, 138: 11, 139: 54, 140: 59, 141: 16, 142: 11, 143: 29, 144: 12, 145: 13, 146: 14, 147: 15, 148: 23, 149: 16, 150: 16, 151: 71, 152: 17, 153: 71, 154: 64, 155: 45, 156: 57, 157: 18, 158: 76, 159: 24, 160: 58, 161: 22, 162: 34, 163: 19, 164: 20, 165: 21, 166: 22, 167: 23, 168: 19, 169: 74, 170: 22, 171: 76, 172: 24, 173: 43, 174: 35, 175: 10, 176: 64, 177: 24, 178: 31, 179: 69, 180: 34, 181: 61, 182: 30, 183: 59, 184: 56, 185: 25, 186: 26, 187: 74, 188: 59, 189: 49, 190: 63, 191: 27, 192: 70, 193: 62, 194: 28, 195: 22, 196: 74, 197: 29, 198: 43, 199: 73, 200: 30, 201: 33, 202: 31, 203: 32, 204: 30, 205: 59, 206: 45, 207: 37, 208: 11, 209: 33, 210: 36, 211: 37, 212: 37, 213: 56, 214: 24, 215: 33, 216: 58, 217: 68, 218: 30, 219: 59, 220: 38, 221: 55, 222: 75, 223: 42, 224: 7, 225: 34, 226: 34, 227: 35, 228: 36, 229: 37, 230: 30, 231: 38, 232: 14, 233: 76, 234: 53, 235: 18, 236: 56, 237: 39, 238: 60, 239: 14, 240: 74, 241: 74, 242: 68, 243: 50, 244: 40, 245: 24, 246: 69, 247: 41, 248: 42, 249: 43, 250: 46, 251: 59, 252: 24, 253: 44, 254: 59, 255: 62, 256: 45, 257: 46, 258: 16, 259: 34, 260: 48, 261: 47, 262: 8, 263: 48, 264: 49, 265: 49, 266: 69, 267: 52, 268: 59, 269: 37, 270: 62, 271: 50, 272: 43, 273: 46, 274: 51, 275: 38, 276: 74, 277: 76, 278: 74, 279: 52, 280: 37, 281: 53, 282: 54, 283: 18, 284: 29, 285: 30, 286: 55, 287: 19, 288: 56, 289: 16, 290: 38, 291: 43, 292: 33, 293: 57, 294: 24, 295: 46, 296: 58, 297: 27, 298: 59, 299: 60, 300: 53, 301: 61, 302: 67, 303: 67, 304: 74, 305: 30, 306: 76, 307: 30, 308: 62, 309: 58, 310: 63, 311: 24, 312: 24, 313: 64, 314: 65, 315: 19, 316: 68, 317: 66, 318: 67, 319: 69, 320: 68, 321: 69, 322: 57, 323: 70, 324: 45, 325: 68, 326: 71, 327: 72, 328: 30, 329: 73, 330: 74, 331: 44, 332: 75, 333: 30, 334: 19, 335: 19, 336: 1, 337: 34, 338: 76, 339: 74}
        # data_category = np.array([m[c] for c in data_category])
        # categories = ["group{}".format(i) for i in range(max(m.values()) + 1)]

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
            data_country = data_country[category_filter]

        if fold is None:
            train_categories, val_categories, train_drawing, val_drawing, train_recognized, val_recognized, train_country, val_country = \
                train_test_split(
                    data_category,
                    data_drawing,
                    data_recognized,
                    data_country,
                    test_size=test_size,
                    stratify=data_category,
                    random_state=42
                )
        else:
            train_indexes, val_indexes = list(kfold_split(3, range(len(data_category)), data_category))[fold]

            train_categories = data_category[train_indexes]
            train_drawing = data_drawing[train_indexes]
            train_recognized = data_recognized[train_indexes]
            train_country = data_country[train_indexes]

            val_categories = data_category[val_indexes]
            val_drawing = data_drawing[val_indexes]
            val_recognized = data_recognized[val_indexes]
            val_country = data_country[val_indexes]

        if False:
            categories_subset = []

            categories_mask = np.array([c in categories_subset for c in categories])

            train_category_filter = np.array([categories_mask[dc] for dc in train_categories])
            train_categories = train_categories[train_category_filter]
            train_drawing = train_drawing[train_category_filter]
            train_recognized = train_recognized[train_category_filter]
            train_country = train_country[train_category_filter]

            val_category_filter = np.array([categories_mask[dc] for dc in val_categories])
            val_categories = val_categories[val_category_filter]
            val_drawing = val_drawing[val_category_filter]
            val_recognized = val_recognized[val_category_filter]
            val_country = val_country[val_category_filter]

        if confusion_set is not None:
            confusion_set_categories = read_confusion_set(
                "/storage/models/quickdraw/seresnext50_confusion/confusion_set_{}.txt".format(confusion_set))

            categories_mask = np.array([c in confusion_set_categories for c in categories])

            train_category_filter = np.array([categories_mask[dc] for dc in train_categories])
            train_categories = train_categories[train_category_filter]
            train_drawing = train_drawing[train_category_filter]
            train_recognized = train_recognized[train_category_filter]
            train_country = train_country[train_category_filter]

            val_category_filter = np.array([categories_mask[dc] for dc in val_categories])
            val_categories = val_categories[val_category_filter]
            val_drawing = val_drawing[val_category_filter]
            val_recognized = val_recognized[val_category_filter]
            val_country = val_country[val_category_filter]

            category_mapping = {}
            for csc in confusion_set_categories:
                category_mapping[categories.index(csc)] = confusion_set_categories.index(csc)
            train_categories = np.array([category_mapping[c] for c in train_categories])
            val_categories = np.array([category_mapping[c] for c in val_categories])
            categories = confusion_set_categories

        if not train_on_unrecognized:
            train_categories = train_categories[train_recognized]
            train_drawing = train_drawing[train_recognized]
            train_country = train_country[train_recognized]
            train_recognized = train_recognized[train_recognized]

        self.train_set_df = {
            "category": train_categories,
            "drawing": train_drawing,
            "country": train_country,
            "recognized": train_recognized
        }
        self.val_set_df = {
            "category": val_categories,
            "drawing": val_drawing,
            "country": val_country,
            "recognized": val_recognized
        }
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
        country = self.df["country"][index]

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
                    # if np.random.rand() < 0.2:
                    #     padding += np.random.randint(5, 50)

            image = draw_temporal_strokes(
                drawing,
                size=self.image_size,
                padding=padding,
                fliplr=fliplr,
                extended_channels=self.use_extended_stroke_channels)

        image = image_to_tensor(image)
        category = category_to_tensor(category)

        # values_channel = calculate_drawing_values_channel(drawing, country, self.image_size)
        # image = torch.cat([torch.from_numpy(values_channel).float().unsqueeze(0), image], dim=0)

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

        countries = read_lines("{}/countries.txt".format(data_dir))
        self.df["country"] = [countries.index(c) if isinstance(c, str) else 255 for c in self.df.countrycode]


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
        country = self.df.iloc[index].country

        image = draw_temporal_strokes(
            drawing,
            size=self.image_size,
            padding=3,
            extended_channels=self.use_extended_stroke_channels)

        image = image_to_tensor(image)

        # values_channel = calculate_drawing_values_channel(drawing, country, self.image_size)
        # image = torch.cat([torch.from_numpy(values_channel).float().unsqueeze(0), image], dim=0)

        return (image,)


def image_to_tensor(image):
    if len(image.shape) == 2:
        image = np.expand_dims(image, 0)
    return torch.from_numpy(image / 255.).float()


def category_to_tensor(category):
    a = np.zeros((68, 68), dtype=np.float32)
    a[category.item()] = 1.0
    return torch.tensor(a).float()


class StratifiedSampler(Sampler):
    def __init__(self, class_vector, batch_size):
        super().__init__(None)
        self.class_vector = class_vector
        self.batch_size = batch_size

    def gen_sample_array(self):
        n_splits = math.ceil(len(self.class_vector) / self.batch_size)
        splitter = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.5)
        train_index, test_index = next(splitter.split(np.zeros(len(self.class_vector)), self.class_vector))
        return np.hstack([train_index, test_index])

    def __iter__(self):
        return iter(self.gen_sample_array())

    def __len__(self):
        return len(self.class_vector)
