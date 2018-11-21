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

        # m = {0: 0, 1: 1, 2: 217, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 43, 10: 10, 11: 11, 12: 213, 13: 13, 14: 14, 15: 15, 16: 16, 17: 74, 18: 15, 19: 19, 20: 213, 21: 21, 22: 80, 23: 23, 24: 24, 25: 25, 26: 26, 27: 24, 28: 28, 29: 29, 30: 116, 31: 31, 32: 32, 33: 32, 34: 34, 35: 35, 36: 36, 37: 37, 38: 38, 39: 39, 40: 40, 41: 41, 42: 42, 43: 43, 44: 74, 45: 45, 46: 217, 47: 47, 48: 48, 49: 49, 50: 31, 51: 51, 52: 51, 53: 53, 54: 182, 55: 55, 56: 56, 57: 57, 58: 58, 59: 59, 60: 217, 61: 61, 62: 62, 63: 80, 64: 64, 65: 65, 66: 66, 67: 67, 68: 68, 69: 69, 70: 213, 71: 319, 72: 72, 73: 47, 74: 74, 75: 72, 76: 76, 77: 213, 78: 182, 79: 79, 80: 80, 81: 81, 82: 172, 83: 83, 84: 84, 85: 243, 86: 74, 87: 87, 88: 182, 89: 213, 90: 80, 91: 91, 92: 92, 93: 93, 94: 94, 95: 95, 96: 96, 97: 97, 98: 116, 99: 99, 100: 100, 101: 8, 102: 102, 103: 34, 104: 104, 105: 105, 106: 106, 107: 107, 108: 64, 109: 109, 110: 110, 111: 161, 112: 112, 113: 56, 114: 217, 115: 115, 116: 116, 117: 117, 118: 118, 119: 119, 120: 120, 121: 121, 122: 161, 123: 123, 124: 124, 125: 125, 126: 126, 127: 127, 128: 128, 129: 23, 130: 161, 131: 32, 132: 132, 133: 65, 134: 134, 135: 10, 136: 8, 137: 137, 138: 121, 139: 139, 140: 140, 141: 141, 142: 142, 143: 143, 144: 36, 145: 161, 146: 80, 147: 69, 148: 148, 149: 149, 150: 213, 151: 151, 152: 14, 153: 153, 154: 154, 155: 155, 156: 156, 157: 110, 158: 158, 159: 159, 160: 24, 161: 161, 162: 162, 163: 163, 164: 76, 165: 109, 166: 161, 167: 148, 168: 168, 169: 21, 170: 170, 171: 171, 172: 172, 173: 81, 174: 174, 175: 175, 176: 176, 177: 172, 178: 178, 179: 179, 180: 180, 181: 181, 182: 182, 183: 80, 184: 213, 185: 185, 186: 28, 187: 21, 188: 188, 189: 189, 190: 190, 191: 74, 192: 192, 193: 161, 194: 38, 195: 195, 196: 21, 197: 143, 198: 198, 199: 199, 200: 182, 201: 201, 202: 43, 203: 74, 204: 204, 205: 80, 206: 206, 207: 207, 208: 208, 209: 116, 210: 34, 211: 211, 212: 212, 213: 213, 214: 172, 215: 116, 216: 24, 217: 217, 218: 182, 219: 219, 220: 74, 221: 221, 222: 222, 223: 223, 224: 217, 225: 213, 226: 213, 227: 227, 228: 34, 229: 213, 230: 230, 231: 74, 232: 232, 233: 80, 234: 234, 235: 235, 236: 236, 237: 43, 238: 238, 239: 239, 240: 21, 241: 241, 242: 242, 243: 243, 244: 134, 245: 245, 246: 246, 247: 217, 248: 223, 249: 81, 250: 250, 251: 251, 252: 252, 253: 91, 254: 254, 255: 255, 256: 206, 257: 257, 258: 258, 259: 259, 260: 260, 261: 93, 262: 262, 263: 107, 264: 264, 265: 127, 266: 266, 267: 267, 268: 268, 269: 269, 270: 270, 271: 243, 272: 272, 273: 273, 274: 51, 275: 74, 276: 21, 277: 277, 278: 278, 279: 279, 280: 213, 281: 234, 282: 282, 283: 283, 284: 284, 285: 182, 286: 286, 287: 287, 288: 213, 289: 289, 290: 74, 291: 291, 292: 116, 293: 156, 294: 294, 295: 295, 296: 24, 297: 297, 298: 80, 299: 66, 300: 300, 301: 181, 302: 302, 303: 303, 304: 21, 305: 305, 306: 80, 307: 307, 308: 161, 309: 309, 310: 310, 311: 311, 312: 172, 313: 154, 314: 45, 315: 287, 316: 316, 317: 42, 318: 318, 319: 319, 320: 217, 321: 319, 322: 156, 323: 323, 324: 206, 325: 217, 326: 326, 327: 65, 328: 182, 329: 15, 330: 21, 331: 115, 332: 222, 333: 333, 334: 334, 335: 335, 336: 336, 337: 337, 338: 80, 339: 21}
        # data_category = np.array([m[c] for c in data_category])

        print("Loaded {} samples".format(len(data_drawing)))

        categories = read_lines("{}/categories.txt".format(data_dir))

        countries = read_lines("{}/countries.txt".format(data_dir))
        country_index_map = {c: countries.index(c) for c in countries}

        data_country = np.array([country_index_map[c] if isinstance(c, str) else 255 for c in data_countrycode],
                                dtype=np.uint8)

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
    return torch.tensor(category.item()).long()


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
