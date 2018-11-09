import argparse
import glob

import numpy as np

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from dataset import TestData, TestDataset, TrainDataset, TrainDataProvider
from models.ensemble import Ensemble
from train import create_model
from utils import str2bool, read_categories

cudnn.enabled = True
cudnn.benchmark = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def predict(model, data_loader, tta=False):
    model.eval()

    result = []
    with torch.no_grad():
        for batch in data_loader:
            images = batch[0].to(device, non_blocking=True)

            if tta:
                predictions1 = F.softmax(model(images), dim=1)
                predictions2 = F.softmax(model(images.flip(3)), dim=1)
                predictions = 0.5 * (predictions1 + predictions2)
            else:
                predictions = F.softmax(model(images), dim=1)

            prediction_scores, prediction_categories = predictions.topk(3, dim=1, sorted=True)
            prediction_scores = prediction_scores.cpu().data.numpy()
            prediction_categories = prediction_categories.cpu().data.numpy()

            result.extend([(ps, pc) for ps, pc in zip(prediction_scores, prediction_categories)])

    return result


def main():
    args = argparser.parse_args()
    print("Arguments:")
    for arg in vars(args):
        print("  {}: {}".format(arg, getattr(args, arg)))
    print()

    input_dir = args.input_dir
    output_dir = args.output_dir
    base_model_dir = args.base_model_dir
    image_size = args.image_size
    augment = args.augment
    use_dummy_image = args.use_dummy_image
    use_progressive_image_sizes = args.use_progressive_image_sizes
    progressive_image_size_min = args.progressive_image_size_min
    progressive_image_size_step = args.progressive_image_size_step
    progressive_image_epoch_step = args.progressive_image_epoch_step
    batch_size = args.batch_size
    batch_iterations = args.batch_iterations
    test_size = args.test_size
    train_on_unrecognized = args.train_on_unrecognized
    num_category_shards = args.num_category_shards
    category_shard = args.category_shard
    exclude_categories = args.exclude_categories
    eval_train_mapk = args.eval_train_mapk
    mapk_topk = args.mapk_topk
    num_shard_preload = args.num_shard_preload
    num_shard_loaders = args.num_shard_loaders
    num_workers = args.num_workers
    pin_memory = args.pin_memory
    epochs_to_train = args.epochs
    lr_scheduler_type = args.lr_scheduler
    lr_patience = args.lr_patience
    lr_min = args.lr_min
    lr_max = args.lr_max
    lr_min_decay = args.lr_min_decay
    lr_max_decay = args.lr_max_decay
    optimizer_type = args.optimizer
    loss_type = args.loss
    loss2_type = args.loss2
    loss2_start_sgdr_cycle = args.loss2_start_sgdr_cycle
    model_type = args.model
    patience = args.patience
    sgdr_cycle_epochs = args.sgdr_cycle_epochs
    sgdr_cycle_epochs_mult = args.sgdr_cycle_epochs_mult
    sgdr_cycle_end_prolongation = args.sgdr_cycle_end_prolongation
    sgdr_cycle_end_patience = args.sgdr_cycle_end_patience
    max_sgdr_cycles = args.max_sgdr_cycles

    use_extended_stroke_channels = model_type in ["cnn", "residual_cnn", "fc_cnn", "hc_fc_cnn"]

    base_model_dirs = [
        "/storage/models/quickdraw/l1",
        "/storage/models/quickdraw/l2",
        "/storage/models/quickdraw/l3",
        "/storage/models/quickdraw/l4"
    ]

    model_categories = [
        ['vase', 'flip flops', 'hospital', 'lollipop', 'hammer', 'toothbrush', 'fork', 'moustache', 'sailboat', 'couch', 'underwear', 'church', 'tooth', 'penguin', 'apple', 'bulldozer', 'drums', 'kangaroo', 'alarm clock', 'submarine', 'spider', 'owl', 'stethoscope', 'mushroom', 'popsicle', 'airplane', 'flamingo', 'backpack', 'hot air balloon', 'toilet', 'candle', 'palm tree', 'camera', 'sock', 'power outlet', 'teapot', 'computer', 'triangle', 'diamond', 'snowflake', 'donut', 'compass', 'stitches', 'eyeglasses', 'paper clip', 'carrot', 'binoculars', 'envelope', 'cactus', 'flashlight', 'sun', 'traffic light', 'television', 'crown', 'pineapple', 'strawberry', 'saw', 'bee', 'megaphone', 'squirrel', 'wristwatch', 'flower', 'fish', 'rain', 'key', 'hourglass', 'clock', 'sheep', 'tennis racquet', 'star', 'parachute', 'giraffe', 'rollerskates', 'The Mona Lisa', 'sword', 'butterfly', 'mermaid', 'wine glass', 'bowtie', 'angel', 'eye', 'stairs', 'scorpion', 'house plant', 'anvil', 'chair', 'umbrella', 'see saw', 'snail', 'The Eiffel Tower', 'ladder', 'camel', 'octopus', 'skateboard', 'harp', 'snowman', 'skull', 'swing set', 'ice cream', 'stop sign', 'headphones', 'helicopter'],
        ['banana', 'parrot', 'tree', 'lipstick', 'teddy-bear', 'horse', 'arm', 'basket', 'necklace', 'baseball bat', 'sandwich', 'zebra', 'telephone', 'elephant', 'hot dog', 'streetlight', 'shorts', 'face', 'table', 'cow', 'postcard', 'boomerang', 'pear', 'shovel', 'zigzag', 'rhinoceros', 'onion', 'picture frame', 'saxophone', 'hat', 'cruise ship', 'train', 'ceiling fan', 'nose', 'belt', 'speedboat', 'bridge', 'barn', 'door', 'skyscraper', 'fence', 'scissors', 'shark', 'rake', 'microphone', 'ear', 'whale', 'fireplace', 'lightning', 'screwdriver', 'jacket', 'crab', 'roller coaster', 'cannon', 'garden', 'helmet', 'dresser', 'bed', 'nail', 'swan', 'fan', 'bat', 'rabbit', 'mountain', 'shoe', 'floor lamp', 'soccer ball', 'mailbox', 'laptop', 'washing machine', 'drill', 'calculator', 'ant', 'chandelier', 'hamburger', 'lighthouse', 'sea turtle', 'goatee', 'pizza', 'crocodile', 'dolphin', 'rainbow', 'frying pan', 'leaf', 'mouth', 'snorkel', 'remote control', 'light bulb', 'axe', 'hand', 'pig', 'sink', 'baseball', 'lion', 'pants', 'windmill', 'castle', 'dumbbell', 'hedgehog', 'tent', 'wine bottle', 'bandage'],
        ['animal migration', 'monkey', 'watermelon', 'radio', 'panda', 'beach', 'dishwasher', 'calendar', 'peas', 'bottlecap', 'bird', 'police car', 'ambulance', 'clarinet', 'mouse', 'snake', 'asparagus', 'cloud', 'finger', 'dragon', 'foot', 'microwave', 'cookie', 'book', 'tiger', 'sleeping bag', 'canoe', 'toothpaste', 'toe', 'broom', 'tractor', 'matches', 'brain', 'bread', 'bracelet', 'purse', 'knee', 'diving board', 'peanut', 'paintbrush', 'lantern', 'firetruck', 'pliers', 'duck', 'map', 't-shirt', 'toaster', 'yoga', 'lobster', 'elbow', 'passport', 'waterslide', 'broccoli', 'moon', 'campfire', 'jail', 'basketball', 'sweater', 'fire hydrant', 'feather', 'flying saucer', 'grass', 'spoon', 'cell phone', 'smiley face', 'beard', 'wheel', 'house'],
        ['camouflage', 'mug', 'cello', 'hurricane', 'bus', 'truck', 'pond', 'birthday cake', 'garden hose', 'cake', 'school bus', 'leg', 'van', 'guitar', 'cup', 'pool', 'hockey stick', 'bear', 'marker', 'blackberry', 'squiggle', 'tornado', 'crayon', 'circle', 'pickup truck', 'coffee cup', 'cooler', 'square', 'river', 'paint can', 'oven', 'string bean', 'The Great Wall of China', 'hockey puck', 'car', 'spreadsheet', 'trombone', 'bucket', 'trumpet', 'eraser', 'line', 'pencil', 'pillow', 'blueberry', 'frog', 'bush', 'keyboard', 'steak', 'potato', 'ocean', 'bicycle', 'mosquito', 'stereo', 'dog', 'suitcase', 'violin', 'octagon', 'bathtub', 'raccoon', 'hot tub', 'cat', 'bench', 'piano', 'stove', 'golf club', 'motorbike', 'grapes', 'hexagon']
    ]

    categories = read_categories("{}/categories.txt".format(input_dir))

    test_data = TestData(input_dir)
    test_set = TestDataset(test_data.df, image_size, use_extended_stroke_channels)
    test_set_data_loader = \
        DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    all_model_predictions = []
    for base_model_dir in base_model_dirs:
        print("Processing model dir '{}'".format(base_model_dir), flush=True)

        ms = []
        for model_file_path in glob.glob("{}/model-*.pth".format(base_model_dir)):
            m = create_model(type=model_type, input_size=image_size, num_classes=len(categories)).to(device)
            m.load_state_dict(torch.load(model_file_path, map_location=device))
            ms.append(m)
        model = Ensemble(ms)

        print("Predicting...", flush=True)
        all_model_predictions.append(predict(model, test_set_data_loader, tta=True))

    final_predictions = all_model_predictions[0].copy()
    cumulative_categories = model_categories[0].copy()
    for m in range(1, len(all_model_predictions)):
        model_predictions = all_model_predictions[m]
        for p in range(len(model_predictions)):
            final_prediction_scores = final_predictions[p][0]
            final_prediction_categories = final_predictions[p][1]
            current_prediction_scores = model_predictions[p][0]
            current_prediction_categories = model_predictions[p][1]
            for r in range(len(final_prediction_scores)):
                final_prediction_score = final_prediction_scores[r]
                final_prediction_category = final_prediction_categories[r]
                current_prediction_score = current_prediction_scores[r]
                current_prediction_category = current_prediction_categories[r]
                final_category_contained = categories[final_prediction_category] in cumulative_categories
                current_category_contained = categories[current_prediction_category] in model_categories[m]
                if final_category_contained == current_category_contained:
                    if current_prediction_score > final_prediction_score:
                        final_prediction_scores[r] = current_prediction_score
                        final_prediction_category[r] = current_prediction_category
                else:
                    if current_category_contained:
                        final_prediction_scores[r] = current_prediction_score
                        final_prediction_category[r] = current_prediction_category
        cumulative_categories.extend(model_categories[m])

    words = np.array([c.replace(" ", "_") for c in categories])

    submission_df = test_data.df.copy()
    submission_df["word"] = [" ".join(words[fp[1]]) for fp in final_predictions]
    submission_df.to_csv("{}/submission.csv".format(output_dir), columns=["word"])


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--input_dir", default="/storage/kaggle/quickdraw")
    argparser.add_argument("--output_dir", default="/artifacts")
    argparser.add_argument("--base_model_dir", default=None)
    argparser.add_argument("--image_size", default=128, type=int)
    argparser.add_argument("--augment", default=False, type=str2bool)
    argparser.add_argument("--use_dummy_image", default=False, type=str2bool)
    argparser.add_argument("--use_progressive_image_sizes", default=False, type=str2bool)
    argparser.add_argument("--progressive_image_size_min", default=32, type=int)
    argparser.add_argument("--progressive_image_size_step", default=16, type=int)
    argparser.add_argument("--progressive_image_epoch_step", default=7, type=int)
    argparser.add_argument("--epochs", default=500, type=int)
    argparser.add_argument("--batch_size", default=256, type=int)
    argparser.add_argument("--batch_iterations", default=1, type=int)
    argparser.add_argument("--test_size", default=0.1, type=float)
    argparser.add_argument("--train_on_unrecognized", default=True, type=str2bool)
    argparser.add_argument("--num_category_shards", default=1, type=int)
    argparser.add_argument("--category_shard", default=0, type=int)
    argparser.add_argument("--exclude_categories", default=False, type=str2bool)
    argparser.add_argument("--eval_train_mapk", default=True, type=str2bool)
    argparser.add_argument("--mapk_topk", default=3, type=int)
    argparser.add_argument("--num_shard_preload", default=1, type=int)
    argparser.add_argument("--num_shard_loaders", default=1, type=int)
    argparser.add_argument("--num_workers", default=8, type=int)
    argparser.add_argument("--pin_memory", default=True, type=str2bool)
    argparser.add_argument("--lr_scheduler", default="cosine_annealing")
    argparser.add_argument("--lr_patience", default=3, type=int)
    argparser.add_argument("--lr_min", default=0.01, type=float)
    argparser.add_argument("--lr_max", default=0.1, type=float)
    argparser.add_argument("--lr_min_decay", default=1.0, type=float)
    argparser.add_argument("--lr_max_decay", default=1.0, type=float)
    argparser.add_argument("--model", default="seresnext50")
    argparser.add_argument("--patience", default=5, type=int)
    argparser.add_argument("--optimizer", default="sgd")
    argparser.add_argument("--loss", default="cce")
    argparser.add_argument("--loss2", default=None)
    argparser.add_argument("--loss2_start_sgdr_cycle", default=None, type=int)
    argparser.add_argument("--sgdr_cycle_epochs", default=5, type=int)
    argparser.add_argument("--sgdr_cycle_epochs_mult", default=1.0, type=float)
    argparser.add_argument("--sgdr_cycle_end_prolongation", default=0, type=int)
    argparser.add_argument("--sgdr_cycle_end_patience", default=1, type=int)
    argparser.add_argument("--max_sgdr_cycles", default=None, type=int)

    main()
