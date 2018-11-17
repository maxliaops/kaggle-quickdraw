import argparse
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from dataset import TrainDataProvider, TrainDataset
from train import calculate_confusion, load_ensemble_model, create_criterion
from utils import str2bool, read_lines, pack_confusion_sets, save_confusion_set

cudnn.enabled = True
cudnn.benchmark = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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

    train_data_provider = TrainDataProvider(
        input_dir,
        50,
        num_shard_preload=num_shard_preload,
        num_workers=num_shard_loaders,
        test_size=test_size,
        fold=None,
        train_on_unrecognized=train_on_unrecognized,
        confusion_set=None,
        num_category_shards=num_category_shards,
        category_shard=category_shard)

    train_data = train_data_provider.get_next()

    val_set = TrainDataset(train_data.val_set_df, image_size, use_extended_stroke_channels, False, use_dummy_image)
    val_set_data_loader = \
        DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    categories = read_lines("{}/categories.txt".format(input_dir))

    criterion = create_criterion(loss_type, len(categories))

    model_dir = "/storage/models/quickdraw/seresnext50"
    model = load_ensemble_model(model_dir, 3, val_set_data_loader, criterion, model_type, image_size, len(categories))

    confusion = np.zeros((len(categories), len(categories)), dtype=np.float32)
    predictions = []
    for i in range(50):
        start_time = time.time()

        c, p = calculate_confusion(model, val_set_data_loader, len(categories), scale=False)
        confusion += c
        predictions.extend(p)
        train_data = train_data_provider.get_next()
        val_set.df = train_data.val_set_df

        end_time = time.time()
        duration_time = end_time - start_time
        print("[{:02d}/{:02d}] {}s".format(i + 1, 50, duration_time), flush=True)

    np.save("{}/confusion.npy".format(output_dir), confusion)
    np.save("{}/predictions.npy".format(output_dir), np.array(predictions))

    for c in range(confusion.shape[0]):
        category_count = confusion[c, :].sum()
        if category_count != 0:
            confusion[c, :] /= category_count

    confusion_bitmap = confusion > 0.01
    for i in range(confusion_bitmap.shape[0]):
        confusion_bitmap[i, i] = True

    confusion_sets, confusion_set_source_categories = pack_confusion_sets(confusion_bitmap, 68)

    for i, confusion_set in enumerate(confusion_sets):
        save_confusion_set("{}/confusion_set_{}.txt".format(output_dir, i), confusion_set, categories)

    category_confusion_set_mapping = np.full((len(categories),), -1, dtype=np.int32)
    for i, m in enumerate(confusion_set_source_categories):
        category_confusion_set_mapping[m] = i
    np.save("{}/category_confusion_set_mapping.npy".format(output_dir), category_confusion_set_mapping)


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
    argparser.add_argument("--batch_size", default=64, type=int)
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
