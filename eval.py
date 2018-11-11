import argparse

import torch
import numpy as np
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from dataset import TrainDataset, TrainData
from train import load_ensemble_model, create_criterion, evaluate, predict
from utils import str2bool, read_categories, read_confusion_set

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

    train_data = TrainData(
        data_dir=input_dir,
        shard=0,
        test_size=test_size,
        train_on_unrecognized=train_on_unrecognized,
        confusion_set=None,
        num_category_shards=num_category_shards,
        category_shard=category_shard)

    val_set = TrainDataset(train_data.val_set_df, image_size, use_extended_stroke_channels, False, use_dummy_image)
    val_set_data_loader = \
        DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    categories = train_data.categories
    criterion = create_criterion(loss_type, len(categories))
    model_dir = "/storage/models/quickdraw/seresnext50"
    model = load_ensemble_model(model_dir, 3, val_set_data_loader, criterion, model_type, image_size, len(categories))

    predicted_words = predict(model, val_set_data_loader, categories, tta=True)
    prediction_mask = []
    for i, p in enumerate(predicted_words):
        cond1 = p.split(" ")[0] in ['angel', 'arm', 'bat', 'bathtub', 'bottlecap', 'hospital', 'police_car', 'spider',
                                    'sun', 'tent', 'triangle', 'windmill']
        cond2 = True  # train_data.val_set_df["category"][i] in [3, 8, 19, 20, 36, 147, 224, 272, 291, 302, 318, 333]
        prediction_mask.append(cond1 and cond2)
    print("matched {} of {}".format(sum(prediction_mask), len(prediction_mask)), flush=True)
    df = {
        "category": train_data.val_set_df["category"][prediction_mask],
        "drawing": train_data.val_set_df["drawing"][prediction_mask]
    }
    val_set = TrainDataset(df, image_size, use_extended_stroke_channels, False, use_dummy_image)
    val_set_data_loader = \
        DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    loss_avg, mapk_avg, accuracy_top1_avg, accuracy_top3_avg, accuracy_top5_avg, accuracy_top10_avg = \
        evaluate(model, val_set_data_loader, criterion, mapk_topk)
    print(
        "loss: {:.3f}, map@3: {:.3f}, acc@1: {:.3f}, acc@3: {:.3f}, acc@5: {:.3f}, acc@10: {:.3f}"
            .format(loss_avg, mapk_avg, accuracy_top1_avg, accuracy_top3_avg, accuracy_top5_avg, accuracy_top10_avg),
        flush=True)

    confusion_set_categories = read_confusion_set(
        "/storage/models/quickdraw/seresnext50_confusion/confusion_set_{}.txt".format(0))
    category_mapping = {}
    for csc in confusion_set_categories:
        category_mapping[categories.index(csc)] = confusion_set_categories.index(csc)
    df["category"] = np.array([category_mapping[c] if c in category_mapping else -1 for c in df["category"]])
    categories = confusion_set_categories
    criterion = create_criterion(loss_type, len(categories))
    model_dir = "/storage/models/quickdraw/seresnext50_cs_0"
    model = load_ensemble_model(model_dir, 3, val_set_data_loader, criterion, "seresnext50_cs", image_size, len(categories))
    loss_avg, mapk_avg, accuracy_top1_avg, accuracy_top3_avg, accuracy_top5_avg, accuracy_top10_avg = \
        evaluate(model, val_set_data_loader, criterion, mapk_topk)
    print(
        "loss: {:.3f}, map@3: {:.3f}, acc@1: {:.3f}, acc@3: {:.3f}, acc@5: {:.3f}, acc@10: {:.3f}"
            .format(loss_avg, mapk_avg, accuracy_top1_avg, accuracy_top3_avg, accuracy_top5_avg, accuracy_top10_avg),
        flush=True)


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
    argparser.add_argument("--batch_size", default=128, type=int)
    argparser.add_argument("--batch_iterations", default=1, type=int)
    argparser.add_argument("--test_size", default=0.1, type=float)
    argparser.add_argument("--train_on_unrecognized", default=True, type=str2bool)
    argparser.add_argument("--num_category_shards", default=1, type=int)
    argparser.add_argument("--category_shard", default=0, type=int)
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
