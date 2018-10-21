import argparse
import datetime
import glob
import os
import time
from math import ceil

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from dataset import TrainData, TrainDataset
from metrics import accuracy
from models import ResNet34, SimpleCnn
from utils import get_learning_rate

cudnn.enabled = True
cudnn.benchmark = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def create_model(type):
    if type == "resnet":
        model = ResNet34()
    elif type == "cnn":
        model = SimpleCnn()
    else:
        raise Exception("Unsupported model type: '{}".format(type))

    return nn.DataParallel(model)


def evaluate(model, data_loader, criterion):
    model.eval()

    loss_sum = 0.0
    accuracy_sum = 0.0
    step_count = 0

    with torch.no_grad():
        for batch in data_loader:
            images, categories = \
                batch[0].to(device, non_blocking=True), \
                batch[1].to(device, non_blocking=True)

            prediction_logits = model(images)
            loss = criterion(prediction_logits, categories)

            loss_sum += loss.item()
            accuracy_sum += accuracy(prediction_logits, categories).item()

            step_count += 1

    loss_avg = loss_sum / step_count
    accuracy_avg = accuracy_sum / step_count

    return loss_avg, accuracy_avg


def create_optimizer(type, model, lr):
    if type == "adam":
        return optim.Adam(model.parameters(), lr=lr)
    elif type == "sgd":
        return optim.SGD(model.parameters(), lr=lr, weight_decay=1e-4, momentum=0.9, nesterov=True)
    else:
        raise Exception("Unsupported optimizer type: '{}".format(type))


def main():
    args = argparser.parse_args()
    print("Arguments:")
    for arg in vars(args):
        print("  {}: {}".format(arg, getattr(args, arg)))
    print()

    input_dir = args.input_dir
    output_dir = args.output_dir
    image_size_target = args.image_size
    batch_size = args.batch_size
    batch_iterations = args.batch_iterations
    num_loaders = args.num_loaders
    num_workers = args.num_workers
    epochs_to_train = args.epochs
    max_epoch_iterations = args.max_epoch_iterations
    lr_min = args.lr_min
    lr_max = args.lr_max
    lr_min_decay = args.lr_min_decay
    lr_max_decay = args.lr_max_decay
    optimizer_type = args.optimizer
    loss_type = args.loss
    model_type = args.model
    patience = args.patience
    sgdr_cycle_epochs = args.sgdr_cycle_epochs
    sgdr_cycle_epochs_mult = args.sgdr_cycle_epochs_mult
    sgdr_cycle_end_prolongation = args.sgdr_cycle_end_prolongation
    sgdr_cycle_end_patience = args.sgdr_cycle_end_patience
    max_sgdr_cycles = args.max_sgdr_cycles

    load_start_time = time.time()

    train_data = TrainData(input_dir, num_loaders)

    train_set = TrainDataset(train_data.train_set_df, image_size_target)
    train_set_data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_set = TrainDataset(train_data.val_set_df, image_size_target)
    val_set_data_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)

    load_end_time = time.time()
    print()
    print("Load time: %s" % str(datetime.timedelta(seconds=load_end_time - load_start_time)))

    model = create_model(type=model_type).to(device)
    torch.save(model.state_dict(), "{}/model.pth".format(output_dir))

    epoch_iterations = ceil(len(train_set) / (batch_size * batch_iterations))
    if max_epoch_iterations > 0:
        epoch_iterations = min(epoch_iterations, max_epoch_iterations)

    print("train_set_samples: {}, val_set_samples: {}, samples_per_epoch: {}".format(
        len(train_set),
        len(val_set),
        min(len(train_set), epoch_iterations * batch_size * batch_iterations)),
        flush=True)
    print()

    global_val_accuracy_best_avg = float("-inf")
    sgdr_cycle_val_accuracy_best_avg = float("-inf")

    optimizer = create_optimizer(optimizer_type, model, lr_max)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=sgdr_cycle_epochs, eta_min=lr_min)

    optim_summary_writer = SummaryWriter(log_dir="{}/logs/optim".format(output_dir))
    train_summary_writer = SummaryWriter(log_dir="{}/logs/train".format(output_dir))
    val_summary_writer = SummaryWriter(log_dir="{}/logs/val".format(output_dir))

    current_sgdr_cycle_epochs = sgdr_cycle_epochs
    sgdr_next_cycle_end_epoch = current_sgdr_cycle_epochs + sgdr_cycle_end_prolongation
    sgdr_iterations = 0
    sgdr_cycle_count = 0
    batch_count = 0
    epoch_of_last_improval = 0

    ensemble_model_index = 0
    for model_file_path in glob.glob("{}/model-*.pth".format(output_dir)):
        model_file_name = os.path.basename(model_file_path)
        model_index = int(model_file_name.replace("model-", "").replace(".pth", ""))
        ensemble_model_index = max(ensemble_model_index, model_index + 1)

    print('{"chart": "best_val_accuracy", "axis": "epoch"}')
    print('{"chart": "val_accuracy", "axis": "epoch"}')
    print('{"chart": "val_loss", "axis": "epoch"}')
    print('{"chart": "sgdr_cycle", "axis": "epoch"}')
    print('{"chart": "accuracy", "axis": "epoch"}')
    print('{"chart": "loss", "axis": "epoch"}')
    print('{"chart": "lr_scaled", "axis": "epoch"}')

    train_start_time = time.time()

    if loss_type == "cce":
        criterion = nn.CrossEntropyLoss()
    else:
        raise Exception("Unsupported loss type: '{}".format(loss_type))

    for epoch in range(epochs_to_train):
        epoch_start_time = time.time()

        model.train()

        train_set_data_loader_iter = iter(train_set_data_loader)

        train_loss_sum = 0.0
        train_accuracy_sum = 0.0

        epoch_batch_iter_count = 0

        for _ in range(epoch_iterations):
            lr_scheduler.step(epoch=min(current_sgdr_cycle_epochs, sgdr_iterations / epoch_iterations))

            optimizer.zero_grad()

            for _ in range(batch_iterations):
                try:
                    batch = next(train_set_data_loader_iter)
                except StopIteration:
                    break

                images, categories = \
                    batch[0].to(device, non_blocking=True), \
                    batch[1].to(device, non_blocking=True)

                prediction_logits = model(images)
                loss = criterion(prediction_logits, categories)
                loss.backward()

                with torch.no_grad():
                    train_loss_sum += loss.item()
                    train_accuracy_sum += accuracy(prediction_logits, categories).item()

                epoch_batch_iter_count += 1

            optimizer.step()

            sgdr_iterations += 1
            batch_count += 1

            optim_summary_writer.add_scalar("lr", get_learning_rate(optimizer), batch_count + 1)

        train_loss_avg = train_loss_sum / epoch_batch_iter_count
        train_accuracy_avg = train_accuracy_sum / epoch_batch_iter_count

        val_loss_avg, val_accuracy_avg = evaluate(model, val_set_data_loader, criterion)

        model_improved_within_sgdr_cycle = val_accuracy_avg > sgdr_cycle_val_accuracy_best_avg
        if model_improved_within_sgdr_cycle:
            torch.save(model.state_dict(), "{}/model-{}.pth".format(output_dir, ensemble_model_index))
            sgdr_cycle_val_accuracy_best_avg = val_accuracy_avg

        model_improved = val_accuracy_avg > global_val_accuracy_best_avg
        ckpt_saved = False
        if model_improved:
            torch.save(model.state_dict(), "{}/model.pth".format(output_dir))
            global_val_accuracy_best_avg = val_accuracy_avg
            epoch_of_last_improval = epoch
            ckpt_saved = True

        sgdr_reset = False
        if (epoch + 1 >= sgdr_next_cycle_end_epoch) and (epoch - epoch_of_last_improval >= sgdr_cycle_end_patience):
            sgdr_iterations = 0
            current_sgdr_cycle_epochs = int(current_sgdr_cycle_epochs * sgdr_cycle_epochs_mult)
            sgdr_next_cycle_end_epoch = epoch + 1 + current_sgdr_cycle_epochs + sgdr_cycle_end_prolongation

            ensemble_model_index += 1
            sgdr_cycle_val_accuracy_best_avg = float("-inf")
            sgdr_cycle_count += 1
            sgdr_reset = True

            new_lr_min = lr_min * (lr_min_decay ** sgdr_cycle_count)
            new_lr_max = lr_max * (lr_max_decay ** sgdr_cycle_count)

            optimizer = create_optimizer(optimizer_type, model, new_lr_max)
            lr_scheduler = CosineAnnealingLR(optimizer, T_max=current_sgdr_cycle_epochs, eta_min=new_lr_min)

        optim_summary_writer.add_scalar("sgdr_cycle", sgdr_cycle_count, epoch + 1)

        train_summary_writer.add_scalar("loss", train_loss_avg, epoch + 1)
        train_summary_writer.add_scalar("accuracy", train_accuracy_avg, epoch + 1)
        val_summary_writer.add_scalar("loss", val_loss_avg, epoch + 1)
        val_summary_writer.add_scalar("accuracy", val_accuracy_avg, epoch + 1)

        epoch_end_time = time.time()
        epoch_duration_time = epoch_end_time - epoch_start_time

        print(
            "[%03d/%03d] %ds, lr: %.6f, loss: %.4f, val_loss: %.4f, acc: %.4f, val_acc: %.4f, ckpt: %d, rst: %d" % (
                epoch + 1,
                epochs_to_train,
                epoch_duration_time,
                get_learning_rate(optimizer),
                train_loss_avg,
                val_loss_avg,
                train_accuracy_avg,
                val_accuracy_avg,
                int(ckpt_saved),
                int(sgdr_reset)),
            flush=True)

        print('{"chart": "best_val_accuracy", "x": %d, "y": %.4f}' % (epoch + 1, global_val_accuracy_best_avg))
        print('{"chart": "val_loss", "x": %d, "y": %.4f}' % (epoch + 1, val_loss_avg))
        print('{"chart": "val_accuracy", "x": %d, "y": %.4f}' % (epoch + 1, val_accuracy_avg))
        print('{"chart": "sgdr_cycle", "x": %d, "y": %d}' % (epoch + 1, sgdr_cycle_count))
        print('{"chart": "loss", "x": %d, "y": %.4f}' % (epoch + 1, train_loss_avg))
        print('{"chart": "accuracy", "x": %d, "y": %.4f}' % (epoch + 1, train_accuracy_avg))
        print('{"chart": "lr_scaled", "x": %d, "y": %.4f}' % (epoch + 1, 1000 * get_learning_rate(optimizer)))

        if sgdr_reset and epoch - epoch_of_last_improval >= patience:
            print("early abort due to lack of improval")
            break

        if max_sgdr_cycles is not None and sgdr_cycle_count >= max_sgdr_cycles:
            print("early abort due to maximum number of sgdr cycles reached")
            break

    optim_summary_writer.close()
    train_summary_writer.close()
    val_summary_writer.close()

    train_end_time = time.time()
    print()
    print("Train time: %s" % str(datetime.timedelta(seconds=train_end_time - train_start_time)))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--input_dir", default="/storage/kaggle/quickdraw")
    argparser.add_argument("--output_dir", default="/artifacts")
    argparser.add_argument("--image_size", default=64, type=int)
    argparser.add_argument("--epochs", default=500, type=int)
    argparser.add_argument("--max_epoch_iterations", default=0, type=int)
    argparser.add_argument("--batch_size", default=64, type=int)
    argparser.add_argument("--batch_iterations", default=1, type=int)
    argparser.add_argument("--num_loaders", default=8, type=int)
    argparser.add_argument("--num_workers", default=16, type=int)
    argparser.add_argument("--lr_min", default=0.0001, type=float)
    argparser.add_argument("--lr_max", default=0.002, type=float)
    argparser.add_argument("--lr_min_decay", default=1.0, type=float)
    argparser.add_argument("--lr_max_decay", default=1.0, type=float)
    argparser.add_argument("--model", default="cnn")
    argparser.add_argument("--patience", default=30, type=int)
    argparser.add_argument("--optimizer", default="adam")
    argparser.add_argument("--loss", default="cce")
    argparser.add_argument("--sgdr_cycle_epochs", default=20, type=int)
    argparser.add_argument("--sgdr_cycle_epochs_mult", default=1.0, type=float)
    argparser.add_argument("--sgdr_cycle_end_prolongation", default=3, type=int)
    argparser.add_argument("--sgdr_cycle_end_patience", default=3, type=int)
    argparser.add_argument("--max_sgdr_cycles", default=None, type=int)

    main()
