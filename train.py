import argparse
import datetime
import glob
import os
import shutil
import sys
import time
from math import ceil

import numpy as np
import psutil
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader

from dataset import TrainDataProvider, TrainDataset, TestData, TestDataset, StratifiedSampler
from metrics import accuracy, mapk, FocalLoss, CceCenterLoss, SoftCrossEntropyLoss
from metrics.smooth_topk_loss.svm import SmoothSVM
from models import ResNet, SimpleCnn, ResidualCnn, FcCnn, HcFcCnn, MobileNetV2, Drn, SeNet, NasNet, SeResNext50Cs, \
    StackNet
from models.ensemble import Ensemble
from swa_utils import moving_average
from utils import get_learning_rate, str2bool

cudnn.enabled = True
cudnn.benchmark = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class_weights = [0.0030502994322052983, 0.0024824986950179166, 0.002977493633314952, 0.0030123374143810142, 0.0027731585961971715, 0.0025069014123580632, 0.0025394718982390996, 0.0029114674846666745, 0.0024332506718945214, 0.003381818293745507, 0.002497043760670782, 0.002530821305942098, 0.006194950673417428, 0.0029696477472781362, 0.003040562486457045, 0.0027234277493176644, 0.0024907469341848253, 0.0023830973542284164, 0.0026916016167273004, 0.002376176880390815, 0.003507231764395526, 0.002513459768378581, 0.002711095625880311, 0.0033234770898820077, 0.0022906366049330225, 0.0024320234948477376, 0.00384486639351315, 0.0025890418038665695, 0.0025454267245644774, 0.002498411761313099, 0.0026871556146397715, 0.0029166980753578845, 0.0025781380340410462, 0.002572605678502266, 0.0024013239510216338, 0.0028704274653971783, 0.0030821658001086716, 0.0026209886423959614, 0.002402370069159876, 0.002877488762830312, 0.002425585844766248, 0.0026758494916841553, 0.0026721478428872987, 0.0023522972221197897, 0.002495876936593512, 0.0037702298878808803, 0.0033437154523256905, 0.002424579961941015, 0.00237386334989278, 0.0026490125379069456, 0.002512795885713927, 0.00258260415378508, 0.006477503159025307, 0.0024422633820086067, 0.0025905908634174277, 0.003474520454918957, 0.0026835947894384474, 0.0028475536899513856, 0.0028445159238191828, 0.0024899019926116297, 0.003676783373416758, 0.0026647646629500907, 0.00246509692214139, 0.0024785355166864996, 0.002321839090171742, 0.003012116120159463, 0.0024368517324088548, 0.004480322809525686, 0.003369747699842714, 0.00330382213947696, 0.002471977160665982, 0.00253912989807852, 0.0024249018444450895, 0.002419449959532328, 0.0036902219679618674, 0.002567194028902514, 0.0024922758760791788, 0.002642514534855942, 0.005460817152249559, 0.0024073190126600212, 0.002476141515562446, 0.0025535341401358535, 0.0026143498157494252, 0.0025736920319535178, 0.002697556443052678, 0.002482719989239468, 0.002629800175945, 0.0026472220664780313, 0.0034151532505737204, 0.005838928506254549, 0.003061082496091793, 0.002446568560500603, 0.0028315802706866894, 0.0024187458415546653, 0.0025018719982318996, 0.002482418224391898, 0.0027518137626457327, 0.0027621341204326203, 0.002725540103250653, 0.0031780867863228663, 0.00247239963145258, 0.002539914486682202, 0.0025543187287395347, 0.002713127509187281, 0.0023807033531043627, 0.002532571542058003, 0.004541802367803912, 0.003252341056481548, 0.0027391798743608095, 0.0024123081914731756, 0.002603747810771472, 0.003378901233552332, 0.002760987414011855, 0.0031297038224291714, 0.004439866202294825, 0.002698783620099462, 0.002506036353128363, 0.004823469676525586, 0.002444657383132661, 0.0033466727478318747, 0.0029133987796911213, 0.003057199788386395, 0.004085614388904356, 0.002536373779137383, 0.0031996529140958566, 0.002491048699032395, 0.003189191732713436, 0.0024511956214966738, 0.0025586037895750265, 0.0038223949711974504, 0.0039197845463364855, 0.0031243726434554376, 0.0024759001036843897, 0.0024231918436421938, 0.002608696754271617, 0.0023942425359319954, 0.00586978899133269, 0.005741639519398038, 0.004478391514501239, 0.002392110064342502, 0.0024247207855365477, 0.003217577746041504, 0.0024523222102609344, 0.002865458404240528, 0.004089939685052857, 0.002617508287820656, 0.003586696507588913, 0.0033686613463914626, 0.0025418658993631533, 0.003661393366190697, 0.0024197316067233934, 0.0027351362254033735, 0.002724333043860374, 0.002474391279446541, 0.002740930110476714, 0.0024771473983876786, 0.0043076730814027376, 0.0024167541935607045, 0.00350992753036715, 0.0032382385792717847, 0.003777411891253042, 0.005382277821255386, 0.00252253283146218, 0.0030158781219258336, 0.005260787293623775, 0.0025261942449460273, 0.0023498227503697174, 0.002431802200626186, 0.003236991284568496, 0.0030490320198455047, 0.002887869473586714, 0.002433210436581512, 0.002567475676093579, 0.002819992500540008, 0.0025921399229682862, 0.0026163615813998907, 0.0024267727865000224, 0.006420268426269564, 0.002896318889318669, 0.0027628382384102834, 0.003627293938415307, 0.002425585844766248, 0.002626018056522125, 0.0025676768526586256, 0.0024475342080128265, 0.002475055162111194, 0.003418613487492521, 0.0025859235671083476, 0.003597560042101427, 0.003619649228943538, 0.0026984818552518923, 0.0030763517973788263, 0.002860066872297281, 0.003190519498042743, 0.0024257870213312944, 0.003974705748594193, 0.0026453310067665937, 0.0032082431534233443, 0.003020706359486951, 0.0026615056025963363, 0.004162544307378157, 0.003412598308197629, 0.0037620420016834855, 0.0024834442248736354, 0.0024535292696512136, 0.0022856273084633635, 0.0029022535979875423, 0.00255753755378028, 0.0025613599085161642, 0.0037324288113086334, 0.0030229796546719766, 0.002547418372558438, 0.002351834516020183, 0.0032521398799165015, 0.002454374211224409, 0.005105680161972885, 0.0023511505156990245, 0.002630182411418589, 0.0024618177441311315, 0.003757374705374406, 0.00238903206289729, 0.0025161354166936996, 0.002622758996168371, 0.003471281512221708, 0.002615778169361256, 0.0024467093840961356, 0.0026844799663246523, 0.0025490479027353154, 0.0025289101285741558, 0.00662281299195843, 0.003409178306591838, 0.002480909400154049, 0.0031240306432948587, 0.0024058303060786766, 0.0027305292820638074, 0.002709445978046929, 0.0025518241393329578, 0.0031109742842233374, 0.002406956894842937, 0.0037918563686233846, 0.0026811002000318705, 0.0028882919443733116, 0.002409531954875533, 0.0027461808188244292, 0.002650139126671206, 0.002439386557128441, 0.0023760360567952827, 0.0024551789174845954, 0.0024823176361093748, 0.0033332743885997745, 0.0023399449810259315, 0.002411624191152017, 0.002654243128598156, 0.002535830602411757, 0.0025372589560235873, 0.00241876595921117, 0.0025141035333867295, 0.0023576686364065328, 0.004192720792135139, 0.002589806274813746, 0.0025383251918183343, 0.003695794558813657, 0.002407902424698656, 0.002502354821988011, 0.0026908773810931327, 0.0024598462137936754, 0.003108841812633844, 0.0023474287492456633, 0.00684058662362132, 0.0025217281252019937, 0.004138503707855094, 0.0037937876636478314, 0.004213582801930466, 0.002515270357464, 0.003424025137092273, 0.0025176241232750442, 0.0023827553540678374, 0.003156118305419783, 0.002594795453626901, 0.00276857177051411, 0.0024551990351411, 0.002463286333055971, 0.003093974864476904, 0.002518569653130763, 0.0024103768964487287, 0.002344411100769965, 0.0024604095081758055, 0.002480104693893863, 0.002395670889543826, 0.0025018719982318996, 0.0025437167237615816, 0.0026913602048492446, 0.0030596541424799625, 0.002417820429355451, 0.0024011831274261012, 0.0024906061105892927, 0.0025754825033824317, 0.002550999315416267, 0.0036124873432278807, 0.002572746502097799, 0.002477227869013697, 0.004650216418707497, 0.002646015007087752, 0.0027118802144839925, 0.003883009470245976, 0.0024419414995045323, 0.0024355843200490612, 0.0024630851564909247, 0.003091118157253243, 0.002613042168076623, 0.002515994593098167, 0.0025112468261630685, 0.0026361573554004713, 0.00288227676507842, 0.002347267807993626, 0.0025211648308198636, 0.002574013914457592, 0.00291144736701017, 0.0024778917516783505, 0.0037169180981435446, 0.002642534652512447, 0.0034108883073947333, 0.002519394477047454, 0.0024962792897236055, 0.002505613882341765, 0.003337700273030799, 0.0025443806064262353, 0.0043707620522013355, 0.002431238906244056, 0.002674421138072325, 0.003729089280328861, 0.0023437472181053113, 0.0027492588202696414, 0.002427074551347592, 0.0025423286054627606, 0.0026616061908788595, 0.003272036242199605, 0.005641835825478445, 0.0029091740718251435, 0.0024155672518269295]
class_weights = torch.tensor(class_weights).to(device)


def create_model(type, input_size, num_classes):
    if type == "resnet":
        model = ResNet(num_classes=num_classes)
    elif type in ["seresnext50", "seresnext101", "seresnet50", "seresnet101", "seresnet152", "senet154"]:
        model = SeNet(type=type, num_classes=num_classes)
    elif type == "nasnet":
        model = NasNet(num_classes=num_classes)
    elif type == "cnn":
        model = SimpleCnn(num_classes=num_classes)
    elif type == "residual_cnn":
        model = ResidualCnn(num_classes=num_classes)
    elif type == "fc_cnn":
        model = FcCnn(num_classes=num_classes)
    elif type == "hc_fc_cnn":
        model = HcFcCnn(num_classes=num_classes)
    elif type == "mobilenetv2":
        model = MobileNetV2(input_size=input_size, n_class=num_classes)
    elif type in ["drn_d_38", "drn_d_54", "drn_d_105"]:
        model = Drn(type=type, num_classes=num_classes)
    elif type == "seresnext50_cs":
        model = SeResNext50Cs(num_classes=num_classes)
    elif type == "stack":
        model = StackNet(num_classes=num_classes)
    else:
        raise Exception("Unsupported model type: '{}".format(type))

    return nn.DataParallel(model)


def zero_item_tensor():
    return torch.tensor(0.0).float().to(device, non_blocking=True)


def evaluate(model, data_loader, criterion, mapk_topk):
    model.eval()

    loss_sum_t = zero_item_tensor()
    mapk_sum_t = zero_item_tensor()
    accuracy_top1_sum_t = zero_item_tensor()
    accuracy_top3_sum_t = zero_item_tensor()
    accuracy_top5_sum_t = zero_item_tensor()
    accuracy_top10_sum_t = zero_item_tensor()
    step_count = 0

    with torch.no_grad():
        for batch in data_loader:
            images, categories = \
                batch[0].to(device, non_blocking=True), \
                batch[1].to(device, non_blocking=True)

            prediction_logits = model(images)
            if prediction_logits.size(1) == len(class_weights):
                print("foooo.......", flush=True)
                criterion.weight = class_weights
            loss = criterion(prediction_logits, categories)

            num_categories = prediction_logits.size(1)

            loss_sum_t += loss
            mapk_sum_t += mapk(prediction_logits, categories, topk=min(mapk_topk, num_categories))
            accuracy_top1_sum_t += accuracy(prediction_logits, categories, topk=min(1, num_categories))
            accuracy_top3_sum_t += accuracy(prediction_logits, categories, topk=min(3, num_categories))
            accuracy_top5_sum_t += accuracy(prediction_logits, categories, topk=min(5, num_categories))
            accuracy_top10_sum_t += accuracy(prediction_logits, categories, topk=min(10, num_categories))

            step_count += 1

    loss_avg = loss_sum_t.item() / step_count
    mapk_avg = mapk_sum_t.item() / step_count
    accuracy_top1_avg = accuracy_top1_sum_t.item() / step_count
    accuracy_top3_avg = accuracy_top3_sum_t.item() / step_count
    accuracy_top5_avg = accuracy_top5_sum_t.item() / step_count
    accuracy_top10_avg = accuracy_top10_sum_t.item() / step_count

    return loss_avg, mapk_avg, accuracy_top1_avg, accuracy_top3_avg, accuracy_top5_avg, accuracy_top10_avg


def create_criterion(loss_type, num_classes):
    if loss_type == "cce":
        criterion = nn.CrossEntropyLoss()
    elif loss_type == "scce":
        criterion = SoftCrossEntropyLoss()
    elif loss_type == "focal":
        criterion = FocalLoss()
    elif loss_type == "topk_svm":
        criterion = SmoothSVM(n_classes=num_classes, k=3, tau=1., alpha=1.)
    elif loss_type == "center":
        criterion = CceCenterLoss(num_classes=num_classes, alpha=0.5)
    else:
        raise Exception("Unsupported loss type: '{}".format(loss_type))
    return criterion


def create_optimizer(type, model, lr):
    if type == "adam":
        return optim.Adam(model.parameters(), lr=lr)
    elif type == "sgd":
        return optim.SGD(model.parameters(), lr=lr, weight_decay=1e-4, momentum=0.9, nesterov=True)
    else:
        raise Exception("Unsupported optimizer type: '{}".format(type))


def predict(model, data_loader, categories, tta=False):
    categories = np.array([c.replace(" ", "_") for c in categories])

    model.eval()

    all_predictions = []
    predicted_words = []
    with torch.no_grad():
        for batch in data_loader:
            images = batch[0].to(device, non_blocking=True)

            if tta:
                predictions1 = F.softmax(model(images), dim=1)
                predictions2 = F.softmax(model(images.flip(3)), dim=1)
                predictions = 0.5 * (predictions1 + predictions2)
            else:
                predictions = F.softmax(model(images), dim=1)

            _, prediction_categories = predictions.topk(3, dim=1, sorted=True)

            all_predictions.extend(predictions.cpu().data.numpy())
            predicted_words.extend([" ".join(categories[pc.cpu().data.numpy()]) for pc in prediction_categories])

    return all_predictions, predicted_words


def calculate_confusion(model, data_loader, num_categories, scale=True):
    confusion = np.zeros((num_categories, num_categories), dtype=np.float32)

    model.eval()

    all_predictions = []
    with torch.no_grad():
        for batch in data_loader:
            images, categories = \
                batch[0].to(device, non_blocking=True), \
                batch[1].to(device, non_blocking=True)

            predictions = F.softmax(model(images), dim=1)
            _, prediction_categories = predictions.topk(3, dim=1, sorted=True)

            for bpc, bc in zip(prediction_categories[:, 0], categories):
                confusion[bpc, bc] += 1

            all_predictions.extend(predictions.cpu().data.numpy())

    if scale:
        for c in range(confusion.shape[0]):
            category_count = confusion[c, :].sum()
            if category_count != 0:
                confusion[c, :] /= category_count

    return confusion, all_predictions


def find_sorted_model_files(base_dir):
    return sorted(glob.glob("{}/model-*.pth".format(base_dir)), key=lambda e: int(os.path.basename(e)[6:-4]))


def load_ensemble_model(base_dir, ensemble_model_count, data_loader, criterion, model_type, input_size, num_classes):
    ensemble_model_candidates = find_sorted_model_files(base_dir)[-(2 * ensemble_model_count):]
    if os.path.isfile("{}/swa_model.pth".format(base_dir)):
        ensemble_model_candidates.append("{}/swa_model.pth".format(base_dir))

    score_to_model = {}
    for model_file_path in ensemble_model_candidates:
        model_file_name = os.path.basename(model_file_path)
        model = create_model(type=model_type, input_size=input_size, num_classes=num_classes).to(device)
        model.load_state_dict(torch.load(model_file_path, map_location=device))

        val_loss_avg, val_mapk_avg, _, _, _, _ = evaluate(model, data_loader, criterion, 3)
        print("ensemble '%s': val_loss=%.4f, val_mapk=%.4f" % (model_file_name, val_loss_avg, val_mapk_avg))

        if len(score_to_model) < ensemble_model_count or min(score_to_model.keys()) < val_mapk_avg:
            if len(score_to_model) >= ensemble_model_count:
                del score_to_model[min(score_to_model.keys())]
            score_to_model[val_mapk_avg] = model

    ensemble = Ensemble(list(score_to_model.values()))

    val_loss_avg, val_mapk_avg, _, _, _, _ = evaluate(ensemble, data_loader, criterion, 3)
    print("ensemble: val_loss=%.4f, val_mapk=%.4f" % (val_loss_avg, val_mapk_avg))

    return ensemble


def check_model_improved(old_score, new_score, threshold=1e-3):
    return new_score - old_score > threshold


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
    fold = args.fold
    train_on_unrecognized = args.train_on_unrecognized
    confusion_set = args.confusion_set
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
    print("use_extended_stroke_channels: {}".format(use_extended_stroke_channels), flush=True)

    progressive_image_sizes = list(range(progressive_image_size_min, image_size + 1, progressive_image_size_step))

    train_data_provider = TrainDataProvider(
        input_dir,
        50,
        num_shard_preload=num_shard_preload,
        num_workers=num_shard_loaders,
        test_size=test_size,
        fold=fold,
        train_on_unrecognized=train_on_unrecognized,
        confusion_set=confusion_set,
        num_category_shards=num_category_shards,
        category_shard=category_shard)

    train_data = train_data_provider.get_next()

    train_set = TrainDataset(train_data.train_set_df, image_size, use_extended_stroke_channels, augment,
                             use_dummy_image)
    stratified_sampler = StratifiedSampler(train_data.train_set_df["category"], batch_size * batch_iterations)
    train_set_data_loader = \
        DataLoader(train_set, batch_size=batch_size, shuffle=False, sampler=stratified_sampler, num_workers=num_workers,
                   pin_memory=pin_memory)

    val_set = TrainDataset(train_data.val_set_df, image_size, use_extended_stroke_channels, False, use_dummy_image)
    val_set_data_loader = \
        DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    if base_model_dir:
        for model_file_path in glob.glob("{}/model*.pth".format(base_model_dir)):
            shutil.copyfile(model_file_path, "{}/{}".format(output_dir, os.path.basename(model_file_path)))
        model = create_model(type=model_type, input_size=image_size, num_classes=len(train_data.categories)).to(device)
        model.load_state_dict(torch.load("{}/model.pth".format(output_dir), map_location=device))
    else:
        model = create_model(type=model_type, input_size=image_size, num_classes=len(train_data.categories)).to(device)

    torch.save(model.state_dict(), "{}/model.pth".format(output_dir))

    ensemble_model_index = 0
    for model_file_path in glob.glob("{}/model-*.pth".format(output_dir)):
        model_file_name = os.path.basename(model_file_path)
        model_index = int(model_file_name.replace("model-", "").replace(".pth", ""))
        ensemble_model_index = max(ensemble_model_index, model_index + 1)

    if confusion_set is not None:
        shutil.copyfile(
            "/storage/models/quickdraw/seresnext50_confusion/confusion_set_{}.txt".format(confusion_set),
            "{}/confusion_set.txt".format(output_dir))

    epoch_iterations = ceil(len(train_set) / batch_size)

    print("train_set_samples: {}, val_set_samples: {}".format(len(train_set), len(val_set)), flush=True)
    print()

    global_val_mapk_best_avg = float("-inf")
    sgdr_cycle_val_mapk_best_avg = float("-inf")

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

    lr_scheduler_plateau = ReduceLROnPlateau(optimizer, mode="max", min_lr=lr_min, patience=lr_patience, factor=0.8, threshold=1e-3)

    print('{"chart": "best_val_mapk", "axis": "epoch"}')
    print('{"chart": "val_mapk", "axis": "epoch"}')
    print('{"chart": "val_loss", "axis": "epoch"}')
    print('{"chart": "val_accuracy@1", "axis": "epoch"}')
    print('{"chart": "val_accuracy@3", "axis": "epoch"}')
    print('{"chart": "val_accuracy@5", "axis": "epoch"}')
    print('{"chart": "val_accuracy@10", "axis": "epoch"}')
    print('{"chart": "sgdr_cycle", "axis": "epoch"}')
    print('{"chart": "mapk", "axis": "epoch"}')
    print('{"chart": "loss", "axis": "epoch"}')
    print('{"chart": "lr_scaled", "axis": "epoch"}')
    print('{"chart": "mem_used", "axis": "epoch"}')
    print('{"chart": "epoch_time", "axis": "epoch"}')

    train_start_time = time.time()

    criterion = create_criterion(loss_type, len(train_data.categories))

    if loss_type == "center":
        optimizer_centloss = torch.optim.SGD(criterion.center.parameters(), lr=0.01)

    for epoch in range(epochs_to_train):
        epoch_start_time = time.time()

        print("memory used: {:.2f} GB".format(psutil.virtual_memory().used / 2 ** 30), flush=True)

        if use_progressive_image_sizes:
            next_image_size = \
                progressive_image_sizes[min(epoch // progressive_image_epoch_step, len(progressive_image_sizes) - 1)]

            if train_set.image_size != next_image_size:
                print("changing image size to {}".format(next_image_size), flush=True)
                train_set.image_size = next_image_size
                val_set.image_size = next_image_size

        model.train()

        train_loss_sum_t = zero_item_tensor()
        train_mapk_sum_t = zero_item_tensor()

        epoch_batch_iter_count = 0

        for b, batch in enumerate(train_set_data_loader):
            images, categories = \
                batch[0].to(device, non_blocking=True), \
                batch[1].to(device, non_blocking=True)

            if lr_scheduler_type == "cosine_annealing":
                lr_scheduler.step(epoch=min(current_sgdr_cycle_epochs, sgdr_iterations / epoch_iterations))

            if b % batch_iterations == 0:
                optimizer.zero_grad()

            prediction_logits = model(images)
            if prediction_logits.size(1) == len(class_weights):
                criterion.weight = class_weights
            loss = criterion(prediction_logits, categories)
            loss.backward()

            with torch.no_grad():
                train_loss_sum_t += loss
                if eval_train_mapk:
                    train_mapk_sum_t += mapk(prediction_logits, categories, topk=min(mapk_topk, len(train_data.categories)))

            if (b + 1) % batch_iterations == 0 or (b + 1) == len(train_set_data_loader):
                optimizer.step()
                if loss_type == "center":
                    for param in criterion.center.parameters():
                        param.grad.data *= (1. / 0.5)
                    optimizer_centloss.step()

            sgdr_iterations += 1
            batch_count += 1
            epoch_batch_iter_count += 1

            optim_summary_writer.add_scalar("lr", get_learning_rate(optimizer), batch_count + 1)

        # TODO: recalculate epoch_iterations and maybe other values?
        train_data = train_data_provider.get_next()
        train_set.df = train_data.train_set_df
        val_set.df = train_data.val_set_df
        epoch_iterations = ceil(len(train_set) / batch_size)
        stratified_sampler.class_vector = train_data.train_set_df["category"]

        train_loss_avg = train_loss_sum_t.item() / epoch_batch_iter_count
        train_mapk_avg = train_mapk_sum_t.item() / epoch_batch_iter_count

        val_loss_avg, val_mapk_avg, val_accuracy_top1_avg, val_accuracy_top3_avg, val_accuracy_top5_avg, val_accuracy_top10_avg = \
            evaluate(model, val_set_data_loader, criterion, mapk_topk)

        if lr_scheduler_type == "reduce_on_plateau":
            lr_scheduler_plateau.step(val_mapk_avg)

        model_improved_within_sgdr_cycle = check_model_improved(sgdr_cycle_val_mapk_best_avg, val_mapk_avg)
        if model_improved_within_sgdr_cycle:
            torch.save(model.state_dict(), "{}/model-{}.pth".format(output_dir, ensemble_model_index))
            sgdr_cycle_val_mapk_best_avg = val_mapk_avg

        model_improved = check_model_improved(global_val_mapk_best_avg, val_mapk_avg)
        ckpt_saved = False
        if model_improved:
            torch.save(model.state_dict(), "{}/model.pth".format(output_dir))
            global_val_mapk_best_avg = val_mapk_avg
            epoch_of_last_improval = epoch
            ckpt_saved = True

        sgdr_reset = False
        if (epoch + 1 >= sgdr_next_cycle_end_epoch) and (epoch - epoch_of_last_improval >= sgdr_cycle_end_patience):
            sgdr_iterations = 0
            current_sgdr_cycle_epochs = int(current_sgdr_cycle_epochs * sgdr_cycle_epochs_mult)
            sgdr_next_cycle_end_epoch = epoch + 1 + current_sgdr_cycle_epochs + sgdr_cycle_end_prolongation

            ensemble_model_index += 1
            sgdr_cycle_val_mapk_best_avg = float("-inf")
            sgdr_cycle_count += 1
            sgdr_reset = True

            new_lr_min = lr_min * (lr_min_decay ** sgdr_cycle_count)
            new_lr_max = lr_max * (lr_max_decay ** sgdr_cycle_count)
            new_lr_max = max(new_lr_max, new_lr_min)

            optimizer = create_optimizer(optimizer_type, model, new_lr_max)
            lr_scheduler = CosineAnnealingLR(optimizer, T_max=current_sgdr_cycle_epochs, eta_min=new_lr_min)
            if loss2_type is not None and sgdr_cycle_count >= loss2_start_sgdr_cycle:
                print("switching to loss type '{}'".format(loss2_type), flush=True)
                criterion = create_criterion(loss2_type, len(train_data.categories))

        optim_summary_writer.add_scalar("sgdr_cycle", sgdr_cycle_count, epoch + 1)

        train_summary_writer.add_scalar("loss", train_loss_avg, epoch + 1)
        train_summary_writer.add_scalar("mapk", train_mapk_avg, epoch + 1)
        val_summary_writer.add_scalar("loss", val_loss_avg, epoch + 1)
        val_summary_writer.add_scalar("mapk", val_mapk_avg, epoch + 1)

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
                train_mapk_avg,
                val_mapk_avg,
                int(ckpt_saved),
                int(sgdr_reset)))

        print('{"chart": "best_val_mapk", "x": %d, "y": %.4f}' % (epoch + 1, global_val_mapk_best_avg))
        print('{"chart": "val_loss", "x": %d, "y": %.4f}' % (epoch + 1, val_loss_avg))
        print('{"chart": "val_mapk", "x": %d, "y": %.4f}' % (epoch + 1, val_mapk_avg))
        print('{"chart": "val_accuracy@1", "x": %d, "y": %.4f}' % (epoch + 1, val_accuracy_top1_avg))
        print('{"chart": "val_accuracy@3", "x": %d, "y": %.4f}' % (epoch + 1, val_accuracy_top3_avg))
        print('{"chart": "val_accuracy@5", "x": %d, "y": %.4f}' % (epoch + 1, val_accuracy_top5_avg))
        print('{"chart": "val_accuracy@10", "x": %d, "y": %.4f}' % (epoch + 1, val_accuracy_top10_avg))
        print('{"chart": "sgdr_cycle", "x": %d, "y": %d}' % (epoch + 1, sgdr_cycle_count))
        print('{"chart": "loss", "x": %d, "y": %.4f}' % (epoch + 1, train_loss_avg))
        print('{"chart": "mapk", "x": %d, "y": %.4f}' % (epoch + 1, train_mapk_avg))
        print('{"chart": "lr_scaled", "x": %d, "y": %.4f}' % (epoch + 1, 1000 * get_learning_rate(optimizer)))
        print('{"chart": "mem_used", "x": %d, "y": %.2f}' % (epoch + 1, psutil.virtual_memory().used / 2 ** 30))
        print('{"chart": "epoch_time", "x": %d, "y": %d}' % (epoch + 1, epoch_duration_time))

        sys.stdout.flush()

        if (sgdr_reset or lr_scheduler_type == "reduce_on_plateau") and epoch - epoch_of_last_improval >= patience:
            print("early abort due to lack of improval", flush=True)
            break

        if max_sgdr_cycles is not None and sgdr_cycle_count >= max_sgdr_cycles:
            print("early abort due to maximum number of sgdr cycles reached", flush=True)
            break

    optim_summary_writer.close()
    train_summary_writer.close()
    val_summary_writer.close()

    train_end_time = time.time()
    print()
    print("Train time: %s" % str(datetime.timedelta(seconds=train_end_time - train_start_time)), flush=True)

    if False:
        swa_model = create_model(type=model_type, input_size=image_size, num_classes=len(train_data.categories)).to(
            device)
        swa_update_count = 0
        for f in find_sorted_model_files(output_dir):
            print("merging model '{}' into swa model".format(f), flush=True)
            m = create_model(type=model_type, input_size=image_size, num_classes=len(train_data.categories)).to(device)
            m.load_state_dict(torch.load(f, map_location=device))
            swa_update_count += 1
            moving_average(swa_model, m, 1.0 / swa_update_count)
            # bn_update(train_set_data_loader, swa_model)
        torch.save(swa_model.state_dict(), "{}/swa_model.pth".format(output_dir))

    test_data = TestData(input_dir)
    test_set = TestDataset(test_data.df, image_size, use_extended_stroke_channels)
    test_set_data_loader = \
        DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    model.load_state_dict(torch.load("{}/model.pth".format(output_dir), map_location=device))
    model = Ensemble([model])

    categories = train_data.categories

    submission_df = test_data.df.copy()
    predictions, predicted_words = predict(model, test_set_data_loader, categories, tta=False)
    submission_df["word"] = predicted_words
    np.save("{}/submission_predictions.npy".format(output_dir), np.array(predictions))
    submission_df.to_csv("{}/submission.csv".format(output_dir), columns=["word"])

    submission_df = test_data.df.copy()
    predictions, predicted_words = predict(model, test_set_data_loader, categories, tta=True)
    submission_df["word"] = predicted_words
    np.save("{}/submission_predictions_tta.npy".format(output_dir), np.array(predictions))
    submission_df.to_csv("{}/submission_tta.csv".format(output_dir), columns=["word"])

    val_set_data_loader = \
        DataLoader(val_set, batch_size=64, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    model = load_ensemble_model(output_dir, 3, val_set_data_loader, criterion, model_type, image_size, len(categories))
    submission_df = test_data.df.copy()
    predictions, predicted_words = predict(model, test_set_data_loader, categories, tta=True)
    submission_df["word"] = predicted_words
    np.save("{}/submission_predictions_ensemble_tta.npy".format(output_dir), np.array(predictions))
    submission_df.to_csv("{}/submission_ensemble_tta.csv".format(output_dir), columns=["word"])

    confusion, _ = calculate_confusion(model, val_set_data_loader, len(categories))
    precisions = np.array([confusion[c, c] for c in range(confusion.shape[0])])
    percentiles = np.percentile(precisions, q=np.linspace(0, 100, 10))

    print()
    print("Category precision percentiles:")
    print(percentiles)

    print()
    print("Categories sorted by precision:")
    print(np.array(categories)[np.argsort(precisions)])


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--input_dir", default="/storage/kaggle/quickdraw")
    argparser.add_argument("--output_dir", default="/artifacts")
    argparser.add_argument("--base_model_dir", default=None)
    argparser.add_argument("--image_size", default=64, type=int)
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
    argparser.add_argument("--fold", default=None, type=int)
    argparser.add_argument("--train_on_unrecognized", default=True, type=str2bool)
    argparser.add_argument("--confusion_set", default=None, type=int)
    argparser.add_argument("--num_category_shards", default=1, type=int)
    argparser.add_argument("--category_shard", default=0, type=int)
    argparser.add_argument("--eval_train_mapk", default=True, type=str2bool)
    argparser.add_argument("--mapk_topk", default=3, type=int)
    argparser.add_argument("--num_shard_preload", default=1, type=int)
    argparser.add_argument("--num_shard_loaders", default=1, type=int)
    argparser.add_argument("--num_workers", default=8, type=int)
    argparser.add_argument("--pin_memory", default=True, type=str2bool)
    argparser.add_argument("--lr_scheduler", default="cosine_annealing")
    argparser.add_argument("--lr_patience", default=2, type=int)
    argparser.add_argument("--lr_min", default=0.01, type=float)
    argparser.add_argument("--lr_max", default=0.1, type=float)
    argparser.add_argument("--lr_min_decay", default=1.0, type=float)
    argparser.add_argument("--lr_max_decay", default=1.0, type=float)
    argparser.add_argument("--model", default="cnn")
    argparser.add_argument("--patience", default=5, type=int)
    argparser.add_argument("--optimizer", default="sgd")
    argparser.add_argument("--loss", default="cce")
    argparser.add_argument("--loss2", default=None)
    argparser.add_argument("--loss2_start_sgdr_cycle", default=None, type=int)
    argparser.add_argument("--sgdr_cycle_epochs", default=5, type=int)
    argparser.add_argument("--sgdr_cycle_epochs_mult", default=1.0, type=float)
    argparser.add_argument("--sgdr_cycle_end_prolongation", default=0, type=int)
    argparser.add_argument("--sgdr_cycle_end_patience", default=2, type=int)
    argparser.add_argument("--max_sgdr_cycles", default=None, type=int)

    main()
