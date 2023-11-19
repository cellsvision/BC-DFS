import time
import os, sys
from tqdm import tqdm

from monai.data.image_reader import TIOReader
from monai.data import DataLoader, decollate_batch
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    CropForegroundd,
    EnsureTyped,
    EnsureType,
    LoadImaged,
    NormalizeIntensityd,
    OneOf,
    Orientationd,
    RandAdjustContrastd,
    RandCropByPosNegLabeld,
    RandCropByLabelClassesd,
    RandHistogramShiftd,
    RandScaleIntensityd,
    RandSpatialCropd,
    Spacingd,
    SpatialPadd,
)
from monai.networks.nets import BasicUNet,HighResNet,UNet,FlexibleUNet,DiNTS
from monai.losses import DiceLoss,DiceFocalLoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from transforms import Resample_TIO,select_foreground,CombineSeq,LoadPickle
from dataloader import SegDataset

from configs import configs as config


torch.backends.cudnn.benchmark = False



roi_size = config['roi_size']

max_epochs = 600
val_interval = 1

train_csv_path = 'data/sample_seg_data_train.csv'
test_csv_path = 'data/sample_seg_data_val.csv'
pretrained_weight_path = None

input_sequence_names = ['T1C','T2WI'] 
workdir = 'ckpts'
log_id = config['log_id']
ckpt_dir = workdir+'/'+log_id+'_ckpt'
os.makedirs(ckpt_dir,exist_ok=True)


transform_train = config['transform_train']

transform_val = config['transform_val']

device = torch.device("cuda")

model = config['network'].to(device)


if pretrained_weight_path is not None:
    stdict = torch.load(pretrained_weight_path)#['state_dict']
    new_stdict = {}
    for i,(k,v) in enumerate(stdict.items()):
        if 'module.' in k:
            new_stdict[k[7:]] = stdict[k]
        else:
            new_stdict[k] = stdict[k]
    model.load_state_dict(
        new_stdict, # strict=False
    )
sys.stdout.flush()
model = nn.DataParallel(model)




train_ds = SegDataset(
    csv_path=train_csv_path,
    input_sequence=input_sequence_names,
    transform=transform_train,
    phase='train',
    cache_rate=0,
    num_workers=16,
)
train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=3)

val_ds = SegDataset(
    csv_path=test_csv_path,
    input_sequence=input_sequence_names,
    transform=transform_val,
    phase='val',
    cache_rate=0,
    num_workers=4,
)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4)


loss_function = config['loss']
optimizer = config['optimizer'](model.parameters(), config['lr']) 
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

dice_metric = DiceMetric(include_background=True, reduction="mean")
dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")
post_trans = Compose(
    [EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold=0.5)]
)

def inference(input):
    def _compute(input):
        return sliding_window_inference(
            inputs=input,
            roi_size=roi_size,
            sw_batch_size=config['batch_size'],
            predictor=model,
            overlap=0.1,
        )

    return _compute(input)

best_metric = -1
best_metric_epoch = -1
best_metrics_epochs_and_time = [[], [], []]



total_start = time.time()
writer = SummaryWriter(log_dir=workdir+'/'+log_id,flush_secs=20)
for epoch in range(max_epochs):
    epoch_start = time.time()
    print("-" * 10 + f"\nepoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        step_start = time.time()
        step += 1
        inputs, labels = (
            batch_data["image"].to(device),
            batch_data["label"].to(device),
        )
        labels = labels[:,0:1]
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)

        loss = loss.requires_grad_()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        print(
            f"{step}/{len(train_ds) // train_loader.batch_size}"
            f", train_loss: {loss.item():.4f}"
            f", step time: {(time.time() - step_start):.4f}"
        )
        writer.add_scalar('step_Loss/tain', loss, epoch*len(train_ds)//train_loader.batch_size + step)
    lr_scheduler.step()
    epoch_loss /= step
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    writer.add_scalar('epoch_Loss/tain', epoch_loss, epoch)

    torch.save(
        model.state_dict(),
        os.path.join(ckpt_dir, f"epoch_{epoch}.pth"),
    )

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            for val_data in tqdm(val_loader):
                val_inputs, val_labels = (
                    val_data["image"].to(device),
                    val_data["label"].to(device),
                )
                val_labels = val_labels[:,0:1]
                val_outputs = inference(val_inputs)
                val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                dice_metric(y_pred=val_outputs, y=val_labels)
                dice_metric_batch(y_pred=val_outputs, y=val_labels)

            metric = dice_metric.aggregate().item()
            dice_metric.reset()
            dice_metric_batch.reset()

            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                best_metrics_epochs_and_time[0].append(best_metric)
                best_metrics_epochs_and_time[1].append(best_metric_epoch)
                best_metrics_epochs_and_time[2].append(time.time() - total_start)
                torch.save(
                    model.state_dict(),
                    os.path.join(ckpt_dir, f"best_{metric}.pth"),
                )
                print("saved new best metric model")
            print(
                f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                f"\nbest mean dice: {best_metric:.4f}"
                f" at epoch: {best_metric_epoch}")
            writer.add_scalar('epoch_metrics/mean_dice', metric, epoch)

    print(f"time consuming of epoch {epoch + 1} is: {(time.time() - epoch_start):.4f}")
total_time = time.time() - total_start