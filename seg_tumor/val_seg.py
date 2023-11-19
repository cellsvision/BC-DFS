import time
import argparse
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from tqdm import tqdm
import pandas as pd
import numpy as np

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
from monai.metrics import DiceMetric, compute_meaniou,compute_fp_tp_probs,compute_froc_curve_data,compute_froc_score

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from transforms import Resample_TIO,select_foreground,CombineSeq,LoadPickle
from dataloader import SegDataset

from configs import configs as config


def parse_args():
    parser = argparse.ArgumentParser(description='Train classification network')
    
    parser.add_argument('--ckpt', help='ckpt path', required=False, type=str,default='ckpts/seg_ckpt.pth')
    args = parser.parse_args()
    return args

args = parse_args()
ckpt = args.ckpt

torch.backends.cudnn.benchmark = False
no_cuda = False

roi_size = config['roi_size']

max_epochs = 600
val_interval = 1

test_csv_path = 'data/sample_seg_data_val.csv'
THRESH = 0.5
input_sequence_names = ['T1C','T2WI'] 

save_csv_path = 'seg_tumor/result_tmp.csv'

transform_val = config['transform_val']


device = torch.device("cuda")

model = config['network'].to(device)

stdict = torch.load(ckpt,map_location='cpu') if no_cuda else torch.load(ckpt)#['state_dict']
new_stdict = {}
for i,(k,v) in enumerate(stdict.items()):
    if 'module.' in k:
        new_stdict[k[7:]] = stdict[k]
    else:
        new_stdict[k] = stdict[k]
model.load_state_dict(
    new_stdict, 
)
sys.stdout.flush()
model = nn.DataParallel(model)

val_ds = SegDataset(
    csv_path=test_csv_path,
    input_sequence=input_sequence_names,
    transform=transform_val,
    phase='val',
    cache_rate=0,
    num_workers=4,
)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4)


post_trans = Compose(
    [EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold=0.5)]
)

# define inference method
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

def calc_on_target(df):
    iou_list = df[f'IOU_th_{THRESH}'].values
    plane_iou_list = df[f'max_plane_IOU_th_{THRESH}'].values
    info = {
        f'max_plane_iou_th_{THRESH} on target': np.mean(plane_iou_list>0.5),
        f'mean_iou_th_{THRESH}': np.mean(iou_list),
    }
    for k,v in info.items():
        print(k,v)
    


result_df = pd.DataFrame()
model.eval()
with torch.no_grad():
    for val_data in tqdm(val_loader):
        val_inputs, val_labels = (
            val_data["image"].to(device),
            val_data["label"].to(device),
        )
        val_labels = val_labels[:,0:1]
        val_outputs = inference(val_inputs)
        if not no_cuda: 
            val_outputs = val_outputs.cpu()
            val_labels = val_labels.cpu()
        ID = val_data['seg_ID'][0]
        info = {
            'ID':ID,
        }

        val_outputs_threshed = Activations(sigmoid=True)(val_outputs)
        val_outputs_threshed = AsDiscrete(threshold=THRESH)(val_outputs_threshed)
        iou = compute_meaniou(val_outputs_threshed,val_labels).numpy()[0,0]
        info[f'IOU_th_{THRESH}'] = iou

        max_plane_idx = np.argmax(np.sum(val_labels.numpy(),axis=(0,1,2,3)))
        max_plane_iou = compute_meaniou(val_outputs_threshed[:,:,:,:,max_plane_idx],val_labels[:,:,:,:,max_plane_idx]).numpy()[0,0]
        info[f'max_plane_IOU_th_{THRESH}'] = max_plane_iou

        result_df = result_df.append(info,ignore_index=True)

    calc_on_target(result_df)
    result_df.to_csv(save_csv_path,index=False)