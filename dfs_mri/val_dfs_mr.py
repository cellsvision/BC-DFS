import os
import argparse
import time
import numpy as np
import pandas as pd
import sys
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split

from monai.data import DataLoader
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
    RandFlipd,
    RandHistogramShiftd,
    RandScaleIntensityd,
    RandSpatialCropd,
    ScaleIntensityRangePercentilesd,
    Spacingd,
    SpatialPadd,
)
from monai.networks.nets import DenseNet121,DenseNet169,EfficientNet,DenseNet,EfficientNetBN,Regressor

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from dfs_transform import LoadROIPickle
from dfs_dataloader import DFSDataset
from utils import make_riskset, c_index, get_timeDependent_auc

import CoxPHLoss

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_list', help='val csv path', required=False, type=str,default='data/sample_dfsmr_val.csv')
    parser.add_argument('--ROI_dir', help='ROI dir path', required=False, type=str,default='data/roi_data')
    parser.add_argument('--ckpt', help='ckpt path', required=False, type=str,default='ckpts/dfsmr_ckpt.pth')
    parser.add_argument('--result_path', help='result path', required=False, type=str,default='dfs_mri/result_tmp.csv')
    args = parser.parse_args()
    return args

args = parse_args()
val_list = args.val_list
roi_dir = args.ROI_dir
ckpt = args.ckpt
result_path = args.result_path

input_size =  (100,136,35)

transform_val = Compose([
    LoadROIPickle(keys=['roi_path']),        
    ScaleIntensityRangePercentilesd(keys="roi", lower=0, upper=100, b_min=0, b_max=1, clip=False, relative=False,channel_wise=True),
    SpatialPadd(keys=["roi"],spatial_size=input_size),       
    EnsureTyped(keys=["roi"]),
])

val_df = pd.read_csv(val_list,dtype={'ID':str,'dfs_status':np.float})

val_ds = DFSDataset(
    data_df = val_df,
    transform = transform_val,
    cache_rate = 0.0,
    num_workers = 12,
    roi_dir = roi_dir,
    phase = 'test',
)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=True, num_workers=4)


device = torch.device("cuda")
model = DenseNet121(
        spatial_dims=3,
        in_channels=2,
        out_channels=1,
        pretrained=False, progress=True).to(device)
stdict = torch.load(ckpt)#['state_dict']
new_stdict = {}
for i,(k,v) in enumerate(stdict.items()):
    if 'module.' in k:
        new_stdict[k[7:]] = stdict[k]
    else:
        new_stdict[k] = stdict[k]
model.load_state_dict(
    new_stdict, strict=True
)
sys.stdout.flush()
model = nn.DataParallel(model)


model.eval()
val_outputs_list, status_list, time_list, id_list = [], [], [], []
with torch.no_grad():
    for val_data in tqdm(val_loader):
        val_inputs, status, dfs_time, ID = (
            val_data["roi"].to(device),
            val_data['dfs_status'].to(device),
            val_data['dfs_time'],
            val_data['dfs_ID'],
        )
        val_inputs = val_inputs.float()
        val_outputs = model(val_inputs)
        val_outputs_list.append(val_outputs)
        status_list.append(status)
        time_list.append(dfs_time)
        id_list.extend(ID)
    val_outputs = torch.cat(val_outputs_list,0)
    status = torch.cat(status_list,0)
    dfs_time = torch.cat(time_list,0)
result_df = pd.DataFrame()
result_df['ID'] = id_list
result_df['status'] = status.cpu().numpy()
result_df['dfs_time'] = dfs_time.numpy()
result_df['risk_score'] = val_outputs.cpu().numpy()
result_df.to_csv(result_path,index=False)

aucs_val = get_timeDependent_auc(result_df['dfs_time'].values, 
                    result_df['status'].values, 
                    result_df['risk_score'].values, 
                    times=[12,24,36,60])
print(aucs_val)

