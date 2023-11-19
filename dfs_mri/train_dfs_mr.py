import os
import argparse
import time
import numpy as np
import pandas as pd
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
    parser = argparse.ArgumentParser(description='')
    
    parser.add_argument('--train_list', help='train csv path', required=False, type=str,default='data/sample_dfsmr_train.csv')
    parser.add_argument('--val_list', help='val csv path', required=False, type=str,default='data/sample_dfsmr_val.csv')
    parser.add_argument('--ROI_dir', help='ROI dir path', required=False, type=str,default='data/roi_data')
    args = parser.parse_args()
    return args

args = parse_args()
train_list = args.train_list
val_list = args.val_list
roi_dir = args.ROI_dir


workdir = './ckpts'
os.makedirs(workdir,exist_ok=True)

log_id = 'dfsmr_000'

input_size =  (100,136,35)
max_epochs = 100


train_df = pd.read_csv(train_list,dtype={'ID':str,'dfs_status':np.float})
val_df = pd.read_csv(val_list,dtype={'ID':str,'dfs_status':np.float})


def eval_ds(model,data_loader,cohort_name, writer, device, loss_function,epoch,tmp_result):
    model.eval()
    val_outputs_list, status_list, time_list = [], [], []
    with torch.no_grad():
        for val_data in tqdm(data_loader):
            val_inputs, status, dfs_time = (
                val_data["roi"].to(device),
                val_data['dfs_status'].to(device),
                val_data['dfs_time'],
            )
            val_inputs = val_inputs.float()
            val_outputs = model(val_inputs)
            val_outputs_list.append(val_outputs)
            status_list.append(status)
            time_list.append(dfs_time)

        val_outputs = torch.cat(val_outputs_list,0)
        status = torch.cat(status_list,0)
        dfs_time = torch.cat(time_list,0)
        riskset = make_riskset(dfs_time.numpy())
        riskset = torch.tensor(riskset).to(device)
        loss = loss_function(val_outputs,[status, riskset])
        val_c_index = c_index(val_outputs, dfs_time, status)

        aucs_val = get_timeDependent_auc(dfs_time.detach().cpu().numpy(), 
                            status.detach().cpu().numpy(), 
                            val_outputs.detach().cpu().numpy(), 
                            times=[12,24,36,60])
        writer.add_scalar(f'epoch_Loss/{cohort_name}', loss, epoch)
        writer.add_scalar(f'C-Index/{cohort_name}', val_c_index, epoch)
        writer.add_scalar(f'auc_{cohort_name}/1-y', aucs_val['12_auc'], epoch)
        writer.add_scalar(f'auc_{cohort_name}/2-y', aucs_val['24_auc'], epoch)
        writer.add_scalar(f'auc_{cohort_name}/3-y', aucs_val['36_auc'], epoch)
        writer.add_scalar(f'auc_{cohort_name}/5-y', aucs_val['60_auc'], epoch)
        writer.add_scalar(f'auc_{cohort_name}/mean', np.nanmean([aucs_val[k] for k in aucs_val.keys()]), epoch)

        tmp_result[f'{cohort_name}_c_index'] = val_c_index
        tmp_result[f'{cohort_name}_auc_1y'] = aucs_val['12_auc']
        tmp_result[f'{cohort_name}_auc_2y'] = aucs_val['24_auc']
        tmp_result[f'{cohort_name}_auc_3y'] = aucs_val['36_auc']
        tmp_result[f'{cohort_name}_auc_5y'] = aucs_val['60_auc']

        return tmp_result

def do_train():

    transform = Compose([
        LoadROIPickle(keys=['roi_path']),       
        RandAdjustContrastd(keys=["roi"],prob=1.0),
        ScaleIntensityRangePercentilesd(keys="roi", lower=0, upper=100, b_min=0, b_max=1, clip=False, relative=False,channel_wise=True),
        RandFlipd(keys=["roi"],prob=0.75),
        SpatialPadd(keys=["roi"],spatial_size=input_size),  
        RandSpatialCropd(keys=["roi"],roi_size=input_size,random_size=False),     
        EnsureTyped(keys=["roi",'roi_path','dfs_ID','dfs_time','dfs_status']),
    ])

    transform_val = Compose([
        LoadROIPickle(keys=['roi_path']),        
        ScaleIntensityRangePercentilesd(keys="roi", lower=0, upper=100, b_min=0, b_max=1, clip=False, relative=False,channel_wise=True),
        SpatialPadd(keys=["roi"],spatial_size=input_size),       
        EnsureTyped(keys=["roi"]),
    ])

    train_ds = DFSDataset(
        data_df = train_df,
        transform = transform,
        cache_rate = 1.0,
        num_workers = 12,
        roi_dir = roi_dir,
        phase = 'train',
    )
    train_loader = DataLoader(train_ds, batch_size=84, shuffle=True, num_workers=4)

    val_ds = DFSDataset(
        data_df = val_df,
        transform = transform_val,
        cache_rate = 1.0,
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
            pretrained=False, progress=True,
            # norm='instance'
            ).to(device)
    model = nn.DataParallel(model)


    optimizer = torch.optim.Adam(model.parameters(), 2e-6) 
    loss_function  = CoxPHLoss.coxPHLoss


    best_c_index_mean = 0
    best_epoch = -1
    writer = SummaryWriter(log_dir=workdir+'/'+log_id,flush_secs=20)
    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        
        train_outputs_list, status_list, time_list = [], [], []
        for batch_data in train_loader:
            step_start = time.time()
            step += 1
            inputs, status, dfs_time = (
                batch_data["roi"].to(device),
                batch_data['dfs_status'].to(device),
                batch_data['dfs_time'],
            )
            inputs = inputs.float()
            riskset = make_riskset(dfs_time.numpy())
            riskset = torch.tensor(riskset).to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            train_outputs_list.append(outputs)
            status_list.append(status)
            time_list.append(dfs_time)
            try:
                loss = loss_function(outputs, [status, riskset])
            except Exception as e:
                print(e)
                continue
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print(
                f"{step}/{len(train_ds) // train_loader.batch_size}"
                f", train_loss: {loss.item():.4f}"
                f", step time: {(time.time() - step_start):.4f}"
            )
            writer.add_scalar('step_Loss/tain', loss, epoch*len(train_ds)//train_loader.batch_size + step)
        train_outputs_list = torch.cat(train_outputs_list,0)
        status = torch.cat(status_list,0)
        dfs_time = torch.cat(time_list,0)    
        train_c_index = c_index(train_outputs_list, dfs_time, status)
        epoch_loss /= step
        writer.add_scalar('epoch_Loss/tain', epoch_loss, epoch)
        writer.add_scalar('C-Index/train', train_c_index, epoch)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        
        
        if epoch>=4:
            tmp_result ={'log_id':log_id, 'epoch':epoch, 'train_c_index':train_c_index,}
                
            tmp_result = eval_ds(model,val_loader,'val', writer, device, loss_function,epoch,tmp_result)

            print(tmp_result)
            if tmp_result['val_c_index']>best_c_index_mean:  
                best_epoch = epoch
                best_c_index_mean = tmp_result['val_c_index']
                torch.save(
                    model.state_dict(),
                    os.path.join(workdir, f"epoch_{log_id}_{epoch}_{tmp_result['val_c_index']}.pth"),
                )
                tmp_result['c_index_mean'] = tmp_result['val_c_index']

    return tmp_result




tmp_result = do_train()