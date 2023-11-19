import time
import argparse
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from tqdm import tqdm
import pandas as pd
import numpy as np
from pickle import dump
from copy import deepcopy 
from multiprocessing import Pool, freeze_support
from itertools import repeat
from skimage import measure

from monai.data.image_reader import TIOReader
from monai.data import DataLoader, decollate_batch
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    CropForegroundd,
    EnsureTyped,
    EnsureType,
    Invertd,
    LoadImaged,
    NormalizeIntensityd,
    SpatialPadd,
)
from monai.networks.nets import BasicUNet
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, compute_meaniou,compute_fp_tp_probs,compute_froc_curve_data,compute_froc_score

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

import torchio as tio

from transforms import Resample_TIO,select_foreground,CombineSeq,LoadPickle,ToSpacing

from configs import configs as config

def parse_args():
    parser = argparse.ArgumentParser(description='')
    
    parser.add_argument('--infer_list', help='infer csv path', required=False, type=str,default='data/sample_seg_data_val.csv')
    parser.add_argument('--ckpt', help='ckpt path', required=False, type=str,default='ckpts/seg_ckpt.pth')
    parser.add_argument('--ROI_dir', help='ROI dir path', required=False, type=str,default='data/roi_data')
    args = parser.parse_args()
    return args

args = parse_args()
infer_list = args.infer_list
ckpt = args.ckpt

input_sequence_names = ['T1C','T2WI'] 
pixdim = (1.0, 1.0, 3.0)
THRESH = 0.5

crop_size = [100,136,35]
crop_spacing = (0.6,0.6,2)

roi_size =  config['roi_size']
no_cuda = False
device = torch.device("cpu") if no_cuda else torch.device("cuda")


saved_dir = args.ROI_dir
os.makedirs(saved_dir, exist_ok=True)


df = pd.read_csv(infer_list,dtype={'ID':str,'dyn_fix':str}).dropna(how='any',subset=input_sequence_names)

transform = Compose([
        LoadImaged(keys=input_sequence_names,reader=TIOReader),
        ToSpacing(keys=["T1C"],pixdim=pixdim,to_ras=True),
        Resample_TIO(source_key="T1C",keys=input_sequence_names,mode=['linear']*len(input_sequence_names)),
        CombineSeq(keys=input_sequence_names),
        CropForegroundd(source_key="image", keys=["image"], select_fn=select_foreground,constant_values=0),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        SpatialPadd(keys=["image"],spatial_size=roi_size),
        EnsureTyped(keys=["image"]),
        ])


infer_info = deepcopy(config)
infer_info['ckpt'] = ckpt
infer_info['infer_transform'] = transform
infer_info['THRESH'] = THRESH
infer_info['crop_size'] = crop_size
infer_info['crop_spacing'] = crop_spacing
with open(saved_dir+'/infer_info.pkl','wb')as f:
    dump(infer_info,f)

model = config['network'].to(device)
stdict = torch.load(ckpt,map_location='cpu') if no_cuda else torch.load(ckpt)#['state_dict']
new_stdict = {}
for i,(k,v) in enumerate(stdict.items()):
    if 'module.' in k:
        # print(stdict[k].dtype)
        # print('pre-trained key:',k)
        new_stdict[k[7:]] = stdict[k]
    else:
        new_stdict[k] = stdict[k]
model.load_state_dict(
    new_stdict, 
)
sys.stdout.flush()
# model = nn.DataParallel(model)
model.eval()

def inference(input):

    def _compute(input):
        return sliding_window_inference(
            inputs=input,
            roi_size=roi_size,
            sw_batch_size=6,
            predictor=model,
            overlap=0.1,
        )

    return _compute(input)

def get_crop_roi(t1c_image,t2_image,infer_mask,infer_conf,ID=''):
    # todo: add side
    t1c_image_1 = tio.ToCanonical()(tio.Resample(crop_spacing)(t1c_image))
    t2_image_1 = tio.Resample(t1c_image_1)(t2_image)
    infer_conf_1 = tio.Resample(t1c_image_1)(infer_conf)
    infer_mask_1 = tio.Resample(t1c_image_1,image_interpolation='nearest')(infer_mask)

    t1c_arr, t2_arr, mask_arr, conf_arr = t1c_image_1.numpy()[0], t2_image_1.numpy()[0], infer_mask_1.numpy()[0], infer_conf_1.numpy()[0]

    label = measure.label(mask_arr)
    regions = measure.regionprops_table(label,properties=('label', 'area','bbox'))
    rois = []
    roi_infos = []
    for i in np.argsort(regions['area'])[::-1]:
        if regions['area'][i]>50 or (regions['area'][i]==np.max(regions['area'])): #50
            h = regions['bbox-3'][i]-regions['bbox-0'][i]
            w = regions['bbox-4'][i]-regions['bbox-1'][i]
            d = regions['bbox-5'][i]-regions['bbox-2'][i]
            if (h>crop_size[0]) or (w>crop_size[1]) or (d>crop_size[2]):
                print(f'{ID} tumor size {h} {w} {d} is larger than crop size {crop_size}')
                h,w,d = min(crop_size[0],h), min(crop_size[1],w), min(crop_size[2],d)

            center = [
                (regions['bbox-3'][i]+regions['bbox-0'][i])//2,
                (regions['bbox-4'][i]+regions['bbox-1'][i])//2,
                (regions['bbox-5'][i]+regions['bbox-2'][i])//2,
            ]
            a_min,a_max = center[0] - (crop_size[0]//2), center[0] + (crop_size[0]//2)
            b_min,b_max = center[1] - (crop_size[1]//2), center[1] + (crop_size[1]//2)
            c_min,c_max = center[2] - (crop_size[2]//2), center[2] + (crop_size[2]//2)

            if c_min<0:
                c_min = 0 
                c_max = crop_size[2]
            if c_max>t1c_arr.shape[2]:
                t1c_arr.shape[2]-crop_size[2]
                c_max = t1c_arr.shape[2]

            if a_min<0:
                a_min = 0 
                a_max = crop_size[0]
            if a_max>t1c_arr.shape[0]:
                t1c_arr.shape[0]-crop_size[0]
                a_max = t1c_arr.shape[0]

            if b_min<0:
                b_min = 0 
                b_max = crop_size[1]
            if b_max>t1c_arr.shape[1]:
                t1c_arr.shape[1]-crop_size[1]
                b_max = t1c_arr.shape[1]

            crop_t1c = t1c_arr[a_min:a_max, b_min:b_max, c_min:c_max]
            crop_t2 = t2_arr[a_min:a_max, b_min:b_max, c_min:c_max]
            crop_mask = mask_arr[a_min:a_max, b_min:b_max, c_min:c_max]
            crop_conf = conf_arr[a_min:a_max, b_min:b_max, c_min:c_max]

            crop_arr = np.stack([crop_t1c,crop_t2,crop_mask,crop_conf],axis=0)

            roi_info = {
                'center': center,
                'size': [h,w,d],
                'area': regions['area'][i],
                'image_size':t1c_arr.shape
            }
            roi_infos.append(roi_info)
            rois.append(crop_arr)
    # rois = np.stack(rois,axis=0)
    print(len(rois))

    result = {
        'rois':rois,
        'roi_infos':roi_infos
    }

    return result


def process_data(row):
    i,row = row
    ID = row['ID']
    if ('T1C*' in row['T1C']) or ('T1C_' in row['T1C']):
        if isinstance(row['dyn_fix'],str):
            t1c_path = f"{os.path.dirname(row['T1C'])}/T1C_{int(float(row['dyn_fix']))}"
        else:
            print(f'{ID} T1C is T1C* without dyn_fix id')
    else:
        t1c_path = row['T1C']
    t2_path = row['T2WI']

    infer_from_raw(t1c_path,t2_path,saved_dir=saved_dir)

def infer_from_raw(t1c_path,t2_path,saved_dir=saved_dir,mask_path=None,save_intermediate_results=False):
    data = [{'T1C':t1c_path,'T2WI':t2_path}]

    ID = os.path.basename(os.path.dirname(t1c_path))
    print(f'---------{ID}-------------')
    if os.path.exists(saved_dir+f'/{ID}_roi.pkl'):
        return
    val_data = transform(data)
    val_inputs = val_data[0]["image"].to(device)
    val_inputs = torch.unsqueeze(val_inputs,0)
    with torch.no_grad():
        val_outputs = inference(val_inputs)


    val_data = val_data[0]
    val_data["pred"] = Activations(sigmoid=True)(val_outputs[0])

    invert_post = Compose([
        EnsureTyped(keys="pred"),
        Invertd(
            keys="pred",
            transform=transform,
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
        ),
        ])
    invert_post_data = invert_post(val_data)
    val_outputs_invert = invert_post_data['pred']
    val_outputs_threshed = AsDiscrete(threshold=0.5)(val_outputs_invert)
    roi_size = np.sum(val_outputs_threshed)
    if roi_size<50:
        val_outputs_threshed = AsDiscrete(threshold=0.5)(val_outputs_invert)
        if np.sum(val_outputs_threshed)<50: 
            print(f'------------------ {ID} no tumor detected -------------------')
            return 

    
    t1c_image = tio.ScalarImage(t1c_path)
    t2_image = tio.ScalarImage(t2_path)

    im_affine=invert_post_data['image_meta_dict']['affine']
    infer = tio.LabelMap(tensor=val_outputs_threshed.cpu().numpy(),affine=im_affine)
    infer_conf = tio.ScalarImage(tensor=val_outputs_invert.cpu().numpy(),affine=im_affine)
    if mask_path is not None:
        mask = tio.LabelMap(mask_path)

    if save_intermediate_results:
        t1c_image.save(path=f'{saved_dir}/{ID}_image.nrrd')
        infer.save(path=f'{saved_dir}/{ID}_infer.seg.nrrd')
        if mask_path is not None:
            mask.save(path=f'{saved_dir}/{ID}_mask.seg.nrrd')

    result = get_crop_roi(t1c_image,t2_image,infer,infer_conf,ID=ID)

    with open(saved_dir+f'/{ID}_roi.pkl','wb') as f:
        dump(result,f)


if __name__=="__main__":
    # with Pool(12) as pool:
    #     pool.map(process_data,df.iterrows())
    for i,row in tqdm(df.iterrows()):
        ID = row['ID']
        if ('T1C*' in row['T1C']) or ('T1C_' in row['T1C']):
            if isinstance(row['dyn_fix'],str):
                t1c_path = f"{os.path.dirname(row['T1C'])}/T1C_{int(float(row['dyn_fix']))}"
            else:
                print(f'{ID} T1C is T1C* without dyn_fix id')
        else:
            t1c_path = row['T1C']
        t2_path = row['T2WI']
        infer_from_raw(t1c_path,t2_path,saved_dir=saved_dir)
        # print(row)