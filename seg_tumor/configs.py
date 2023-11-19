from easydict import EasyDict as edict
from monai.networks.nets import BasicUNet,HighResNet,UNet,FlexibleUNet,AttentionUnet
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
    RandGaussianNoised,
    RandFlipd,
    RandHistogramShiftd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
    SpatialPadd,
)
from monai.losses import DiceLoss,DiceFocalLoss,GeneralizedDiceLoss

import torch
import torch_optimizer as optim
from monai.data.image_reader import TIOReader



from transforms import Resample_TIO,select_foreground,CombineSeq,LoadPickle,ToSpacing


configs = edict()   
configs['log_id'] = '000'           
configs['network'] = UNet( 
                    spatial_dims=3,
                    in_channels=2,
                    out_channels=1, 
                    strides=((2,2,1),(2,2,1),(2,2,1),(2,2,1)),
                    channels=[32,64,128,128,32],
                    )
configs['roi_size'] = [256,256,32]
input_sequence_names = ['T1C','T2WI'] 
pixdim =  (1.0, 1.0, 3.0)

configs['transform_train'] = Compose([
    LoadImaged(keys=input_sequence_names+["label"],reader=TIOReader),
    ToSpacing(keys=["T1C"],pixdim=pixdim,to_ras=True),
    Resample_TIO(source_key="T1C",keys=input_sequence_names+["label"],mode=['linear']*len(input_sequence_names)+['nearest']),
    CropForegroundd(source_key="T1C", keys=input_sequence_names+["label"], select_fn=select_foreground),
    CombineSeq(keys=input_sequence_names),
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    SpatialPadd(keys=["image", "label"],spatial_size=configs['roi_size']),
    RandSpatialCropd(keys=["image", "label"],roi_size=configs['roi_size'],random_size=False),
    OneOf([ 
        RandScaleIntensityd(keys=["image"],prob=0.8,factors=0.4), 
        RandAdjustContrastd(keys=["image"],prob=0.8), 
        RandHistogramShiftd(keys=["image"],prob=0.8), 
        RandShiftIntensityd(keys=["image"],prob=0.8,offsets=0.2)
    ]),
    RandFlipd(keys=["image"],prob=0.6),
    EnsureTyped(keys=["image", "label"]),
    ])
configs['transform_val'] = Compose([
    LoadImaged(keys=input_sequence_names+["label"],reader=TIOReader),
    ToSpacing(keys=["T1C"],pixdim=pixdim,to_ras=True),
    Resample_TIO(source_key="T1C",keys=input_sequence_names+["label"],mode=['linear']*len(input_sequence_names)+['nearest']),
    CropForegroundd(source_key="T1C", keys=input_sequence_names+["label"], select_fn=select_foreground),
    CombineSeq(keys=input_sequence_names),
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    SpatialPadd(keys=["image", "label"],spatial_size=configs['roi_size']),
    EnsureTyped(keys=["image", "label"]),
    ])
configs['optimizer'] = optim.RAdam
configs['lr'] = 1e-4
configs['batch_size'] = 16
configs['loss'] = DiceLoss(to_onehot_y=False, squared_pred=True, sigmoid=True, reduction='mean', smooth_nr=1e-07, smooth_dr=1e-07)

