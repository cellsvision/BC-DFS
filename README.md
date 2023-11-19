## Environment
Tested on Ubuntu 16.04, python3.7


## Requirements
- Python packages
  - torchio==0.18.91
  - monai==1.1.0
  - torch==1.8.1
  - pyqlib==0.9.1

## Train Tumor Segmentation Model
To train the tumor segmentation model, Add images and label pathes to  ./data/sample_seg_data_train.csv and ./data/sample_seg_data_val.csv

Then run the following command

```bash
python ./seg_tumor/train_seg.py 
```

## Validate Tumor Segmentation Model
To validate the tumor segmentation model, Add images and label pathes to ./data/sample_seg_data_val.csv

Then run the following command

```bash
python ./seg_tumor/val_seg.py --ckpt [ckpt_path]
```

This will give you an output of:
```bash
max_plane_iou_th_0.5 on target xxx 
mean_iou_th_0.5 xxx
```

And the detail list will be save as .csv at ./seg_tumor/result_tmp.csv

## Crop Tumor ROI by Segmentation Model
To crop tumor ROIs by segmentation model, run the following command

```bash
python ./seg_tumor/infer_crop_roi.py --ckpt [ckpt_path] --infer_list [infer_list_path] --ROI_dir [ROI_dir]
```


## Train DFS MR Model
To train the DFS MR model, run the following command

```bash
python ./dfs_mri/train_dfs_mr.py  --train_list [train_list_path] --val_list [val_list_path] --ROI_dir [ROI_dir]
```

## Validate or test DFS MR Model
To validate the tumor segmentation model, run the following command

```bash
python ./dfs_mri/val_dfs_mr.py --ckpt [ckpt_path] --val_list [infer_list_path] --ROI_dir [ROI_dir] --result_path [result_path]
```

This will give you an output of:
{'12_auc': xxx, '24_auc': xxx, '36_auc': xxx, '60_auc': xxx}

And the detail list will be save as .csv at result_path




## Train DFS MR-clinical Model
To train the DFS MR-clinical model, run the following command

```bash
python ./dfs_clinical/train_dfs_mrclinic.py  --train_list [train_list_path] --val_list [val_list_path] 
```


## Validate or test DFS MR Model
To validate the tumor segmentation model, run the following command

```bash
python ./dfs_clinical/val_dfs_mrclinic.py --ckpt [ckpt_path] --val_list [infer_list_path]  --result_path [result_path]
```

This will give you an output of:
{'test_c_index': xxx, 'test_auc_1y': xxx, 'test_auc_2y': xxx, 'test_auc_3y': xxx, 'test_auc_4y': xxx, 'test_auc_5y': xxx}

And the detail list will be save as .csv at result_path


