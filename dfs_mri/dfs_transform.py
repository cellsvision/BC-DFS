from monai.transforms import MapTransform,FillHoles
from pickle import load
import numpy as np

class LoadROIPickle(MapTransform):
    def __call__(self, data):
        d = dict(data)
        with open(d['roi_path'],'rb') as f:
            data_pkl = load(f)
        rois = data_pkl['rois']
        # print(np.unique(rois[0]))
        # d['roi'] = (rois[0][0:2]).astype(np.float)

        mask = FillHoles()(np.round(rois[0][2:3]))
        # t1c = (rois[0][0:1] - np.min(rois[0][0:1])) * mask
        # t2 = (rois[0][1:2] - np.min(rois[0][1:2])) * mask
        t1c = rois[0][0:1] * mask
        t2 = rois[0][1:2] * mask
        d['roi'] = np.concatenate([t1c,t2],axis=0)
        d['n_roi'] = len(rois)
        return d