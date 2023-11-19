import torchio as tio
from pickle import load
import numpy as np
from monai.transforms import MapTransform
from skimage import exposure
from skimage.filters import threshold_otsu,gaussian

class Resample_TIO(MapTransform):
    """
    """

    def __init__(self,source_key,mode,**kwargs) -> None:
        super().__init__(**kwargs)
        self.source_key = source_key
        self.mode = mode

    def __call__(self, data):
        d = dict(data)
        ref_im = tio.ScalarImage(tensor=data[self.source_key],affine=data[self.source_key+'_meta_dict']['affine'])

        for i,k in enumerate(self.keys):
            tmp_tar_im = tio.ScalarImage(tensor=data[k],affine=data[k+'_meta_dict']['affine'])
            tmp_tar_im = tio.Resample(ref_im,image_interpolation=self.mode[i])(tmp_tar_im)
            d[k] = tmp_tar_im.numpy()
            d[k+'_meta_dict']['affine'] = tmp_tar_im.affine
            d[k+'_meta_dict']['spatial_shape'] = tmp_tar_im.shape[1:]

        return d

class ToSpacing(MapTransform):
    """
    """

    def __init__(self,pixdim,to_ras,**kwargs) -> None:
        super().__init__(**kwargs)
        self.pixdim = pixdim
        self.to_ras = to_ras

    def __call__(self, data):
        d = dict(data)
        for k in self.keys:
            ref_im = tio.ScalarImage(tensor=data[k],affine=data[k+'_meta_dict']['affine'])
            if self.to_ras:
                ref_im = tio.ToCanonical()(ref_im)
            im = tio.Resample(self.pixdim,)(ref_im)
            d[k] = im.numpy()
            d[k+'_meta_dict']['affine'] = im.affine
            d[k+'_meta_dict']['spatial_shape'] = im.shape[1:]
        return d

class CombineSeq(MapTransform):
    def __call__(self, data):
        d = dict(data)
        ims = []
        for k in self.keys:
            ims.append(data[k])
        d['image'] = np.concatenate(ims,0)
        d['image_meta_dict'] = data[self.keys[0]+'_meta_dict']

        for k in self.keys:
            d.pop(k)
        return d

class LoadPickle(MapTransform):
    def __call__(self, data):
        d = dict(data)
        with open(d['preprocessed_path'],'rb') as f:
            data_pkl = load(f)
        for k,v in data_pkl.items():
            d[k] = v
        return d




def get_binary(image):
    #     image_blur = median(image)
    image_blur = gaussian(image,sigma=4)
    # image_blur = image
    thresh = threshold_otsu(image_blur)
    binary = image_blur > thresh
    return binary

def select_foreground(img):
    # print(img.shape,'-'*20)
    if len(img.shape)==4:
        image = img[0]
    else:
        image = img
    binary = exposure.equalize_hist(image) ####    
    binary = get_binary(binary)
    if len(img.shape)==4:
        binary = np.repeat(np.expand_dims(binary,0),img.shape[0],axis=0)
    return binary