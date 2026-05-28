import os
import glob
import re
import nibabel as nib
from pathlib import Path
import numpy as np
import h5py
# from prefetch_generator import BackgroundGenerator
from monai.transforms import (
    AsDiscrete,
    NormalizeIntensityd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Transposed,
    LabelFilterd,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    EnsureTyped,
    RandAffined,
    SpatialPadd,
    Resized,
    RandScaleIntensityd,
    HistogramNormalized,
    CenterSpatialCrop,
    RandCropByLabelClassesd,
    EnsureChannelFirstd,
ScaleIntensityRangePercentilesd,    RandSpatialCropd,
    ToTensord,
    SaveImaged,
    ThresholdIntensityd,
    CenterSpatialCropd)

from monai.data import (
    ThreadDataLoader,
    DataLoader,
    CacheDataset,
    Dataset,
    load_decathlon_datalist,
    decollate_batch,
    set_track_meta
    
)

import warnings
warnings.filterwarnings('ignore')  
def get_a_file(task_dir, filename )
    data_root = task_dir
    image_root = os.path.join(data_root, 'image')
    label_root = os.path.join(data_root, 'label')
    
    image = os.path.join(image_root, filename)
    label = os.path.join(label_root, filename.replace('_0000', ''))

    file = []
    file.append({'image' image, 'label' label})

    return file





def get_files(task_dir, mode = 'train')

    data_root = task_dir
    files = []


    image_root = os.path.join(data_root, mode, 'image')
    mask_root = os.path.join(data_root, mode, 'label')

    for image_name in os.listdir(image_root)
        label_name = image_name.split('_0000')[0] + r'.nii.gz'

        image = os.path.join(image_root, image_name)
        label = os.path.join(mask_root, label_name)

        if os.path.exists(label)
            files.append({'image' image, 'label' label})
        
    return files



def transform_to_nii(files, mode = 'train')
    if mode == 'train'
        sample_atch = (96,96,96)
        num_samples= 6
        train_transforms = Compose(
            [
                LoadImaged(keys=[image, label]),
                EnsureChannelFirstd(keys=[image, label]),  
                ScaleIntensityRangePercentilesd(keys=[image],lower = 5, upper = 95, b_min =0, b_max=255, clip=True),
                Spacingd(
                    keys=[image, label],
                    pixdim=(2, 2, 2.0),                
                    mode=(bilinear, nearest),
                ),
                CropForegroundd(keys=[image, label], source_key=label), #取label 的 value  0

                NormalizeIntensityd(keys='image',nonzero =True), 
                SpatialPadd(keys=['image', 'label'], spatial_size=sample_patch,method ='end',mode='constant'), #对低于spatial_size 的维度pad

                RandCropByPosNegLabeld(
                                    keys=[image, label],
                                    label_key=label,
                                    spatial_size=sample_patch,
                                    pos=5,
                                    neg=1,
                                    num_samples=num_samples,
                                ),
                                
                                
                RandFlipd(
                    keys=[image, label],
                    spatial_axis=[0],
                    prob=0.10,
                ),
                RandFlipd(
                    keys=[image, label],
                    spatial_axis=[1],
                    prob=0.10,
                ),
                RandFlipd(
                    keys=[image, label],
                    spatial_axis=[2],
                    prob=0.10,
                ),
                RandRotate90d(
                    keys=[image, label],
                    prob=0.10,
                    max_k=3,
                ),
                RandScaleIntensityd(keys=[image], factors=0.1, prob=0.1),
                RandShiftIntensityd(
                    keys=[image],
                    offsets=0.10,
                    prob=0.1,
                ),
                RandAffined(
                    keys=['image', 'label'],
                    mode=('bilinear', 'nearest'),
                    prob=0.1, spatial_size= None,
                    rotate_range=(0, 0, np.pi  15),
                    scale_range=(0.1, 0.1, 0.1)),

                SaveImaged(
                    keys = [image, label],
                    output_dir = .output,
                    output_postfix = transformed,
                    output_ext = .nii.gz,
                    separate_folder = False
                )

                # ToTensord(keys=[image, label]), 
            

            ])
        
        if isinstance(files, dict)
            train_transforms(files)
        elif isinstance(files, list)
            for f in files
                train_transforms(f)
        # data = train_transforms(files)

        # # if data.shape[0] = 4 and data.ndim == 4
        # #     data = np.transpose(data, (1, 2, 3, 0))
        
        # img = data['image'].squeeze(0)
        # label = data['label'].squeeze(0)

        # # img_affine = data['image'].affine
        # # label_affine = data['label'].affine
        # img_affine = data['image_meta_dict'][afiine]
        # label_affine = data['image_meta_dict'][affine]


        # nii_img = nib.Nifti1Image(img, img_affine)
        # nii_label = nib.Nifti1Image(label, label_affine)

        # nib.save(nii_img, img.nii.gz)
        # nib.save(nii_label, label.nii.gz)






    if mode == 'val'
        val_transforms = Compose(
        [
            LoadImaged(keys=[image, label] ),
            EnsureChannelFirstd(keys=[image, label]),  
            ScaleIntensityRangePercentilesd(keys=[image],lower = 5, upper = 95, b_min =0.0, b_max=255.0, clip=True),         
            Spacingd(
                        keys=[image, label],
                        pixdim=(2, 2, 2.0),  
                        mode=(bilinear, nearest),
                    ),
            CropForegroundd(keys=[image, label], source_key=label), #取label 的 value  0
            NormalizeIntensityd(keys='image',nonzero=True),            
            SpatialPadd(keys=['image', 'label'], spatial_size=sample_patch,method ='end',mode='constant'), #对低于spatial_size 的维度pad
            ToTensord(keys=[image, label]),

        ]
    )
    
file = get_a_file(task_dir='Dcode_fieldprocess_niidatalabeled', filename='FLARE22_Tr_0016_0000.nii.gz')

transform_to_nii(file)

