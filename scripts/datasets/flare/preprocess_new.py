import os, warnings, glob, torch
import numpy as np
from monai.transforms import (
    NormalizeIntensityd, Compose, CropForegroundd, LoadImaged, RandFlipd, RandCropByPosNegLabeld, Invertd, SaveImaged,
    RandShiftIntensityd, Spacingd, RandRotate90d, RandAffined, SpatialPadd, Resized, RandScaleIntensityd, Activationsd,
    EnsureChannelFirstd, ScaleIntensityRangePercentilesd, ToTensord, AsDiscreted, RemoveSmallObjectsd, )
import nibabel as nib

######################## split #############################
# train_l_img_path  = "E:/FLARE25/train_gt_label/imagesTr"
# train_l_mask_path = "E:/FLARE25/train_gt_label/labelsTr"

val_img_path  = "E:/FLARE25/validation/Validation-Public-Images"
val_mask_path = "E:/FLARE25/validation/Validation-Public-Labels"

img_data_path   = val_img_path
mask_data_path  = val_mask_path
# img_save_path  =  "E:/FLARE25/test/nii_train_l_img_6"
# mask_save_path =  "E:/FLARE25/test/nii_train_l_mask_6"
img_save_path  =  "E:/FLARE25/test/nii_val_img_6"
mask_save_path =  "E:/FLARE25/test/nii_val_mask_6"
os.makedirs(img_save_path,  exist_ok=True)
os.makedirs(mask_save_path, exist_ok=True)

img_list = sorted(glob.glob(os.path.join(img_data_path, '*.nii.gz')))
######################## split #############################

transforms1 = Compose([
            # CropForegroundd(keys=["image"], source_key="pred_ROI"),  # 取label 的 value > 0
            Spacingd(
                keys=["image"],
                pixdim=(1.5, 1.5, 2),
                mode=("bilinear"),),
            NormalizeIntensityd(keys=['image'], nonzero=True),
            # SpatialPadd(keys=['image'], spatial_size=(112, 112, 112), method='end', mode='constant'),# 对低于spatial_size 的维度pad
            ])

transforms2 = Compose([
                LoadImaged(keys=["image", "label"] ),
                EnsureChannelFirstd(keys=["image", "label"]),  
                ScaleIntensityRangePercentilesd(keys=["image"],lower = 5, upper = 95, b_min =0.0, b_max=255.0, clip=True),
                Spacingd(
                        keys=["image", "label"],
                        pixdim=(2, 2, 2.0),  
                        mode=("bilinear", "nearest"),),
                CropForegroundd(keys=["image", "label"], source_key="label"),    #取label 的 value > 0
                NormalizeIntensityd(keys='image',nonzero=True), 
                # SpatialPadd(keys=['image', 'label'], spatial_size=(112, 112, 112), method ='end',mode='constant'), #对低于spatial_size 的维度pad
                ToTensord(keys=["image", "label"]),])

for nii_name in img_list:
    print(f"{'*'*15} process for {nii_name.split('/')[-1]} starts! {'*'*15}")
    data_files=[{"image": nii_name}]
    saveimg_path = os.path.join(img_save_path, nii_name.split('\\')[-1])
    img1 = Compose([LoadImaged(keys=["image"]),
            # EnsureChannelFirstd(keys=["image"]),
            # ScaleIntensityRangePercentilesd(keys=["image"], lower=5, upper=95, b_min=0, b_max=255, clip=True),
            ])(data_files)
    img2 = Compose([LoadImaged(keys=["image"]),
            # EnsureChannelFirstd(keys=["image"]),    # make input (H, W, D) to (1, H, W, D)
            ScaleIntensityRangePercentilesd(keys=["image"], lower=5, upper=95, b_min=0, b_max=255, clip=True),
            Spacingd(keys=["image"], pixdim=(1.5, 1.5, 2), mode=("bilinear"),),
            # Spacingd(keys=["image", "label"], pixdim=(2, 2, 2.0), mode=("bilinear", "nearest"),),
            SaveImaged(keys=["image"], output_dir=img_save_path, output_postfix='', output_ext='.nii.gz', separate_folder=False),
            ])(data_files)
    break





