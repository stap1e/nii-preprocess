import h5py
import numpy as np
import os
import SimpleITK as sitk


def main():
    nii_imgpath = 'E:/FLARE-Lab/FLARE25/images_preprocess2'
    nii_maskpath = 'E:/FLARE-Lab/FLARE25/labels_preprocess2'

    h5_save_path = 'E:/FLARE-Lab/FLARE25/test2/train_l_h5'
    os.makedirs(h5_save_path, exist_ok=True)
    
    imgnii_names = sorted(os.listdir(nii_imgpath))
    masknii_names = sorted(os.listdir(nii_maskpath))
    for img_name in imgnii_names:
        id = imgnii_names.index(img_name)
        mask_name = masknii_names[id]
        sitk_img = sitk.ReadImage(os.path.join(nii_imgpath, img_name))
        sitk_mask = sitk.ReadImage(os.path.join(nii_maskpath, mask_name))
        img = sitk.GetArrayFromImage(sitk_img)
        mask = sitk.GetArrayFromImage(sitk_mask)
        print(f"{'sitk img type:':20s}{type(img)}, max: {img.max():3f}, min: {img.min():3f}")

        # save to h5 file
        savepath = os.path.join(h5_save_path, img_name.split('.')[0] + '.h5')
        f = h5py.File(savepath, 'w')
        print(f"saving the {mask_name:20s} nii file to {savepath:20s}")
        print(f"saving the {img_name:20s} nii file to {savepath:20s}")
        f.create_dataset('image', data=img, compression="gzip")
        f.create_dataset('label', data=mask, compression="gzip")
        f.close()


if __name__ == "__main__":
    main()
