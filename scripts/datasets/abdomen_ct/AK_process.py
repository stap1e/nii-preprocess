import os
import SimpleITK as sitk
import numpy as np

def main():
    img_path = "E:/AbdomenCT-1K/Image"
    mask_path = "E:/AbdomenCT-1K/Mask"
    niiimgnames = sorted(os.listdir(img_path))
    niimasknames = sorted(os.listdir(mask_path))
    unlabeled_names = [name for name in niiimgnames if (name.split('_0000.nii.gz')[0] + '.nii.gz') not in  niimasknames]
    print(f"Total unlabeled images num: {len(unlabeled_names)}")
    for name in unlabeled_names:
        print(name)

    for niiimgname in niiimgnames:
        id = niiimgnames.index(niiimgname)
        niimaskname = niimasknames[id]
        imgfilepath = os.path.join(img_path, niiimgname)
        maskfilepath = os.path.join(mask_path, niimaskname)

        img_sitk  = sitk.ReadImage(imgfilepath,  sitk.sitkFloat32)           # Reading CT
        image     = sitk.GetArrayFromImage(img_sitk).astype(np.float32)

        mask_sitk = sitk.ReadImage(maskfilepath, sitk.sitkInt32)         # Reading CT
        mask      = sitk.GetArrayFromImage(mask_sitk).astype(np.float32)

        print(f"img spacing: {img_sitk.GetSpacing()}, size: {img_sitk.GetSize()}")


if __name__ == "__main__":
    main()