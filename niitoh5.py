import h5py
import os
import SimpleITK as sitk


def main():
    mode = 'l'
    if mode == 'l':
        # nii_imgpath = 'E:/FLARE-Lab/FLARE25/images_preprocess2' # flare
        # nii_maskpath = 'E:/FLARE-Lab/FLARE25/labels_preprocess2' # flare
        # h5_save_path = 'E:/FLARE-Lab/FLARE25/labeled_h5' # flare
        nii_imgpath = 'E:/AbdomenCT-1K/test1/imgs' # AK
        nii_maskpath = 'E:/AbdomenCT-1K/test1/labels' # AK
        h5_save_path = 'E:/AbdomenCT-1K/test1/labeled_h5' # AK
        masknii_names = sorted(os.listdir(nii_maskpath))
    elif mode == 'u':
        # nii_imgpath = 'E:/FLARE-Lab/FLARE25/unlabeled_preprocess2' # flare
        # h5_save_path = 'E:/FLARE-Lab/FLARE25/unlabeled_h5' # flare
        nii_imgpath = 'E:/AbdomenCT-1K/unlabeled_preprocess' # AK
        h5_save_path = 'E:/AbdomenCT-1K/test1/unlabeled_h5' # AK

    os.makedirs(h5_save_path, exist_ok=True)
    imgnii_names = sorted(os.listdir(nii_imgpath))
    for img_name in imgnii_names:
        id = imgnii_names.index(img_name)
        sitk_img = sitk.ReadImage(os.path.join(nii_imgpath, img_name))
        img = sitk.GetArrayFromImage(sitk_img)

        if mode == 'l':
            mask_name = masknii_names[id]
            sitk_mask = sitk.ReadImage(os.path.join(nii_maskpath, mask_name))
            mask = sitk.GetArrayFromImage(sitk_mask)

        print(f"{'sitk img type:':20s}{type(img)}, max: {img.max():3f}, min: {img.min():3f}")

        # save to h5 file
        savepath = os.path.join(h5_save_path, img_name.split('.')[0] + '.h5')
        f = h5py.File(savepath, 'w')
        print(f"saving the {img_name:20s} nii file to {savepath:20s}")
        f.create_dataset('image', data=img, compression="gzip")
        if mode == 'l':
            print(f"saving the {mask_name:20s} nii file to {savepath:20s}")
            f.create_dataset('label', data=mask, compression="gzip")
        f.close()


if __name__ == "__main__":
    main()
