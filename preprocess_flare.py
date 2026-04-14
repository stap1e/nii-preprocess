from skimage.transform import resize
import numpy as np
import os
import SimpleITK as sitk

from preprocess_notuse import cureImage


def _build_temp_output_path(final_path):
    if final_path.endswith(".nii.gz"):
        return final_path[:-7] + ".tmp.nii.gz"

    root, ext = os.path.splitext(final_path)
    return root + ".tmp" + ext


def remove_file_if_exists(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)


def safe_write_image(image, final_path):
    temp_path = _build_temp_output_path(final_path)
    remove_file_if_exists(temp_path)
    try:
        sitk.WriteImage(image, temp_path)
        os.replace(temp_path, final_path)
    except Exception:
        remove_file_if_exists(temp_path)
        raise


def main():
    index = 0
    mode = 'train_u' # or 'train_l'

    if mode == 'train_l':
        # img_path = 'E:/FLARE-Lab/FLARE25/imagesTr' # flare
        # mask_path = 'E:/FLARE-Lab/FLARE25/labelsTr' # flare
        img_path = "E:/AbdomenCT-1K/Image" # AK
        mask_path = "E:/AbdomenCT-1K/Mask" # AK
        # pathtoimg = 'E:/FLARE-Lab/FLARE25/images_preprocess2' # flare
        # pathtomask = 'E:/FLARE-Lab/FLARE25/labels_preprocess2' # flare
        pathtoimg = 'E:/AbdomenCT-1K/test1/imgs' # AK
        pathtomask = 'E:/AbdomenCT-1K/test1/labels' # AK
        niiimgnames = sorted(os.listdir(img_path))
        niimasknames = sorted(os.listdir(mask_path))
    elif mode == 'train_u':
        img_path = "E:/FLARE-Lab/FLARE22/FLARE22_UnlabeledCase0-500" # flare
        # img_path = 'E:/AbdomenCT-1K/unlabeled' # AK
        pathtoimg = 'E:/FLARE-Lab/FLARE25/unlabeled_preprocess_26_all' # FLARE
        # pathtoimg = 'E:/AbdomenCT-1K/unlabeled_preprocess' # AK
        niiimgnames = sorted(os.listdir(img_path))

    os.makedirs(pathtoimg, exist_ok=True)
    if mode == 'train_l':
        os.makedirs(pathtomask, exist_ok=True)

    # batch process nii data
    for sample_idx, niiimgname in enumerate(niiimgnames):
        imgfilepath = os.path.join(img_path, niiimgname)
        target_img_path = os.path.join(pathtoimg, niiimgname)

        niimaskname = None
        target_mask_path = None
        if mode == 'train_l':
            niimaskname = niimasknames[sample_idx]
            target_mask_path = os.path.join(pathtomask, niimaskname)
            print(f"{'='*15} Processing img-{niiimgname}, mask-{niimaskname} {'='*15}")
            if niiimgname.split('_0000.nii.gz')[0] != niimaskname.split('.')[0]:
                print(f"This itertion is not match for img and mask....")
                continue
            maskfilepath = os.path.join(mask_path, niimaskname)
        else:
            print(f"{'='*15} Processing img-{niiimgname} {'='*15}")

        remove_file_if_exists(_build_temp_output_path(target_img_path))
        if target_mask_path is not None:
            remove_file_if_exists(_build_temp_output_path(target_mask_path))

        img_exists = os.path.exists(target_img_path)
        if target_mask_path is None:
            if img_exists:
                print(f"...跳过已存在样本 img : {(niiimgname):50s}")
                continue
        else:
            mask_exists = os.path.exists(target_mask_path)
            if img_exists and mask_exists:
                print(f"...跳过已存在样本 img : {(niiimgname):50s}")
                print(f"...跳过已存在样本 mask: {(niimaskname):50s}")
                continue
            if img_exists != mask_exists:
                print("...检测到历史半成品，删除后重新生成当前样本")
                remove_file_if_exists(target_img_path)
                remove_file_if_exists(target_mask_path)

        img_sitk  = sitk.ReadImage(imgfilepath,  sitk.sitkFloat32)           # Reading CT
        image     = sitk.GetArrayFromImage(img_sitk).astype(np.float32)      # Converting sitk_metadata to image Array
        if mode == 'train_l':
            mask_sitk = sitk.ReadImage(maskfilepath, sitk.sitkInt32)         # Reading CT
            mask      = sitk.GetArrayFromImage(mask_sitk).astype(np.float32) # Converting sitk_metadata to image Array

        # normalise image， intensity clipping to [-40, 325]
        np_img = np.clip(image, -40., 325.).astype(np.float32)

        ### 1. clip data(both iamge and mask)
        """Whitening. Normalises image to zero mean and unit variance."""
        np_img = np_img.astype(np.float32)
        mean, std = np.mean(np_img), np.std(np_img)
        np_img = (np_img - mean) / std
        
        """Image normalization. Normalises image to fit [0, 1] range. or [0, 255] or [-1, 1]"""
        minimum, maximum = np.min(np_img), np.max(np_img)
        np_img = (np_img - minimum) / (maximum - minimum + 1e-7)

        img_final = sitk.GetImageFromArray(np_img)
        cureImage(img_final, img_sitk)

        ### 2. set up spacing
        outspacing = [1.254798173904419, 1.254798173904419, 2.5] # flare and AK
        img_outsize , mask_outsize = [0, 0, 0], [0, 0, 0]
        img_spacing , imgsize  = img_final.GetSpacing(), img_final.GetSize()

        img_outsize[0] = round(imgsize[0] * img_spacing[0] / outspacing[0])
        img_outsize[1] = round(imgsize[1] * img_spacing[1] / outspacing[1])
        img_outsize[2] = round(imgsize[2] * img_spacing[2] / outspacing[2])
        img_outsize = [img_outsize[2], img_outsize[1], img_outsize[0]]
        if mode == 'train_l':
            mask_spacing, masksize = mask_sitk.GetSpacing(), mask_sitk.GetSize()
            mask_outsize[0] = round(masksize[0] * mask_spacing[0] / outspacing[0])
            mask_outsize[1] = round(masksize[1] * mask_spacing[1] / outspacing[1])
            mask_outsize[2] = round(masksize[2] * mask_spacing[2] / outspacing[2])
            mask_outsize = [mask_outsize[2], mask_outsize[1], mask_outsize[0]]

            mask_final = resize(mask, output_shape=mask_outsize, order=0, anti_aliasing=False)
            mask_final_sitk = sitk.GetImageFromArray(mask_final)
            mask_final_sitk.SetSpacing(outspacing)
            mask_final_sitk.SetDirection(mask_sitk.GetDirection())
            mask_final_sitk.SetOrigin(mask_sitk.GetOrigin())

        image_data = sitk.GetArrayFromImage(img_final)
        img_final = resize(image_data, output_shape=img_outsize, order=3, mode='edge', anti_aliasing=False)
        img_final_sitk = sitk.GetImageFromArray(img_final)
        img_final_sitk.SetSpacing(outspacing)
        img_final_sitk.SetDirection(img_sitk.GetDirection())
        img_final_sitk.SetOrigin(img_sitk.GetOrigin())
        
        ### 3. save to nii
        if mode == 'train_l':
            try:
                safe_write_image(mask_final_sitk, target_mask_path)
                safe_write_image(img_final_sitk, target_img_path)
            except Exception:
                remove_file_if_exists(target_img_path)
                remove_file_if_exists(target_mask_path)
                raise
            print(f"...保存完成 mask: {(niimaskname):50s} nii.gz文件 to {target_mask_path}")
        else:
            safe_write_image(img_final_sitk, target_img_path)
        print(f"...保存完成 img : {(niiimgname):50s} nii.gz文件 to {target_img_path}")

        ### 4. debugg and select final method
        index += 1
        # if index > 49:
        #     break
    
    print(f"{'='*20} All done, total {index} cases {'='*20}")



if __name__ == "__main__":
    main()      
