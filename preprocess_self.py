from skimage.transform import resize
import numpy as np
import os
import SimpleITK as sitk
import h5py

from preprocess import cureImage

def main():
    index = 0
    img_path = 'E:/FLARE-Lab/FLARE25/imagesTr'
    mask_path = 'E:/FLARE-Lab/FLARE25/labelsTr'

    niiimgnames = sorted(os.listdir(img_path))
    niimasknames = sorted(os.listdir(mask_path))

    # batch process nii data
    for niiimgname in niiimgnames:
        id = niiimgnames.index(niiimgname)
        niimaskname = niimasknames[id]
        print(f"{'='*15} Processing img-{niiimgname}, mask-{niimaskname} {'='*15}")
        if niiimgname.split('_0000')[0] != niimaskname.split('.')[0]:
            print(f"This itertion is not match for img and mask....")
            continue

        # for labeled
        imgfilepath = os.path.join(img_path, niiimgname)
        maskfilepath = os.path.join(mask_path, niimaskname)

        img_sitk  = sitk.ReadImage(imgfilepath,  sitk.sitkFloat32)       # Reading CT
        image     = sitk.GetArrayFromImage(img_sitk).astype(np.float32)  # Converting sitk_metadata to image Array
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
        outspacing = [1.254798173904419, 1.254798173904419, 2.5]
        img_outsize , mask_outsize = [0, 0, 0], [0, 0, 0]
        img_spacing , imgsize  = img_final.GetSpacing(), img_final.GetSize()
        mask_spacing, masksize = mask_sitk.GetSpacing(), mask_sitk.GetSize()

        img_outsize[0] = round(imgsize[0] * img_spacing[0] / outspacing[0])
        img_outsize[1] = round(imgsize[1] * img_spacing[1] / outspacing[1])
        img_outsize[2] = round(imgsize[2] * img_spacing[2] / outspacing[2])
        img_outsize = [img_outsize[2], img_outsize[1], img_outsize[0]]
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
        pathtomask = 'E:/FLARE-Lab/FLARE25/labels_preprocess'
        os.makedirs(pathtomask, exist_ok=True)
        sitk.WriteImage(mask_final_sitk, os.path.join(pathtomask, niimaskname))
        print(f"...保存完成 mask: {(niimaskname):50s} nii.gz文件 to {os.path.join(pathtomask, niimaskname)}")

        pathtoimg = 'E:/FLARE-Lab/FLARE25/images_preprocess'
        os.makedirs(pathtoimg, exist_ok=True)
        sitk.WriteImage(img_final_sitk, os.path.join(pathtoimg, niiimgname))
        print(f"...保存完成 img : {(niiimgname):50s} nii.gz文件 to {os.path.join(pathtoimg, niiimgname)}")

        ### 4. debugg and select final method
        index += 1
        if index > 49:
            break
    
    print(f"{'='*20} All done, total {index} cases {'='*20}")




if __name__ == "__main__":
    main()      
