from skimage.transform import resize
import numpy as np
import os
import SimpleITK as sitk
import h5py

def resampleVolume(outspacing, vol, mask=False):
    """
    将体数据重采样的指定的spacing大小
    paras:
    outpacing: 指定的spacing, 例如[1,1,1]
    vol: sitk读取的image信息, 这里是体数据
    return: 重采样后的数据
    """  
    outsize = [0, 0, 0]
    # 读取文件的size和spacing信息
    inputsize = vol.GetSize()
    inputspacing = vol.GetSpacing()

    # 计算改变spacing后的size，用物理尺寸/体素的大小
    outsize[0] = round(inputsize[0] * inputspacing[0] / outspacing[0])
    outsize[1] = round(inputsize[1] * inputspacing[1] / outspacing[1])
    outsize[2] = round(inputsize[2] * inputspacing[2] / outspacing[2])
    outsize = [outsize[2], outsize[1], outsize[0]]

    # 设定重采样的一些参数
    image_data = sitk.GetArrayFromImage(vol)
    if mask:
        im = resize(image_data, output_shape=outsize, order=0, anti_aliasing=False)
    else:
        im = resize(image_data, output_shape=outsize, order=3, mode='edge', anti_aliasing=False)
    image = sitk.GetImageFromArray(im)
    image.SetSpacing(outspacing)
    image.SetDirection(vol.GetDirection())
    image.SetOrigin(vol.GetOrigin())

    return image

def normalize_data(data, ifzhao=True):  # flare
    if not ifzhao:
        data = np.clip(data, -22.0, 325.0)
        data = np.array(data, dtype=np.float32)
        
        data -= -100
        data /=  100
        # z-score
        # data -= np.mean(data)
        # data /= np.std(data, ddof=1)
        # print(f"z-score process")
    else:
        data = np.clip(data, -22.0, 325.0)
        data = np.array(data, dtype=np.float32)
        data -= 214.68231
        data /= 100.240135
        print(f"zhao process")
    
    return data

def cureImage(new, old):
    new.SetSpacing(old.GetSpacing())
    new.SetOrigin(old.GetOrigin())
    new.SetDirection(old.GetDirection())
    return new

def save_nii_file(arr, path, direction=False, ifzhao=True, mask=False):
    # saveimg = sitk.GetImageFromArray(arr)   # will destroy info from nii
    saveimg = arr

    # vol_resampled = resampleVolume([1.22250766, 1.22250766, 2.5], saveimg, mask=mask)
    vol_resampled = resampleVolume([1.254798173904419, 1.254798173904419, 2.5], saveimg, mask=mask)
    if mask:
        savemask = vol_resampled
        savemask.SetSpacing(vol_resampled.GetSpacing())   # need change
        if direction:
            savemask.SetDirection(direction)
        else:
            savemask.SetDirection(vol_resampled.GetDirection())
        print(f'mask direction: {savemask.GetDirection()}')
        savemask.SetOrigin(vol_resampled.GetOrigin())
        sitk.WriteImage(savemask, path)
        print(f"save mask successfully")
        return None
    
    resize_imgarr = sitk.GetArrayFromImage(vol_resampled)
    nor_resize_imgarr = normalize_data(resize_imgarr, ifzhao=ifzhao)
    print(f"final arrary min: {nor_resize_imgarr.min():4f}, max: {nor_resize_imgarr.max():4f}")
    nor_resize_img = sitk.GetImageFromArray(nor_resize_imgarr)

    # save nii file
    nor_resize_img.SetSpacing(vol_resampled.GetSpacing())
    nor_resize_img.SetOrigin(vol_resampled.GetOrigin())
    if direction:
        nor_resize_img.SetDirection(direction)
    else:
        nor_resize_img.SetDirection(vol_resampled.GetDirection())
    print(f'img direction: {nor_resize_img.GetDirection()}')
    sitk.WriteImage(nor_resize_img, path)

def main():
    # h5_save_path = "E:/FLARE25/train_gt_label/h5_test"

    # for val
    img_r_path = 'E:/FLARE25/test/nii_val_img1'
    mask_r_path = 'E:/FLARE25/test/nii_val_mask1'

    # for train_l
    # img_r_path = 'E:/FLARE25/test/nii_train_l_img2'
    # mask_r_path = 'E:/FLARE25/test/nii_train_l_mask2'

    # for train_u
    # img_r_path = 'E:/FLARE25/test/nii_train_u_img1'
    os.makedirs(img_r_path, exist_ok=True)
    os.makedirs(mask_r_path, exist_ok=True)

    # for train_l
    # trainimg_path  = 'E:/FLARE25/train_gt_label/imagesTr'
    # trainmask_path = 'E:/FLARE25/train_gt_label/labelsTr'

    # for val
    trainimg_path  = 'E:/FLARE25/validation/Validation-Public-Images'
    trainmask_path = 'E:/FLARE25/validation/Validation-Public-Labels'

    # for train_u
    # trainunlabeled_path = 'E:/FLARE25/train_pseudo_label/imagesTr'    # 2000 data
    # trainunlabeled_path = 'E:/FLARE25/coreset_train_50_random'

    # for train unlabeled
    # niiimgnames = sorted(os.listdir(trainunlabeled_path))
    # niimasknames = None

    # for train labeled
    niiimgnames = sorted(os.listdir(trainimg_path))
    niimasknames = sorted(os.listdir(trainmask_path))

    # batch process nii data
    for niiimgname in niiimgnames:
        id = niiimgnames.index(niiimgname)

        # for unlabeled
        # print(f"{'='*15} Processing img-{niiimgname} {'='*15}")

        # for labeled
        niimaskname = niimasknames[id]
        print(f"{'='*15} Processing img-{niiimgname}, mask-{niimaskname} {'='*15}")
        if niiimgname.split('_0000')[0] != niimaskname.split('.')[0]:
            print(f"This itertion is not match for img and mask....")
            continue

        # # for unlabeled 
        # imgfilepath = os.path.join(trainunlabeled_path, niiimgname)

        # for labeled
        imgfilepath = os.path.join(trainimg_path, niiimgname)
        maskfilepath = os.path.join(trainmask_path, niimaskname)

        # get sitk img and mask
        sitk_img = sitk.ReadImage(imgfilepath)
        img_sitk = sitk.GetArrayFromImage(sitk_img).astype(np.float32)

        # for labeled
        sitk_mask = sitk.ReadImage(maskfilepath)
        mask_arrary = sitk.GetArrayFromImage(sitk_mask)

        # get origin direction and transform the window and spacing, finally, saving to nii file
        directions = np.asarray(sitk_img.GetDirection())

        """ 1.first step """
        img1 = np.clip(img_sitk, -1000, 450)
        # for labeled
        # min_organ_value = img_sitk[mask_arrary!=0].min()
        # max_organ_value = img_sitk[mask_arrary!=0].max()
        
        # for unlabeled 
        # min_organ_value = -1024.0
        # max_organ_value = 3071.0

        # img1 = np.clip(img_sitk, min_organ_value, max_organ_value)

        sitk_img1 = sitk.GetImageFromArray(img1)
        cureImage(sitk_img1, sitk_img)

        # save clip file
        new_direciton = [-1, 0, 0, 0, -1, 0, 0, 0, 1]
        savepath     = os.path.join(img_r_path,  niiimgname)
        save_nii_file(sitk_img1,      savepath, ifzhao=False)
        # save_nii_file(sitk_img,      savepath, direction=new_direciton, ifzhao=False)  
        print(f"...保存完成 {(niiimgname):50s} nii.gz文件 to {savepath.split('nii_mid_results')[-1]}")

        # for labeled
        savemaskpath = os.path.join(mask_r_path, niimaskname)
        # save_nii_file(sitk_mask, savemaskpath, direction=new_direciton, ifzhao=False, mask=True)
        save_nii_file(sitk_mask, savemaskpath, ifzhao=False, mask=True)

        # min_mean += min_organ_value
        # max_mean += max_organ_value
        # if min_final >= min_organ_value:
        #     min_final = min_organ_value
        # if max_final <= max_organ_value:
        #     max_final = max_organ_value

    # print(f"min mean : {(min_mean / tmp):3f}, max mean : {(max_mean / tmp):3f}")
    # print(f"min final: {min_final:3f}     , max final: {max_final:3f}")    

if __name__ == "__main__":
    main()














# # get flare22 lab info
# nii_test_path = "D:/PythonProject/ecai25/test"
# test_name_list = os.listdir(nii_test_path)
# times_id = 0
# for test_name in test_name_list:
#     path_test = os.path.join(nii_test_path, test_name)
#     test_file = sitk.ReadImage(path_test)
#     print(f"Direction: {test_file.GetDirection()}, spacing: {test_file.GetSpacing()}")
#     times_id += 1 
#     if times_id >= 1:
#         break