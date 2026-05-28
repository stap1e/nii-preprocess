from skimage.transform import resize
import numpy as np
import os
import SimpleITK as sitk

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

def normalize_data(data, ifzhao=False):  # flare
    if not ifzhao:
        data -= np.mean(data)
        data /= np.std(data, ddof=1)
    else:
        data = np.clip(data, -22.0, 325.0)
        data = np.array(data, dtype=np.float32)
        data -= 214.68231
        data /= 100.240135
    
    return data

def cureImage(new, old):
    new.SetSpacing(old.GetSpacing())
    new.SetOrigin(old.GetOrigin())
    new.SetDirection(old.GetDirection())
    return new

def save_nii_file(arr, path, direction=False, ifzhao=True, mask=False):
    # saveimg = sitk.GetImageFromArray(arr)   # will destroy info from nii
    print(f"{'*'*15}  saving new nii to {path}  {'*'*15}")
    saveimg = arr

    vol_resampled = resampleVolume([1.22250766, 1.22250766, 2.5], saveimg, mask=mask)
    if mask:
        savemask = vol_resampled
        savemask.SetSpacing(vol_resampled.GetSpacing())  
        if direction:
            savemask.SetDirection(direction)
        else:
            savemask.SetDirection(vol_resampled.GetDirection())
        savemask.SetOrigin(vol_resampled.GetOrigin())
        sitk.WriteImage(savemask, path)
        print(f"save mask successfully")
        return None
    
    resize_imgarr = sitk.GetArrayFromImage(vol_resampled)
    nor_resize_imgarr = normalize_data(resize_imgarr, ifzhao=ifzhao)
    nor_resize_img = sitk.GetImageFromArray(nor_resize_imgarr)

    # save nii file
    nor_resize_img.SetSpacing(vol_resampled.GetSpacing())
    nor_resize_img.SetOrigin(vol_resampled.GetOrigin())
    if direction:
        nor_resize_img.SetDirection(direction)
    else:
        nor_resize_img.SetDirection(vol_resampled.GetDirection())
    print(f'final direction: {nor_resize_img.GetDirection()}, final spacing: {nor_resize_img.GetSpacing()}')
    sitk.WriteImage(nor_resize_img, path)

    # save just spacing nii file
    # sitk.WriteImage(vol_resampled, path)

niiimg_path = "E:/FLARE25/train_gt_label/img_test"
niimask_path = "E:/FLARE25/train_gt_label/mask_test"
midresult_path = 'E:/FLARE25/train_gt_label/nii_mid_result'
os.makedirs(midresult_path, exist_ok=True)

# just select one to test
niiimgnames = sorted(os.listdir(niiimg_path))
niimasknames = sorted(os.listdir(niimask_path))
# batch process nii data
for niiimgname in niiimgnames:
    id = niiimgnames.index(niiimgname)
    niimaskname = niimasknames[id]
    print(f"{'='*15} Processing img-{niiimgname}, mask-{niimaskname} {'='*15}")

    # just select one to test and get sitk img and mask
    imgfilepath = os.path.join(niiimg_path, niiimgname)
    maskfilepath = os.path.join(niimask_path, niimaskname)
    sitk_img = sitk.ReadImage(imgfilepath)
    sitk_mask = sitk.ReadImage(maskfilepath)
    img_sitk = sitk.GetArrayFromImage(sitk_img).astype(np.float32)
    mask_arrary = sitk.GetArrayFromImage(sitk_mask)

    # # save zhao clip file
    # img1 = np.clip(img_sitk, -325, 325)
    # savepath = os.path.join(midresult_path, niiimgname)
    # img1_sitk = sitk.GetImageFromArray(img1)
    # cureImage(img1_sitk, sitk_img)
    # save_nii_file(img1_sitk, savepath)
    # save_nii_file(sitk_mask, os.path.join(midresult_path, niimaskname), mask=True)
    # print(f"...保存完成 {(niiimgname+'  zhao cliped    '):50s} nii.gz文件 to {savepath.split('nii_mid_results')[-1]}")
    
    # save new clip file
    min_organ_value = img_sitk[mask_arrary!=0].min()
    max_organ_value = img_sitk[mask_arrary!=0].max()
    img2 = np.clip(img_sitk, min_organ_value, max_organ_value)
    img2_sitk = sitk.GetImageFromArray(img2)
    cureImage(img2_sitk, sitk_img)
    save_nii_file(img2_sitk, os.path.join(midresult_path,  niiimgname.split('.')[0] + '_new_clip.nii.gz'), ifzhao=False)
    save_nii_file(sitk_mask, os.path.join(midresult_path, niimaskname.split('.')[0] + '_new_clip.nii.gz'), ifzhao=False, mask=True)
    print(f"...保存完成 {(niiimgname+'   new cliped    '):50s} nii.gz文件 to {niiimgname.split('.')[0] + '_new_clip.nii.gz'}")































    # # set new direction and save new direction nii file
    # new_direction = [-1, 0, 0, 0, -1, 0, 0, 0, 1]
    # new_directionname = niiimgname.split('.')[0] + '_new_direction.nii.gz'
    # savepath_new_direction = os.path.join(midresult_path, new_directionname)  
    # save_nii_file(sitk_img, savepath_new_direction, direction=new_direction)
    # print(f"...保存完成 {(niiimgname+'    new direction'):50s} nii.gz文件 to {savepath_new_direction.split('nii_mid_results')[-1]}")

# print(f"{'change arrary img size':30s}:{img1_sitk.GetSize()}, spacing: {img1_sitk.GetSpacing()}, direction: {img1_sitk.GetDirection()}, origin: {img1_sitk.GetOrigin()}")
# print(f"{'new change arrary img size':30s}:{img1_sitk.GetSize()}, spacing: {img1_sitk.GetSpacing()}, direction: {img1_sitk.GetDirection()}, origin: {img1_sitk.GetOrigin()}")
# print(f"{'origin img size':30s}:{sitk_img.GetSize()}, spacing: {sitk_img.GetSpacing()}, direction: {sitk_img.GetDirection()}, origin: {sitk_img.GetOrigin()}")