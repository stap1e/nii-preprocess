import h5py, os
import SimpleITK as sitk

imgnii_path = 'E:/FLARE25/train_gt_label/img_val'
masknii_path = 'E:/FLARE25/train_gt_label/mask_val'

# imgnii_path = 'E:/FLARE25/train_gt_label/nii_val_results_l'
h5_save_path = 'E:/FLARE25/train_gt_label/valh5'
# h5_read_path = 'E:/FLARE25/train_gt_label/labeled_onlyh5'
os.makedirs(h5_save_path, exist_ok=True)

imgnii_names = sorted(os.listdir(imgnii_path))
# h5names = sorted(os.listdir(h5_read_path))

for img_name in imgnii_names:
    id = imgnii_names.index(img_name)
    maskname = img_name.split('_0000')[0] + '.nii.gz'
    sitk_img = sitk.ReadImage(os.path.join(imgnii_path, img_name))
    sitk_mask = sitk.ReadImage(os.path.join(masknii_path, maskname))
    img = sitk.GetArrayFromImage(sitk_img)
    mask = sitk.GetArrayFromImage(sitk_mask)
    print(f"{'sitk img type:':20s}{type(img)}, max: {img.max():3f}, min: {img.min():3f}")

    # save to h5 file
    savepath = os.path.join(h5_save_path, img_name.split('.')[0] + '.h5')
    f = h5py.File(savepath, 'w')
    print(f"saving the {maskname:20s} nii file to {savepath:20s}")
    print(f"saving the {img_name:20s} nii file to {savepath:20s}")
    f.create_dataset('image', data=img, compression="gzip")
    f.create_dataset('label', data=mask, compression="gzip")
    f.close()












    # h5_path = os.path.join(h5_read_path, h5names[id])
    # h5data = h5py.File(h5_path, 'r')
    # img_h5 = h5data['image'][:]
    # mask_h5 = h5data['label'][:]
    # print(f"{'h5 img type: ':20s}{type(img_h5)}, max: {img_h5.max():3f}. min: {img_h5.min():3f}")
