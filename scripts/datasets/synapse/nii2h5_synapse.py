import h5py
import os
import SimpleITK as sitk

def main():
    # 路径配置
    nii_imgpath = "D:/data/Synapse_preprocessed/img" 
    nii_maskpath = "D:/data/Synapse_preprocessed/label" 
    h5_save_path = "D:/data/Synapse_preprocessed/h5" 
    
    os.makedirs(h5_save_path, exist_ok=True)
    
    # 获取所有的 .nii 文件
    imgnii_names = sorted([f for f in os.listdir(nii_imgpath) if f.endswith('.nii')])
    masknii_names = sorted([f for f in os.listdir(nii_maskpath) if f.endswith('.nii')])

    print("🔄 开始将 3D CT 转换为 H5 格式（完整 3D Volume 保存）...\n")

    for img_name in imgnii_names:
        # 1. 智能匹配文件名 (安全策略：img0001.nii -> label0001.nii)
        base_num = img_name.replace('img', '').replace('.nii', '')
        mask_name = f"label{base_num}.nii"
        
        if mask_name not in masknii_names:
            print(f"⚠️ 找不到 {img_name} 对应的标签 {mask_name}，已跳过。")
            continue

        # 2. 读取 3D 图像和标签
        sitk_img = sitk.ReadImage(os.path.join(nii_imgpath, img_name))
        img = sitk.GetArrayFromImage(sitk_img)  # 这里的 img 直接是 3D 数组 (Z, Y, X)

        sitk_mask = sitk.ReadImage(os.path.join(nii_maskpath, mask_name))
        mask = sitk.GetArrayFromImage(sitk_mask) # 这里的 mask 直接是 3D 数组 (Z, Y, X)

        print(f"📦 处理数据: {img_name} & {mask_name}")
        print(f"   - Image | shape: {img.shape}, dtype: {img.dtype}, max: {img.max():.3f}, min: {img.min():.3f}")
        print(f"   - Mask  | shape: {mask.shape}, dtype: {mask.dtype}, max: {mask.max()}, min: {mask.min()}")

        # 3. 构造保存的 H5 文件名 (例如: case0001.h5)
        h5_filename = f"case{base_num}.h5"
        savepath = os.path.join(h5_save_path, h5_filename)

        # 4. 写入 h5 文件
        # 使用 with 语句，写入完毕后会自动调用 f.close()，防止内存泄漏或文件损坏
        with h5py.File(savepath, 'w') as f:
            f.create_dataset('image', data=img, compression="gzip")
            f.create_dataset('label', data=mask, compression="gzip")
            
        print(f"   ✅ 成功保存至: {savepath}\n")

    print("🎉 所有 3D 数据 H5 转换完成！可以用于 3D 半监督网络(如 V-Net) 的训练了。")

if __name__ == "__main__":
    main()