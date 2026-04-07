import os
import numpy as np
import SimpleITK as sitk
from skimage.transform import resize

def preprocess_synapse():
    # ==========================================
    # 1. 配置路径与预处理参数
    # ==========================================
    # 原始数据路径 (只读)
    img_in_dir = "D:/data/Synapse/img"
    mask_in_dir = "D:/data/Synapse/label"
    
    # 预处理后保存的全新路径 (写入)
    out_base_dir = "D:/data/Synapse_preprocessed"
    img_out_dir = os.path.join(out_base_dir, "img")
    mask_out_dir = os.path.join(out_base_dir, "label")
    
    # 创建输出文件夹
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(mask_out_dir, exist_ok=True)

    # 参数设置 (参考 AMOS 和 Synapse 的常用设定)
    # Synapse 腹部 CT 常用的窗宽窗位是 [-125, 275]，如果你想严格保持 AMOS 的逻辑，可以改为 [-40, 325]
    CLIP_MIN, CLIP_MAX = -125.0, 275.0 
    TARGET_SPACING = [1.25, 1.25, 2.5] # 目标物理分辨率 [X, Y, Z]

    # ==========================================
    # 2. 获取并匹配文件列表
    # ==========================================
    nii_img_names = sorted([f for f in os.listdir(img_in_dir) if f.endswith('.nii')])
    nii_mask_names = sorted([f for f in os.listdir(mask_in_dir) if f.endswith('.nii')])

    labeled_pairs = []
    for img_name in nii_img_names:
        base_num = img_name.replace('img', '').replace('.nii', '')
        expected_mask = f"label{base_num}.nii"
        if expected_mask in nii_mask_names:
            labeled_pairs.append((img_name, expected_mask))

    print(f"🔍 扫描到 {len(labeled_pairs)} 对匹配的有标签数据，准备开始预处理...\n")

    # ==========================================
    # 3. 执行批量预处理
    # ==========================================
    for img_name, mask_name in labeled_pairs:
        print(f"{'='*15} Processing {img_name} & {mask_name} {'='*15}")
        
        img_path = os.path.join(img_in_dir, img_name)
        mask_path = os.path.join(mask_in_dir, mask_name)

        # --- 读取数据 ---
        img_sitk = sitk.ReadImage(img_path, sitk.sitkFloat32)
        mask_sitk = sitk.ReadImage(mask_path, sitk.sitkInt32)

        # 异常数据跳过检测
        if len(img_sitk.GetDirection()) != 9:
            print(f"⚠️ 跳过 {img_name}: 方向矩阵长度异常")
            continue

        image_array = sitk.GetArrayFromImage(img_sitk).astype(np.float32)
        mask_array = sitk.GetArrayFromImage(mask_sitk).astype(np.uint8)

        # --- 步骤 A: 图像灰度截断与归一化 (仅对 Image 进行) ---
        # 1. 截断 (Clipping)
        np_img = np.clip(image_array, CLIP_MIN, CLIP_MAX)
        
        # 2. Z-score 零均值单位方差 (Whitening)
        mean, std = np.mean(np_img), np.std(np_img)
        np_img = (np_img - mean) / (std + 1e-8)
        
        # 3. Min-Max 缩放到 [0, 1] 区间
        minimum, maximum = np.min(np_img), np.max(np_img)
        np_img = (np_img - minimum) / (maximum - minimum + 1e-8)

        # --- 步骤 B: 物理空间插值重采样 (Resize) ---
        img_spacing = img_sitk.GetSpacing() # (X, Y, Z)
        img_size = img_sitk.GetSize()       # (X, Y, Z)

        # 计算新的 Shape
        out_size_x = round(img_size[0] * img_spacing[0] / TARGET_SPACING[0])
        out_size_y = round(img_size[1] * img_spacing[1] / TARGET_SPACING[1])
        out_size_z = round(img_size[2] * img_spacing[2] / TARGET_SPACING[2])
        
        # 注意：skimage 的 resize 接受的 shape 顺序必须是 (Z, Y, X)
        target_shape_zyx = (out_size_z, out_size_y, out_size_x)

        # 执行 Resize (Image 用三次插值 order=3，Mask 必须用最近邻 order=0 防止标签改变)
        img_resized = resize(np_img, output_shape=target_shape_zyx, order=3, mode='edge', anti_aliasing=False)
        mask_resized = resize(mask_array, output_shape=target_shape_zyx, order=0, anti_aliasing=False)

        # --- 步骤 C: 重新打包为 SimpleITK 对象并保存 ---
        # 转换回 SITK 对象
        final_img_sitk = sitk.GetImageFromArray(img_resized.astype(np.float32))
        final_mask_sitk = sitk.GetImageFromArray(mask_resized.astype(np.uint8))

        # 继承原始物理坐标信息，并更新 Spacing
        for sitk_obj in [final_img_sitk, final_mask_sitk]:
            sitk_obj.SetSpacing(TARGET_SPACING)
            sitk_obj.SetDirection(img_sitk.GetDirection())
            sitk_obj.SetOrigin(img_sitk.GetOrigin())

        # 保存到新文件夹
        sitk.WriteImage(final_img_sitk, os.path.join(img_out_dir, img_name))
        sitk.WriteImage(final_mask_sitk, os.path.join(mask_out_dir, mask_name))
        
        print(f"✅ 完成! Size: {img_size} -> {final_img_sitk.GetSize()[::-1]} (Z,Y,X)")

    print(f"\n🎉 所有预处理任务结束！预处理后的数据已安全存放在: {out_base_dir}")

if __name__ == "__main__":
    preprocess_synapse()