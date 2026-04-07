import os
import SimpleITK as sitk
import numpy as np

def main():
    img_path = "D:/data/Synapse/img"
    mask_path = "D:/data/Synapse/label"
    
    # 获取所有的 .nii 文件 (加一个 if 条件防止读取到隐藏文件)
    niiimgnames = sorted([f for f in os.listdir(img_path) if f.endswith('.nii')])
    niimasknames = sorted([f for f in os.listdir(mask_path) if f.endswith('.nii')])

    unlabeled_names = []
    labeled_pairs = []

    # 1. 安全地划分有标签和无标签数据
    for img_name in niiimgnames:
        # 提取数字标识：将 "img0001.nii" 变成 "0001"
        base_num = img_name.replace('img', '').replace('.nii', '')
        
        # 拼接出期望的 mask 名字："label0001.nii"
        expected_mask_name = f"label{base_num}.nii"

        # 检查这个 mask 是否存在于 mask 文件夹中
        if expected_mask_name in niimasknames:
            labeled_pairs.append((img_name, expected_mask_name))
        else:
            unlabeled_names.append(img_name)

    # 2. 打印无标签数据信息
    print(f"Total unlabeled images num: {len(unlabeled_names)}")
    for name in unlabeled_names:
        print(f"Unlabeled: {name}")

    print("-" * 40)
    print(f"Total labeled pairs num: {len(labeled_pairs)}")

    # 3. 遍历【有标签】的数据对，进行读取
    # 这样写绝对不会出现索引越界和张冠李戴的问题
    for img_name, mask_name in labeled_pairs:
        imgfilepath = os.path.join(img_path, img_name)
        maskfilepath = os.path.join(mask_path, mask_name)

        # 读取 CT 图像
        img_sitk = sitk.ReadImage(imgfilepath, sitk.sitkFloat32)
        image = sitk.GetArrayFromImage(img_sitk).astype(np.float32)

        # 读取 标签图像
        mask_sitk = sitk.ReadImage(maskfilepath, sitk.sitkInt32)
        mask = sitk.GetArrayFromImage(mask_sitk).astype(np.float32)

        print(f"✅ Loaded: {img_name} & {mask_name} | img spacing: {img_sitk.GetSpacing()}, size: {img_sitk.GetSize()}")

if __name__ == "__main__":
    main()