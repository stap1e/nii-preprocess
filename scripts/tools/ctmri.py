import os, shutil
import SimpleITK as sitk
import numpy as np


def main():
    path = "E:/AMOS-Lab/amos22_labeled/labels" # AMOS
    mri_txt = "E:/AMOS-Lab/mri-id.txt" 
    names = os.listdir(path)
    
    # 1. get name
    mri_num = 0
    num = 0
    for name in names:
        img_path = os.path.join(path, name)
        img_sitk = sitk.ReadImage(img_path,  sitk.sitkFloat32)
        img_np = sitk.GetArrayFromImage(img_sitk).astype(np.float32)
        num += 1
        print(f"{num}-th shape: {img_np.shape}, min: {img_np.min()}, max: {img_np.max()}")
        if img_np.min() == 0:
            with open(mri_txt, 'a') as f:
                f.write(name)
                f.write('\n')
            mri_num += 1
    print(f"mri num: {mri_num}")

    # 2. commented the 1. corresponding code
    # mri_path = "E:/AMOS-Lab/amos22_labeled_MRI/labels" # AMOS
    # os.makedirs(mri_path, exist_ok=True)
    # with open(mri_txt, 'r') as f:
    #     mri_names = f.readlines()
    # mri_names = [n.split('\n')[0] for n in mri_names]
    # for mri_name in mri_names:
    #     des_path = os.path.join(mri_path, mri_name)
    #     src_path = os.path.join(path, mri_name)
    #     try:
    #         shutil.move(src_path, des_path)
    #         print(f"成功移动文件：'{mri_name}'")
    #     except Exception as e:
    #         print(f"移动文件 '{mri_name}' 时出错：{e}")

if __name__ == '__main__':
    main()