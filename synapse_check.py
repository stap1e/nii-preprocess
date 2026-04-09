import os
import numpy as np
import nibabel as nib
from pathlib import Path
from collections import defaultdict

def check_nii_unique_values_with_names(dataset_dir):
    """
    遍历指定目录下所有的 .nii 和 .nii.gz 文件，统计全局 Unique 值，
    并打印出每种独特值组合对应的具体文件名。
    """
    dataset_path = Path(dataset_dir)
    nii_files = list(dataset_path.rglob("*.nii")) + list(dataset_path.rglob("*.nii.gz"))
    
    if not nii_files:
        print("未找到任何 .nii 或 .nii.gz 文件，请检查路径！")
        return

    print(f"共找到 {len(nii_files)} 个 NIfTI 文件。开始读取和统计...\n")

    global_unique_values = set()
    
    # 【关键修改】：将原来的 defaultdict(int) 改为 defaultdict(list)
    # 这样键依然是 unique 值的组合，但值变成了拥有该组合的“文件名列表”
    value_combinations_files = defaultdict(list)

    for i, file_path in enumerate(nii_files):
        try:
            img = nib.load(str(file_path))
            data = np.asanyarray(img.dataobj) 
            unique_vals = np.unique(data)
            
            global_unique_values.update(unique_vals)
            
            val_tuple = tuple(unique_vals.tolist())
            # 【关键修改】：将当前文件名追加到对应的列表里
            value_combinations_files[val_tuple].append(file_path.name)
            
            if (i + 1) % 50 == 0:
                print(f"  已处理 {i + 1} / {len(nii_files)} 个文件...")
                
        except Exception as e:
            print(f"读取文件出错 {file_path.name}: {e}")

    # --- 打印最终统计结果 ---
    print("\n" + "="*60)
    print("【全局唯一像素值 (Global Unique Values)】:")
    print(sorted(list(global_unique_values)))
    
    print("\n【像素值组合统计及具体文件名】:")
    # 按照拥有该组合的文件数量降序排列
    sorted_combinations = sorted(value_combinations_files.items(), key=lambda x: len(x[1]), reverse=True)
    
    for vals, files in sorted_combinations:
        count = len(files)
        print(f"  包含像素值 {list(vals)} 的 NIfTI 数量: {count} 个")
        
        # 智能打印策略：如果这种组合的文件很少（比如你遇到的 2 个），就把名字全打印出来
        if count <= 5:
            for f in files:
                print(f"    -> 异常/少数派文件: {f}")
        # 如果文件很多（比如正常的 28 个），只打印前 2 个作为示例，避免刷屏
        else:
            print(f"    -> 示例文件: {files[0]}, {files[1]} ... (省略其余 {count-2} 个)")
            
    print("="*60)

# ==========================================
# 运行入口
# ==========================================
if __name__ == "__main__":
    dataset_directory = "D:\data\Synapse_preprocessed\label"
    check_nii_unique_values_with_names(dataset_directory)
