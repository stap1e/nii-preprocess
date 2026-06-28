import os
from collections import Counter

import nibabel as nib
import numpy as np


def diagnose_nifti_directory(directory_path):
    print(f"[{'='*15} 开始诊断 {'='*15}]")
    print(f"数据目录: {directory_path}\n")

    nii_files = sorted(
        os.path.join(directory_path, filename)
        for filename in os.listdir(directory_path)
        if filename.endswith((".nii", ".nii.gz"))
    )

    if not nii_files:
        print("未找到 .nii 或 .nii.gz 文件")
        return

    print(f"共找到 {len(nii_files)} 个 NIfTI 文件")
    print(f"基准文件: {nii_files[0]}\n")

    try:
        base_img = nib.load(nii_files[0])
    except Exception as e:
        print(f"读取基准文件失败，请检查路径: {e}")
        return

    base_ornt = nib.aff2axcodes(base_img.affine)
    base_spacing = np.round(base_img.header.get_zooms(), 4)
    base_shape = base_img.shape
    base_dtype = base_img.get_data_dtype()

    issues_found = 0
    warning_found = 0
    orientation_counter = Counter()
    spacing_counter = Counter()
    shape_counter = Counter()
    dtype_counter = Counter()
    failed_files = []
    orientation_mismatch_files = []
    spacing_mismatch_files = []
    shape_mismatch_files = []
    dtype_mismatch_files = []

    for file_path in nii_files:
        print("-" * 80)
        print(f"检查文件: {file_path}")

        try:
            img = nib.load(file_path)
        except Exception as e:
            print(f"   ❌ 读取文件失败: {e}")
            issues_found += 1
            failed_files.append(file_path)
            continue

        # 1. 检查坐标系方向 (Orientation) - 这是导致“左右颠倒”的元凶
        ornt = nib.aff2axcodes(img.affine)
        orientation_counter[ornt] += 1
        print("1. 坐标系方向 (Orientation):")
        print(f"   - 当前文件: {ornt}")
        print(f"   - 基准文件: {base_ornt}")
        if ornt != base_ornt:
            print("   ❌ [警告] 坐标系方向不一致！(例如：一个可能从左到右，另一个从右到左)")
            issues_found += 1
            orientation_mismatch_files.append(file_path)
        else:
            print("   ✅ 一致")

        # 2. 检查体素间距 (Voxel Spacing) - 决定了 CNN 看到的器官物理尺度是否一致
        spacing = np.round(img.header.get_zooms(), 4)
        spacing_counter[tuple(spacing)] += 1
        print("2. 体素间距 (Spacing, mm):")
        print(f"   - 当前文件: {spacing}")
        print(f"   - 基准文件: {base_spacing}")
        if not np.allclose(spacing, base_spacing, atol=1e-3):
            print("   ❌ [警告] 体素间距不一致！建议输入网络前进行重采样 (Resample)。")
            issues_found += 1
            spacing_mismatch_files.append(file_path)
        else:
            print("   ✅ 一致")

        # 3. 检查图像尺寸 (Shape)
        shape = img.shape
        shape_counter[shape] += 1
        print("3. 图像矩阵尺寸 (Shape):")
        print(f"   - 当前文件: {shape}")
        print(f"   - 基准文件: {base_shape}")
        if shape != base_shape:
            print("   ⚠️ [提示] 矩阵尺寸不同（这在医学图像中很常见，通常靠裁剪/Padding解决）。")
            warning_found += 1
            shape_mismatch_files.append(file_path)
        else:
            print("   ✅ 一致")

        # 4. 检查数据类型 (Data Type)
        dtype = img.get_data_dtype()
        dtype_counter[str(dtype)] += 1
        print("4. 数据类型 (Data Type):")
        print(f"   - 当前文件: {dtype}")
        print(f"   - 基准文件: {base_dtype}")
        if dtype != base_dtype:
            print("   ⚠️ [提示] 数据类型不同，注意网络前向传播时的 Cast 操作。")
            warning_found += 1
            dtype_mismatch_files.append(file_path)
        else:
            print("   ✅ 一致")

    print("\n")
    print("=" * 80)
    print("所有项总结:")
    print(f"- 检查文件总数: {len(nii_files)}")
    print(f"- 成功读取文件数: {len(nii_files) - len(failed_files)}")
    print(f"- 读取失败文件数: {len(failed_files)}")
    print(f"- 坐标系方向不一致文件数: {len(orientation_mismatch_files)}")
    print(f"- 体素间距不一致文件数: {len(spacing_mismatch_files)}")
    print(f"- 图像尺寸不同文件数: {len(shape_mismatch_files)}")
    print(f"- 数据类型不同文件数: {len(dtype_mismatch_files)}")
    print(f"- 严重问题总数: {issues_found}")
    print(f"- 提示项总数: {warning_found}")

    print("\n坐标系方向分布:")
    for value, count in orientation_counter.most_common():
        print(f"  {value}: {count}")

    print("\n体素间距分布:")
    for value, count in spacing_counter.most_common():
        print(f"  {value}: {count}")

    print("\n图像尺寸分布:")
    for value, count in shape_counter.most_common():
        print(f"  {value}: {count}")

    print("\n数据类型分布:")
    for value, count in dtype_counter.most_common():
        print(f"  {value}: {count}")

    if failed_files:
        print("\n读取失败文件:")
        for file_path in failed_files:
            print(f"  - {file_path}")

    if orientation_mismatch_files:
        print("\n坐标系方向不一致文件:")
        for file_path in orientation_mismatch_files:
            print(f"  - {file_path}")

    if spacing_mismatch_files:
        print("\n体素间距不一致文件:")
        for file_path in spacing_mismatch_files:
            print(f"  - {file_path}")

    if shape_mismatch_files:
        print("\n图像尺寸不同文件:")
        for file_path in shape_mismatch_files:
            print(f"  - {file_path}")

    if dtype_mismatch_files:
        print("\n数据类型不同文件:")
        for file_path in dtype_mismatch_files:
            print(f"  - {file_path}")

    print("=" * 80)

    if issues_found > 0:
        print(f"🚨 诊断结束：发现 {issues_found} 个严重的不一致问题，需在预处理阶段修复！")
    elif warning_found > 0:
        print(f"⚠️ 诊断结束：未发现严重问题，但发现 {warning_found} 个提示项。")
    else:
        print("🎉 诊断结束：所有数据的底层物理属性高度一致。")


# 测试调用
if __name__ == '__main__':
    diagnose_nifti_directory("F:/FLARE-Lab/FLARE25/test6/imgs")
