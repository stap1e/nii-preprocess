import nibabel as nib
import numpy as np

def diagnose_nifti_consistency(file_path_1, file_path_2):
    print(f"[{'='*15} 开始诊断 {'='*15}]")
    print(f"数据 A: {file_path_1}")
    print(f"数据 B: {file_path_2}\n")

    try:
        img1 = nib.load(file_path_1)
        img2 = nib.load(file_path_2)
    except Exception as e:
        print(f"读取文件失败，请检查路径: {e}")
        return

    issues_found = 0

    # 1. 检查坐标系方向 (Orientation) - 这是导致“左右颠倒”的元凶
    ornt1 = nib.aff2axcodes(img1.affine)
    ornt2 = nib.aff2axcodes(img2.affine)
    print("1. 坐标系方向 (Orientation):")
    print(f"   - 数据 A: {ornt1}")
    print(f"   - 数据 B: {ornt2}")
    if ornt1 != ornt2:
        print("   ❌ [警告] 坐标系方向不一致！(例如：一个可能从左到右，另一个从右到左)")
        issues_found += 1
    else:
        print("   ✅ 一致")
    print("-" * 40)

    # 2. 检查体素间距 (Voxel Spacing) - 决定了 CNN 看到的器官物理尺度是否一致
    spacing1 = np.round(img1.header.get_zooms(), 4)
    spacing2 = np.round(img2.header.get_zooms(), 4)
    print("2. 体素间距 (Spacing, mm):")
    print(f"   - 数据 A: {spacing1}")
    print(f"   - 数据 B: {spacing2}")
    if not np.allclose(spacing1, spacing2, atol=1e-3):
        print("   ❌ [警告] 体素间距不一致！建议输入网络前进行重采样 (Resample)。")
        issues_found += 1
    else:
        print("   ✅ 一致")
    print("-" * 40)

    # 3. 检查图像尺寸 (Shape)
    shape1 = img1.shape
    shape2 = img2.shape
    print("3. 图像矩阵尺寸 (Shape):")
    print(f"   - 数据 A: {shape1}")
    print(f"   - 数据 B: {shape2}")
    if shape1 != shape2:
        print("   ⚠️ [提示] 矩阵尺寸不同（这在医学图像中很常见，通常靠裁剪/Padding解决）。")
    else:
        print("   ✅ 一致")
    print("-" * 40)

    # 4. 检查数据类型 (Data Type)
    dtype1 = img1.get_data_dtype()
    dtype2 = img2.get_data_dtype()
    print("4. 数据类型 (Data Type):")
    print(f"   - 数据 A: {dtype1}")
    print(f"   - 数据 B: {dtype2}")
    if dtype1 != dtype2:
        print("   ⚠️ [提示] 数据类型不同，注意网络前向传播时的 Cast 操作。")
    else:
        print("   ✅ 一致")
    print("\n")
    
    # 总结
    if issues_found > 0:
        print(f"🚨 诊断结束：发现 {issues_found} 个严重的不一致问题，需在预处理阶段修复！")
    else:
        print("🎉 诊断结束：这两个数据的底层物理属性高度一致。")

# 测试调用
if __name__ == '__main__':
    diagnose_nifti_consistency("D:/data/invest/AMOS_origin/labeled/amos_0001.nii.gz", "D:/data/invest/AMOS_origin/labeled/amos_0403.nii.gz")