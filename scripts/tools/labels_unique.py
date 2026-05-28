import h5py, os, torch
import SimpleITK as sitk

def main():
    # label_path = "E:/FLARE-Lab/FLARE25/test_labels" # flare
    label_path = "E:/AbdomenCT-1K/test1/labels" # AK
    labelnii_names = sorted(os.listdir(label_path)) 

    for i, label_name in enumerate(labelnii_names):
        sitk_label = sitk.ReadImage(os.path.join(label_path, label_name))
        label = sitk.GetArrayFromImage(sitk_label)
        label = torch.from_numpy(label).long()
        print(f"{(i+1):04d}-th, file nme: {label_name},  test label unique: {label.unique().cpu().tolist()}, shape: {label.shape}")

if __name__ == '__main__':
    main()