import os

image_path = "E:/FLARE-Lab/FLARE25/Validation-Public-Images"
mask_path  = "E:/FLARE-Lab/FLARE25/Validation-Public-Labels"

def main():
    image_list = os.listdir(image_path)
    mask_list  = os.listdir(mask_path)
    image_list.sort()
    mask_list.sort()
    i = 51
    for imgname in image_list:
        index = image_list.index(imgname)
        maskname = mask_list[index]
        new_imgname = f"FLARE22_Tr_{i:04d}_0000.nii.gz"
        new_maskname = f"FLARETs_{i:04d}.nii.gz"

        os.rename(os.path.join(image_path, imgname), os.path.join(image_path, new_imgname))
        os.rename(os.path.join(mask_path, maskname), os.path.join(mask_path, new_maskname))
        print(f"Renamed: {imgname} -> {new_imgname}, {maskname} -> {new_maskname}")
        i += 1

    print(f"Total images renamed: {i}")

if __name__ == "__main__":
    main()
