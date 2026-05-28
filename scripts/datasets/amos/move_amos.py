import os
import SimpleITK as sitk
import numpy as np
from pathlib import Path

def main():
    unlabel_path = "E:/AMOS-Lab/AMOS22/unlabeled_preprocess_new"
    file_names = os.listdir(unlabel_path)
    for name in file_names:
        with open("E:/AMOS-Lab/AMOS22/train_u.txt", 'a') as f:
            f.write(name.split('.nii.gz')[0])
            f.write('\n')
    pass

if __name__ == '__main__':
    main()

