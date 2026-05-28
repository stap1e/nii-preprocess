import os, shutil


def main():
    unlabeled_path = "E:/AbdomenCT-1K/test2/labeled_notuse_h5" # AK
    os.makedirs(unlabeled_path, exist_ok=True)
    unlabeled_names_file = "E:/AbdomenCT-1K/test2/notuse.txt" # AK
    labeled_path = "E:/AbdomenCT-1K/test2/labeled_h5" # AK
    index = 0
    with open(unlabeled_names_file, 'r') as f:
        file_names = f.readlines()
    file_names = [item.strip() for item in file_names]
    linu_f_names = sorted(os.listdir(labeled_path)) 
    linu_f_names = [name.split('.h5')[0] for name in linu_f_names]
    names_file = [name for name in linu_f_names if name in file_names]
    for name in names_file:
        index += 1
        destination_path = os.path.join(unlabeled_path, name + '.h5')
        source_path = os.path.join(labeled_path, name + '.h5')
        try:
            shutil.move(source_path, destination_path)
            print(f"成功移动文件：'{name + '.h5'}'")
        except Exception as e:
            print(f"移动文件 '{name + '.h5'}' 时出错：{e}")
    print(f"Total moved files num: {index}")


if __name__ == '__main__':
    main()