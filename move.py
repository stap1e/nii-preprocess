import os, shutil


def main():
    h5_path = "E:/AbdomenCT-1K/test1/labeled_h5" # AK
    unlabeled_names_file = "E:/AbdomenCT-1K/test1/notuse.txt" # AK
    unlabeled_path = "E:/AbdomenCT-1K/test1/labeled_h5_notuse" # AK
    os.makedirs(unlabeled_path, exist_ok=True)
    index = 0
    file_names = sorted(os.listdir(h5_path))
    with open(unlabeled_names_file, 'r') as f:
        names_file = f.readlines()
    names_file = [name.strip() for name in names_file]
    for name in file_names:
        if name.split('.')[-1] != 'h5':
            continue
        if name.split('.h5')[0] in names_file:
            index += 1
            destination_path = os.path.join(unlabeled_path, name)
            source_path = os.path.join(h5_path, name)
            try:
                shutil.move(source_path, destination_path)
                print(f"成功移动文件：'{name}'")
            except Exception as e:
                print(f"移动文件 '{name}' 时出错：{e}")
    print(f"Total moved files num: {index}")


if __name__ == '__main__':
    main()