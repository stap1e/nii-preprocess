import os, random

# 300 CT, 60 MRI; 区分后进行preprocess
def main(mode):
    # 1. fix the seed for all datasets
    random.seed(2025)

    # 2. get splits
    if mode == 'tl':
        path = "E:/AMOS-Lab/amos22_labeled/images" # AK
        names_file = sorted(os.listdir(path))
        names_file = [name.split('.nii')[0] for name in names_file]
        ids = [int(name.split('_')[1]) for name in names_file]
        group1 = random.sample(ids, 30) # test
        group2 = random.sample([i for i in ids if i not in group1], 30) # val
        group3 = random.sample([i for i in ids if i not in group1 and i not in group2], 240) # train_l
        group4 = random.sample([i for i in ids if i not in group1 and i not in group2 and i not in group3], 60) # notuse
        print(f"Group1: {group1}, len   : {len(group1)}")
        print(f"Group2: {group2}, len   : {len(group2)}")
        print(f"Group3: {group3}, len   : {len(group3)}")
        print(f"Group4: {group4}, len   : {len(group4)}")
    elif mode == 'tu':
        path = "E:/AMOS-Lab/unlabeled" # AK
    else:
        raise ValueError(f"error mode: {mode}")
    
    # 3. get name to txt file
    names  = os.listdir(path)
    for name in names:
        if name.split('.nii')[-1] != '.gz':
            continue
        name = name.split('.nii')[0]
        if mode != 'tu':
            name_id = int(name.split('_')[1]) # AK
            if name_id in group1:
                with open("E:/AMOS-Lab/amos22_labeled/test.txt", 'a') as f:
                    f.write(name.split('.nii.gz')[0])
                    f.write('\n')
            elif name_id in group2:
                with open("E:/AMOS-Lab/amos22_labeled/val.txt", 'a') as f:
                    f.write(name.split('.nii.gz')[0])
                    f.write('\n')
            elif name_id in group3:
                with open("E:/AMOS-Lab/amos22_labeled/train_l.txt", 'a') as f:
                    f.write(name.split('.nii.gz')[0])
                    f.write('\n')
            elif name_id in group4:
                with open("E:/AMOS-Lab/notuse.txt", 'a') as f:
                    f.write(name.split('.h5')[0])
                    f.write('\n')
            else:
                raise ValueError(f"here are an error id: {name_id}")
        else:
            with open("E:/AMOS-Lab/unlabeled/train_u.txt", 'a') as f:
                f.write(name.split('.nii.gz')[0])
                f.write('\n')

if __name__ == '__main__':
    mode = 'tu' # You need to fill a string among "tl", "tu"
    main(mode)