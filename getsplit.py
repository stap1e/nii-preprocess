import os
import random

def main(mode):
    # 1. get splits and path
    random.seed(2025)

    if mode == 'tl':
        ids = list(range(1, 101))                
        group1 = random.sample(ids, 10)
        group2 = [i for i in ids if i not in group1]
        print("Group1:", group1)
        print("Group2:", group2)
        path = "E:/FLARE-Lab/FLARE25/test4/train_l_h5"
    elif mode == 'val':
        ids = list(range(51, 101))                
        group1 = random.sample(ids, 10)
        group2 = [i for i in ids if i not in group1]
        print("Group1:", group1)
        print("Group2:", group2)
        path = "E:/FLARE-Lab/FLARE25/test4/val_h5"
    elif mode == 'tu':
        path = "E:/FLARE-Lab/FLARE25/test4/unlabeled_h5"
    else:
        raise ValueError(f"error mode: {mode}")


    # 2. get name txt file
    names  = os.listdir(path)
    for name in names:
        if name.split('.')[-1] != 'h5':
            continue
        if mode != 'tu':
            name_id = int(name.split('_')[2])
            if name_id in group1:
                with open("E:/FLARE-Lab/FLARE25/test4/train_l_h5/train_l.txt", 'a') as f:
                    f.write(name.split('.h5')[0])
                    f.write('\n')
            elif name_id in group2:
                with open("E:/FLARE-Lab/FLARE25/test4/train_l_h5/val.txt", 'a') as f:
                    f.write(name.split('.h5')[0])
                    f.write('\n')
            else:
                raise ValueError(f"here are an error id: {name_id}")
        else:
            with open("E:/FLARE-Lab/FLARE25/test4/unlabeled_h5/train_u.txt", 'a') as f:
                f.write(name.split('.h5')[0])
                f.write('\n')

if __name__ == '__main__':
    mode = 'You need to fill a string among "tl", "tu" or "val"'
    main(mode)