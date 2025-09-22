import os
import random

def main(mode):
    # 1. get splits and path
    random.seed(2025)

    if mode == 'tl':
        # ids = list(range(1, 1001))  # flare 
        names_file = sorted(os.listdir("E:/AbdomenCT-1K/test1/labeled_h5")) # AK    
        ids = [int(name.split('_')[1]) for name in names_file if name.split('.')[-1] == 'h5'] 
        print(f"Total labeled h5 files num: {len(ids)}")      
        group1 = random.sample(ids, 50) # train_l
        group2 = random.sample([i for i in ids if i not in group1], 50) # val
        group_test = random.sample([i for i in ids if i not in group1 and i not in group2], 50)
        group3 = random.sample([i for i in ids if i not in group1 and i not in group2 and i not in group_test], 738) # train_u
        group4 = [i for i in ids if i not in group1 and i not in group2 and i not in group_test and i not in group3]
        print(f"Group1: {group1}, len   : {len(group1)}")
        print(f"Group2: {group2}, len   : {len(group2)}")
        print(f"Group3: {group3}, len   : {len(group3)}")
        print(f"Group_test: {group_test}, len   : {len(group_test)}")
        # path = "E:/FLARE-Lab/FLARE25/test4/train_l_h5" # flare
        path = "E:/AbdomenCT-1K/test1/labeled_h5" # AK
    elif mode == 'val':
        ids = list(range(51, 101))                
        group1 = random.sample(ids, 10)
        group2 = [i for i in ids if i not in group1]
        print("Group1:", group1)
        print("Group2:", group2)
        path = "E:/FLARE-Lab/FLARE25/test4/val_h5"
    elif mode == 'tu':
        path = "E:/FLARE-Lab/FLARE25/test4/unlabeled_h5" # flare
        path = "E:/AbdomenCT-1K/test1/unlabeled_h5" # AK
    else:
        raise ValueError(f"error mode: {mode}")


    # 2. get name txt file
    names  = os.listdir(path)
    for name in names:
        if name.split('.')[-1] != 'h5':
            continue
        if mode != 'tu':
            name_id = int(name.split('_')[1])
            if name_id in group1:
                with open("E:/AbdomenCT-1K/test1/train_l.txt", 'a') as f:
                    f.write(name.split('.h5')[0])
                    f.write('\n')
            elif name_id in group2:
                with open("E:/AbdomenCT-1K/test1/val.txt", 'a') as f:
                    f.write(name.split('.h5')[0])
                    f.write('\n')
            elif name_id in group3:
                with open("E:/AbdomenCT-1K/test1/train_u.txt", 'a') as f:
                    f.write(name.split('.h5')[0])
                    f.write('\n')
            elif name_id in group_test:
                with open("E:/AbdomenCT-1K/test1/test.txt", 'a') as f:
                    f.write(name.split('.h5')[0])
                    f.write('\n')
            elif name_id in group4:
                with open("E:/AbdomenCT-1K/test1/notuse.txt", 'a') as f:
                    f.write(name.split('.h5')[0])
                    f.write('\n')
            else:
                raise ValueError(f"here are an error id: {name_id}")
        else:
            # with open("E:/FLARE-Lab/FLARE25/test4/unlabeled_h5/train_u.txt", 'a') as f:
            #     f.write(name.split('.h5')[0])
            #     f.write('\n')
            with open("E:/AbdomenCT-1K/test1/train_u.txt", 'a') as f:
                f.write(name.split('.h5')[0])
                f.write('\n')

if __name__ == '__main__':
    mode = 'tl' # 'You need to fill a string among "tl", "tu" or "val"'
    main(mode)