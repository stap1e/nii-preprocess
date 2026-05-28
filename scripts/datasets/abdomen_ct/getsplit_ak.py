import os
import random

def main(mode):
    # 1. get splits and path
    random.seed(2025)

    if mode == 'tl':
        names_file = sorted(os.listdir("E:/AbdomenCT-1K/test2/labeled_h5")) # AK    
        ids = [int(name.split('_')[1]) for name in names_file if name.split('.')[-1] == 'h5'] # AK    
        print(f"Total labeled h5 files num: {len(ids)}") # AK    
        group1 = random.sample(ids, 160) # train_l AK
        group2 = random.sample([i for i in ids if i not in group1], 20) # val
        group3 = random.sample([i for i in ids if i not in group1 and i not in group2], 20) # test
        group4 = random.sample([i for i in ids if i not in group1 and i not in group2 and i not in group3], 738) # train_u
        group5 = list(set(ids) - set(group2) - set(group1) - set(group3) - set(group4)) # not_use
        
        print(f"Group1: {group1}, len   : {len(group1)}")
        print(f"Group2: {group2}, len   : {len(group2)}")
        print(f"Group3: {group3}, len   : {len(group3)}")
        print(f"Group4: {group4}, len   : {len(group4)}")
        print(f"Group5: {group5}, len   : {len(group5)}")
        path = "E:/AbdomenCT-1K/test2/labeled_h5" # AK
    elif mode == 'tu':
        path = "E:/AbdomenCT-1K/test2/unlabeled_h5" # AK
    else:
        raise ValueError(f"error mode: {mode}")

    # 2. get name txt file
    names  = os.listdir(path)
    for name in names:
        if name.split('.')[-1] != 'h5':
            continue
        if mode != 'tu':
            name_id = int(name.split('_')[1]) # AK
            if name_id in group1:
                with open("E:/AbdomenCT-1K/test2/train_l.txt", 'a') as f:
                    f.write(name.split('.h5')[0])
                    f.write('\n')
            elif name_id in group2:
                with open("E:/AbdomenCT-1K/test2/val.txt", 'a') as f:
                    f.write(name.split('.h5')[0])
                    f.write('\n')
            elif name_id in group3:
                with open("E:/AbdomenCT-1K/test2/test.txt", 'a') as f:
                    f.write(name.split('.h5')[0])
                    f.write('\n')
            elif name_id in group4:
                with open("E:/AbdomenCT-1K/test2/train_u.txt", 'a') as f:
                    f.write(name.split('.h5')[0])
                    f.write('\n')
            elif name_id in group5:
                with open("E:/AbdomenCT-1K/test2/notuse.txt", 'a') as f:
                    f.write(name.split('.h5')[0])
                    f.write('\n')
            else:
                raise ValueError(f"here are an error id: {name_id}")
        else:
            with open("E:/AbdomenCT-1K/test2/train_u.txt", 'a') as f:
                f.write(name.split('.h5')[0])
                f.write('\n')

if __name__ == '__main__':
    mode = 'tl' # 'You need to fill a string among "tl", "tu" or "val"'
    main(mode)