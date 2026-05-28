import os
import random

def main(mode):
    # 1. get splits and path
    random.seed(2025)

    if mode == 'tl':
        ids1 = list(range(1, 51)) # flare
        ids2 = list(range(51, 101)) # flare
        # names_file = sorted(os.listdir("E:/AbdomenCT-1K/test1/labeled_h5")) # AK    
        # ids = [int(name.split('_')[1]) for name in names_file if name.split('.')[-1] == 'h5'] # AK    
        # print(f"Total labeled h5 files num: {len(ids)}") # AK
        print(f"Total labeled h5 files num: {len(ids1)+len(ids2)}")      
        # group1 = random.sample(ids, 50) # train_l AK
        # group2 = random.sample([i for i in ids if i not in group1], 50)
        # group_test = random.sample([i for i in ids if i not in group1 and i not in group2], 50)
        # group3 = random.sample([i for i in ids if i not in group1 and i not in group2 and i not in group_test], 738) # train_u
        # group4 = [i for i in ids if i not in group1 and i not in group2 and i not in group_test and i not in group3]
        group1_1, group1_2 = random.sample(ids1, 10), random.sample(ids2, 10) # test
        group2_1, group2_2 = random.sample([i for i in ids1 if i not in group1_1], 10), random.sample([i for i in ids2 if i not in group1_2], 10) # val
        group3_1, group3_2 = random.sample([i for i in ids1 if i not in group1_1 and i not in group2_1], 30), random.sample([i for i in ids2 if i not in group1_2 and i not in group2_2], 30) # val
        
        # print(f"Group1: {group1}, len   : {len(group1)}")
        # print(f"Group2: {group2}, len   : {len(group2)}")
        # print(f"Group3: {group3}, len   : {len(group3)}")
        # print(f"Group_test: {group_test}, len   : {len(group_test)}")
        print(f"Group1_1: {group1_1}, Group1_2: {group1_2}")
        print(f"Group2_1: {group2_1}, Group2_2: {group2_2}")
        print(f"Group3_1: {group3_1}, Group3_2: {group3_2}")
        path = "E:/FLARE-Lab/FLARE25/test4/labeled_h5" # flare
        # path = "E:/AbdomenCT-1K/test1/labeled_h5" # AK
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
            # name_id = int(name.split('_')[1]) # AK
            name_id = int(name.split('_')[2]) # flare
            # if name_id in group1:
            #     with open("E:/AbdomenCT-1K/test1/train_l.txt", 'a') as f:
            #         f.write(name.split('.h5')[0])
            #         f.write('\n')
            # elif name_id in group2:
            #     with open("E:/AbdomenCT-1K/test1/val.txt", 'a') as f:
            #         f.write(name.split('.h5')[0])
            #         f.write('\n')
            # elif name_id in group3:
            #     with open("E:/AbdomenCT-1K/test1/train_u.txt", 'a') as f:
            #         f.write(name.split('.h5')[0])
            #         f.write('\n')
            # elif name_id in group_test:
            #     with open("E:/AbdomenCT-1K/test1/test.txt", 'a') as f:
            #         f.write(name.split('.h5')[0])
            #         f.write('\n')
            # elif name_id in group4:
            #     with open("E:/AbdomenCT-1K/test1/notuse.txt", 'a') as f:
            #         f.write(name.split('.h5')[0])
            #         f.write('\n')
            if name_id in group1_1:
                with open("E:/FLARE-Lab/FLARE25/latest-txt/test.txt", 'a') as f:
                    f.write(name.split('.h5')[0])
                    f.write('\n')
            elif name_id in group1_2:
                with open("E:/FLARE-Lab/FLARE25/latest-txt/test.txt", 'a') as f:
                    f.write(name.split('.h5')[0])
                    f.write('\n')
            elif name_id in group2_2:
                with open("E:/FLARE-Lab/FLARE25/latest-txt/val.txt", 'a') as f:
                    f.write(name.split('.h5')[0])
                    f.write('\n')
            elif name_id in group2_1:
                with open("E:/FLARE-Lab/FLARE25/latest-txt/val.txt", 'a') as f:
                    f.write(name.split('.h5')[0])
                    f.write('\n')
            elif name_id in group3_1:
                with open("E:/FLARE-Lab/FLARE25/latest-txt/train_l.txt", 'a') as f:
                    f.write(name.split('.h5')[0])
                    f.write('\n')
            elif name_id in group3_2:
                with open("E:/FLARE-Lab/FLARE25/latest-txt/train_l.txt", 'a') as f:
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