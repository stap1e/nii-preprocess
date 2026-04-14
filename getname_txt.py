import os

def main():
    h5path = "E:/FLARE-Lab/FLARE25/test4/unlabeled_h5_all"
    savepath = 'E:/FLARE-Lab/FLARE25/test4/train_u.txt'

    h5names = sorted(os.listdir(h5path))
    with open(savepath, 'a') as f:
        for h5name in h5names:
            f.write(h5name.split('.')[0] + '\n')

if __name__ == "__main__":
    main()