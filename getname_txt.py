import os

def main():
    h5path = 'E:/FLARE-Lab/FLARE25/test2/train_l_h5'
    savepath = 'E:/FLARE-Lab/FLARE25/test2/train_l.txt'

    h5names = sorted(os.listdir(h5path))
    with open(savepath, 'a') as f:
        for h5name in h5names:
            f.write(h5name.split('.')[0] + '\n')

if __name__ == "__main__":
    main()