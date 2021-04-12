import os
import glob
import random


if __name__ == '__main__':
    traindata_path = './data/train'
    labels = os.listdir(traindata_path)

    for index, label in enumerate(labels):
        imglist = glob.glob(os.path.join(traindata_path,label, '*.png'))
        random.shuffle(imglist)

        with open('./data/train.txt', 'a')as f:
            for img in imglist:
                f.write(img + ' ' + str(index))
                f.write('\n')
    

    validdata_path = './data/valid'
    labels = os.listdir(validdata_path)

    for index, label in enumerate(labels):
        imglist = glob.glob(os.path.join(validdata_path,label, '*.png'))
        random.shuffle(imglist)

        with open('./data/valid.txt', 'a')as f:
            for img in imglist:
                f.write(img + ' ' + str(index))
                f.write('\n')