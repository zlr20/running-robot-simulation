# -*- coding:utf-8 -*-

import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from PIL import Image


class  SelfCustomDataset(Dataset):
    def __init__(self, label_file):
        with open(label_file, 'r') as f:
            self.imgs = list(map(lambda line: line.strip().split(' '), f))
        self.transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, index):
        img_path, label = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        img =self.transform(img)
        return img, torch.from_numpy(np.array(int(label)))
 
    def __len__(self):
        return len(self.imgs)




train_datasets = SelfCustomDataset('./data/train.txt')
train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True, num_workers=1)

val_datasets = SelfCustomDataset('./data/valid.txt')
val_dataloader = torch.utils.data.DataLoader(val_datasets, batch_size=64, shuffle=True, num_workers=1)


##进行数据提取函数的测试
if __name__ =="__main__":
    for images, labels in train_dataloader:
        print(images.shape)