# -*- coding:utf-8 -*-

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import mobilenet_v2
from dataset import train_dataloader,val_dataloader


##创建训练模型参数保存的文件夹
save_folder = './model'
os.makedirs(save_folder, exist_ok=True)


model=mobilenet_v2(pretrained=False,num_classes=2)

device = torch.device("cuda:0")
model = model.to(device)

##定义优化器与损失函数
optimizer = optim.Adam(model.parameters(),lr=0.01)
criterion = nn.CrossEntropyLoss()
#criterion = nn.BCEWithLogitsLoss()
# criterion = LabelSmoothSoftmaxCE()
# criterion = LabelSmoothingCrossEntropy()

print("Start Training...")
for epoch in range(20):
    valid_loss = 0.
    for i, data in enumerate(val_dataloader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        valid_loss += loss.item()
    print(valid_loss/(i+1))

    for i, data in enumerate(train_dataloader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

print("Done Training!")

torch.save(model, './model/1.pth') 