import torch
from torchvision import transforms
import numpy as np
import os
from PIL import Image
from dataset import train_dataloader,val_dataloader


device = torch.device("cpu")
model = torch.load('./model/1.pth').to(device)
model.eval()

# criterion = torch.nn.CrossEntropyLoss()

# for i, data in enumerate(val_dataloader):
#     inputs, labels = data
#     inputs, labels = inputs.to(device), labels.to(device)
#     with torch.no_grad():
#         outputs = model(inputs)
#     pred = np.argmax(outputs.numpy(),axis=1)
#     gt = labels.numpy()
#     print(pred)
#     print(gt)




transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


img = Image.open('data/valid/1/600.png').convert('RGB')
img = transform(img)
img = torch.unsqueeze(img, 0)
with torch.no_grad():
    output = model(img)
pred = np.argmax(output.numpy(),axis=1)
print(pred)