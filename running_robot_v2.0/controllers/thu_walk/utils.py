import numpy as np
import torch
from torchvision import transforms

'''
陀螺仪数模转换
The gyroscope returns values between 0 and 1024, corresponding to values between -1600 [deg/sec] and +1600 [deg/sec]
Similar to the values returned by the real robot
'''
def gyroDA(gyroData):
    gyroData = np.array(gyroData)
    return 3.125 * gyroData - 1600


'''
加载关卡预训练模型
'''
def load_model(model_path,gpu=False):
    if gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = torch.load(model_path).to(device)
    model.eval()
    return model


transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

'''
调用关卡预训练模型
'''
def call_model(img,model):
    img = transform(img)
    img = torch.unsqueeze(img, 0)
    with torch.no_grad():
        output = model(img)
    pred = np.argmax(output.numpy(),axis=1).item()
    return pred
