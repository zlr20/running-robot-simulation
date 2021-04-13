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
陀螺仪积分更新角度
传入当前角度，仿真周期间隔(ms)以及陀螺仪传感器句柄
'''
def updateAngle(angle,mTimeStep,mGyro):
	gyroD = mGyro.getValues() #数字信号
	gyroA = gyroDA(gyroD) #模拟信号
	angle += gyroA * mTimeStep /1000
	return angle

def updateAngle2(angle,mTimeStep,mGyro):
	gyroD = mGyro.getValues() #数字信号
	gyroA = gyroDA(gyroD) #模拟信号
	omega = gyroA * mTimeStep /1000
	angle += omega
	return angle,omega
'''
摄像头数据采集
可选模式 rgb rgba gray
'''
def getImage(mCamera,mode='rgb'):
	cameraData = mCamera.getImage()
	mCameraHeight, mCameraWidth = mCamera.getHeight(), mCamera.getWidth()
	rgba_raw = np.frombuffer(cameraData, np.uint8).reshape((mCameraHeight, mCameraWidth, 4))
	if mode == 'rgb':
		rgb_raw = rgba_raw[...,:3]
		return rgb_raw
	elif mode == 'rgba':
		return rgba_raw
	else:
		return rgba_raw
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

def softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x

transform=transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		])
'''
调用关卡预训练模型
'''
def call_classifier(img,model):
	img = transform(img)
	img = torch.unsqueeze(img, 0)
	with torch.no_grad():
		output = model(img).numpy().flatten()
	prob = softmax(output)
	pred = np.argmax(output).item()
	return pred, prob[pred]
