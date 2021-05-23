import numpy as np
import torch
from torchvision import transforms
import cv2
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

'''
加速度计数模转换
The accelerometer returns values between 0 and 1024 corresponding to values between -3 [g] to +3 [g] like on the real robot
'''
def accDA(accData):
    accData = np.array(accData)
    return 6 * 9.8 * accData/1024  - 3 * 9.8

'''
加速度计积分更新速度
传入当前速度，仿真周期间隔(ms)以及加速度计传感器句柄
'''
def updateVelocity(velocity,mTimeStep,mAccelerometer):
    accD = mAccelerometer.getValues() #数字信号
    accA = accDA(accD) #模拟信号
    #print(accA)
    velocity += accA * mTimeStep /1000
    return velocity


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
if 1:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def load_model(model_path):
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
        output = model(img.to(device)).cpu().numpy().flatten()
    prob = softmax(output)
    pred = np.argmax(output).item()
    return pred, prob[pred]

def call_segmentor(img,model):
    img = transform(img)
    img = torch.unsqueeze(img, 0)
    with torch.no_grad():
        output = model(img.to(device)).cpu().numpy()
    res = output.reshape(120,160)
    res_max = min(np.max(res),-5)
    res_min = max(np.min(res),-10)
    gray = np.clip(255*(res-res_min)/(res_max-res_min),0,255)
    gray = gray.astype(np.uint8)
    _, binary = cv2.threshold(gray,200,255,cv2.THRESH_BINARY)
    return binary


'''
HSV空间颜色检测，用于检测蓝色的obstacle，即需要翻越的那个坎
'''
def obstacleDetect(img):
    h,w = img.shape[:2]
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    blue1 = np.array([110,250,0]) 
    blue2 = np.array([130,255,255]) 
    mask=cv2.inRange(hsv,blue1,blue2)
    contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 80,-1 # 默认在视野之外，用于调速，没看见（为负）则可以快速走
    # 找面积最大的轮廓
    area = [cv2.contourArea(contour) for contour in contours]
    max_idx = np.argmax(np.array(area))
    target = contours[max_idx]
    x,y,w,h = cv2.boundingRect(target)
    return int(x+w/2), int(y+h/2)

'''
第一次转角的程序，通过HSV分割赛道，然后计算赛道中心线，转动到中心线与视野垂直为止
'''
import scipy.linalg as la
def cornerTurn(img):
    h,w = img.shape[:2]
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    yellow1 = np.array([15,80,0]) 
    yellow2 = np.array([25,170,255]) 
    mask=cv2.inRange(hsv,yellow1,yellow2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel,iterations=3) # 闭运算：先膨胀后腐蚀，用来连接被误分为许多小块的对象

    centerLine = []
    for i in range(h):
        tmp = mask[i,:]
        white = np.where(tmp==255)[0]
        if white.size == 0:
            continue
        mid = int((white[0]+white[-1])/2)
        #mask[i,mid] = 0
        centerLine.append([i,mid])
    
    centerLine = np.array(centerLine)
    if centerLine.size > 0:
        x,y = centerLine[:,0], centerLine[:,1]
        A = np.vstack([x**0, x**1])
        sol, r, rank, s = la.lstsq(A.T, y)
        return sol[1]
    else:
        return 1000


'''
HSV空间颜色检测，用于检测黑色的门，检测出两个柱子的中间位置用于配准
'''
def detectDoor(img):
    h,w = img.shape[:2]
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    black1 = np.array([0,0,0]) 
    black2 = np.array([1,1,1])
    mask=cv2.inRange(hsv,black1,black2)
    contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) < 2:
        return -1 # 理论上不应该有这种情况才对
    # 找面积最大的两个轮廓
    area = [cv2.contourArea(contour) for contour in contours]
    idxSorted = np.argsort(np.array(area))

    target = contours[idxSorted[-1]]
    x1,y1,w1,h1 = cv2.boundingRect(target)

    target = contours[idxSorted[-2]]
    x2,y2,w2,h2 = cv2.boundingRect(target)
    
    # center1 = min(int(x1+w1/2),int(x2+w2/2))
    # center2 = max(int(x1+w1/2),int(x2+w2/2))
    # print(center1,center2)

    return int(x1/2+w1/4+x2/2+w2/4)
    