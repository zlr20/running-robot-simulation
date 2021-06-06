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
        return rgb_raw.astype(np.uint8)
    elif mode == 'rgba':
        return rgba_raw
    else:
        return rgba_raw
'''
加载关卡预训练模型
'''
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
def load_model(model_path):
    if torch.cuda.is_available():
        model = torch.load(model_path,map_location='cuda')
    else:
        model = torch.load(model_path,map_location='cpu')
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
    blue1 = np.array([110,250,50]) 
    blue2 = np.array([130,255,255]) 
    mask=cv2.inRange(hsv,blue1,blue2)
    # cv2.imshow('image',mask)
    # cv2.waitKey(0)
    contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 80,-1 # 默认在视野之外，用于调速，没看见（为负）则可以快速走
    # 找面积最大的轮廓
    cnt = max(contours, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(cnt)
    #print(int(x+w/2), int(y+h/2))
    # 有一种特殊情况，球洞的边缘也是这种蓝色
    if cv2.contourArea(cnt) < 2000:
        return 80,-1 # 默认在视野之外，用于调速，没看见（为负）则可以快速走
    return int(x+w/2), int(y+h/2)

'''
第一次转角的程序，通过HSV分割赛道，然后计算赛道中心线，计算斜率
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
        # scipy调用拟合函数，但不知道为啥这么慢
        # A = np.vstack([x**0, x**1])
        # sol, r, rank, s = la.lstsq(A.T, y)

        #直接用(ATA)-1ATY计算
        A = np.vstack([x**0, x**1]).T
        ATA_1 = np.linalg.pinv(np.dot(A.T,A))
        sol = np.dot(ATA_1,A.T)
        sol = np.dot(sol,y.reshape(-1,1))
        
        return sol[1]
    else:
        return 1000

'''
符合条件的黄色小砖块个数计算
'''
def calculateBrickNum(img):

    img = img[90:120][:][:]
    img = cv2.resize(img, (img.shape[1]*4, img.shape[0] * 4), cv2.INTER_NEAREST)
    # img = cv2.GaussianBlur(img, (5, 5), 0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dst = cv2.Canny(gray, 40, 100, apertureSize=3)

    k = np.ones((8, 8), np.uint8)

    close = cv2.dilate(dst, k, iterations=2)
    close = cv2.erode(close,k,iterations=1)
    close=~close

    cnts = cv2.findContours(close.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    count=0
    for cnt in cnts:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        area = cv2.contourArea(box)
        if 1000 < area < 5000:
            count+=1
            cv2.drawContours(close, [cnt], 0, (120, 80, 10), 3)
    # cv2.imshow('image',close)
    # cv2.waitKey(0)
    # print(count)
    return count

if __name__ == '__main__':
    img = cv2.imread('log/keySteps/0黄色砖块分割点.png')
    print(calculateBrickNum(img))
