# -*- coding: UTF-8 -*-
# 清华大学Thu-bot队仿真赛代码
# 在本代码中我们定义关卡编号如下：1-上下开合横杆 2-回字陷阱 3-地雷路段 4-翻越障碍物 5-窄桥 6-踢球进洞 7-阶梯 8-水平开合横杆 9-窄门

# 系统库
import os
import sys
import time
from math import radians
import itertools
from typing import Tuple
libraryPath = os.path.join(os.environ.get("WEBOTS_HOME"), 'projects', 'robots', 'robotis', 'darwin-op', 'libraries','python37')
libraryPath = libraryPath.replace('/', os.sep)
sys.path.append(libraryPath)
from managers import RobotisOp2GaitManager, RobotisOp2MotionManager
from controller import Robot, Motion

# 第三方库
import numpy as np
import cv2
import torch
import torchvision

# 自编写库
from utils import *

# 赛道路面材料和对应可能出现的关卡
raceTrackInfo = [
    {'material':'草地','possible_stage':[1],'hsv':{'low':[20,100,100],'high':[55,200,200]}},
    {'material':'灰色','possible_stage':[3,9,6],'hsv':{'low':[35,0,150],'high':[40,20,255]}},
    {'material':'黄色砖块','possible_stage':[3,9,7,6],'hsv':{'low':[15,100,50],'high':[34,255,255]}}, # 楼梯前也是砖块
    {'material':'绿色','possible_stage':[2,5],'hsv':{'low':[35, 43, 35],'high':[90, 255, 255]}},
    {'material':'白色','possible_stage':[3,9,6],'hsv':{'low':[10, 5, 200],'high':[25, 30, 255]}},
    {'material':'蓝色碎花','possible_stage':[8],'hsv':{'low':[100, 10, 100],'high':[150, 80, 200]}},
]

# 注意有三关会随机交换赛道材料，分别是地雷、门和踢球
stageInfo = {
    1 : {'name':'上下开合横杆','trackInfo':[raceTrackInfo[0]]}, 
    2 : {'name':'回字陷阱','trackInfo':[raceTrackInfo[3]]}, 
    3 : {'name':'地雷路段','trackInfo':[raceTrackInfo[1],raceTrackInfo[2],raceTrackInfo[4]]}, 
    4 : {'name':'翻越障碍','trackInfo':[]}, 
    5 : {'name':'窄桥','trackInfo':[raceTrackInfo[3]]}, 
    6 : {'name':'踢球','trackInfo':[raceTrackInfo[1],raceTrackInfo[2],raceTrackInfo[4]]}, 
    7 : {'name':'楼梯','trackInfo':[raceTrackInfo[2]]}, 
    8 : {'name':'水平开合横杆','trackInfo':[raceTrackInfo[5]]}, 
    9 : {'name':'窄门','trackInfo':[raceTrackInfo[1],raceTrackInfo[2],raceTrackInfo[4]]}, 
}



class Walk():
    def __init__(self):
        self.robot = Robot()  # 初始化Robot类以控制机器人
        self.mTimeStep = int(self.robot.getBasicTimeStep())  # 获取当前每一个仿真步所仿真时间mTimeStep
        self.HeadLed = self.robot.getDevice('HeadLed')  # 获取头部LED灯
        self.EyeLed = self.robot.getDevice('EyeLed')  # 获取眼部LED灯
        self.HeadLed.set(0xff0000)  # 点亮头部LED灯并设置一个颜色
        self.EyeLed.set(0xa0a0ff)  # 点亮眼部LED灯并设置一个颜色
        self.mAccelerometer = self.robot.getDevice('Accelerometer')  # 获取加速度传感器
        self.mAccelerometer.enable(self.mTimeStep)  # 激活传感器，并以mTimeStep为周期更新数值

        self.mGyro = self.robot.getDevice('Gyro')  # 获取陀螺仪
        self.mGyro.enable(self.mTimeStep)  # 激活陀螺仪，并以mTimeStep为周期更新数值

        self.positionSensors = []  # 初始化关节角度传感器
        self.positionSensorNames = ('ShoulderR', 'ShoulderL', 'ArmUpperR', 'ArmUpperL',
                                    'ArmLowerR', 'ArmLowerL', 'PelvYR', 'PelvYL',
                                    'PelvR', 'PelvL', 'LegUpperR', 'LegUpperL',
                                    'LegLowerR', 'LegLowerL', 'AnkleR', 'AnkleL',
                                    'FootR', 'FootL', 'Neck', 'Head')  # 初始化各传感器名

        # 获取各传感器并激活，以mTimeStep为周期更新数值
        for i in range(0, len(self.positionSensorNames)):
            self.positionSensors.append(self.robot.getDevice(self.positionSensorNames[i] + 'S'))
            self.positionSensors[i].enable(self.mTimeStep)

        # 控制各个电机
        self.motors = []
        self.motorNames = ('ShoulderR', 'ShoulderL', 'ArmUpperR', 'ArmUpperL',
                           'ArmLowerR', 'ArmLowerL', 'PelvYR', 'PelvYL',
                           'PelvR', 'PelvL', 'LegUpperR', 'LegUpperL',
                           'LegLowerR', 'LegLowerL', 'AnkleR', 'AnkleL',
                           'FootR', 'FootL', 'Neck', 'Head')
        for i in range(0, len(self.motorNames)):
            self.motors.append(self.robot.getDevice(self.motorNames[i]))

        self.mCamera = self.robot.getDevice("Camera")  # 获取并初始化摄像头
        self.mCamera.enable(self.mTimeStep)
        self.mCameraHeight, self.mCameraWidth = self.mCamera.getHeight(), self.mCamera.getWidth()

        self.mKeyboard = self.robot.getKeyboard()  # 初始化键盘读入类
        self.mKeyboard.enable(self.mTimeStep)  # 以mTimeStep为周期从键盘读取

        self.mMotionManager = RobotisOp2MotionManager(self.robot)  # 初始化机器人动作组控制器
        self.mGaitManager = RobotisOp2GaitManager(self.robot, "config.ini")  # 初始化机器人步态控制器

        # 初始化变量组
        self.vX, self.vY, self.vA = 0., 0., 0.  # 前进、水平、转动方向的设定速度（-1，1）
        self.angle = np.array([0., 0., 0.])  # 角度， 由陀螺仪积分获得
        self.velocity = np.array([0., 0., 0.])  # 速度， 由加速度计积分获得

        # 剩余关卡
        self.stageLeft = {1,2,3,5,6,7,8,9}
        # 已经通过关卡
        self.stagePass = []
        # 当前关卡，默认第一关
        self.stageNow = 1
        # 这关转弯
        self.cornerAlready = False
        # 这关之后有蓝色障碍物
        self.obstacleBehind = False
        # 当前道路材料，默认草地
        self.materialInfo = {'material':'草地','possible_stage':[1],'hsv':{'low':[20,100,100],'high':[55,200,200]}}
        # 已经转弯的次数
        self.turnCount = 0

        self.hshhCount = 0

    # 执行一个仿真步，同时每步更新机器人RPY角度
    def myStep(self):
        #print(self.angle[-1])
        ret = self.robot.step(self.mTimeStep)
        self.angle = updateAngle(self.angle, self.mTimeStep, self.mGyro)
        if ret == -1:
            exit(0)
    
    # 输入为等待毫秒数，使机器人等待一段时间
    def wait(self, ms):  
        startTime = self.robot.getTime()
        s = ms / 1000.0  
        while (s + startTime >= self.robot.getTime()):  
            self.myStep()

    # 设置前进、水平、转动方向的速度指令，大小在-1到1之间。
    def setMoveCommand(self, vX=0., vY=0., vA=0.):
        self.vX = np.clip(vX, -1, 1)
        self.mGaitManager.setXAmplitude(self.vX)  # 设置x向速度（直线方向）
        self.vY = np.clip(vY, -1, 1)
        self.mGaitManager.setYAmplitude(self.vY)  # 设置y向速度（水平方向）
        self.vA = np.clip(vA, -1, 1)
        self.mGaitManager.setAAmplitude(self.vA)  # 设置转动速度

    # 检查偏航是否超过指定threshold，若是则修正回0°附近，由eps指定
    def checkIfYaw(self, threshold=7.5, eps=1,initAngle=0):
        if np.abs(self.angle[-1]) > threshold:
            print('Current Yaw is %.3f, begin correction' % self.angle[-1])
            while np.abs(self.angle[-1]) > eps:
                u = -0.02 * self.angle[-1]
                u = np.clip(u, -1, 1)
                self.setMoveCommand(vX=self.vX,vA=u)
                self.mGaitManager.step(self.mTimeStep)
                self.myStep()
            print('Current Yaw is %.3f, finish correction' % self.angle[-1])
        else:
            pass

    # 比赛开始前准备动作，保持机器人运行稳定和传感器读数稳定
    def prepare(self):
        # 仿真一个步长，刷新传感器读数
        self.robot.step(self.mTimeStep)
        # 准备动作
        self.mMotionManager.playPage(9)  # 执行动作组9号动作，初始化站立姿势，准备行走
        self.wait(200)
        # 开始运动
        self.mGaitManager.setBalanceEnable(True)
        self.mGaitManager.start()
        self.wait(200)
        self.setMoveCommand(vX=0., vY=0., vA=0.)
        print('~~~~~~~~~~~准备就绪~~~~~~~~~~~')

    # 键盘控制程序
    def keyBoardControl(self, collet_data=1, rotation=False):
        self.isWalking = True
        ns = itertools.count(0)
        for n in ns:
            self.mGaitManager.setXAmplitude(0.0)  # 前进为0
            self.mGaitManager.setAAmplitude(0.0)  # 转体为0
            key = 0  # 初始键盘读入默认为0
            key = self.mKeyboard.getKey()  # 从键盘读取输入
            if collet_data and n % 33 == 0:
                rgb_raw = getImage(self.mCamera)
                fname = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time())) 
                cv2.imwrite('./tmp/' + fname + '.png', rgb_raw)
            if key == 49:
                pos = self.positionSensors[19].getValue()
                print(f'Head Pos: {pos}')
                self.motors[19].setPosition(np.clip(pos + 0.05, -0.25, 0.4))
            elif key == 50:
                pos = self.positionSensors[19].getValue()
                print(f'Head Pos: {pos}')
                self.motors[19].setPosition(np.clip(pos - 0.05, -0.25, 0.4))
            # elif key == 51:
            #     if collet_data == 1:
            #         collet_data = 0
            #     if collet_data == 0:
            #         collet_data = 1
            if rotation:
                if key == 32:  # 如果读取到空格，则改变行走状态
                    if (self.isWalking):  # 如果当前机器人正在走路，则使机器人停止
                        self.mGaitManager.stop()
                        self.isWalking = False
                        self.wait(200)
                    else:  # 如果机器人当前停止，则开始走路
                        self.mGaitManager.start()
                        self.isWalking = True
                        self.wait(200)
                elif key == 315:  # 如果读取到‘↑’，则前进
                    self.mGaitManager.setXAmplitude(1.0)
                elif key == 317:  # 如果读取到‘↓’，则后退
                    self.mGaitManager.setXAmplitude(-1.0)
                elif key == 316:  # 如果读取到‘←’，则左转
                    self.mGaitManager.setAAmplitude(-0.5)
                elif key == 314:  # 如果读取到‘→’，则右转
                    self.mGaitManager.setAAmplitude(0.5)
            else:
                self.checkIfYaw()
                if key == 32:  # 如果读取到空格，则改变行走状态
                    if (self.isWalking):  # 如果当前机器人正在走路，则使机器人停止
                        self.mGaitManager.stop()
                        self.isWalking = False
                        self.wait(200)
                    else:  # 如果机器人当前停止，则开始走路
                        self.mGaitManager.start()
                        self.isWalking = True
                        self.wait(200)
                elif key == 315:  # 如果读取到‘↑’，则前进
                    self.mGaitManager.setXAmplitude(1.0)
                elif key == 317:  # 如果读取到‘↓’，则后退
                    self.mGaitManager.setXAmplitude(-1.0)
                elif key == 316:  # 如果读取到‘←’，则左转
                    self.mGaitManager.setYAmplitude(-1)
                elif key == 314:  # 如果读取到‘→’，则右转
                    self.mGaitManager.setYAmplitude(1)
            self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
            self.myStep()  # 仿真一个步长

    # 上下开合横杆（成功率100%）
    def stage1(self):
        print("~~~~~~~~~~~上下横杆关开始~~~~~~~~~~~")
        stage1model = load_model('./pretrain/1.pth')
        crossBarDownFlag = False
        goFlag = False
        # 判断何时启动
        ns = itertools.count(0)
        for n in ns:
            self.checkIfYaw()
            if n % 100 == 0:
                rgb_raw = getImage(self.mCamera)
                pred, prob = call_classifier(rgb_raw, stage1model)
                if not crossBarDownFlag:
                    if pred == 1:
                        crossBarDownFlag = True
                        print('横杆已经落下，判别概率为%.3f' % prob)
                    else:
                        print('横杆已经打开，判别概率为%.3f，但需要等待横杆落下先' % prob)
                else:
                    if pred == 0:
                        goFlag = True
                        print('横杆已经打开，判别概率为%.3f，机器人启动' % prob)
                    else:
                        print('横杆已经落下，判别概率为%.3f' % prob)
            if goFlag:
                self.setMoveCommand(vX = 1.)
                break
            self.mGaitManager.step(self.mTimeStep)
            self.myStep()
        del stage1model

        # 走完第一关
        self.motors[19].setPosition(-0.2)
        ns = itertools.count(0)
        for n in ns:
            # 调角速度，保持垂直。这里没有用checkIfYaw，虽然功能相同，但是后者占用的mystep没有计数。
            u = -0.02 * (self.angle[-1])
            u = np.clip(u, -1, 1)
            self.setMoveCommand(vX=1.0,vA=u)
            self.mGaitManager.step(self.mTimeStep)
            self.myStep()
            if n % 5 == 0 and np.abs(self.positionSensors[19].getValue()+0.2)<0.05:
                rgb_raw = getImage(self.mCamera)
                hsv = cv2.cvtColor(rgb_raw,cv2.COLOR_BGR2HSV)
                low = self.materialInfo['hsv']['low']
                high = self.materialInfo['hsv']['high']
                mask=cv2.inRange(hsv[self.mCameraHeight//2:,:],np.array(low),np.array(high))
                road = np.where(mask==255)[0]
                num = len(road)
                # 判定关卡结束条件，当前材料的hsv基本消失在视野中下段。或者行进步数超限1000。
                if num < 500 or n > 1000:
                    break
        self.stageLeft.remove(1)
        print("~~~~~~~~~~~上下横杆关结束~~~~~~~~~~~")

    # 回字陷阱，这版没有用神经网络，纯图像处理，更稳定。（成功率100%）
    def stage2(self):
        print("~~~~~~~~~~~回字陷阱关开始~~~~~~~~~~~") 
        ## 方案一：写死，先走两步，确保进入回字，然后转向90，走到边缘，再转回来。
        # self.setMoveCommand(vX=0.)
        # self.checkIfYaw(threshold=3.0)
        # self.setMoveCommand(vX=1.)
        # ns = itertools.repeat(0,400)
        # for n in ns:
        #     self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
        #     self.myStep()  # 仿真一个步长
        # self.setMoveCommand(vX=0.)
        ## 方案二：动态控制第一次转向，回字离视野底部较近时转向90
        self.motors[19].setPosition(-0.1)
        self.setMoveCommand(vX=1.0)
        ns = itertools.count(0)
        for n in ns:
            u = -0.02 * (self.angle[-1])
            u = np.clip(u, -1, 1)
            self.setMoveCommand(vX=1.0,vA=u)
            self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
            self.myStep()  # 仿真一个步长
            if n % 5 == 0 and np.abs(self.positionSensors[19].getValue()+0.1)<0.05:
                rgb_raw = getImage(self.mCamera)
                hsv = cv2.cvtColor(rgb_raw,cv2.COLOR_BGR2HSV)
                low = self.materialInfo['hsv']['low']
                high = self.materialInfo['hsv']['high']
                mask=cv2.inRange(hsv[self.mCameraHeight//2:,:],np.array(low),np.array(high))
                mask = 255 - mask
                contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                turnFlag = False
                # 由于视野较低，回字出现的特点是，很扁的长方形。当长方形最低点来到视野中间时，可以转向。
                for contour in contours:
                    x,y,w,h = cv2.boundingRect(contour)
                    if w >= 0.7*h and y + h > 30:
                        turnFlag = True
                if turnFlag:break

        # 向右转向90,走到边缘
        ns = itertools.count(0)
        #self.setMoveCommand(vX=1.)
        for n in ns:
            u = -0.02 * (self.angle[-1]+90)
            u = np.clip(u, -1, 1)
            self.setMoveCommand(vX=1.0,vA=u)
            self.mGaitManager.step(self.mTimeStep)
            self.myStep()
            if n % 5 == 0:
                rgb_raw = getImage(self.mCamera)
                hsv = cv2.cvtColor(rgb_raw,cv2.COLOR_BGR2HSV)
                low = self.materialInfo['hsv']['low']
                high = self.materialInfo['hsv']['high']
                mask=cv2.inRange(hsv[self.mCameraHeight//2:,:],np.array(low),np.array(high))
                road = np.where(mask==255)[0]
                num = len(road)
                #看不到绿色，即走到边缘
                if num < 20:
                    break
        # 回正
        self.setMoveCommand(vX=0.)
        self.checkIfYaw()
        # 向前走，走完窄路
        headpos = 0.4
        self.motors[19].setPosition(headpos)
        ns = itertools.count(0)
        for n in ns:
            self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
            self.myStep()  # 仿真一个步长
            u = -0.02 * (self.angle[-1])
            u = np.clip(u, -1, 1)
            self.setMoveCommand(vX=1.0,vA=u)
            if n % 5 == 0 and np.abs(self.positionSensors[19].getValue()-headpos)<0.05:
                rgb_raw = getImage(self.mCamera)
                hsv = cv2.cvtColor(rgb_raw,cv2.COLOR_BGR2HSV)
                low = self.materialInfo['hsv']['low']
                high = self.materialInfo['hsv']['high']
                # 只看右侧一半，防止下一关是绿色窄桥造成干扰
                # fix the bug : 一半还是有可能看见绿色窄桥，故再加20
                mask=cv2.inRange(hsv[self.mCameraHeight//2+20:,self.mCameraWidth//2:],np.array(low),np.array(high))
                road = np.where(mask==255)[0]
                num = len(road)
                # cv2.imshow('mask',mask)
                # cv2.waitKey(0)
                if num < 50 or n >= 330:
                    print(f'走了{n}步') # 限制在330
                    break
        cv2.imwrite('log/keySteps/2窄桥上决策rgb.png',rgb_raw)
        cv2.imwrite('log/keySteps/2窄桥上决策mask一半.png',mask)
        # 走完摘路后先判断一下，前面是不是窄桥，若是，则写死多少步停。否则可以根据绿色动态停。
        mask=cv2.inRange(hsv[self.mCameraHeight//2:,:],np.array(low),np.array(high))
        cv2.imwrite('log/keySteps/2窄桥上决策mask.png',mask)
        road = np.where(mask==255)[0]
        num = len(road)
        if num < 1500:
            print('下一关不是窄桥，执行动态出关程序')
            # 向左转向60前进，动态停
            self.motors[18].setPosition(-radians(30))
            self.motors[19].setPosition(-0.1)
            ns = itertools.count(0)
            for n in ns:
                u = -0.02 * (self.angle[-1]-30)
                u = np.clip(u, -1, 1)
                self.setMoveCommand(vX=1.,vA=u)
                self.mGaitManager.step(self.mTimeStep)
                self.myStep()
                if n % 5 == 0 and np.abs(self.positionSensors[19].getValue()+0.1)<0.05 and np.abs(self.positionSensors[18].getValue()+radians(30))<0.05:
                    rgb_raw = getImage(self.mCamera)
                    hsv = cv2.cvtColor(rgb_raw,cv2.COLOR_BGR2HSV)
                    low = self.materialInfo['hsv']['low']
                    high = self.materialInfo['hsv']['high']
                    mask=cv2.inRange(hsv[self.mCameraHeight//2:,:],np.array(low),np.array(high))
                    road = np.where(mask==255)[0]
                    num = len(road)
                    # cv2.imshow('image',mask)
                    # cv2.waitKey(0)
                    if num < 50 and n > 200:
                        cv2.imwrite('log/keySteps/2出回字时rgb.png',rgb_raw)
                        cv2.imwrite('log/keySteps/2出回字时mask.png',mask)
                        break
        else:
            # 向右转向45前进，写死停
            print('下一关是窄桥，执行固定步长出关程序')
            while np.abs(self.angle[-1]-45) > 1:
                u = -0.02 * (self.angle[-1]-45)
                u = np.clip(u, -1, 1)
                self.setMoveCommand(vX=1.,vA=u)
                self.mGaitManager.step(self.mTimeStep)
                self.myStep()
        self.setMoveCommand(vX=0.)
        self.motors[18].setPosition(0)
        self.motors[19].setPosition(0)
        self.checkIfYaw(threshold=3.0)
        self.stageLeft.remove(2)
        print("~~~~~~~~~~~回字陷阱关结束~~~~~~~~~~~")

    # 地雷路段，没有用神经网络，纯图像处理，更稳定。（成功率90%）
    def stage3(self):
        print("~~~~~~~~~~~地雷关开始~~~~~~~~~~~")
        self.motors[19].setPosition(-0.2)
        self.setMoveCommand(vX=1.0)
        state = '正常行走状态' # -> '避雷转弯状态' -> '避雷斜走状态'  状态转换机
        turnRightCount = 0 # 记录向右转的次数，向左右的次数最好均衡
        ns = itertools.count(0)
        for n in ns:
            self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
            self.myStep()  # 仿真一个步长
            if state == '避雷行走状态':
                count += 1
            if n % 5 == 0 and np.abs(self.positionSensors[19].getValue()+0.2)<0.05:
                rgb_raw = getImage(self.mCamera)
                low =  [0,0,15]
                high = [255,255,255]
                hsv = cv2.cvtColor(rgb_raw,cv2.COLOR_BGR2HSV)
                mask=cv2.inRange(hsv,np.array(low),np.array(high))
                mask = 255 - mask
                contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                dilei = []
                for contour in contours:
                    if cv2.contourArea(contour) > 5:
                        M = cv2.moments(contour)  # 计算第一条轮廓的各阶矩,字典形式
                        center_x = int(M["m10"] / M["m00"])
                        center_y = int(M["m01"] / M["m00"])
                        if 25 < center_x < 135 and center_y > 75 and state=='正常行走状态':
                            dilei.append((center_x,center_y))
                        elif 20 < center_x < 140 and center_y > 60 and state=='避雷转弯状态':
                            dilei.append((center_x,center_y))
                        elif 30 < center_x < 130 and center_y > 80 and state=='避雷行走状态':
                            dilei.append((center_x,center_y))

                #print(state)
                #cv2.imshow('image',mask)
                #cv2.waitKey(0)
                if state == '正常行走状态':
                    # 在正常行走状态下，有雷则要么需要后退，要么需要转向
                    if len(dilei):
                        dilei = sorted(dilei,key = lambda x:x[1],reverse=True)
                        # 第一种情况，常发生在转弯之后，面前的雷离我们太近，直接转弯会撞到，必须要退一段
                        if dilei[0][1] > 100:
                            u = -0.02 * (self.angle[-1])
                            u = np.clip(u, -1, 1)
                            self.setMoveCommand(vX=-1.0,vA=u)
                            continue
                        # 否则判断转向哪里
                        else:
                            # 前方有地雷必须转弯，则扭头看一看左右距离道路边缘情况，再决定往哪里转
                            self.setMoveCommand(vX=0.0)
                            # 左边的路面情况
                            self.motors[18].setPosition(radians(90))
                            while True:
                                self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
                                self.myStep()  # 仿真一个步长
                                self.checkIfYaw()
                                if np.abs(self.positionSensors[18].getValue()-radians(90))<0.05:
                                    leftImg = getImage(self.mCamera)
                                    low = self.materialInfo['hsv']['low']
                                    high = self.materialInfo['hsv']['high']
                                    lefthsv = cv2.cvtColor(leftImg,cv2.COLOR_BGR2HSV)
                                    leftMask=cv2.inRange(lefthsv,np.array(low),np.array(high))
                                    leftMask = cv2.medianBlur(leftMask,3)
                                    if len(np.where(leftMask==255)[0]):
                                        leftValue = 120 - min(np.where(leftMask==255)[0])
                                    else:
                                        leftValue = 1
                                    break
                            self.motors[18].setPosition(radians(-90))
                            while True:
                                self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
                                self.myStep()  # 仿真一个步长
                                self.checkIfYaw()
                                if np.abs(self.positionSensors[18].getValue()+radians(90))<0.05:
                                    rightImg = getImage(self.mCamera)
                                    low = self.materialInfo['hsv']['low']
                                    high = self.materialInfo['hsv']['high']
                                    righthsv = cv2.cvtColor(rightImg,cv2.COLOR_BGR2HSV)
                                    rightMask=cv2.inRange(righthsv,np.array(low),np.array(high))
                                    rightMask = cv2.medianBlur(rightMask,3)
                                    if len(np.where(rightMask==255)[0]):
                                        rightValue = 120 - min(np.where(rightMask==255)[0])
                                    else:
                                        rightValue = 1
                                    break
                            ratio = leftValue / (leftValue + rightValue)
                            # cv2.imshow('left',leftMask)
                            # cv2.imshow('right',rightMask)
                            # cv2.waitKey(0)

                            # 视角回正
                            self.motors[18].setPosition(radians(0))
                            while True:
                                self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
                                self.myStep()  # 仿真一个步长
                                self.checkIfYaw()
                                if np.abs(self.positionSensors[18].getValue())<0.05:
                                    break
                            if ratio<0.3:
                                cv2.imwrite('log/keySteps/3地雷只能右转正视角mask.png',rgb_raw)
                                cv2.imwrite('log/keySteps/3地雷只能右转正视角rgb.png',mask)
                                cv2.imwrite('log/keySteps/3地雷只能右转左视角mask.png',leftMask)
                                cv2.imwrite('log/keySteps/3地雷只能右转左视角rgb.png',leftImg)
                                cv2.imwrite('log/keySteps/3地雷只能右转右视角mask.png',rightMask)
                                cv2.imwrite('log/keySteps/3地雷只能右转右视角rgb.png',rightImg)
                                turnFlag = 'right'
                                print('不允许左转')
                            # 不管雷哪，道路不允许右转，则只能左转
                            elif ratio>0.7:
                                cv2.imwrite('log/keySteps/3地雷只能左转正视角mask.png',rgb_raw)
                                cv2.imwrite('log/keySteps/3地雷只能左转正视角rgb.png',mask)
                                cv2.imwrite('log/keySteps/3地雷只能左转左视角mask.png',leftMask)
                                cv2.imwrite('log/keySteps/3地雷只能左转左视角rgb.png',leftImg)
                                cv2.imwrite('log/keySteps/3地雷只能左转右视角mask.png',rightMask)
                                cv2.imwrite('log/keySteps/3地雷只能左转右视角rgb.png',rightImg)
                                turnFlag = 'left'
                                print('不允许右转')
                            elif 70<dilei[0][0]< 90 and turnRightCount <= -1:
                                turnFlag = 'right'
                                print('之前左转次数多，右转')
                            elif 70<dilei[0][0]< 90 and turnRightCount >= 1:
                                turnFlag = 'left'
                                print('之前右转次数多，左转')
                            elif dilei[0][0] < 80:
                                turnFlag = 'right'
                                print('地雷在左，右转')
                            else:
                                turnFlag = 'left'
                                print('地雷在右，左转')

                        if turnFlag == 'left':
                            turnRightCount -= 1
                            self.setMoveCommand(vA=1.0)
                        elif turnFlag == 'right':
                            turnRightCount += 1
                            self.setMoveCommand(vA=-1.0)
                        state = '避雷转弯状态'
                
                    # 正常行走状态下，若视野无雷，则继续走，并判别关卡结束状态
                    else:
                        u = -0.02 * (self.angle[-1])
                        u = np.clip(u, -1, 1)
                        self.setMoveCommand(vX=1.0,vA=u)
                        # 若本关后接高障碍物，则进stage4()
                        if self.obstacleBehind:
                            ob_x, ob_y = obstacleDetect(rgb_raw)
                            if ob_y > 40:
                                #print(ob_y)
                                self.stage4()
                                self.obstacleBehind = False
                        # 若本关后无障碍物，则正常结束
                        elif self.materialInfo['material'] == '黄色砖块':
                            if self.checkIfPassBrick(rgb_raw): break
                        else:
                            hsv = cv2.cvtColor(rgb_raw,cv2.COLOR_BGR2HSV)
                            low = self.materialInfo['hsv']['low']
                            high = self.materialInfo['hsv']['high']
                            mask=cv2.inRange(hsv[self.mCameraHeight//2:,:],np.array(low),np.array(high))
                            road = np.where(mask==255)[0]
                            num = len(road)
                            if num < 250: break

                elif state == '避雷转弯状态':
                    if np.abs(self.angle[-1])<20:
                        continue # 最少转20度再看
                    elif len(dilei):
                        if turnFlag == 'left':
                            self.setMoveCommand(vA=1.0)
                        elif turnFlag == 'right':
                            self.setMoveCommand(vA=-1.0)
                    else:
                        # cv2.imshow('wulei',mask)
                        # cv2.imshow('wulei2',rgb_raw)
                        # cv2.waitKey(0)
                        self.setMoveCommand(vX=1.0,vA=0.0) # 斜向走，避雷
                        state = '避雷行走状态'
                        count = 0
                
                else:# '避雷行走状态'
                    # 避雷斜向走过程中遇到雷，停
                    if len(dilei) or count >= 250:
                        self.setMoveCommand(vX=0.0)
                        self.checkIfYaw(eps=5)
                        state = '正常行走状态'
                        if len(dilei):
                            print('停止避雷行走，原因是前方有雷')
                        else:
                            print('停止避雷行走，原因是达到上限')

        self.stageLeft.remove(3)
        print("~~~~~~~~~~~地雷关结束~~~~~~~~~~~")
        
    # 翻越障碍(跨越式)
    def stage4(self):
        print('########Stage4_Start########')
        # 调整头部位置，低下头看
        self.motors[19].setPosition(-0.3)
        ns = itertools.count(0)
        for n in ns:
            u = -0.02 * (self.angle[-1])
            u = np.clip(u, -1, 1)
            if n%5 ==0 and np.abs(self.positionSensors[19].getValue()+0.3)<0.05:
                rgb_raw = getImage(self.mCamera)
                ob_x, ob_y = obstacleDetect(rgb_raw)
                if ob_y > 0:
                    self.setMoveCommand(vX=0.5,vA=u)
                if ob_y > 90:  # 蓝色障碍物出现在正确位置，跳出循环，准备空翻
                    self.setMoveCommand(vX=0.,vA=u)
                    break
            self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
            self.myStep()  # 仿真一个步长
        
        # 侧身九十度
        while np.abs(self.angle[-1]+90) > 5:
            u = -0.02 * (self.angle[-1]+90)
            u = np.clip(u, -1, 1)
            self.setMoveCommand(vA=u)
            self.mGaitManager.step(self.mTimeStep)
            self.myStep()
        
        self.mGaitManager.stop()
        self.wait(200)
        
        # 侧向怼障碍
        duiMotion = Motion('motion/obstacle_duidui.motion')
        duiMotion.setLoop(False)
        for i in range(3):
            duiMotion.play()
            while not duiMotion.isOver():
                self.myStep()
            self.wait(200)
        
        # 跨越障碍物
        self.mMotionManager.playPage(9)
        kuaMotion = Motion('./motion/obstacle_kuayue.motion')
        kuaMotion.setLoop(False)
        kuaMotion.play()
        while not kuaMotion.isOver():
            self.myStep()

        self.angle[-1] -= 5
        self.mGaitManager.start()
        self.wait(200)
        self.motors[19].setPosition(-0.2)
        self.checkIfYaw()
        print('########Stage4_End########')

    # 过门（成功率90%）
    def stage9(self):
        print("~~~~~~~~~~~窄门关开始~~~~~~~~~~~")
        # 向右转向90,走到边缘
        self.motors[19].setPosition(-0.1)
        ns = itertools.count(0)
        for n in ns:
            u = -0.02 * (self.angle[-1]+90)
            u = np.clip(u, -1, 1)
            self.setMoveCommand(vX=1.0,vA=u)
            self.mGaitManager.step(self.mTimeStep)
            self.myStep()
            if n % 5 == 0:
                rgb_raw = getImage(self.mCamera)
                hsv = cv2.cvtColor(rgb_raw,cv2.COLOR_BGR2HSV)
                low = self.materialInfo['hsv']['low']
                high = self.materialInfo['hsv']['high']
                mask=cv2.inRange(hsv[self.mCameraHeight//2:,:],np.array(low),np.array(high))
                road = np.where(mask==255)[0]
                num = len(road)
                if num < 20:
                    break
        # 回正，正好配准右侧柱子
        self.setMoveCommand(vX=0.)
        self.checkIfYaw(eps=5)

        # 继续向前走，配准右侧柱子停下位置
        headpos = -0.2
        self.motors[19].setPosition(headpos)
        self.setMoveCommand(vX=1.0,vA=0.0)
        ns = itertools.count(0)
        for n in ns:
            u = -0.02 * (self.angle[-1])
            u = np.clip(u, -1, 1)
            self.setMoveCommand(vX=1.0,vA=u)
            if n % 5 == 0 and np.abs(self.positionSensors[19].getValue()-headpos)<0.05:
                rgb_raw = getImage(self.mCamera)
                low =  [0,0,15]
                high = [255,255,255]
                hsv = cv2.cvtColor(rgb_raw,cv2.COLOR_BGR2HSV)
                mask=cv2.inRange(hsv,np.array(low),np.array(high))
                mask = 255 - mask
                contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                # 视野里有黑色柱子，则按从左到右顺序排序
                if len(contours):
                    contours = sorted(contours,key=lambda cnt : tuple(cnt[cnt[:,:,0].argmin()][0])[0])
                    cnt = contours[0] # 优先取左边那个
                    bottommost=tuple(cnt[cnt[:,:,1].argmax()][0]) # 计算最下点
                    if bottommost[1] > 30 :
                        break
            self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
            self.myStep()  # 仿真一个步长
        
        # 然后斜走，配准第二根柱子（左侧）停下的位置
        ns = itertools.count(0)
        for n in ns:
            u = -0.02 * (self.angle[-1]-45)
            u = np.clip(u, -1, 1)
            self.setMoveCommand(vX=1.0,vA=u)
            self.mGaitManager.step(self.mTimeStep)
            self.myStep()
            if n % 5 == 0 and np.abs(self.positionSensors[19].getValue()-headpos)<0.05 and np.abs(self.angle[-1]-45) < 10:
                rgb_raw = getImage(self.mCamera)
                low =  [0,0,15]
                high = [255,255,255]
                hsv = cv2.cvtColor(rgb_raw,cv2.COLOR_BGR2HSV)
                mask=cv2.inRange(hsv,np.array(low),np.array(high))
                mask = 255 - mask
                contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                # 视野里有黑色柱子，则按从左到右顺序排序
                if len(contours):
                    contours = sorted(contours,key=lambda cnt : tuple(cnt[cnt[:,:,0].argmin()][0])[0])
                    cnt = contours[0] # 优先取左边那个
                    bottommost=tuple(cnt[cnt[:,:,1].argmax()][0]) # 计算最下点
                    if bottommost[1] > 60 or n >=165:
                        print(f'走了{n}步') # 限制在165步
                        break
            self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
            self.myStep()  # 仿真一个步长
        
        # 向左转向90
        while np.abs(self.angle[-1]-90) > 1:
            u = -0.05 * (self.angle[-1]-90)
            u = np.clip(u, -1, 1)
            self.setMoveCommand(vA=u)
            self.mGaitManager.step(self.mTimeStep)
            self.myStep()
        ns = itertools.repeat(0,1000)
        for n in ns:
            a = 0.02 * (75 - self.angle[-1])
            self.setMoveCommand(vY=-1.0,vA=a)
            self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
            self.myStep()  # 仿真一个步长
    
        self.setMoveCommand(vX=0.0)
        self.checkIfYaw(eps=10)
        
        # 通过这关
        self.motors[19].setPosition(-0.2)
        ns = itertools.count(0)
        self.setMoveCommand(vX=1.)
        for n in ns:
            u = -0.02 * (self.angle[-1])
            u = np.clip(u, -1, 1)
            self.setMoveCommand(vX=1.0,vA=u)
            self.mGaitManager.step(self.mTimeStep)
            self.myStep()
            if n % 5 == 0 and  np.abs(self.positionSensors[19].getValue()+0.2)<0.05:
                rgb_raw = getImage(self.mCamera)
                hsv = cv2.cvtColor(rgb_raw,cv2.COLOR_BGR2HSV)
                # 若本关后接高障碍物，则进stage4()
                if self.obstacleBehind:
                    ob_x, ob_y = obstacleDetect(rgb_raw)
                    if ob_y > 40:
                        #print(ob_y)
                        self.stage4()
                        self.obstacleBehind = False
                # 若本关后无障碍物，则正常结束
                elif self.materialInfo['material'] == '黄色砖块':
                    if self.checkIfPassBrick(rgb_raw): break
                else:
                    hsv = cv2.cvtColor(rgb_raw,cv2.COLOR_BGR2HSV)
                    low = self.materialInfo['hsv']['low']
                    high = self.materialInfo['hsv']['high']
                    mask=cv2.inRange(hsv[self.mCameraHeight//2:,:],np.array(low),np.array(high))
                    road = np.where(mask==255)[0]
                    num = len(road)
                    if num < 250: break
        
        print("~~~~~~~~~~~窄门关结束~~~~~~~~~~~")
        self.stageLeft.remove(9)
        
    # 过桥（成功率100%）
    def stage5(self):
        print("~~~~~~~~~~~窄桥关开始~~~~~~~~~~~")
        self.motors[19].setPosition(-0.2)
        # 先看一下是不是太太太靠边了
        ns = itertools.count(0)
        self.setMoveCommand(vX=0.0)
        for n in ns:
            if n % 5 == 0 and np.abs(self.positionSensors[19].getValue()+0.2)<0.05:
                # img processing
                rgb_raw = getImage(self.mCamera)
                hsv = cv2.cvtColor(rgb_raw, cv2.COLOR_BGR2HSV)
                low = self.materialInfo['hsv']['low']
                high = self.materialInfo['hsv']['high']
                mask = cv2.inRange(hsv, np.array(low), np.array(high))
                cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
                if len(cnts)>0:
                    try:
                        cnt = max(cnts, key=cv2.contourArea)
                        M = cv2.moments(cnt)  # 计算第一条轮廓的各阶矩,字典形式
                        center_x = int(M["m10"] / M["m00"])
                        center_y = int(M["m01"] / M["m00"])
                        mid_point = [center_x,center_y]
                    except:
                        pass
                        #cv2.imwrite('mask.png',mask)
                        #cv2.imwrite('rgb.png',rgb_raw)
                else:
                    mid_point = [-1,60]
                # 根据绿色质心灵活调整
                if 35 <= mid_point[0] <= 125:
                    self.setMoveCommand(vX=1.0)
                    break
                elif  0 <= mid_point[0] < 35:
                    self.setMoveCommand(vX=0.0,vY=1.0)
                elif 125 < mid_point[0] <= 160 :
                    self.setMoveCommand(vX=0.0,vY=-1.0)
                else:
                    self.setMoveCommand(vX=-1.0)
                print(mid_point[0])
            self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
            self.myStep()  # 仿真一个步长
        
        # 正常配准
        ns = itertools.count(0)
        for n in ns:
            if n % 5 == 0:
                # img processing
                rgb_raw = getImage(self.mCamera)
                hsv = cv2.cvtColor(rgb_raw, cv2.COLOR_BGR2HSV)
                low = self.materialInfo['hsv']['low']
                high = self.materialInfo['hsv']['high']
                mask = cv2.inRange(hsv, np.array(low), np.array(high))
                cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
                if len(cnts)>0:
                    try:
                        cnt = max(cnts, key=cv2.contourArea)
                        # 注意绿色不能太小，否则桥快到尽头；不能太大，否则进入到绿色回字
                        if not 800 < cv2.contourArea(cnt) < 16000:
                            print('调整结束')
                            self.setMoveCommand(vX=0.0,vA=0.0)
                            self.checkIfYaw()
                            cv2.imwrite('log/keySteps/5窄桥调整结束图像.png',rgb_raw)
                            cv2.imwrite('log/keySteps/5窄桥调整结束图像mask.png',mask)
                            break
                        M = cv2.moments(cnt)  # 计算第一条轮廓的各阶矩,字典形式
                        center_x = int(M["m10"] / M["m00"])
                        center_y = int(M["m01"] / M["m00"])
                        mid_point = [center_x,center_y]
                    except:
                        pass
                        #cv2.imwrite('mask.png',mask)
                        #cv2.imwrite('rgb.png',rgb_raw)
                else:
                    mid_point = [80,60]
            # 根据绿色质心灵活调整
            if 60 < mid_point[0] < 70:
                self.setMoveCommand(vX=0.8,vA=0.4)
            elif 90 < mid_point[0] < 100:
                self.setMoveCommand(vX=0.8,vA=-0.4)
            elif mid_point[0] < 60:
                self.setMoveCommand(vX=0.2,vA=0.8)
            elif mid_point[0] > 100:
                self.setMoveCommand(vX=0.2,vA=-0.8)
            else:
                self.setMoveCommand(vX=1.0,vA=0.0)

            self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
            self.myStep()  # 仿真一个步长
        
        # 判断合适结束独木桥，此时必然已经调整结束。低头看路，注意绿色不能太小，否则桥快到尽头；不能太大，否则进入到绿色回字
        ns = itertools.count(0)
        for n in ns:
            u = -0.02 * (self.angle[-1])
            u = np.clip(u, -1, 1)
            self.setMoveCommand(vX=1.0,vA=u)
            self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
            self.myStep()  # 仿真一个步长        
            if n % 5 == 0:
                rgb_raw = getImage(self.mCamera)
                hsv = cv2.cvtColor(rgb_raw, cv2.COLOR_BGR2HSV)
                low = self.materialInfo['hsv']['low']
                high = self.materialInfo['hsv']['high']
                mask=cv2.inRange(hsv[100:,:],np.array(low),np.array(high))
                road = np.where(mask==255)[0]
                num = len(road)
                if num < 100 or num > 3000: # 大于3000代表走到了回字，视野底部基本全绿
                    cv2.imwrite('log/keySteps/5窄桥判别通关图像.png',rgb_raw)
                    cv2.imwrite('log/keySteps/5窄桥判别通关图像mask.png',mask)
                    break
        
        # 通过后在往前一小段，防止站站在窄桥上开始下一关调整。但最后一关前不能这样。
        print(self.stageLeft)
        if len(self.stageLeft)> 2:
            ns = itertools.repeat(0,250)
            for n in ns:
                u = -0.02 * (self.angle[-1])
                u = np.clip(u, -1, 1)
                self.setMoveCommand(vX=1.,vA=u)
                self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
                self.myStep()  # 仿真一个步长
        self.setMoveCommand(vX=0.)
        self.checkIfYaw()
        self.stageLeft.remove(5)
        print("~~~~~~~~~~~窄桥关结束~~~~~~~~~~~")

    # 踢球进洞（成功率75%）
    def stage6(self):
        print("~~~~~~~~~~~踢球关开始~~~~~~~~~~~")
        # 三种情况，一种是洞在左前，一种洞在右前，一种洞在右后
        # 右后是一种特殊情况，一定是出现在第二个转弯后，或者说踢球关出现在所有长形关顺序的最后一个（这样一定会被安排在第二个转弯之后）
        if 3 not in self.stageLeft and 7 not in self.stageLeft and 9 not in self.stageLeft:
            print('特殊情况，回身踢球')
            self.motors[19].setPosition(-0.2)
            while np.abs(self.angle[-1]-90) > 20:
                u = -0.02 * (self.angle[-1]-90)
                u = np.clip(u, -1, 1)
                self.setMoveCommand(vA=u)
                self.mGaitManager.step(self.mTimeStep)
                self.myStep()
            # 走到道路边缘
            ns = itertools.count(0)
            for n in ns:
                u = -0.02 * (self.angle[-1]-90)
                u = np.clip(u, -1, 1)
                self.setMoveCommand(vX=1.,vA=u)
                self.mGaitManager.step(self.mTimeStep)
                self.myStep()
                if n % 5 == 0:
                    rgb_raw = getImage(self.mCamera)
                    # 所谓的回身黄黄
                    if self.checkIfHSHH(rgb_raw):
                        break
                    else:
                        hsv = cv2.cvtColor(rgb_raw,cv2.COLOR_BGR2HSV)
                        low = self.materialInfo['hsv']['low']
                        high = self.materialInfo['hsv']['high']
                        mask=cv2.inRange(hsv[self.mCameraHeight//2:,:],np.array(low),np.array(high))
                        road = np.where(mask==255)[0]
                        num = len(road)
                        if num < 100:
                            break
            # 转180度
            self.setMoveCommand(vX=0.)
            while np.abs(self.angle[-1]-175) > 10:
                u = -0.02 * (self.angle[-1]-175)
                u = np.clip(u, -1, 1)
                self.setMoveCommand(vA=u)
                self.mGaitManager.step(self.mTimeStep)
                self.myStep()

            self.motors[18].setPosition(radians(45))
            self.motors[19].setPosition(0.1)
            # 向后倒车，直到洞和球垂直出现在视野中央
            self.setMoveCommand(vX=-1.0,vY=0.,vA=0.)
            ns = itertools.count(0)
            for n in ns:
                u = -0.02 * (self.angle[-1]-175)
                u = np.clip(u, -1, 1)
                self.setMoveCommand(vX=-1.,vA=u)
                self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
                self.myStep()  # 仿真一个步长
                if n % 5 == 0 and np.abs(self.angle[-1]-175)<1:
                    rgb_raw = getImage(self.mCamera)
                    hsv = cv2.cvtColor(rgb_raw,cv2.COLOR_BGR2HSV)
                    low = [95,115,0] #球和洞的hsv
                    high = [255,255,50]
                    mask=cv2.inRange(hsv,np.array(low),np.array(high))
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
                    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel,iterations=3) # 闭运算：先膨胀后腐蚀，用来连接被误分为许多小块的对象
                    contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                    ballHoleInfo = []
                    for contour in contours:
                        area = cv2.contourArea(contour)
                        if area > 5:
                            x,y,w,h = cv2.boundingRect(contour)
                            ballHoleInfo.append({'center':(int(x+w/2),int(y+w/2)),'area':w*h})
                    if len(ballHoleInfo)>=1 and 75 < ballHoleInfo[0]['center'][0] < 95 :
                        cv2.imwrite('log/keySteps/6踢球进洞mask.png',mask)
                        cv2.imwrite('log/keySteps/6踢球进洞rgb.png',rgb_raw)
                        break
                
            # 向左转向45
            self.motors[18].setPosition(0)
            self.motors[19].setPosition(0.1)
            while np.abs(self.angle[-1]-220) > 1:
                u = -0.02 * (self.angle[-1]-220)
                u = np.clip(u, -1, 1)
                self.setMoveCommand(vA=u)
                self.mGaitManager.step(self.mTimeStep)
                self.myStep()
            

            # 前进，好像直接能把球撞进去
            ns = itertools.count(0)
            bias = 0.
            forwardFlag = False
            for n in ns:
                u = -0.02 * (self.angle[-1]-220)
                u = np.clip(u, -1, 1)
                if forwardFlag:
                    self.setMoveCommand(vX=1.0,vY=0.0,vA=u)
                else:
                    self.setMoveCommand(vX=0.0,vY=0.05*bias,vA=u)
                self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
                self.myStep()  # 仿真一个步长
                # 配准球洞
                if n % 5 == 0:
                    rgb_raw = getImage(self.mCamera)
                    hsv = cv2.cvtColor(rgb_raw,cv2.COLOR_BGR2HSV)
                    low = [95,115,0] #球和洞的hsv
                    high = [255,255,50]
                    mask=cv2.inRange(hsv,np.array(low),np.array(high))
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
                    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel,iterations=3) # 闭运算：先膨胀后腐蚀，用来连接被误分为许多小块的对象
                    contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                    if len(contours):
                        # fix the bug : 有时候会除0
                        try:
                            cnt = max(contours, key=cv2.contourArea)
                            M = cv2.moments(cnt)  # 计算第一条轮廓的各阶矩,字典形式
                            center_x = int(M["m10"] / M["m00"])
                            center_y = int(M["m01"] / M["m00"])
                            #print(center_x)
                        except:
                            break
                        bias = 90 - center_x
                        if -2 < bias < 2:
                            forwardFlag = True
                        if center_y > 90:
                            break
                    else:
                        bias = 0.
            # 转身离去
            print('踢球完毕')
            self.setMoveCommand(vX=0.)
            self.checkIfYaw(eps=10)
          
        # 如果不是右后，则按正常流程处理。
        else:
            # 先上下抬头，同时找到球和洞，并判定左右关系
            ns = itertools.count(0)
            self.setMoveCommand(vX=0.)
            headPos = 0.5
            self.motors[19].setPosition(headPos)
            while not np.abs(self.positionSensors[19].getValue()-headPos) < 0.05:
                u = -0.02 * (self.angle[-1])
                u = np.clip(u, -1, 1)
                self.setMoveCommand(vA=u)
                self.mGaitManager.step(self.mTimeStep)
                self.myStep()
            # 从0.5降到0.3，若还没有找到，则代表转弯之后离球太近，要往后退
            backwardFlag = False
            headPos = 0.35
            self.motors[19].setPosition(headPos)
            while True:
                u = -0.02 * (self.angle[-1])
                u = np.clip(u, -1, 1)
                x = -1. if backwardFlag else 0.
                self.setMoveCommand(vX=x,vA=u)
                self.mGaitManager.step(self.mTimeStep)
                self.myStep()
                rgb_raw = getImage(self.mCamera)
                hsv = cv2.cvtColor(rgb_raw, cv2.COLOR_BGR2HSV)
                low = [95,115,0] #球和洞的hsv
                high = [255,255,50]
                mask = cv2.inRange(hsv, np.array(low), np.array(high))
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel,iterations=3) # 闭运算：先膨胀后腐蚀，用来连接被误分为许多小块的对象
                contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                ballHoleInfo = []
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 5:
                        x,y,w,h = cv2.boundingRect(contour)
                        ballHoleInfo.append({'center':(int(x+w/2),int(y+w/2)),'area':w*h})
                if len(ballHoleInfo) == 2:
                    print(f'同时检测到球和洞时头部位置 ： {self.positionSensors[19].getValue()}')
                    break #检测到球和洞

                if np.abs(self.positionSensors[19].getValue()-headPos) < 0.05:
                    backwardFlag = True      
            
            # 判定左右关系，用面积和中心
            ballHoleInfo = sorted(ballHoleInfo,key= lambda x:x['area'])
            if ballHoleInfo[0]['center'][0] < ballHoleInfo[1]['center'][0]:
                case = 1
                print('洞在右侧')
            else:
                case = 2
                print('洞在左侧')
        
            # 若太远，先走一段。不要在上一关结束的地方直接转。
            if ballHoleInfo[0]['center'][1] < 70:
                ns = itertools.repeat(0,(400-ballHoleInfo[1]['center'][1]))
                for n in ns:
                    u = -0.02 * (self.angle[-1])
                    u = np.clip(u, -1, 1)
                    self.setMoveCommand(vX=1.0,vA=u)
                    self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
                    self.myStep()  # 仿真一个步长
            else:
                print('不可以先走一段')

            # 若洞在右侧
            if case == 1:
                # 向左转向90
                self.motors[19].setPosition(-0.15)
                while np.abs(self.angle[-1]-90) > 20:
                    u = -0.02 * (self.angle[-1]-90)
                    u = np.clip(u, -1, 1)
                    self.setMoveCommand(vA=u)
                    self.mGaitManager.step(self.mTimeStep)
                    self.myStep()
                # 走到道路边缘
                ns = itertools.count(0)
                for n in ns:
                    u = -0.02 * (self.angle[-1]-90)
                    u = np.clip(u, -1, 1)
                    self.setMoveCommand(vX=1.,vA=u)
                    self.mGaitManager.step(self.mTimeStep)
                    self.myStep()
                    if n % 5 == 0:
                        rgb_raw = getImage(self.mCamera)
                        # 所谓的回身黄黄
                        if self.checkIfHSHH(rgb_raw): 
                            break
                        else:
                            hsv = cv2.cvtColor(rgb_raw,cv2.COLOR_BGR2HSV)
                            
                            low = self.materialInfo['hsv']['low']
                            high = self.materialInfo['hsv']['high']
                            mask=cv2.inRange(hsv[self.mCameraHeight//2:,:],np.array(low),np.array(high))
                            road = np.where(mask==255)[0]
                            num = len(road)
                            if num < 100:
                                break
                        # if self.materialInfo['material'] =='白色' or self.materialInfo['material'] =='灰色' and self.turnCount == 1:
                        #     cv2.imshow('image',mask)
                        #     cv2.waitKey(0)
                # 回正，侧头
                self.setMoveCommand(vX=0.)
                self.motors[18].setPosition(radians(-45))
                self.motors[19].setPosition(0.1)
                self.checkIfYaw(eps=10)

                # 向前走，直到洞和球垂直出现在视野中央
                ns = itertools.count(0)
                for n in ns:
                    u = -0.02 * (self.angle[-1])
                    u = np.clip(u, -1, 1)
                    self.setMoveCommand(vX=1.,vA=u)
                    self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
                    self.myStep()  # 仿真一个步长
                    if n % 5 == 0 and np.abs(self.angle[-1])<5:
                        rgb_raw = getImage(self.mCamera)
                        hsv = cv2.cvtColor(rgb_raw,cv2.COLOR_BGR2HSV)
                        low = [95,115,0] #球和洞的hsv
                        high = [255,255,50]
                        mask=cv2.inRange(hsv,np.array(low),np.array(high))
                        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
                        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel,iterations=3) # 闭运算：先膨胀后腐蚀，用来连接被误分为许多小块的对象
                        contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                        ballHoleInfo = []
                        for contour in contours:
                            area = cv2.contourArea(contour)
                            if area > 5:
                                x,y,w,h = cv2.boundingRect(contour)
                                ballHoleInfo.append({'center':(int(x+w/2),int(y+w/2)),'area':w*h})
                            
                        # if len(ballHoleInfo) >= 1:
                        #     print(ballHoleInfo)
                        if len(ballHoleInfo)>=1 and 75 < ballHoleInfo[0]['center'][0] < 95 :
                            cv2.imwrite('log/keySteps/6踢球进洞mask.png',mask)
                            cv2.imwrite('log/keySteps/6踢球进洞rgb.png',rgb_raw)
                            break
                    
                # 向右转向45
                self.motors[18].setPosition(0)
                self.motors[19].setPosition(0.1)
                while np.abs(self.angle[-1]+45) > 1:
                    u = -0.02 * (self.angle[-1]+45)
                    u = np.clip(u, -1, 1)
                    self.setMoveCommand(vA=u)
                    self.mGaitManager.step(self.mTimeStep)
                    self.myStep()
                
                # 前进，好像直接能把球撞进去
                ns = itertools.count(0)
                bias = 0.
                forwardFlag = False
                for n in ns:
                    u = -0.02 * (self.angle[-1]+45)
                    u = np.clip(u, -1, 1)
                    if forwardFlag:
                        self.setMoveCommand(vX=1.0,vY=0.0,vA=u)
                    else:
                        self.setMoveCommand(vX=0.0,vY=0.05*bias,vA=u)
                    self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
                    self.myStep()  # 仿真一个步长
                    # 配准球洞
                    if n % 5 == 0:
                        rgb_raw = getImage(self.mCamera)
                        hsv = cv2.cvtColor(rgb_raw,cv2.COLOR_BGR2HSV)
                        low = [95,115,0] #球和洞的hsv
                        high = [255,255,50]
                        mask=cv2.inRange(hsv,np.array(low),np.array(high))
                        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
                        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel,iterations=3) # 闭运算：先膨胀后腐蚀，用来连接被误分为许多小块的对象
                        contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                        if len(contours):
                            cnt = max(contours, key=cv2.contourArea)
                            M = cv2.moments(cnt)  # 计算第一条轮廓的各阶矩,字典形式
                            center_x = int(M["m10"] / M["m00"])
                            center_y = int(M["m01"] / M["m00"])
                            #print(center_x)
                            bias = 70 - center_x
                            if -2 < bias < 2:
                                forwardFlag = True
                            if center_y > 90:
                                break
                        else:
                            bias = 0.
                # 停下，踢球
                # self.mGaitManager.stop()
                # self.wait(200)
                # self.mMotionManager.playPage(9)  # 执行动作组9号动作，初始化站立姿势，准备行走
                # kickMotion = Motion('motion/kick.motion')
                # kickMotion.setLoop(False)
                # kickMotion.play()
                # while not kickMotion.isOver():
                #     self.myStep()
                
                # 转身离去
                print('踢球完毕')
                self.setMoveCommand(vX=0.)
                self.checkIfYaw(eps=10)
                
            
            # 若洞在左侧
            elif case == 2:
                # 向右转向90
                self.motors[19].setPosition(-0.15)
                while np.abs(self.angle[-1]+90) > 20:
                    u = -0.02 * (self.angle[-1]+90)
                    u = np.clip(u, -1, 1)
                    self.setMoveCommand(vA=u)
                    self.mGaitManager.step(self.mTimeStep)
                    self.myStep()
                # 走到道路边缘
                ns = itertools.count(0)
                for n in ns:
                    u = -0.02 * (self.angle[-1]+90)
                    u = np.clip(u, -1, 1)
                    self.setMoveCommand(vX=1.,vA=u)
                    self.mGaitManager.step(self.mTimeStep)
                    self.myStep()
                    if n % 5 == 0:
                        rgb_raw = getImage(self.mCamera)
                        hsv = cv2.cvtColor(rgb_raw,cv2.COLOR_BGR2HSV)
                        low = self.materialInfo['hsv']['low']
                        high = self.materialInfo['hsv']['high']
                        mask=cv2.inRange(hsv[self.mCameraHeight//2:,:],np.array(low),np.array(high))
                        road = np.where(mask==255)[0]
                        num = len(road)
                        if num < 100:
                            break
                # 回正，侧头
                self.setMoveCommand(vX=0.)
                self.motors[18].setPosition(radians(45))
                self.motors[19].setPosition(0.1)
                self.checkIfYaw(eps=10)

                # 向前走，直到洞和球垂直出现在视野中央
                ns = itertools.count(0)
                for n in ns:
                    u = -0.02 * (self.angle[-1])
                    u = np.clip(u, -1, 1)
                    self.setMoveCommand(vX=1.,vA=u)
                    self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
                    self.myStep()  # 仿真一个步长
                    if n % 5 == 0 and np.abs(self.angle[-1])<5:
                        rgb_raw = getImage(self.mCamera)
                        hsv = cv2.cvtColor(rgb_raw,cv2.COLOR_BGR2HSV)
                        low = [95,115,0] #球和洞的hsv
                        high = [255,255,50]
                        mask=cv2.inRange(hsv,np.array(low),np.array(high))
                        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
                        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel,iterations=3) # 闭运算：先膨胀后腐蚀，用来连接被误分为许多小块的对象
                        contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                        ballHoleInfo = []
                        for contour in contours:
                            area = cv2.contourArea(contour)
                            if area > 5:
                                x,y,w,h = cv2.boundingRect(contour)
                                ballHoleInfo.append({'center':(int(x+w/2),int(y+w/2)),'area':w*h})
                        # if len(ballHoleInfo) >= 1:
                        #     print(ballHoleInfo)
                        if len(ballHoleInfo)>=1 and 65 < ballHoleInfo[0]['center'][0] < 85 :
                            cv2.imwrite('log/keySteps/6踢球进洞mask.png',mask)
                            cv2.imwrite('log/keySteps/6踢球进洞rgb.png',rgb_raw)
                            break
                    
                # 向左转向45
                self.motors[18].setPosition(0)
                self.motors[19].setPosition(0.1)
                while np.abs(self.angle[-1]-45) > 1:
                    u = -0.02 * (self.angle[-1]-45)
                    u = np.clip(u, -1, 1)
                    self.setMoveCommand(vA=u)
                    self.mGaitManager.step(self.mTimeStep)
                    self.myStep()
                
                # 前进，好像直接能把球撞进去
                ns = itertools.count(0)
                bias = 0.
                forwardFlag = False
                for n in ns:
                    u = -0.02 * (self.angle[-1]-45)
                    u = np.clip(u, -1, 1)
                    if forwardFlag:
                        self.setMoveCommand(vX=1.0,vY=0.0,vA=u)
                    else:
                        self.setMoveCommand(vX=0.0,vY=0.05*bias,vA=u)
                    self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
                    self.myStep()  # 仿真一个步长
                    # 配准球洞
                    if n % 5 == 0:
                        rgb_raw = getImage(self.mCamera)
                        hsv = cv2.cvtColor(rgb_raw,cv2.COLOR_BGR2HSV)
                        low = [95,115,0] #球和洞的hsv
                        high = [255,255,50]
                        mask=cv2.inRange(hsv,np.array(low),np.array(high))
                        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
                        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel,iterations=3) # 闭运算：先膨胀后腐蚀，用来连接被误分为许多小块的对象
                        contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                        if len(contours):
                            cnt = max(contours, key=cv2.contourArea)
                            M = cv2.moments(cnt)  # 计算第一条轮廓的各阶矩,字典形式
                            center_x = int(M["m10"] / M["m00"])
                            center_y = int(M["m01"] / M["m00"])
                            #print(center_x)
                            bias = 90 - center_x
                            if -2 < bias < 2:
                                forwardFlag = True
                            if center_y > 90:
                                break
                        else:
                            bias = 0.
                # 停下，踢球
                # self.mGaitManager.stop()
                # self.wait(200)
                # self.mMotionManager.playPage(9)  # 执行动作组9号动作，初始化站立姿势，准备行走
                # kickMotion = Motion('motion/kick.motion')
                # kickMotion.setLoop(False)
                # kickMotion.play()
                # while not kickMotion.isOver():
                #     self.myStep()
                
                # 转身离去
                print('踢球完毕')
                self.setMoveCommand(vX=0.)
                self.checkIfYaw(eps=10)


        # 踢球转身后，通过这关
        self.motors[19].setPosition(-0.2)
        ns = itertools.count(0)
        for n in ns:
            u = -0.02 * (self.angle[-1])
            u = np.clip(u, -1, 1)
            self.setMoveCommand(vX=1.,vA=u)
            self.checkIfYaw()
            self.mGaitManager.step(self.mTimeStep)
            self.myStep()
            if n % 5 == 0:
                rgb_raw = getImage(self.mCamera)
                # 若本关后接高障碍物，则进stage4()
                if self.obstacleBehind:
                    ob_x, ob_y = obstacleDetect(rgb_raw)
                    if ob_y > 40:
                        #print(ob_y)
                        self.stage4()
                        self.obstacleBehind = False
                # 若本关后无障碍物，则正常结束
                elif self.materialInfo['material'] == '黄色砖块':
                    if self.checkIfPassBrick(rgb_raw): break
                else:
                    hsv = cv2.cvtColor(rgb_raw,cv2.COLOR_BGR2HSV)
                    low = self.materialInfo['hsv']['low']
                    high = self.materialInfo['hsv']['high']
                    mask=cv2.inRange(hsv[self.mCameraHeight//2:,:],np.array(low),np.array(high))
                    road = np.where(mask==255)[0]
                    num = len(road)
                    if num < 250: break
        print("~~~~~~~~~~~踢球关结束~~~~~~~~~~~")
        self.stageLeft.remove(6)
        
    # 走楼梯（成功率95%）
    def stage7(self):
        print("~~~~~~~~~~~楼梯关开始~~~~~~~~~~~")
        # 直行直到看到第一块蓝色楼梯
        self.motors[19].setPosition(-0.2)
        backwardFlag = False
        ns = itertools.count(0)
        for n in ns:
            u = -0.02 * (self.angle[-1])
            u = np.clip(u, -1, 1)
            if not backwardFlag:
                self.setMoveCommand(vX=1.0,vA=u)
            else:
                self.setMoveCommand(vX=-1.0,vA=u)
            self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
            self.myStep()  # 仿真一个步长
            if n % 5 == 0 and np.abs(self.positionSensors[19].getValue()+0.2)<0.05:
                # img processing
                rgb_raw = getImage(self.mCamera)
                hsv = cv2.cvtColor(rgb_raw, cv2.COLOR_BGR2HSV)
                low = [100,110,150] # 第一块蓝色阶梯的hsv
                high = [110,200,255]
                mask = cv2.inRange(hsv, np.array(low), np.array(high))
                cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
                if len(cnts)>0:
                    cnt = max(cnts, key=cv2.contourArea)
                    bottommost=tuple(cnt[cnt[:,:,1].argmax()][0])
                    if bottommost[1] > 40 : backwardFlag = True
                    if bottommost[1] > 15 and not backwardFlag : break
                    if bottommost[1] < 15 and backwardFlag: break
        
        # 若太偏，先修正一下
        leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
        rightmost=tuple(cnt[cnt[:,:,0].argmax()][0])
        midX = 0.5*(leftmost[0]+rightmost[0]) # 视野中蓝色楼梯中点

        if midX < 65:
            direction = 60
        elif midX > 90:
            direction = -60
        else:
            direction = 0

        while(np.abs(self.angle[-1]-direction)>5):
            u = -0.02 * (self.angle[-1]-direction)
            u = np.clip(u, -1, 1)
            self.setMoveCommand(vX=1.0,vA=u)
            self.mGaitManager.step(self.mTimeStep)
            self.myStep()

        self.setMoveCommand(vX=0.0)
        self.checkIfYaw(eps=5)

        # 配准上楼的位置
        ns = itertools.count(0)
        for n in ns:
            u = -0.02 * (self.angle[-1])
            u = np.clip(u, -1, 1)
            self.setMoveCommand(vX=1.0,vA=u)
            self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
            self.myStep()  # 仿真一个步长
            if n % 5 == 0:
                # img processing
                rgb_raw = getImage(self.mCamera)
                hsv = cv2.cvtColor(rgb_raw, cv2.COLOR_BGR2HSV)
                low = [100,110,150] # 第一块蓝色阶梯的hsv
                high = [110,200,255]
                mask = cv2.inRange(hsv, np.array(low), np.array(high))
                cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
                if len(cnts):
                    cnt = max(cnts, key=cv2.contourArea)
                    bottommost=tuple(cnt[cnt[:,:,1].argmax()][0])
                    if bottommost[1] >= 115:
                        print('配准结束，准备上楼梯')
                        break
        # 停下
        for _ in range(50):
            u = -0.02 * (self.angle[-1])
            u = np.clip(u, -1, 1)
            self.setMoveCommand(vX=0.0,vA=u)
            self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
            self.myStep()  # 仿真一个步长

        # 停下来，怼楼梯，反怼 南科大方案，待优化，这里整整浪费了5-7秒钟
        self.mGaitManager.stop()
        self.wait(200)

        stepMotion = Motion('motion/stair_duidui.motion')
        stepMotion.setLoop(False)
        for i in range(3):
            stepMotion.play()
            while not stepMotion.isOver():
                self.myStep()
        stepMotion = Motion('motion/stair_fandui.motion')
        stepMotion.setLoop(False)
        for i in range(2):
            stepMotion.play()
            while not stepMotion.isOver():
                self.myStep()
        
        # 上楼梯
        stepMotion = Motion('motion/stair_up.motion')
        stepMotion.setLoop(False)
        stepMotion.play()
        while not stepMotion.isOver():
            self.myStep()

        # 下楼梯
        stepMotion = Motion('motion/stair_down_v2.motion')
        stepMotion.setLoop(False)
        stepMotion.play()
        while not stepMotion.isOver():
            self.myStep()
        stepMotion.play()
        while not stepMotion.isOver():
            self.myStep()
        #self.mMotionManager.playPage(9)
        stepMotion = Motion('motion/stair_fandui.motion')
        stepMotion.setLoop(False)
        for i in range(4):
            stepMotion.play()
            while not stepMotion.isOver():
                self.myStep()
        self.angle = np.array([0., 0., 0.])

        # 抬头看下一关，是不是还是黄色的
        self.motors[19].setPosition(0.6)
        ns = itertools.repeat(0, 200)
        for n in ns:
            rgb_raw = getImage(self.mCamera)
            if np.abs(self.positionSensors[19].getValue()-0.6)<0.05:
                break
        cv2.imwrite('log/keySteps/7楼梯上决策.png',rgb_raw)
        hsv = cv2.cvtColor(rgb_raw, cv2.COLOR_BGR2HSV)
        low = self.materialInfo['hsv']['low']
        high = self.materialInfo['hsv']['high']
        mask = cv2.inRange(hsv,np.array(low),np.array(high))
        road = np.where(mask==255)[0]
        farest = np.where(road<5)[0]
        #print(len(farest))
        self.mGaitManager.start()
        self.wait(200)
        if len(farest) > 100:
            print('接下来还是黄色路面')
            # 写死冲一段
            self.motors[19].setPosition(-0.2)
            ns = itertools.repeat(0,800)
            for n in ns:
                u = -0.02 * (self.angle[-1])
                u = np.clip(u, -1, 1)
                self.setMoveCommand(vX=1.0,vA=u)
                self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
                self.myStep()  # 仿真一个步长
        else:
            # 用黄色结束来作为关卡结束的标志
            print('接下来不是黄色路面')
            low =  [0,180,205] # 大红
            high = [255,255,255]
            hsv = cv2.cvtColor(rgb_raw,cv2.COLOR_BGR2HSV)
            mask=cv2.inRange(hsv,np.array(low),np.array(high))
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1,1))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel,iterations=3)
            # 下最后一个红色坡,并结束这关
            self.motors[19].setPosition(-0.2)
            ns = itertools.count(0)
            for n in ns:
                u = -0.02 * (self.angle[-1])
                u = np.clip(u, -1, 1)
                self.setMoveCommand(vX=1.0,vA=u)
                self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
                self.myStep()  # 仿真一个步长
                if n > 100 and n % 5 == 0 and np.abs(self.positionSensors[19].getValue()+0.2)<0.05:
                    # img processing
                    rgb_raw = getImage(self.mCamera)
                    hsv = cv2.cvtColor(rgb_raw, cv2.COLOR_BGR2HSV)
                    low = self.materialInfo['hsv']['low']
                    high = self.materialInfo['hsv']['high']
                    mask1=cv2.inRange(hsv[self.mCameraHeight//2:,:],np.array(low),np.array(high)) # 黄色mask
                    low =  [0,180,205] # 大红
                    high = [255,255,255]
                    mask2=cv2.inRange(hsv[self.mCameraHeight//2:,:],np.array(low),np.array(high)) # 红色mask
                    mask = cv2.bitwise_or(mask1, mask2)  #取并集
                    road = np.where(mask==255)[0]
                    num = len(road)
                    if num < 200:
                        cv2.imwrite('log/keySteps/7楼梯关结束.png',rgb_raw)
                        break
        self.stageLeft.remove(7)
        print("~~~~~~~~~~~楼梯关结束~~~~~~~~~~~")

    # 水平开合横杆（成功率95%）
    def stage8(self):
        print("~~~~~~~~~~~水平横杆关开始~~~~~~~~~~~")
        stage8model = load_model('./pretrain/8.pth')
        crossBarCloseFlag = False
        goFlag = False
        self.setMoveCommand(vX=0.0)
        # 先看一下左右道路距离，若太偏则不能直接直走，会撞到杆子
        self.motors[18].setPosition(radians(90))
        self.motors[19].setPosition(-0.2)
        while True:
            u = -0.02 * (self.angle[-1])
            u = np.clip(u, -1, 1)
            self.setMoveCommand(vX=0.0,vA=u)
            self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
            self.myStep()  # 仿真一个步长
            if np.abs(self.positionSensors[18].getValue()-radians(90))<0.05:
                leftImg = getImage(self.mCamera)
                lefthsv = cv2.cvtColor(leftImg,cv2.COLOR_BGR2HSV)
                low = [10,0,125]
                high = [167,80,175]
                leftMask=cv2.inRange(lefthsv,np.array(low),np.array(high))
                leftMask = cv2.medianBlur(leftMask,3)[10:,:]
                if len(np.where(leftMask==255)[0]) > 50:
                    leftValue = 110 - min(np.where(leftMask==255)[0])
                else:
                    leftValue = 1
                break
        self.motors[18].setPosition(radians(-90))
        while True:
            self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
            self.myStep()  # 仿真一个步长
            self.checkIfYaw()
            if np.abs(self.positionSensors[18].getValue()+radians(90))<0.05:
                rightImg = getImage(self.mCamera)
                low = [25,15,125]
                high = [167,80,175]
                righthsv = cv2.cvtColor(rightImg,cv2.COLOR_BGR2HSV)
                rightMask=cv2.inRange(righthsv,np.array(low),np.array(high))
                rightMask = cv2.medianBlur(rightMask,3)[10:,:]
                if len(np.where(rightMask==255)[0]) > 50:
                    rightValue = 110 - min(np.where(rightMask==255)[0])
                else:
                    rightValue = 1
                break
        ratio = leftValue / (leftValue + rightValue)
        cv2.imwrite('log/keySteps/8左视角rgb.png',leftImg)
        cv2.imwrite('log/keySteps/8左视角mask.png',leftMask)
        cv2.imwrite('log/keySteps/8右视角rgb.png',rightImg)
        cv2.imwrite('log/keySteps/8右视角mask.png',rightMask)
        # cv2.imshow('left',leftMask)
        # cv2.imshow('right',rightMask)
        # cv2.waitKey(0)

        # 视角回正
        self.motors[18].setPosition(radians(0))
        self.motors[19].setPosition(0.35)
        while True:
            u = -0.02 * (self.angle[-1])
            u = np.clip(u, -1, 1)
            self.setMoveCommand(vX=0.0,vA=u)
            self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
            self.myStep()  # 仿真一个步长
            if np.abs(self.positionSensors[18].getValue())<0.05:
                break
        # 分类网络判别
        ns = itertools.count(0)
        for n in ns:
            u = -0.02 * (self.angle[-1])
            u = np.clip(u, -1, 1)
            self.setMoveCommand(vX=0.0,vA=u)
            if n % 100 == 0 and np.abs(self.positionSensors[19].getValue()-0.35)<0.05:
                rgb_raw = getImage(self.mCamera)
                fname = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time())) 
                cv2.imwrite('./log/last/' + fname + '.png', rgb_raw)
                pred, prob = call_classifier(rgb_raw, stage8model)
                if not crossBarCloseFlag:
                    if pred == 1:
                        crossBarCloseFlag = True
                        print('CrossBar already Close with probablity %.3f' % prob)
                    else:
                        print('Wait for CrossBar Close with probablity %.3f ...' % prob)
                else:
                    if pred == 0 and prob > 0.85:
                        goFlag = True
                        print('CrossBar already Open with probablity %.3f, Go Go Go!' % prob)
                    else:
                        print('Wait for CrossBar Open with probablity %.3f ...' % prob)
            if goFlag:
                cv2.imwrite('log/keySteps/8可通行.png',rgb_raw)
                break
            self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
            self.myStep()  # 仿真一个步长
        
        del stage8model

        # 走完最后一关
        self.motors[19].setPosition(-0.2)
        # 如果太边缘了，需要转回来一点
        print(f'第八关预调整,ratio = {ratio}')
        if ratio < 0.35:
            direction = -25
        elif ratio > 0.65:
            direction = 25
        else:
            direction = 0
        while(np.abs(self.angle[-1]-direction)>0.5):
            u = -0.02 * (self.angle[-1]-direction)
            u = np.clip(u, -1, 1)
            self.setMoveCommand(vX=1.0,vA=u)
            self.mGaitManager.step(self.mTimeStep)
            self.myStep()

        # 快速通过
        self.motors[19].setPosition(0)
        ns = itertools.count(0)
        for n in ns:
            # 调角速度，保持垂直。这里没有用checkIfYaw，虽然功能相同，但是后者占用的mystep没有计数。
            u = -0.02 * (self.angle[-1])
            u = np.clip(u, -1, 1)
            self.setMoveCommand(vX=1.0,vA=u)
            self.mGaitManager.step(self.mTimeStep)
            self.myStep()
            if n % 5 == 0 and np.abs(self.positionSensors[19].getValue())<0.05:
                rgb_raw = getImage(self.mCamera)
                hsv = cv2.cvtColor(rgb_raw,cv2.COLOR_BGR2HSV)
                low = self.materialInfo['hsv']['low']
                high = self.materialInfo['hsv']['high']
                mask=cv2.inRange(hsv[self.mCameraHeight//2:,:],np.array(low),np.array(high))
                road = np.where(mask==255)[0]
                num = len(road)
                # 判定关卡结束条件，当前材料的hsv基本消失在视野中下段。或者行进步数超限1000。
                if num < 500 or n > 1500:
                    break
        self.stageLeft.remove(8)
        print("~~~~~~~~~~~水平横杆关结束~~~~~~~~~~~")

    '''
    下面是不同关卡之间的连贯
    '''
    def whereAmI(self,img):
        # 判定当前我在哪种花色的地面上
        hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        maxNum = 0
        currentInfo = {}
        for info in raceTrackInfo:
            low = info['hsv']['low']
            high = info['hsv']['high']
            mask=cv2.inRange(hsv,np.array(low),np.array(high))
            road = np.where(mask==255)[0]
            num = len(road)
            if num > maxNum:
                currentInfo = info
                maxNum = num
        return currentInfo
    
    def checkIfCorner(self,MaterialInfo,judgeCorner=True):
        if not judgeCorner or self.turnCount==2:
            return False
        # 抬头看，长形是不是够长，如果只有一块的长度（正常长形占两块），则代表要转弯了；
        self.motors[19].setPosition(0.5)
        # 重复50个空循环等待电机到位
        ns = itertools.repeat(0, 1000)
        for n in ns:
            self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
            self.myStep()  # 仿真一个步长
            if np.abs(self.positionSensors[19].getValue()-0.5)<0.05:
                rgb_raw = getImage(self.mCamera)
                break
        low = MaterialInfo['hsv']['low']
        high = MaterialInfo['hsv']['high']
        hsv = cv2.cvtColor(rgb_raw,cv2.COLOR_BGR2HSV)
        mask=cv2.inRange(hsv,np.array(low),np.array(high))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel,iterations=3)
        contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        # 找面积最大的轮廓
        area = [cv2.contourArea(contour) for contour in contours]
        idxSorted = np.argsort(np.array(area))
        points = contours[idxSorted[-1]].reshape(-1,2)
        x_min,y_min = np.min(points,axis=0)
        # 若轮廓上界在视野下方，则判断不是完整长形，要转弯
        if y_min > 50:
            fname = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time())) 
            cv2.imwrite('./log/corner/' + fname + '.png', rgb_raw)
            return True
        else:
            return False
    
    def checkIfObstacle(self,MaterialInfo):
        # 抬头看，视野里是否有蓝白，有则代表这关结束后有蓝色障碍物
        self.motors[19].setPosition(0.5)
        # 重复50个空循环等待电机到位
        ns = itertools.repeat(0, 150)
        for n in ns:
            self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
            self.myStep()  # 仿真一个步长
            if np.abs(self.positionSensors[19].getValue()-0.5)<0.05:
                rgb_raw = getImage(self.mCamera)
                break
        low =  [110,250,0]
        high = [130,255,255]
        hsv = cv2.cvtColor(rgb_raw,cv2.COLOR_BGR2HSV)
        mask=cv2.inRange(hsv,np.array(low),np.array(high))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1,1))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel,iterations=3)
        contours, hierarchy = cv2.findContours(mask[:,40:-40],cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        #cv2.imwrite('rgb.png', rgb_raw)
        #cv2.imwrite('mask.png', mask)
        if len(contours) >= 2:
            return True
        else:
            return False

    def checkIfTrap(self,MaterialInfo):
        # 抬头看，主要区分绿色是回字还是窄桥。根据绿色面积，不知道稳定否
        self.motors[19].setPosition(0.1)
        # 重复50个空循环等待电机到位
        ns = itertools.repeat(0, 150)
        for n in ns:
            self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
            self.myStep()  # 仿真一个步长
            if np.abs(self.positionSensors[19].getValue()-0.1)<0.05:
                rgb_raw = getImage(self.mCamera)
                break
        low = MaterialInfo['hsv']['low']
        high = MaterialInfo['hsv']['high']
        hsv = cv2.cvtColor(rgb_raw,cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv,np.array(low),np.array(high))
        contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        area = [cv2.contourArea(contour) for contour in contours]
        print( max(area))
        if max(area) > 13500:
            fname = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time())) 
            cv2.imwrite('./log/trap/' + fname + '.png', rgb_raw)
            return True
        else:
            fname = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time())) 
            cv2.imwrite('./log/bridge/' + fname + '.png', rgb_raw)
            return False

    def checkIfStairs(self,MaterialInfo):
        # 抬头看，视野里是否有大红，大红是最后一级台阶的颜色，而且只出现在楼梯上
        self.motors[19].setPosition(0.5)
        # 重复50个空循环等待电机到位
        ns = itertools.repeat(0, 150)
        for n in ns:
            self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
            self.myStep()  # 仿真一个步长
            if np.abs(self.positionSensors[19].getValue()-0.5)<0.05:
                rgb_raw = getImage(self.mCamera)
                break
        # pred, prob = call_classifier(rgb_raw, self.stage_classifier)
        # print(pred,prob)
        low =  [0,180,205] # 大红
        high = [255,255,255]
        hsv = cv2.cvtColor(rgb_raw,cv2.COLOR_BGR2HSV)
        mask=cv2.inRange(hsv,np.array(low),np.array(high))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1,1))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel,iterations=3)
        contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        num = 0
        # 不要用len(contours)，原因是可能有噪点。
        for cnt in contours:
            if cv2.contourArea(cnt) > 20:
                num += 1
        if num >= 2:
            fname = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time())) 
            cv2.imwrite('./log/stairs/' + fname + '.png', rgb_raw)
            return True
        else:
            return False
        
    def checkMineOrDoorOrBall(self,MaterialInfo):
        # 抬头看，视野里是否有很多黑色小块
        self.motors[19].setPosition(0.5)
        # 重复50个空循环等待电机到位
        ns = itertools.repeat(0, 150)
        for n in ns:
            self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
            self.myStep()  # 仿真一个步长
            if np.abs(self.positionSensors[19].getValue()-0.5)<0.05:
                rgb_raw = getImage(self.mCamera)
                break
        # pred, prob = call_classifier(rgb_raw, self.stage_classifier)
        # print(pred,prob)
        # 注意判断黑色不是直接检测黑色，而是把不属于黑色的都去掉，再取反。
        low =  [0,0,10]
        high = [255,255,255]
        hsv = cv2.cvtColor(rgb_raw,cv2.COLOR_BGR2HSV)
        mask=cv2.inRange(hsv,np.array(low),np.array(high))
        mask = 255 - mask
        contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        
        blackNums = 0
        blackAreaMax = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if  area > 5:
                blackNums += 1
                if area > blackAreaMax and tuple(contour[contour[:,:,1].argmin()][0])[1]< 60:
                    blackAreaMax = area
        # cv2.imwrite('rgb.png',rgb_raw)
        # cv2.imwrite('mask.png',mask)
        if blackNums >= 3 and blackAreaMax < 60 and 3 in self.stageLeft:
            fname = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time())) 
            cv2.imwrite('./log/mine/' + fname + '.png', rgb_raw)
            return 3 # 地雷关对应编号3
        elif blackAreaMax >= 125 and 9 in self.stageLeft:
            fname = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time())) 
            cv2.imwrite('./log/door/' + fname + '.png', rgb_raw)
            return 9 # 门对应编号9
        else:
            fname = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time())) 
            cv2.imwrite('./log/ball/' + fname + '.png', rgb_raw)
            return 6 # 踢球对应编号6

    # 通过关卡X并判定何时结束
    def passStageX(self,X=1,MaterialInfo=None,maxStep=np.inf,stopAfterFinish=True):
        '''
        X表示关卡序号:
        # 1 上下开合横杆 2 回字陷阱 3 地雷路段 4 翻越障碍
        # 5 窄桥 6 踢球 7 楼梯 8 水平开合横杆 9 窄门

        MaterialInfo表示当前路段材料信息:
        # {'material':'材料名','possible_stage':[可能关卡列表],'hsv':{'low':[hsv下界列表],'high':[hsv上界列表]}}

        maxStep表示最大步数，防止视觉判断错误之后会永远走下去

        stopAfterFinish:
        如果设置为True，则判定完成stageX后会停下，并等1s，实际比赛中不需要
        '''
        #如果是回字或独木桥或地雷或楼梯，直接pass了，因为策略有通关判别条件
        if self.stageNow in [2,3,5,7]:
            pass
        else:
            self.motors[19].setPosition(-0.2)
            # 重复50个空循环等待电机到位
            ns = itertools.repeat(0, 50)
            for n in ns:
                self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
                self.myStep()  # 仿真一个步长

            if not MaterialInfo:
                rgb_raw = getImage(self.mCamera)
                MaterialInfo = self.whereAmI(rgb_raw)
                print(MaterialInfo)

            ns = itertools.count(0)
            self.setMoveCommand(vX=1.)
            for n in ns:
                self.checkIfYaw()
                self.mGaitManager.step(self.mTimeStep)
                self.myStep()
                if n % 5 == 0 and np.abs(self.angle[0]) < 1.0:
                    rgb_raw = getImage(self.mCamera)
                    hsv = cv2.cvtColor(rgb_raw,cv2.COLOR_BGR2HSV)
                    low = MaterialInfo['hsv']['low']
                    high = MaterialInfo['hsv']['high']
                    mask=cv2.inRange(hsv[self.mCameraHeight//2:,:],np.array(low),np.array(high))
                    road = np.where(mask==255)[0]
                    num = len(road)
                    # cv2.imshow('image',mask)
                    # cv2.waitKey(0)
                    #cv2.imwrite('tmp/'+str(n)+'.png',rgb_raw)
                    #cv2.imwrite('tmp/'+str(n)+'_1.png',mask)
                    # 判定关卡结束条件，当前材料的hsv基本消失在视野中下段。或者行进步数超限。
                    if self.materialInfo['material'] == '黄色砖块':
                        brickAreaMean = brickArea(rgb_raw)
                        print(brickAreaMean)
                    if num < 500 or n > maxStep:
                        break
        if stopAfterFinish:
            self.setMoveCommand(vX=0.)
            self.mGaitManager.step(self.mTimeStep)
            self.myStep()
            self.checkIfYaw()
        stageLast = stageInfo[X]['name']
        print(f'已经通过关卡{stageLast}')
    
    # 判定下一关是哪一关
    def judgeNextStage(self,judgeCorner=True):
        # 返回结果储存在res字典中，只有判断是长形关卡（地雷、楼梯、门、踢球）时，turn_flag和obstacle_flag才生效
        # 因为转90度只有可能发生在长形地形处，高障碍物只有可能跟在长形地形后
        res = {'stage_num':-1,'turn_flag':False,'obstacle_flag':False}
        self.motors[19].setPosition(-0.2)
        # 重复50个空循环等待电机到位
        ns = itertools.repeat(0, 1000)
        for n in ns:
            rgb_raw = getImage(self.mCamera)
            self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
            self.myStep()  # 仿真一个步长
            if np.abs(self.positionSensors[19].getValue()+0.2)<0.05:  
                break
        MaterialInfo = self.whereAmI(rgb_raw)
        self.materialInfo = MaterialInfo
        # 绿色材料的地面可能是独木桥或回字陷阱
        if MaterialInfo['material'] == '绿色':
            if (self.checkIfTrap(MaterialInfo) and 2 in self.stageLeft) or 5 not in self.stageLeft:
                self.stageNow = 2
            else:
                self.stageNow = 5
            res['stage_num'] = self.stageNow
        # 这三种材料的地面会随机出现在 门、踢球和地雷关，即长形地形，其中黄色还会引导楼梯
        elif MaterialInfo['material'] == '黄色砖块':
            if self.checkIfStairs(MaterialInfo) and 7 in self.stageLeft: 
                self.stageNow = 7
            else:
                if self.checkIfCorner(MaterialInfo,judgeCorner):
                    res['turn_flag'] = True
                    print('要转弯啦')
                else:
                    # 不然只能在 门、踢球和地雷关 中出
                    self.stageNow =  self.checkMineOrDoorOrBall(MaterialInfo)
                    # 注意这三关后面都可能跟蓝色障碍物
                    if self.checkIfObstacle(MaterialInfo):
                        res['obstacle_flag'] = True
                        print('这关结束有蓝色障碍物哦')
        elif MaterialInfo['material'] == '灰色' or MaterialInfo['material'] == '白色' :
            if self.checkIfCorner(MaterialInfo,judgeCorner):
                res['turn_flag'] = True
                print('要转弯啦')
            else:
                # 不然只能在 门、踢球和地雷关 中出
                self.stageNow =  self.checkMineOrDoorOrBall(MaterialInfo)
                # 注意这三关后面都可能跟蓝色障碍物
                if self.checkIfObstacle(MaterialInfo):
                    res['obstacle_flag'] = True
                    print('这关结束有蓝色障碍物哦')
        # 草地和蓝色碎花只可能对应一种关卡
        else:
            self.stageNow = MaterialInfo['possible_stage'][0]
            res['stage_num'] = self.stageNow
        
        nextStageName = stageInfo[self.stageNow]['name']
        print(f'当前关卡:{nextStageName},当前地面材料:{self.materialInfo}')
        return res

    # 左转90度，进入下一个方向的比赛
    def turn90(self):
        print("~~~~~~~~~~~准备转弯~~~~~~~~~~~")
        # fix the bug : 转弯过程中撞到雷
        # 先看一眼路前面是不是有雷，是的话稍微往右靠靠再转
        if 3 in self.stageLeft:
            self.setMoveCommand(vX=0.0)
            self.motors[19].setPosition(-0.2)
            while np.abs(self.positionSensors[19].getValue()+0.2)>0.05:
                self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
                self.myStep()  # 仿真一个步长
                if self.positionSensors[19].getValue() > 0: continue
                rgb_raw = getImage(self.mCamera)
                low =  [0,0,15]
                high = [255,255,255]
                hsv = cv2.cvtColor(rgb_raw,cv2.COLOR_BGR2HSV)
                mask=cv2.inRange(hsv,np.array(low),np.array(high))
                mask = 255 - mask
                contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                dilei = []
                for contour in contours:
                    if cv2.contourArea(contour) > 5:
                        M = cv2.moments(contour)  # 计算第一条轮廓的各阶矩,字典形式
                        center_x = int(M["m10"] / M["m00"])
                        center_y = int(M["m01"] / M["m00"])
                        if 20 < center_x < 120:
                            dilei.append((center_x,center_y))
                if len(dilei):
                    while np.abs(self.angle[-1]+45) > 1:
                        u = -0.02 * (self.angle[-1]+45)
                        u = np.clip(u, -1, 1)
                        self.setMoveCommand(vX=0.5,vA=u)
                        self.mGaitManager.step(self.mTimeStep)
                        self.myStep()
                    cv2.imwrite('log/keySteps/转弯抬头望雷.png',rgb_raw)
                    break # 已经调整完，跳出
            if not len(dilei): print('无雷，可以安全转弯')
        
        # 正常转弯程序
        self.motors[19].setPosition(0.2)
        ns = itertools.count(0)
        for n in ns:
            u = -0.02 * (self.angle[-1])
            u = np.clip(u, -1, 1)
            self.setMoveCommand(vX=1.0,vA=u)
            self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
            self.myStep()  # 仿真一个步长
            if n % 5 == 0 and np.abs(self.positionSensors[19].getValue()-0.2)<0.05:
                rgb_raw = getImage(self.mCamera)
                hsv = cv2.cvtColor(rgb_raw,cv2.COLOR_BGR2HSV)
                low = self.materialInfo['hsv']['low']
                high = self.materialInfo['hsv']['high']
                mask=cv2.inRange(hsv[self.mCameraHeight//2:,:],np.array(low),np.array(high))
                road = np.where(mask==255)[0]
                num = len(road)
                if num < 500:
                    break
        print("~~~~~~~~~~~开始转弯~~~~~~~~~~~")
        self.setMoveCommand(vX=0.0)
        # for _ in range(50):
        #     self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
        #     self.myStep()  # 仿真一个步长
        # self.mGaitManager.stop()
        # self.wait(200)
        self.angle[-1] -= 87.5
        # self.angle[0]=0
        # self.angle[1]=0
        #self.mGaitManager.start()
        #self.wait(200)
        self.checkIfYaw()
        # self.mGaitManager.stop()
        # self.wait(200)
        # self.angle[0]=0
        # self.angle[1]=0
        # self.mGaitManager.start()
        # self.wait(500)
        self.turnCount += 1
        print("~~~~~~~~~~~转弯结束~~~~~~~~~~~")
             
    # 判断是否通过黄色砖块路面。不能单纯用颜色，有可能下关直接跟楼梯
    def checkIfPassBrick(self,rgb_raw):
        # 如果楼梯已经走完，则直接用黄色消失做判据即可
        if 7 not in self.stageLeft:
            hsv = cv2.cvtColor(rgb_raw,cv2.COLOR_BGR2HSV)
            low = raceTrackInfo[2]['hsv']['low'] # 黄色砖块
            high = raceTrackInfo[2]['hsv']['high']
            mask=cv2.inRange(hsv[self.mCameraHeight//2:,:],np.array(low),np.array(high))
            road = np.where(mask==255)[0]
            num = len(road)
            return True if num < 250 else False
        
        # 如果楼梯还没走，那么下一关是有可能接小黄砖引导的楼梯关卡的
        # 因此判据是 1）黄色消失 or 2）出现很多小黄砖
        else:
            hsv = cv2.cvtColor(rgb_raw,cv2.COLOR_BGR2HSV)
            low = raceTrackInfo[2]['hsv']['low'] # 黄色砖块
            high = raceTrackInfo[2]['hsv']['high']
            mask=cv2.inRange(hsv[self.mCameraHeight//2:,:],np.array(low),np.array(high))
            road = np.where(mask==255)[0]
            num = len(road)
            flag1 =  True if num < 250 else False
            if self.turnCount ==0:
                # 第一次转弯之前，光线的原因，会让影子遮住部分小砖块
                flag2 = True if calculateBrickNum(rgb_raw) > 9 else False
            else:
                flag2 = True if calculateBrickNum(rgb_raw) > 15 else False

            if flag1 or flag2:
                return True

    # 踢球向边缘走的时候，这一关是黄色地砖，上一关是楼梯，也是黄色地砖，而且一定是出现在转弯处。回身黄黄
    def checkIfHSHH(self,rgb_raw):
        # 不是黄砖，或上一关不是楼梯，或不在这关转过弯，不可能回身黄黄
        if self.materialInfo['material'] != '黄色砖块' or self.stagePass[-1]!=7 or not self.cornerAlready:
            return False
        else:
            count = calculateBrickNum(rgb_raw)
            print(f'进入回身黄黄判别程序,小砖块数{count}')
            if count >= 6 or self.hshhCount >= 45:
                return True
            else:
                self.hshhCount += 1
                return False
    
    
    def stop(self):
        self.setMoveCommand(vX=0.)
        for _ in range(50):
            self.mGaitManager.step(self.mTimeStep)
            self.myStep()
        self.mGaitManager.stop()
        self.wait(200)

    # 主函数循环
    def run(self):
        # 准备动作
        self.prepare()
        # ##############################键盘采数据####################################
        # self.motors[19].setPosition(-0.2)
        # self.keyBoardControl()
        # # ##############################完成通关程序############################################
        strategies = {
            1 : self.stage1,
            2 : self.stage2,
            3 : self.stage3,
            4 : self.stage4,
            5 : self.stage5,
            6 : self.stage6,
            7 : self.stage7,
            8 : self.stage8,
            9 : self.stage9,
        }
        while len(self.stageLeft):
            strategies[self.stageNow]()
            self.stagePass.append(self.stageNow)
            print(self.stagePass)
            res = self.judgeNextStage()
            self.cornerAlready = res['turn_flag']
            if self.cornerAlready:
                self.turn90()
                res = self.judgeNextStage(judgeCorner=False)
            self.obstacleBehind = res['obstacle_flag']
        ##########################################################################
        #单独测试任意一关，注意一定要把机器人挪动到上一关未完成处哦
        # self.stageLeft = {6,8}
        # strategies = {
        #     1 : self.stage1,
        #     2 : self.stage2,
        #     3 : self.stage3,
        #     4 : self.stage4,
        #     5 : self.stage5,
        #     6 : self.stage6,
        #     7 : self.stage7,
        #     8 : self.stage8,
        #     9 : self.stage9,
        # }
        # self.passStageX(self.stageNow)
        # res = self.judgeNextStage()
        # if res['turn_flag'] == True:
        #     self.turn90()
        #     res = self.judgeNextStage(judgeCorner=False)
        # self.obstacleBehind = res['obstacle_flag']
        # strategies[self.stageNow]()
        # #self.passStageX(self.stageNow,self.materialInfo)
        # res = self.judgeNextStage()
        # if res['turn_flag'] == True:
        #     self.turn90()
        #     res = self.judgeNextStage(judgeCorner=False)
        ##########################################################################
        # 通过第一关	上下开横杆
        # self.stage1()
        # 通过第二关	回字陷阱
        # self.stage2()
        # 通过第三关	地雷路段
        #self.materialInfo = {'material':'灰色','possible_stage':[3,9,6],'hsv':{'low':[35,0,150],'high':[40,20,255]}}
        #self.stage3()
        # 通过第四关	翻越障碍与过门
        # self.stage4()
        # self.prepare(waitingTime=500)
        # self.stage9()
        # 通过第五关	窄桥路段
        # self.materialInfo =  {'material':'绿色','possible_stage':[2,5],'hsv':{'low':[35, 43, 35],'high':[90, 255, 255]}}
        # self.stage5()
        # 通过第六关	踢球进洞
        # self.stageLeft = {6,8}
        # self.materialInfo = {'material':'白色','possible_stage':[3,9,6],'hsv':{'low':[10, 5, 200],'high':[30, 30, 255]}}
        # self.stage6()
        # 通过第七关	走楼梯
        # self.materialInfo = {'material':'黄色砖块','possible_stage':[3,9,7,6],'hsv':{'low':[15,100,50],'high':[34,255,255]}}
        # self.stage7()
        # # 通过第八关	水平开横杆
        # self.materialInfo = {'material':'蓝色碎花','possible_stage':[8],'hsv':{'low':[100, 10, 100],'high':[150, 80, 200]}}
        # self.stage8()

        # 停下
        self.stop()
        while True:
            self.mMotionManager.playPage(24)
        


if __name__ == '__main__':
    walk = Walk()  # 初始化Walk类
    walk.run()  # 运行控制器
