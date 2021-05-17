# -*- coding: UTF-8 -*-
from controller import Robot, Motion
import os
import sys

libraryPath = os.path.join(os.environ.get("WEBOTS_HOME"), 'projects', 'robots', 'robotis', 'darwin-op', 'libraries',
                           'python37')
libraryPath = libraryPath.replace('/', os.sep)
sys.path.append(libraryPath)
from managers import RobotisOp2GaitManager, RobotisOp2MotionManager

import cv2
import numpy as np
from math import radians
import itertools
from utils import *
from yolo import *


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
        self.fup = 0
        self.fdown = 0  # 定义两个类变量，用于之后判断机器人是否摔倒

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

    # 加载关卡1-7的预训练模型
    # self.model1 = load_model('./pretrain/1.pth')
    # self.model2 = load_model('./pretrain/2.pth')

    # 执行一个仿真步。同时每步更新机器人偏航角度。
    def myStep(self):
        ret = self.robot.step(self.mTimeStep)
        self.angle = updateAngle(self.angle, self.mTimeStep, self.mGyro)
        if ret == -1:
            exit(0)

    # 保持动作若干ms
    def wait(self, ms):
        startTime = self.robot.getTime()
        s = ms / 1000.0
        while s + startTime >= self.robot.getTime():
            self.myStep()

    # 设置前进、水平、转动方向的速度指令，大小在-1到1之间。
    # 硬更新，不建议直接使用
    def setMoveCommand(self, vX=None, vY=None, vA=None):
        if vX != None:
            self.vX = np.clip(vX, -1, 1)
            self.mGaitManager.setXAmplitude(self.vX)  # 设置x向速度（直线方向）
        if vY != None:
            self.vY = np.clip(vY, -1, 1)
            self.mGaitManager.setYAmplitude(self.vY)  # 设置y向速度（水平方向）
        if vA != None:
            self.vA = np.clip(vA, -1, 1)
            self.mGaitManager.setAAmplitude(self.vA)  # 设置转动速度

    # 设置前进速度，软更新 v = (1-α)*v + α*Target
    def setForwardSpeed(self, target, threshold=0.05, tau=0.1):
        current = self.vX
        target = np.clip(target, -1, 1)
        while np.abs(current - target) > threshold:
            current = (1 - tau) * current + tau * target
            self.setMoveCommand(vX=current)
            self.mGaitManager.step(self.mTimeStep)
            self.myStep()
        self.setMoveCommand(vX=target)
        self.mGaitManager.step(self.mTimeStep)
        self.myStep()

    # 设置侧向速度，软更新 v = (1-α)*v + α*Target
    def setSideSpeed(self, target, threshold=0.05, tau=0.1):
        current = self.vY
        target = np.clip(target, -1, 1)
        while np.abs(current - target) > threshold:
            current = (1 - tau) * current + tau * target
            self.setMoveCommand(vY=current)
            self.mGaitManager.step(self.mTimeStep)
            self.myStep()
        self.setMoveCommand(vY=target)
        self.mGaitManager.step(self.mTimeStep)
        self.myStep()

    # 设置转动速度，软更新 v = (1-α)*v + α*Target
    def setRotationSpeed(self, target, threshold=0.05, tau=0.1):
        current = self.vA
        target = np.clip(target, -1, 1)
        while np.abs(current - target) > threshold:
            current = (1 - tau) * current + tau * target
            self.setMoveCommand(vA=current)
            self.mGaitManager.step(self.mTimeStep)
            self.myStep()
        self.setMoveCommand(vA=target)
        self.mGaitManager.step(self.mTimeStep)
        self.myStep()

    # 通过PID控制机器人转体到target角度
    def setRotation(self, target, threshold=0.5, Kp=0.1):
        self.setForwardSpeed(0.)
        self.setSideSpeed(0.)
        self.setRotationSpeed(0.)

        while np.abs(self.angle[-1] - target) > threshold:
            u = Kp * (target - self.angle[-1])
            u = np.clip(u, -1, 1)
            self.setMoveCommand(vA=u)
            self.mGaitManager.step(self.mTimeStep)
            self.myStep()

        self.setForwardSpeed(0.)
        self.setSideSpeed(0.)
        self.setRotationSpeed(0.)
        return self.angle[-1]

    def setRobotStop(self):
        self.setForwardSpeed(0.)
        self.setSideSpeed(0.)
        self.setRotationSpeed(0.)

    def setRobotRun(self, speed=1.):
        self.setForwardSpeed(speed)
        self.setSideSpeed(0.)
        self.setRotationSpeed(0.)

    # 检查偏航是否超过指定threshold，若是则修正回0°附近，由eps指定
    def checkIfYaw(self, threshold=7.5, eps=0.25):
        if np.abs(self.angle[-1]) > threshold:
            print('Current Yaw is %.3f, begin correction' % self.angle[-1])
            while np.abs(self.angle[-1]) > eps:
                u = -0.02 * self.angle[-1]
                u = np.clip(u, -1, 1)
                self.setMoveCommand(vA=u)
                self.mGaitManager.step(self.mTimeStep)
                self.myStep()
            print('Current Yaw is %.3f, finish correction' % self.angle[-1])
        else:
            pass

    # 比赛开始前准备动作，保持机器人运行稳定和传感器读数稳定
    def prepare(self, waitingTime):
        print('########Preparation_Start########')
        # 仿真一个步长，刷新传感器读数
        self.robot.step(self.mTimeStep)
        # 准备动作
        print('Preparing...')
        self.mMotionManager.playPage(9)  # 执行动作组9号动作，初始化站立姿势，准备行走
        self.wait(500)
        # 开始运动
        self.mGaitManager.setBalanceEnable(True)
        self.mGaitManager.start()
        self.setRobotStop()
        self.wait(waitingTime)
        print('Ready to Play!')
        print('Initial Yaw is %.3f' % self.angle[-1])
        print('########Preparation_End########')

    # 构造Z字轨迹
    def Z(self, angle, interval):
        speed = self.vX
        # 左转45度
        self.setRotation(angle)
        self.setForwardSpeed(speed)
        ns = itertools.repeat(0, interval)
        for n in ns:
            self.mGaitManager.step(self.mTimeStep)
            self.myStep()
        self.setRotation(0)
        self.setRobotRun(speed)  # 保存原来的行进速度

    def keyBoardControl(self, collet_data=False, rotation=True):
        self.isWalking = True
        ns = itertools.count(0)
        for n in ns:
            self.mGaitManager.setXAmplitude(0.0)  # 前进为0
            self.mGaitManager.setAAmplitude(0.0)  # 转体为0
            key = 0  # 初始键盘读入默认为0
            key = self.mKeyboard.getKey()  # 从键盘读取输入
            if collet_data and n % 53 == 0:
                rgb_raw = getImage(self.mCamera)
                print('save image')
                cv2.imwrite('./tmp/c_' + str(n) + '.png', rgb_raw)
            if key == 49:
                pos = self.positionSensors[19].getValue()
                self.motors[19].setPosition(np.clip(pos + 0.05, -0.25, 0.4))
            elif key == 50:
                pos = self.positionSensors[19].getValue()
                self.motors[19].setPosition(np.clip(pos - 0.05, -0.25, 0.4))
            elif key == 51:
                if collet_data == 1:
                    collet_data = 0
                if collet_data == 0:
                    collet_data = 1
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

    # 水平开合横杆
    def stage1(self):
        print('########Stage1_Start########')
        crossBarDownFlag = False
        goFlag = False
        ns = itertools.count(0)
        for n in ns:
            self.checkIfYaw()
            if n % 100 == 0:
                rgb_raw = getImage(self.mCamera)
                pred, prob = call_classifier(rgb_raw, self.model1)
                if not crossBarDownFlag:
                    if pred == 1:
                        crossBarDownFlag = True
                        print('CrossBar already Down with probablity %.3f' % prob)
                    else:
                        print('Wait for CrossBar Down with probablity %.3f ...' % prob)
                else:
                    if pred == 0:
                        goFlag = True
                        print('CrossBar already UP with probablity %.3f, Go Go Go!' % prob)
                    else:
                        print('Wait for CrossBar Up with probablity %.3f ...' % prob)
            if goFlag:
                self.setRobotRun()
                # self.motors[18].setPosition(radians(50))
                # self.motors[19].setPosition(radians(30))
                break
            self.mGaitManager.step(self.mTimeStep)
            self.myStep()
        n0 = n
        for n in ns:
            self.checkIfYaw()
            # 持续走一段时间，写死了这里
            if (n - n0) >= 850:
                break
            self.mGaitManager.step(self.mTimeStep)
            self.myStep()
        print('########Stage1_End########')

    # 回字陷阱
    def stage2(self):
        print('########Stage2_Start########')
        self.setRobotStop()
        self.checkIfYaw(threshold=3.0)
        ns = itertools.count(0)
        center_y, center_x = None, None
        for n in ns:
            if n % 5 == 0:
                rgb_raw = getImage(self.mCamera)
                binary = call_segmentor(rgb_raw, self.model2)
                trap = np.argwhere(binary == 255)
                tmp_y, tmp_x = np.mean(trap, axis=0)
                if not center_y and not center_x:
                    center_y, center_x = tmp_y, tmp_x
                else:
                    center_y = 0.9 * center_y + 0.1 * tmp_y
                    center_x = 0.9 * center_x + 0.1 * tmp_x
            if center_x < 75:
                self.setSideSpeed(0.5)
                self.checkIfYaw(threshold=3.0)
            elif center_x > 85:
                self.setSideSpeed(-0.5)
                self.checkIfYaw(threshold=3.0)
            else:
                self.setSideSpeed(0.)
                self.checkIfYaw(threshold=3.5)
                print('Align Trap CenterX : %d CenterY : %d' % (center_x, center_y))
                break
            self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
            self.myStep()  # 仿真一个步长

        self.setRobotRun()
        self.Z(angle=45, interval=275)
        ns = itertools.repeat(0, 500)
        for n in ns:
            self.checkIfYaw(threshold=3.0)
            self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
            self.myStep()  # 仿真一个步长
        self.setRobotRun(0.5)
        self.Z(angle=-45, interval=500)
        print('########Stage2_End########')

    # 地雷路段
    def stage3(self):
        print('########Stage3_Start########')
        self.setRobotRun(1.0)
        self.motors[19].setPosition(radians(0))
        yolonet, yolodecoders = load_yolo('./pretrain/3.pth', class_names=['dilei'])
        ns = itertools.count(0)
        for n in ns:
            if n % 5 == 0 and np.abs(self.angle[0]) < 1.0:
                rgb_raw = getImage(self.mCamera)
                # cv2.imwrite('./tmp/'+str(n)+'.png',rgb_raw)
                res = call_yolo(rgb_raw, yolonet, yolodecoders, class_names=['dilei'])
                if not res:
                    print('%d Mine(s) has been Detected' % len(res))
                # 只看视野一定范围内的地雷，过远的忽略
                centers = []
                for info in res:
                    top, left, down, right = info['bbox']
                    centers.append([(left + right) / 2, (top + down) / 2])
                centers = [center for center in centers if center[1] > 0.5 * self.mCameraHeight]
                if not centers:
                    pass
                else:
                    # 看一下地雷距离视野中轴线的水平距离，判定危险级
                    danger = [center[0] - self.mCameraWidth / 2 for center in centers]
                    danger = sorted(danger, key=lambda x: np.abs(x))
                    if np.abs(danger[0]) < 45:
                        print('Warning!')
                        # self.Z(0,interval=20)
                        if len(danger) == 1:
                            self.Z(30, interval=200) if danger[0] >= 0 else self.Z(-45, interval=100)
                        else:
                            self.Z(30, interval=200) if danger[1] >= 0 else self.Z(-45, interval=100)
                    else:
                        pass
            self.checkIfYaw(threshold=5.)
            self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
            self.myStep()  # 仿真一个步长
        print('########Stage3_End########')

    # 翻越障碍
    def stage4(self):
        print('########Stage4_Start########')
        # 调整头部位置，低下头看
        self.setRobotRun(0.0)
        self.mGaitManager.stop()
        self.motors[19].setPosition(-0.3)
        # 重复50个空循环等待电机到位
        ns = itertools.repeat(0, 50)
        for n in ns:
            self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
            self.myStep()  # 仿真一个步长
        # rgb_raw = getImage(self.mCamera)
        # cv2.imwrite('./tmp/'+str(n)+'.png',rgb_raw)

        self.mGaitManager.start()
        self.setRobotRun(1.0)
        ns = itertools.count(0)
        for n in ns:
            self.checkIfYaw()
            if np.abs(self.angle[0]) < 1.0:
                rgb_raw = getImage(self.mCamera)
                ob_x, ob_y = obstacleDetect(rgb_raw)
                if ob_y > 0:
                    self.setRobotRun(0.5)  # 看到蓝色障碍物注意减速
                if ob_y > 75:  # 蓝色障碍物出现在正确位置，跳出循环，准备空翻
                    break
            self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
            self.myStep()  # 仿真一个步长

        # 空翻与起身
        self.setRobotRun(0.0)
        self.mGaitManager.stop()
        rollMotion = Motion('./motion/roll.motion')
        rollMotion.setLoop(False)
        rollMotion.play()
        while not rollMotion.isOver():
            self.myStep()
        self.mMotionManager.playPage(11)
        self.mMotionManager.playPage(9)

        # 陀螺仪累计数据全部清空
        self.angle = np.array([0., 0., 0.])

        # 往后退两步
        self.mGaitManager.start()
        self.setForwardSpeed(-0.1)
        ns = itertools.repeat(0, 500)
        for n in ns:
            self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
            self.myStep()  # 仿真一个步长

        self.setForwardSpeed(-0.05)
        ns = itertools.count(0)
        for n in ns:
            if np.abs(self.angle[-1]) < 1.0:
                rgb_raw = getImage(self.mCamera)
                cv2.imwrite('./tmp/' + str(n) + '.png', rgb_raw)
            self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
            self.myStep()  # 仿真一个步长
            if n == 1000:
                break

        # # 起身后转向90度，开始下一阶段任务
        # self.mGaitManager.start()
        # self.setRotation(20)
        # self.wait(100)
        # self.setRotation(40)
        # self.wait(100)
        # self.setRotation(80)
        # self.wait(100)

        # # 抬头，准备对齐赛道
        # self.motors[19].setPosition(0.6)
        # # 重复50个空循环等待电机到位
        # ns = itertools.repeat(0,50000)
        # for n in ns:
        # 	self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
        # 	self.myStep()  # 仿真一个步长
        print('########Stage4_End########')

    # 过桥
    def stage5(self):
        ball_color = 'green'
        color_dist = {'red': {'Lower': np.array([0, 60, 60]), 'Upper': np.array([6, 255, 255])},
                      'blue': {'Lower': np.array([100, 80, 46]), 'Upper': np.array([124, 255, 255])},
                      'green': {'Lower': np.array([35, 43, 35]), 'Upper': np.array([90, 255, 255])},
                      }
        ns = itertools.count(1)
        turn_flag = 0
        count = 0
        for n in ns:
            action_flag = 0

            if n % 20 == 0:
                # img processing
                rgb_raw = getImage(self.mCamera)
                rgb_raw = cv2.cvtColor(rgb_raw, cv2.COLOR_BGR2RGB)
                hsv = cv2.cvtColor(rgb_raw, cv2.COLOR_BGR2HSV)
                erode_hsv = cv2.erode(hsv, None, iterations=2)
                inRange_hsv = cv2.inRange(erode_hsv, color_dist[ball_color]['Lower'], color_dist[ball_color]['Upper'])
                cnts = cv2.findContours(inRange_hsv.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

                if cnts:
                    c = max(cnts, key=cv2.contourArea)
                    rect = cv2.minAreaRect(c)
                    box = cv2.boxPoints(rect)

                    mid_point = np.mean(box, 0)

                if turn_flag == 0:
                    if mid_point[0] < 75:
                        self.turn_left(0.2)
                        action_flag = 1
                    if mid_point[0] > 85:
                        self.turn_right(0.2)
                        action_flag = 1
                    if mid_point[1] >= 110 and turn_flag == 0:
                        turn_flag = 1

                if turn_flag == 1:
                    count += 1
                    if count == 600:
                        break

                if not action_flag:
                    self.mGaitManager.setXAmplitude(0.8)  # 前进为0
                    self.mGaitManager.setAAmplitude(0.0)  # 转体为0
                    self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
                    self.myStep()  # 仿真一个步长

    def turn_right(self, vel=0.3):
        self.setMoveCommand(vA=-vel, vX=0., vY=0.)
        self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
        self.myStep()

    def turn_left(self, vel=0.3):
        self.setMoveCommand(vA=vel, vX=0., vY=0.)
        self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
        self.myStep()

    # 踢球进洞
    def stage6(self):
        point_size = 1
        point_color1 = (0, 0, 255)  # BGR
        point_color2 = (0, 255, 0)
        thickness = 4

        print('########Stage6_Start########')
        # self.mMotionManager.playPage(1)
        yolonet_ball, yolodecoders_ball = load_yolo('./pretrain/ball.pth', class_names=['ball', 'hole'])
        yolonet_hole, yolodecoders_hole = load_yolo('./pretrain/hole.pth', class_names=['ball', 'hole'])
        self.setRobotRun(0.0)
        self.setSideSpeed(1.0)
        headPos = -0.1
        self.motors[19].setPosition(headPos)
        ns = itertools.count(0)
        # flag
        turn_flag = 0

        for n in ns:
            action_flag = 0
            if n % 2 == 0:
                rgb_raw = getImage(self.mCamera)
                rgb_raw = cv2.cvtColor(rgb_raw, cv2.COLOR_BGR2RGB)
                res_ball = call_yolo(rgb_raw, yolonet_ball, yolodecoders_ball, class_names=['ball', 'hole'],
                                     confidence=0.8)
                res_hole = call_yolo(rgb_raw, yolonet_hole, yolodecoders_hole, class_names=['ball', 'hole'],
                                     confidence=0.1)

                if res_hole:
                    for info in res_hole:
                        top, left, down, right = info['bbox']
                        center_hole_x, center_hole_y = (left + right) / 2, (top + down) / 2
                        # print('hole',[center_hole_x, center_hole_y])
                        hole_pos = [center_hole_x, center_hole_y]
                    # cv2.circle(rgb_raw, (int(center_hole_x),int(center_hole_y)), point_size, point_color2, thickness)

                ball_pos = []
                if res_ball:
                    for info in res_ball:
                        top, left, down, right = info['bbox']
                        center_ball_x, center_ball_y = (left + right) / 2, (top + down) / 2
                        # print('ball',[center_ball_x, center_ball_y])
                        ball_pos.append([center_ball_x, center_ball_y])

                    if len(ball_pos) > 1:
                        dis1 = np.linalg.norm(np.array(ball_pos[0]) - np.array(hole_pos))
                        dis2 = np.linalg.norm(np.array(ball_pos[1]) - np.array(hole_pos))
                        ball_pos = ball_pos[0] if dis1 >= dis2 else ball_pos[1]
                    else:
                        ball_pos = ball_pos[0]
                # cv2.circle(rgb_raw, (int(ball_pos[0]),int(ball_pos[1])), point_size, point_color1, thickness)

            # cv2.imshow('image', rgb_raw)
            # cv2.waitKey(0)

            if turn_flag == 0:
                if ball_pos:
                    print('ball_pos', ball_pos)
                    if ball_pos[0] < 75:
                        self.turn_left(0.2)
                        action_flag = 1
                    if ball_pos[0] > 85:
                        self.turn_right(0.2)
                        action_flag = 1
                    if ball_pos[1] >= 90 and turn_flag == 0:
                        turn_flag = 1

            if turn_flag == 1:
                if hole_pos:
                    print('hole_pos', hole_pos)
                if ball_pos:
                    print('ball_pos', ball_pos)
                if ball_pos and hole_pos:
                    if ball_pos[0] < hole_pos[0] - 3:
                        self.turn_right(0.2)
                        action_flag = 1
                    elif ball_pos[0] >= hole_pos[0] + 3:
                        self.turn_left(0.2)
                        action_flag = 1
                    else:
                        turn_flag = 3
                elif hole_pos:
                    if hole_pos[0] < 75:
                        self.turn_left(0.2)
                        action_flag = 1
                    elif hole_pos[0] > 85:
                        self.turn_right(0.2)
                        action_flag = 1
                    else:
                        turn_flag = 2

            if turn_flag == 2:
                if ball_pos:
                    print('ball_pos', ball_pos)
                    if ball_pos[0] < 70:
                        self.setMoveCommand(vY=0.5, vX=0., vA=0.)
                        self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
                        self.myStep()
                        action_flag = 1
                    elif ball_pos[0] > 80:
                        self.setMoveCommand(vY=-0.5, vX=0., vA=0.)
                        self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
                        self.myStep()
                        action_flag = 1
                    else:
                        turn_flag = 1
                        print(turn_flag)
                else:
                    self.setMoveCommand(vY=0.5, vX=0., vA=0.)
                    self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
                    self.myStep()
                    action_flag = 1

            if turn_flag == 3:
                self.mMotionManager.playPage(9)  # 执行动作组9号动作，初始化站立姿势，准备行走
                self.wait(2000)
                stepMotion = Motion('motion/kick.motion')
                stepMotion.setLoop(True)
                stepMotion.play()
                while not stepMotion.isOver():
                    self.myStep()
                print('########Stage6_End########')
                self.setRobotStop()
                self.mGaitManager.stop()
                self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
                self.myStep()  # 仿真一个步长

            if not action_flag:
                self.setMoveCommand(vX=0.8)
                self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
                self.myStep()  # 仿真一个步

    # 走楼梯
    def stage7(self):
        print('########Stage7_Start########')
        # 前面把南科大怼楼梯的动作加上就很稳了
        stepMotion = Motion('motion/stair_up.motion')
        stepMotion.setLoop(False)
        stepMotion.play()
        while not stepMotion.isOver():
            self.myStep()
        # 陀螺仪累计数据全部清空
        self.angle = np.array([0., 0., 0.])
        stepMotion = Motion('motion/stair_down.motion')
        stepMotion.setLoop(False)
        stepMotion.play()
        while not stepMotion.isOver():
            self.myStep()
        stepMotion.play()
        while not stepMotion.isOver():
            self.myStep()
        print(self.angle[-1])
        self.mGaitManager.start()
        self.setForwardSpeed(0.0)
        self.checkIfYaw(threshold=3.0)
        self.setForwardSpeed(1.0)
        ns = itertools.repeat(0, 700)
        for n in ns:
            self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
            self.myStep()  # 仿真一个步长
        self.setRobotStop()
        self.mGaitManager.stop()
        print('########Stage7_End########')

    # 上下开合横杆
    def stage8(self):
        print('########Stage8_Start########')
        stage8model = load_model('./pretrain/8.pth')
        crossBarCloseFlag = False
        goFlag = False
        ns = itertools.count(0)
        for n in ns:
            self.checkIfYaw()
            if n % 100 == 0:
                rgb_raw = getImage(self.mCamera)
                pred, prob = call_classifier(rgb_raw, stage8model)
                if not crossBarCloseFlag:
                    if pred == 1:
                        crossBarCloseFlag = True
                        print('CrossBar already Close with probablity %.3f' % prob)
                    else:
                        print('Wait for CrossBar Close with probablity %.3f ...' % prob)
                else:
                    if pred == 0:
                        goFlag = True
                        print('CrossBar already Open with probablity %.3f, Go Go Go!' % prob)
                    else:
                        print('Wait for CrossBar Open with probablity %.3f ...' % prob)
            if goFlag:
                self.mGaitManager.start()
                self.setRobotRun()
                break
            self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
            self.myStep()  # 仿真一个步长
        n0 = n
        for n in ns:
            self.checkIfYaw()
            # 持续走一段时间，写死了这里
            if (n - n0) >= 1000:
                break
            self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
            self.myStep()  # 仿真一个步长
        del stage8model
        print('########Stage8_End########')

    # 主函数循环
    def run(self):
        # 准备动作
        self.prepare(waitingTime=500)

        # 通过第一关	上下开横杆
        # self.stage1()
        # 通过第二关	回字陷阱
        # self.stage2()
        # 通过第三关	地雷路段
        # self.stage3()
        # 通过第四关	翻越障碍与过门
        # self.stage4()
        # 通过第五关	窄桥路段
        self.stage5()
        # 通过第六关	踢球进洞
        self.stage6()
        # 通过第七关	走楼梯
        # self.stage7()
        # 通过第八关	水平开横杆
        # self.stage8()
        # 键盘控制与采集数据
        # self.keyBoardControl(collet_data=0,rotation=1)

        # 停下
        self.mMotionManager.playPage(1)
        self.mGaitManager.stop()

        while True:
            # self.checkIfYaw(threshold=3.0)
            self.mGaitManager.step(self.mTimeStep)
            self.myStep()


if __name__ == '__main__':
    walk = Walk()  # 初始化Walk类
    walk.run()  # 运行控制器
