# -*- coding: UTF-8 -*-
from controller import Robot
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
		self.fdown = 0   # 定义两个类变量，用于之后判断机器人是否摔倒  
		
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
			
		
		
		self.mCamera = self.robot.getDevice("Camera") # 获取并初始化摄像头
		self.mCamera.enable(self. mTimeStep)
		self.mCameraHeight, self.mCameraWidth = self.mCamera.getHeight(), self.mCamera.getWidth()

		self.mKeyboard = self.robot.getKeyboard()  # 初始化键盘读入类
		self.mKeyboard.enable(self.mTimeStep)  # 以mTimeStep为周期从键盘读取

		self.mMotionManager = RobotisOp2MotionManager(self.robot)  # 初始化机器人动作组控制器
		self.mGaitManager = RobotisOp2GaitManager(self.robot, "config.ini")  # 初始化机器人步态控制器

		# 变量组
		self.angle = np.array([0.,0.,0.]) # 角度
		self.velocity = np.array([0.,0.,0.]) # 速度

		# 关卡1-7的预训练模型
		self.model1 = load_model('./pretrain/1.pth')

		self.model7 = load_model('./pretrain/7.pth')

		
	def myStep(self):
		ret = self.robot.step(self.mTimeStep)
		self.angle = updateAngle(self.angle,self.mTimeStep,self.mGyro)
		if ret == -1:
			exit(0)

	def wait(self, ms):
		startTime = self.robot.getTime()
		s = ms / 1000.0
		while s + startTime >= self.robot.getTime():
			self.myStep()
	
	def run(self):
		print('########Thu-bot Simulation########')
		self.robot.step(self.mTimeStep)  # 仿真一个步长，刷新传感器读数

		
		# 准备动作，保持机器人稳定
		print('Preparing...')
		self.mMotionManager.playPage(9)  # 执行动作组9号动作，初始化站立姿势，准备行走
		self.wait(500)  # 等待1s
		self.mGaitManager.stop()
		self.wait(500)  # 等待2s
		print('Ready to Play!')
		# 开始运动
		self.mGaitManager.setBalanceEnable(True)
		self.mGaitManager.start()

		# 通过第一关
		self.stage1()
		# 通过第七关
		#self.stage7()

		# 键盘控制与采集数据
		#self.keyBoardControl()
		
		# 停下
		self.mGaitManager.stop()
		self.wait(200)
		self.mMotionManager.playPage(24)
		while True:
			self.mGaitManager.step(self.mTimeStep)
			self.myStep()

			
	def checkIfYaw(self, threshold=7.5):
		if self.angle[-1] > threshold:
			print('Yaw Anticlockwise %.3f°, Start Correction Program...'%self.angle[-1])
			while np.abs(self.angle[-1]) > 0.25:
				self.mGaitManager.setAAmplitude(-0.25)
				self.mGaitManager.step(self.mTimeStep)
				self.myStep()
			print('Yaw Anticlockwise %.3f°, Finish Correction Program!'%self.angle[-1])
			self.mGaitManager.setAAmplitude(0.0)
			self.mGaitManager.step(self.mTimeStep)
			self.myStep()
		elif self.angle[-1] < -threshold:
			print('Yaw Clockwise %.3f°, Start Correction Program...'%self.angle[-1])
			while np.abs(self.angle[-1]) > 0.25:
				self.mGaitManager.setAAmplitude(0.25)
				self.mGaitManager.step(self.mTimeStep)
				self.myStep()
			print('Yaw Clockwise %.3f°, Finish Correction Program!'%self.angle[-1])
			self.mGaitManager.setAAmplitude(0.0)
			self.mGaitManager.step(self.mTimeStep)
			self.myStep()
		else:
			pass

	
	def checkIfFallen(self):
		acc_tolerance = 60.0
		acc_step = 100  # 计数器上限
		acc = self.mAccelerometer.getValues()  # 通过加速度传感器获取三轴的对应值
		if acc[1] < 512.0 - acc_tolerance :  # 面朝下倒地时y轴的值会变小
			self.fup += 1  # 计数器加1
		else :
			self.fup = 0  # 计数器清零
		if acc[1] > 512.0 + acc_tolerance : # 背朝下倒地时y轴的值会变大
			self.fdown += 1 # 计数器加 1
		else :
			self.fdown = 0 # 计数器清零
		
		if self.fup > acc_step :   # 计数器超过100，即倒地时间超过100个仿真步长
			self.mMotionManager.playPage(10) # 执行面朝下倒地起身动作
			self.mMotionManager.playPage(9) # 恢复准备行走姿势
			self.fup = 0 # 计数器清零
			return True
		elif self.fdown > acc_step :
			self.mMotionManager.playPage(11) # 执行背朝下倒地起身动作
			self.mMotionManager.playPage(9) # 恢复准备行走姿势
			self.fdown = 0 # 计数器清零
			return True
		else:
			return False

	def keyBoardControl(self):
		self.isWalking = True  # 初始时机器人未进入行走状态
		ns = itertools.count(0)
		for n in ns:
			if n % 4 == 0 :
				rgb_raw = getImage(self.mCamera)
				print('save image')
				cv2.imwrite('./tmp/'+str(n)+'.png',rgb_raw)
			self.checkIfFallen()
			self.mGaitManager.setXAmplitude(0.0)  # 前进为0
			self.mGaitManager.setAAmplitude(0.0)  # 转体为0
			key = 0  # 初始键盘读入默认为0
			key = self.mKeyboard.getKey()  # 从键盘读取输入
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


	def stage1(self):
		print('########Stage0_Start########')
		crossBarDownFlag = False
		goFlag = False
		self.mGaitManager.setXAmplitude(0.0)  # 前进为0
		self.mGaitManager.setAAmplitude(0.0)  # 转体为0
		
		ns = itertools.count(0)
		for n in ns:
			self.checkIfYaw()
			if n % 100 == 0:
				rgb_raw = getImage(self.mCamera)
				pred,prob = call_classifier(rgb_raw,self.model1)
				if not crossBarDownFlag:
					if pred == 1:
						crossBarDownFlag = True
						print('CrossBar already Down with probablity %.3f'%prob)
					else:
						print('Wait for CrossBar Down with probablity %.3f ...'%prob)
				else:
					if pred == 0:
						goFlag = True
						print('CrossBar already UP with probablity %.3f, Go Go Go!'%prob)
					else:
						print('Wait for CrossBar Up with probablity %.3f ...'%prob)
			if goFlag:
				self.mGaitManager.setXAmplitude(1.0)  # 前进为0
				self.mGaitManager.setAAmplitude(0.0)  # 转体为0
				#self.motors[18].setPosition(radians(50))
				#self.motors[19].setPosition(radians(30))
				break
			self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
			self.myStep()  # 仿真一个步长
		n0 = n
		for n in ns:
			self.checkIfYaw()
			# 持续走一段时间，写死了这里
			if (n-n0) >= 900:
				break
			self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
			self.myStep()  # 仿真一个步长
		print('########Stage0_End########')

	# def stage2(self):
	# 	print('########Stage2########')
	# 	self.motors[19].setPosition(radians(10))
	# 	self.mGaitManager.setXAmplitude(0.0)  # 前进为0
	# 	self.mGaitManager.setAAmplitude(0.0)  # 转体为0
	# 	ns = itertools.count(0)
	# 	for n in ns:
	# 		self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
	# 		self.myStep()  # 仿真一个步长

	# def stage3(self):
	# 	print('########Stage3########')
	# 	self.mGaitManager.setYAmplitude(1.0)  # 前进为0
	# 	self.mGaitManager.setAAmplitude(0.0)  # 转体为0

	# 	ns = itertools.count(0)
	# 	for n in ns:
	# 		self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
	# 		self.myStep()  # 仿真一个步长
	def stage7(self):
		print('########Stage7_Start########')
		crossBarCloseFlag = False
		goFlag = False
		self.mGaitManager.setXAmplitude(0.0)  # 前进为0
		self.mGaitManager.setAAmplitude(0.0)  # 转体为0
		
		ns = itertools.count(0)
		for n in ns:
			self.checkIfYaw()
			if n % 100 == 0:
				rgb_raw = getImage(self.mCamera)
				pred,prob = call_classifier(rgb_raw,self.model7)
				print(pred,prob)
				if not crossBarCloseFlag:
					if pred == 1:
						crossBarCloseFlag = True
						print('CrossBar already Close with probablity %.3f'%prob)
					else:
						print('Wait for CrossBar Close with probablity %.3f ...'%prob)
				else:
					if pred == 0:
						goFlag = True
						print('CrossBar already Open with probablity %.3f, Go Go Go!'%prob)
					else:
						print('Wait for CrossBar Open with probablity %.3f ...'%prob)
			if goFlag:
				self.mGaitManager.setXAmplitude(1.0)  # 前进为0
				self.mGaitManager.setAAmplitude(0.0)  # 转体为0
				#self.motors[18].setPosition(radians(50))
				#self.motors[19].setPosition(radians(30))
				break
			self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
			self.myStep()  # 仿真一个步长
		n0 = n
		for n in ns:
			self.checkIfYaw()
			# 持续走一段时间，写死了这里
			if (n-n0) >= 1000:
				break
			self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
			self.myStep()  # 仿真一个步长
		print('########Stage7_End########')

if __name__ == '__main__':
	walk = Walk()  # 初始化Walk类
	walk.run()  # 运行控制器