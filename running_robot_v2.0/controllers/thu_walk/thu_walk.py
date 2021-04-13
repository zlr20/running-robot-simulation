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

		# 初始化变量组
		self.vX, self.vY, self.vA = 0.,0.,0. # 前进、水平、转动方向的设定速度（-1，1）
		self.angle = np.array([0.,0.,0.]) # 角度， 由陀螺仪积分获得
		self.velocity = np.array([0.,0.,0.]) # 速度， 由加速度计积分获得

		# 加载关卡1-7的预训练模型
		self.model1 = load_model('./pretrain/1.pth')
		self.model2 = load_model('./pretrain/2.pth')
		self.model7 = load_model('./pretrain/7.pth')


	# 执行一个仿真步。同时每步更新机器人偏航角度。
	def myStep(self):
		ret = self.robot.step(self.mTimeStep)
		self.angle = updateAngle(self.angle,self.mTimeStep,self.mGyro)
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
	def setMoveCommand(self,vX=None,vY=None,vA=None):
		if vX != None:
			self.vX = np.clip(vX,-1,1)
			self.mGaitManager.setXAmplitude(self.vX)  # 设置x向速度（直线方向）
		if vY != None:
			self.vY = np.clip(vY,-1,1)
			self.mGaitManager.setYAmplitude(self.vY)  # 设置y向速度（水平方向）
		if vA != None:
			self.vA = np.clip(vA,-1,1)
			self.mGaitManager.setAAmplitude(self.vA)  # 设置转动速度
	
	# 设置前进速度，软更新 v = (1-α)*v + α*Target
	def setForwardSpeed(self, target, threshold=0.05, tau=0.1):
		current = self.vX
		target = np.clip(target,-1,1)
		while np.abs(current - target) > threshold:
			current = (1-tau)*current + tau*target
			self.setMoveCommand(vX=current)
			self.mGaitManager.step(self.mTimeStep)
			self.myStep()
		self.setMoveCommand(vX=target)
		self.mGaitManager.step(self.mTimeStep)
		self.myStep()

	# 设置侧向速度，软更新 v = (1-α)*v + α*Target
	def setSideSpeed(self, target, threshold=0.05, tau=0.1):
		current = self.vY
		target = np.clip(target,-1,1)
		while np.abs(current - target) > threshold:
			current = (1-tau)*current + tau*target
			self.setMoveCommand(vY=current)
			self.mGaitManager.step(self.mTimeStep)
			self.myStep()
		self.setMoveCommand(vY=target)
		self.mGaitManager.step(self.mTimeStep)
		self.myStep()
	
	# 设置转动速度，软更新 v = (1-α)*v + α*Target
	def setRotationSpeed(self, target, threshold=0.05, tau=0.1):
		current = self.vA
		target = np.clip(target,-1,1)
		while np.abs(current - target) > threshold:
			current = (1-tau)*current + tau*target
			self.setMoveCommand(vA=current)
			self.mGaitManager.step(self.mTimeStep)
			self.myStep()
		self.setMoveCommand(vA=target)
		self.mGaitManager.step(self.mTimeStep)
		self.myStep()
	
	# 通过PID控制机器人转体到target角度			
	def setRotation(self, target, threshold=0.5,Kp=0.1):
		self.setForwardSpeed(0.)
		self.setSideSpeed(0.)
		self.setRotationSpeed(0.)

		while np.abs(self.angle[-1] - target) > threshold:
			u = Kp*(target - self.angle[-1])
			u = np.clip(u,-1,1)
			self.setMoveCommand(vA = u)
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
	
	def setRobotRun(self):
		self.setForwardSpeed(1.)
		self.setSideSpeed(0.)
		self.setRotationSpeed(0.)

	
	# 检查偏航是否超过指定threshold，若是则修正回0°附近，由eps指定
	def checkIfYaw(self, threshold=7.5, eps=0.25):
		if np.abs(self.angle[-1]) > threshold:
			print('Current Yaw is %.3f, begin correction' % self.angle[-1])
			while np.abs(self.angle[-1]) > eps:
				u = -0.1 * self.angle[-1]
				u = np.clip(u,-1,1)
				self.setMoveCommand(vA = u)
				self.mGaitManager.step(self.mTimeStep)
				self.myStep()
			print('Current Yaw is %.3f, finish correction' % self.angle[-1])
		else:
			pass
	
	# 比赛开始前准备动作，保持机器人运行稳定和传感器读数稳定
	def prepare(self,waitingTime):
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
	def Z(self,angle,interval):
		# 左转45度
		self.setRotation(angle)
		self.setForwardSpeed(1.0)
		ns = itertools.repeat(0,interval)
		for n in ns:
			self.mGaitManager.step(self.mTimeStep)
			self.myStep()
		self.setRotation(0)
		self.setRobotRun()
		
	def keyBoardControl(self,collet_data=False,rotation=True):
		self.isWalking = True  # 初始时机器人未进入行走状态
		ns = itertools.count(0)
		for n in ns:
			if n % 200 == 0 and collet_data:
				rgb_raw = getImage(self.mCamera)
				print('save image')
				cv2.imwrite('./tmp/f_'+str(n)+'.png',rgb_raw)
			self.checkIfFallen()
			self.mGaitManager.setXAmplitude(0.0)  # 前进为0
			self.mGaitManager.setAAmplitude(0.0)  # 转体为0
			key = 0  # 初始键盘读入默认为0
			key = self.mKeyboard.getKey()  # 从键盘读取输入
			if rotation :
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


	def stage1(self):
		print('########Stage1_Start########')
		crossBarDownFlag = False
		goFlag = False
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
				self.setRobotRun()
				#self.motors[18].setPosition(radians(50))
				#self.motors[19].setPosition(radians(30))
				break
			self.mGaitManager.step(self.mTimeStep)
			self.myStep()
		n0 = n
		for n in ns:
			self.checkIfYaw()
			# 持续走一段时间，写死了这里
			if (n-n0) >= 850:
				break
			self.mGaitManager.step(self.mTimeStep)
			self.myStep()
		print('########Stage1_End########')

	def stage2(self):
		print('########Stage2_Start########')
		self.setRobotStop()
		self.checkIfYaw(threshold=3.0)
		ns = itertools.count(0)
		center_y,center_x = None, None
		for n in ns:
			if n % 5 == 0:
				rgb_raw = getImage(self.mCamera)
				binary= call_segmentor(rgb_raw,self.model2)
				trap = np.argwhere(binary == 255)
				tmp_y,tmp_x = np.mean(trap,axis=0)
				if not center_y and not center_x:
					center_y, center_x = tmp_y,tmp_x
				else:
					center_y = 0.9*center_y + 0.1*tmp_y
					center_x = 0.9*center_x + 0.1*tmp_x
			if center_x < 75:
				self.setSideSpeed(0.5)
				self.checkIfYaw(threshold=3.0)
			elif center_x > 85:
				self.setSideSpeed(-0.5)
				self.checkIfYaw(threshold=3.0)
			else:
				self.setSideSpeed(0.)
				self.checkIfYaw(threshold=3.5)
				print('Align Trap CenterX : %d CenterY : %d' %(center_x,center_y))
				break
			self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
			self.myStep()  # 仿真一个步长
		
		self.Z(angle=45,interval=275)
		ns = itertools.repeat(0,500)
		for n in ns:
			self.checkIfYaw(threshold=3.0)
			self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
			self.myStep()  # 仿真一个步长
		self.Z(angle=-45,interval=275)
		print('########Stage2_End########')


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
	

	# 主函数循环
	def run(self):
		# 准备动作
		self.prepare(waitingTime=250)
		# 通过第一关
		self.stage1()
		# 通过第二关
		self.stage2()
		# 通过第七关
		#self.stage7()
		# 键盘控制与采集数据
		#self.keyBoardControl(collet_data=1,rotation=0)
		
		# 停下
		self.setRobotStop()
		self.wait(200)
		self.mMotionManager.playPage(24)

		while True:
			#self.checkIfYaw(threshold=3.0)
			self.mGaitManager.step(self.mTimeStep)
			self.myStep()

if __name__ == '__main__':
	walk = Walk()  # 初始化Walk类
	walk.run()  # 运行控制器