
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

# 检查偏航是否超过指定threshold，若是则修正回0°附近
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
