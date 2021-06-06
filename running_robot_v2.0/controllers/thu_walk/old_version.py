
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

# 回字陷阱，神经网络
def stage2_old(self):
        print('########Stage2_Start########')
        self.setRobotStop()
        stage2model = load_model('./pretrain/2.pth')
        self.checkIfYaw(threshold=3.0)
        ns = itertools.count(0)
        center_y, center_x = None, None
        for n in ns:
            if n % 5 == 0:
                rgb_raw = getImage(self.mCamera)
                binary = call_segmentor(rgb_raw, stage2model)
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
        ns = itertools.repeat(0, 550)
        for n in ns:
            self.checkIfYaw(threshold=3.0)
            self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
            self.myStep()  # 仿真一个步长
        self.setRobotRun(0.5)
        self.Z(angle=-45, interval=500)
        del stage2model

def stage3_old(self):
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

# 构造L字轨迹
def L(self,direction='left',nums=2):
    speed = self.vX
    self.setMoveCommand(vX=0.0)
    self.mGaitManager.step(self.mTimeStep)
    self.myStep()
    self.mGaitManager.stop()
    self.mGaitManager.step(self.mTimeStep)
    self.myStep()
    if direction == 'left':
        moveLMotion = Motion('./motion/MoveL.motion')
        moveLMotion.setLoop(False)
        for i in range(nums):
            moveLMotion.play()
            while not moveLMotion.isOver():
                self.myStep()
    elif direction == 'right':
        moveRMotion = Motion('./motion/MoveR.motion')
        moveRMotion.setLoop(False)
        for i in range(nums):
            moveRMotion.play()
            while not moveRMotion.isOver():
                self.myStep()
    else:
        raise
    self.prepare(250)
    self.setMoveCommand(vX=speed)
    self.mGaitManager.step(self.mTimeStep)
    self.myStep()

 # 地雷路段，没有用神经网络，纯图像处理，更稳定。
    def stage3(self):
        print("~~~~~~~~~~~地雷关开始~~~~~~~~~~~")
        # 首先，抬头做全局规划,先移动到雷区中间位置
        # self.motors[19].setPosition(0.4)
        # ns = itertools.count(0)
        # for n in ns:
        #     self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
        #     self.myStep()  # 仿真一个步长
        #     self.checkIfYaw()
        #     if n % 5 == 0 and np.abs(self.angle[0]) < 1.0 and np.abs(self.positionSensors[19].getValue()-0.4)<0.05:
        #         rgb_raw = getImage(self.mCamera)
        #         low =  [0,0,15]
        #         high = [255,255,255]
        #         hsv = cv2.cvtColor(rgb_raw,cv2.COLOR_BGR2HSV)
        #         mask=cv2.inRange(hsv,np.array(low),np.array(high))
        #         mask = 255 - mask
        #         contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        #         dilei = []
        #         for contour in contours:
        #             if cv2.contourArea(contour) > 5:
        #                 M = cv2.moments(contour)  # 计算第一条轮廓的各阶矩,字典形式
        #                 center_x = int(M["m10"] / M["m00"])
        #                 center_y = int(M["m01"] / M["m00"])
        #                 # 变换
        #                 center_x += (center_x-80)*0.5*(1-center_y/120)
        #                 if center_x < 0:
        #                     center_x = 0
        #                 elif center_x > 160:
        #                     center_x = 160
        #                 dilei.append((int(center_x),int(center_y)))
        #         # 按x坐标从左到右排序
        #         dilei = sorted(dilei,key=lambda x: x[0])
        #         # 先用最边界的两个雷，判断机器人偏赛道左还是右
        #         leftMineX,rightMineX = dilei[0][0],dilei[-1][0]
        #         midMineX = 0.5*(leftMineX+rightMineX)
        #         if midMineX > 90:
        #             self.setMoveCommand(vY=-1.0)
        #         elif midMineX < 70:
        #             self.setMoveCommand(vY=1.0)
        #         else:
        #             self.setMoveCommand(vY=0.0)
        #             break
        print(f'雷区预规划完成')

        self.motors[19].setPosition(-0.2)
        self.setMoveCommand(vX=1.0)
        ns = itertools.count(0)
        turnRightFlag = True
        for n in ns:
            self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
            self.myStep()  # 仿真一个步长
            self.checkIfYaw()
            if n % 5 == 0 and np.abs(self.angle[0]) < 1.0 and np.abs(self.positionSensors[19].getValue()+0.2)<0.05:
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
                        if 25 < center_x < 135 and center_y > 75:
                            dilei.append((center_x,center_y))
                if len(dilei):
                    dilei = sorted(dilei,key = lambda x:x[1],reverse=True)
                    if dilei[0][0] < 65:
                        # 向右转向45前进
                        minSteps = 350
                        while np.abs(self.angle[-1]+45) > 1 or minSteps>0:
                            u = -0.02 * (self.angle[-1]+45)
                            u = np.clip(u, -1, 1)
                            self.setMoveCommand(vX=0.5,vA=u)
                            self.mGaitManager.step(self.mTimeStep)
                            self.myStep()
                            minSteps -= 1
                        self.checkIfYaw()
                    elif dilei[0][0] > 95:
                        # 向左转向45前进
                        minSteps = 350
                        while np.abs(self.angle[-1]-45) > 1 or minSteps>0:
                            u = -0.02 * (self.angle[-1]-45)
                            u = np.clip(u, -1, 1)
                            self.setMoveCommand(vX=0.5,vA=u)
                            self.mGaitManager.step(self.mTimeStep)
                            self.myStep()
                            minSteps -= 1
                        self.checkIfYaw()
                     # 中间的雷可以向两边转，但是左右次数应均衡
                    elif turnRightFlag:
                        print('中间雷向右')
                        # 向右转向60前进
                        minSteps = 400
                        while np.abs(self.angle[-1]+60) > 1 or minSteps>0:
                            u = -0.02 * (self.angle[-1]+60)
                            u = np.clip(u, -1, 1)
                            self.setMoveCommand(vX=0.5,vA=u)
                            self.mGaitManager.step(self.mTimeStep)
                            self.myStep()
                            minSteps -= 1
                        self.checkIfYaw()
                        turnRightFlag = 1- turnRightFlag
                    else:
                        print('中间雷向左')
                        # 向左转向60前进
                        minSteps = 400
                        while np.abs(self.angle[-1]-60) > 1 or minSteps>0:
                            u = -0.02 * (self.angle[-1]-60)
                            u = np.clip(u, -1, 1)
                            self.setMoveCommand(vX=0.5,vA=u)
                            self.mGaitManager.step(self.mTimeStep)
                            self.myStep()
                            minSteps -= 1
                        self.checkIfYaw()
                        turnRightFlag = 1- turnRightFlag


                else:
                    self.setMoveCommand(vX=1.0)
                
                # 若地雷后接高障碍物，则进stage4()
                if self.obstacleBehind:
                    ob_x, ob_y = obstacleDetect(rgb_raw)
                    if ob_y > 40:
                        #print(ob_y)
                        self.stage4()
                # 若地雷后无障碍物，则正常结束
                else:
                    hsv = cv2.cvtColor(rgb_raw,cv2.COLOR_BGR2HSV)
                    low = self.materialInfo['hsv']['low']
                    high = self.materialInfo['hsv']['high']
                    mask=cv2.inRange(hsv[self.mCameraHeight//2:,:],np.array(low),np.array(high))
                    road = np.where(mask==255)[0]
                    num = len(road)
                    if num < 500:
                        print("~~~~~~~~~~~地雷关结束~~~~~~~~~~~")
                        break

# 下最后一个红色坡,并结束这关
        self.motors[19].setPosition(-0.2)
        ns = itertools.count(0)
        for n in ns:
            u = -0.02 * (self.angle[-1])
            u = np.clip(u, -1, 1)
            self.setMoveCommand(vX=1.0,vA=u)
            self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
            self.myStep()  # 仿真一个步长
            if n % 5 == 0 and np.abs(self.positionSensors[19].getValue()+0.2)<0.05:
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
                fname = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time())) 
                cv2.imwrite('./tmp/' + fname + '.png', rgb_raw)
                # cv2.imshow('image',mask)
                # cv2.waitKey(0)
                road = np.where(mask==255)[0]
                num = len(road)
                if num < 200:
                    break

 while True:
            self.mGaitManager.step(self.mTimeStep)
            self.myStep()
            if np.abs(self.angle[0])<0.25: break
        self.mMotionManager.playPage(9)
        # stepMotion = Motion('motion/stair_duidui.motion')
        # stepMotion.setLoop(False)
        # for i in range(4):
        #     stepMotion.play()
        #     while not stepMotion.isOver():
        #         self.myStep()
        # stepMotion = Motion('motion/stair_fandui.motion')
        # stepMotion.setLoop(False)
        # for i in range(0):
        #     stepMotion.play()
        #     while not stepMotion.isOver():
        #         self.myStep()
        # rollMotion = Motion('./motion/0603_1.motion')
        # rollMotion.setLoop(False)
        # rollMotion.play()
        # while not rollMotion.isOver():
        #     self.myStep()
        # self.mMotionManager.playPage(1)
        # self.mGaitManager.start()
        # for i in range(50):
        #     self.mGaitManager.step(self.mTimeStep)
        #     self.myStep()
        rollMotion = Motion('./motion/kick.motion')
        rollMotion.setLoop(True)
        rollMotion.play()
        while not rollMotion.isOver():
            self.myStep()
        self.checkIfYaw()
        while True:
            self.mGaitManager.step(self.mTimeStep)
            self.myStep()
            if np.abs(self.angle[0])<0.25: break
        self.mMotionManager.playPage(1)
        self.mGaitManager.stop()
        rollMotion = Motion('./motion/roll_v2.motion')
        rollMotion.setLoop(False)
        rollMotion.play()
        while not rollMotion.isOver():
            self.myStep()
        self.mMotionManager.playPage(9)
        self.mMotionManager.playPage(10)
        self.mMotionManager.playPage(9)


# 过门（成功率95%）
    def stage9(self):
        print("~~~~~~~~~~~窄门关开始~~~~~~~~~~~")
        brickAreaMean = 10000
        # 首先，抬头做全局规划,先大致配准到门中间位置
        self.motors[19].setPosition(0.5)
        ns = itertools.count(0)
        peizhun = False
        for n in ns:
            self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
            self.myStep()  # 仿真一个步长
            self.checkIfYaw()
            if n % 5 == 0 and np.abs(self.angle[0]) < 1.0 and np.abs(self.positionSensors[19].getValue()-0.5)<0.05:
                rgb_raw = getImage(self.mCamera)
                low =  [0,0,15]
                high = [255,255,255]
                hsv = cv2.cvtColor(rgb_raw,cv2.COLOR_BGR2HSV)
                mask=cv2.inRange(hsv,np.array(low),np.array(high))
                mask = 255 - mask
                contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                leftPoint,rightPoint = (0,0), (160,0)
                for cnt in contours:
                    if cv2.contourArea(cnt) > 120:
                        bottommost=tuple(cnt[cnt[:,:,1].argmax()][0])
                        if bottommost[0] > 80:
                            rightPoint = bottommost
                        else:
                            leftPoint = bottommost
                if leftPoint == (0,0) and rightPoint == (160,0):
                    print('失去门框视野,窄门预规划完成')
                    break
                elif leftPoint == (0,0):
                    # 没检测到左点,强制左右一样高
                    leftPoint = (0,rightPoint[1])
                elif rightPoint ==  (160,0):
                    # 没检测到右点,强制左右一样高
                    rightPoint = (160,leftPoint[1])
                else:
                    pass
                midPoint = (int((0.4*leftPoint[0]+0.6*rightPoint[0])),int((leftPoint[1]+rightPoint[1])/2))
                if midPoint[-1] > 115:
                    print('已配准,窄门预规划完成')
                    break
                if midPoint[0] > 90 and not peizhun:
                    self.setMoveCommand(vX=0.0,vY=-1.0)
                elif midPoint[0] < 70 and not peizhun:
                    self.setMoveCommand(vX=0.0,vY=1.0)
                elif peizhun:
                    self.setMoveCommand(vX=1.0,vY=0.0)
                    peizhun = True
                else:
                    self.setMoveCommand(vX=1.0,vY=0.0)
                    #break
        self.setMoveCommand(vX=1.0)
        ns = itertools.repeat(0,300)
        for n in ns:
            self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
            self.myStep()  # 仿真一个步长
            self.checkIfYaw()
        # 向左转向90
        while np.abs(self.angle[-1]-90) > 1:
            u = -0.05 * (self.angle[-1]-90)
            u = np.clip(u, -1, 1)
            self.setMoveCommand(vA=u)
            self.mGaitManager.step(self.mTimeStep)
            self.myStep()
        ns = itertools.repeat(0,1500)
        for n in ns:
            a = 0.02 * (75 - self.angle[-1])
            self.setMoveCommand(vY=-1.0,vA=a)
            self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
            self.myStep()  # 仿真一个步长
    
        self.setMoveCommand(vX=0.0)
        self.checkIfYaw()

        # 通过这关
        self.motors[19].setPosition(-0.2)
        ns = itertools.count(0)
        self.setMoveCommand(vX=1.)
        for n in ns:
            self.checkIfYaw()
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
        for _ in range(50):
            self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
            self.myStep()  # 仿真一个步长
        self.mGaitManager.stop()
        self.wait(200)
        self.angle[-1] -= 90
        self.angle[0]=0
        self.angle[1]=0
        self.mGaitManager.start()
        self.wait(200)
        self.checkIfYaw()
        self.mGaitManager.stop()
        self.wait(200)
        self.angle[0]=0
        self.angle[1]=0
        self.mGaitManager.start()
        self.wait(500)
        self.turnCount += 1
        print("~~~~~~~~~~~转弯结束~~~~~~~~~~~")