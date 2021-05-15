from controller import Robot,Motor
robot = Robot()
#仿真步长
timestep = int(robot.getBasicTimeStep())
wheel1 = robot.getDevice("wheel1")
wheel2 = robot.getDevice("wheel2")
TouchSensor1 = robot.getDevice("t_sensor1")
TouchSensor1.enable(timestep)
TouchSensor2 = robot.getDevice("t_sensor2")
TouchSensor2.enable(timestep)
robot.setCustomData(str(0))
def step(num):
        cnt = 0
        Touch_Flag=0
        while robot.step(timestep) != -1:
            #print(w print(TouchSensor.getValue())heel.getVelocity()) #just for test
            if TouchSensor1.getValue() or TouchSensor2.getValue():
                if Touch_Flag==0:
                    print("barrier touched!")
                    Touch_Flag=1
                    robot.setCustomData(str(Touch_Flag))
            cnt = cnt + 1
            if cnt == num:
                break
while True:
   #角度和停顿时间
    wheel1.setPosition(0)
    wheel2.setPosition(0)
    step(5000//timestep)
    wheel1.setPosition(-1.57)
    wheel2.setPosition(-1.57)
    step(10000//timestep)
print('happy')
