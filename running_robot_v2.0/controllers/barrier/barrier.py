from controller import Robot,Motor#,TouchSensor

robot = Robot()
#仿真步长
timestep = int(robot.getBasicTimeStep())
wheel = robot.getDevice("wheel1")
TouchSensor = robot.getDevice("t_sensor1")
TouchSensor.enable(timestep)
 #wheel.setPosition(0)
robot.setCustomData(str(0))
def step(num):
        cnt = 0
        Touch_Flag=0
        while robot.step(timestep) != -1:
            #print(w print(TouchSensor.getValue())heel.getVelocity()) #just for test
            if TouchSensor.getValue():
                if Touch_Flag==0:
                    print("barrier touched!")
                    Touch_Flag=1
                    robot.setCustomData(str(Touch_Flag))
            cnt = cnt + 1
            if cnt == num:
                break
while True:
   #角度和停顿时间
    wheel.setPosition(0)
    step(5000//timestep)
    wheel.setPosition(-1.57)
    step(10000//timestep)
print('happy')
