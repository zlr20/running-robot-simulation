"""door1 controller."""
from controller import Robot
robot = Robot()
timestep = int(robot.getBasicTimeStep())
#print(robot.getNumberOfDevices())
TouchSensor=[0 for _ in range(3)]
for index in range(3):
    TouchSensor[index]=robot.getDevice("doorsensor"+str(index+1))
    TouchSensor[index].enable(timestep)
robot.setCustomData("0")
touch_flag=0
while robot.step(timestep) != -1:
    for index in range(3):
        if TouchSensor[index].getValue():
            if touch_flag==0:
                print("door touched！")
                robot.setCustomData("1")
                touch_flag=1
                break
    if touch_flag:
        break
for index in range(3):
    TouchSensor[index].disable()


    
