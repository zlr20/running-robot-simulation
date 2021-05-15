"""mine0 controller."""
from controller import Robot
robot = Robot()
timestep = int(robot.getBasicTimeStep())
num=6
mine=[0 for _ in range(num)]
for index in range(num):
    mine[index]=robot.getDevice("minesensor"+str(index))
    mine[index].enable(timestep)
timestep = int(robot.getBasicTimeStep())
robot.setCustomData("0")
touch_flag=0
first=-1
while robot.step(timestep) != -1:
    for index in range(num):
        if mine[index].getValue():
            if touch_flag==0:
                first=index
                touch_flag+=1
                print("mine touched！")
                robot.setCustomData(str(touch_flag))
                break
            if touch_flag==1 and first!=index:
                touch_flag+=1
                print("mine touched！")
                robot.setCustomData(str(touch_flag))
                break            
    if touch_flag==2:
        break
for index in range(num):
    mine[index].disable()
