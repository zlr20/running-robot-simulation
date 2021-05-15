"""position_reset controller."""
from controller import Supervisor, Node, Field
from controller import Robot, Motor
import numpy as np
import math
import copy
import random
import socket
import sys
import os
import time
from urllib import request, parse, error
#import urllib2

# 指定关卡
flag = 5
# 是否随机布置赛道
RANDOM_FLAG = True


bias=-1+2*np.random.random()
scale=0.
noise=bias*scale

if flag==1:
    DARWIN_START_POINT = [-2+noise, 0.8175, -2]
    DEFAULT_DIREC = [0, 1, 0, 0]
elif flag==2:
    DARWIN_START_POINT = [-2+noise, 0.8175, -1.3]
    DEFAULT_DIREC = [0, 1, 0, 0]
elif flag==3:
    DARWIN_START_POINT = [-2+noise, 0.8175, -0.5]
    DEFAULT_DIREC = [0, 1, 0, 0]
elif flag==4:
    DARWIN_START_POINT = [-2+noise, 0.8175, 0.75]
    DEFAULT_DIREC = [0, 1, 0, 0]
elif flag==5:
    DARWIN_START_POINT = [-0.8, 0.82, 1.59+noise]
    DEFAULT_DIREC = [ 0, 1, 0, np.pi/2]
elif flag==6:
    DARWIN_START_POINT = [0.18, 0.82, 1.65+noise]
    #DARWIN_START_POINT = [0.55, 0.82, 1.65+noise] 球在面前
    DEFAULT_DIREC = [ 0, 1, 0, np.pi/2]
elif flag==7:
    DARWIN_START_POINT = [2.19, 0.825, 1.20] # 楼梯前
    #DARWIN_START_POINT = [2.19, 0.87, 0.75] # 楼梯最高点
    DEFAULT_DIREC = [0, 1, 0,np.pi] # [0, 1, 0, pi/2]
else:
    DARWIN_START_POINT = [2.23+noise, 0.818, -0.158]
    DEFAULT_DIREC = [0, 1, 0, np.pi]
    
DARWIN_CENTER_HEIGHT_STAND = 0.32


# 全局变量定义
# 方块定义：
BLOCK_START = -1
BLOCK_START_SIZE = [0.6, 0.5, 1]
BLOCK_START_SET_POS = [-2, 0.25, -1.8]
BLOCK_TRAP = 0
BLOCK_TRAP_SIZE = [0.8, 0.1, 0.8]
BLOCK_SECOND = 1
BLOCK_SECOND_SIZE = [0.6, 0.5, 1.8]
BLOCK_THIRD = 2
BLOCK_THIRD_SIZE = [0.6, 0.5, 1.8]
BLOCK_BRIDGE = 3
BLOCK_BRIDGE_SIZE = [0.3, 0.04, 0.6]
BLOCK_FORTH = 4
BLOCK_FORTH_SIZE = [0.6, 0.5, 1.8]
BLOCK_STEP = 5
BLOCK_STEP_SIZE = [0.6, 0.5, 2.05]
BLOCK_END = 6
BLOCK_END_SIZE = [0.6, 0.5, 1.2]
BLOCK_AMOUNT = 6


RACE_SINGLE_LEN = 4
RACE_MAX_WIDTH = 0.6
ROT_PI = 1.5708
ROBOT_OUT_RACE=0.5
# 关卡定义：
# LevelName:barrier,trap,mine,obscle,door,bridge,ball,step,double_bar
LEVEL_BARRIER = 0
LEVEL_TRAP = 1
LEVEL_MINE = 2
LEVEL_OBSCLE = 3
LEVEL_DOOR = 4
LEVEL_BRIDGE = 5
LEVEL_BALL = 6
LEVEL_STEP = 7
LEVEL_DOUBLEBAR = 8
LEVEL_AMOUNT = 9
# # DARWIN
# DARWIN_START_POINT = [-2, 0.8175, -2]
# DEFAULT_DIREC = [0, 1, 0, 0]
# DARWIN_CENTER_HEIGHT_STAND = 0.32

#重心高度：
RACE_HEIGHT = 0.5
OBSCLE_HEIGHT=0.55
BALL_HEIGHT=0.51843
HOLE_HEIGHT=0.2505
#相对位置：
HOLE_RADIUS=0.05
HOLE_BALL_DISTANCE=0.5
HOLE_POS=[0.2,0,0.4]
UNIT_LEN=0.3
UNIT_ERROR=0.001
# 关卡分：
SCORE_BAR=10
SCORE_TRAP=10
SCORE_MINE=20
SCORE_OBSCLE=20
SCORE_DOOR=10
SCORE_BRIDGE=20
SCORE_BALL=20
SCORE_STEP_UP=20
SCORE_STEP_DOWN=20
SCORE_DOUBLEBAR=20









class runningrobot_env:
    def __init__(self, timestep=None):
        self.supervisor = Supervisor()

        self.darwin = self.supervisor.getFromDef("Darwin")

        self.ball = self.supervisor.getFromDef("ball")
        self.hole = self.supervisor.getFromDef("hole")
        self.target = self.supervisor.getFromDef("targetball")
        self.bar = self.supervisor.getFromDef("Barrier")
        self.double_bar = self.supervisor.getFromDef("double_barrier")

        self.obscle = self.supervisor.getFromDef("obscle")

        self.mine = self.supervisor.getFromDef("mine")
        self.door = self.supervisor.getFromDef("door")

        self.levelendpos = []
        self.levelorder = []

        if timestep is None:
            self.timestep = int(self.supervisor.getBasicTimeStep())
        else:
            self.timestep = timestep

        # self.supervisor = Supervisor()

        self.StartingField = self.supervisor.getFromDef("StartingField")

        self.EndingField = self.supervisor.getFromDef("EndingField")

        # 陷阱关
        self.Trap = self.supervisor.getFromDef("Trap")

        # 地雷关
        self.second = self.supervisor.getFromDef("second")

        self.mine = self.supervisor.getFromDef("mine")

        # 门
        self.third = self.supervisor.getFromDef("third")
        # self.door=self.supervisor.getFromDef("door")

        # 桥
        self.bridge = self.supervisor.getFromDef("bridge_2")

        # 球
        self.forth = self.supervisor.getFromDef("forth")
        self.ball = self.supervisor.getFromDef("ball")

        # 楼梯
        self.Step = self.supervisor.getFromDef("Step")

        # 栏杆2
        self.double_barrier = self.supervisor.getFromDef("double_barrier")


        # 关卡的全部信息
        self.levelendpos = [0 for _ in range(LEVEL_AMOUNT)]
        self.levelorder = [0 for _ in range(LEVEL_AMOUNT)]

        # translation_field=[x,z,y] #importment!
        self.ball_translation_field = self.ball.getField("translation")
        self.hole_translation_field = self.hole.getField("translation")
        # self.target_translation_field = self.target.getField("translation")
        self.darwin_translation_field = self.darwin.getField("translation")
        self.obscle_translation_field = self.obscle.getField("translation")
        self.mine_flag = 0
        self.door_flag = 0
        self.obscle_falldown = 0
        self.bar_flag = 0
        self.d_bar_flag = 0
        self.ball_hit = 0
        self.time = 0
        self.single_action_num = 1
        self.state = 0
        self.scores = 0
        self.end_flag = 0
        self.first_arrival_peak=1
        self.start = 0
        self.log = ""
        self.map = []

    def setpos_darwin(self, pos, rot):
        darwin_pos = self.darwin.getField("translation")
        darwin_pos.setSFVec3f(pos)
        darwin_rot = self.darwin.getField("rotation")
        darwin_rot.setSFRotation(rot)

    def setpos_Trap(self, pos, rot):
        Trap_pos = self.Trap.getField("translation")
        old_pos = Trap_pos.getSFVec3f()
        pos = [pos[0], old_pos[1], pos[2]]
        Trap_pos.setSFVec3f(pos)
        Trap_rot = self.Trap.getField("rotation")
        Trap_rot.setSFRotation(rot)

    def setpos_second(self, pos, rot):
        second_pos = self.second.getField("translation")
        old_pos = second_pos.getSFVec3f()
        pos = [pos[0], old_pos[1], pos[2]]
        second_pos.setSFVec3f(pos)
        second_rot = self.second.getField("rotation")
        second_rot.setSFRotation(rot)

    def setpos_third(self, pos, rot):
        third_pos = self.third.getField("translation")
        old_pos = third_pos.getSFVec3f()
        pos = [pos[0], old_pos[1], pos[2]]
        third_pos.setSFVec3f(pos)
        third_rot = self.third.getField("rotation")
        third_rot.setSFRotation(rot)

    def setpos_bridge(self, pos, rot):
        bridge_pos = self.bridge.getField("translation")
        old_pos = bridge_pos.getSFVec3f()
        pos = [pos[0], old_pos[1], pos[2]]
        bridge_pos.setSFVec3f(pos)
        bridge_rot = self.bridge.getField("rotation")
        bridge_rot.setSFRotation(rot)

    def setpos_startbox(self, pos, rot):
        startbox_pos = self.StartingField.getField("translation")
        old_pos = startbox_pos.getSFVec3f()
        pos = [pos[0], old_pos[1], pos[2]]
        startbox_pos.setSFVec3f(pos)
        startbox_rot = self.StartingField.getField("rotation")
        startbox_rot.setSFRotation(rot)

    def setpos_forth(self, pos, rot):
        forth_pos = self.forth.getField("translation")
        old_pos = forth_pos.getSFVec3f()
        pos = [pos[0], old_pos[1], pos[2]]
        forth_pos.setSFVec3f(pos)
        forth_rot = self.forth.getField("rotation")
        forth_rot.setSFRotation(rot)

    def setpos_Step(self, pos, rot):
        Step_pos = self.Step.getField("translation")
        old_pos = Step_pos.getSFVec3f()
        pos = [pos[0], old_pos[1], pos[2]]
        Step_pos.setSFVec3f(pos)
        Step_rot = self.Step.getField("rotation")
        Step_rot.setSFRotation(rot)

    def setpos_EndingField(self, pos, rot):
        EndingField_pos = self.EndingField.getField("translation")
        old_pos = EndingField_pos.getSFVec3f()
        pos = [pos[0], old_pos[1], pos[2]]
        EndingField_pos.setSFVec3f(pos)
        EndingField_rot = self.EndingField.getField("rotation")
        EndingField_rot.setSFRotation(rot)

    # 依附于现有通路的关卡
    def set_sub_bar(self, pos, rot):
        bar_pos = self.bar.getField("translation")
        old_pos = bar_pos.getSFVec3f()
        pos = [pos[0], old_pos[1], pos[2]]
        bar_pos.setSFVec3f(pos)
        bar_rot = self.bar.getField("rotation")
        bar_rot.setSFRotation(rot)

    def set_sub_doublebar(self, pos, rot):
        doublebar_pos = self.double_barrier.getField("translation")
        old_pos = doublebar_pos.getSFVec3f()
        pos = [pos[0], old_pos[1], pos[2]]
        doublebar_pos.setSFVec3f(pos)
        doublebar_rot = self.double_barrier.getField("rotation")
        doublebar_rot.setSFRotation(rot)

    def set_sub_door(self, pos, rot):
        door_pos = self.door.getField("translation")
        old_pos = door_pos.getSFVec3f()
        pos = [pos[0], old_pos[1], pos[2]]
        door_pos.setSFVec3f(pos)
        door_rot = self.door.getField("rotation")
        door_rot.setSFRotation(rot)

    def set_sub_ball(self, pos, rot):
        ball_pos = self.ball.getField("translation")
        pos = [pos[0],BALL_HEIGHT, pos[2]]
        self.ball.resetPhysics
        ball_pos.setSFVec3f(pos)
        ball_rot = self.ball.getField("rotation")
        ball_rot.setSFRotation(rot)

        hole_pos = self.hole.getField("translation")
        pos = [pos[0] + HOLE_POS[0],HOLE_HEIGHT, pos[2] + HOLE_POS[0]]
        hole_pos.setSFVec3f(pos)
        hole_rot = self.hole.getField("rotation")
        hole_rot.setSFRotation(rot)   

    def set_sub_mine(self, pos, rot):
        mine_pos = self.mine.getField("translation")
        old_pos = mine_pos.getSFVec3f()
        pos = [pos[0], old_pos[1], pos[2]]
        mine_pos.setSFVec3f(pos)
        mine_rot = self.mine.getField("rotation")
        mine_rot.setSFRotation(rot)

    def set_sub_obscle(self, pos, rot):
        obscle_pos = self.obscle.getField("translation")
        pos = [pos[0], OBSCLE_HEIGHT, pos[2]]
        obscle_pos.setSFVec3f(pos)
        obscle_rot = self.obscle.getField("rotation")
        obscle_rot.setSFRotation(rot)

    def setpos_one(self, size, last_point, last_direc, the_direc):
        x = last_point[0]
        y = last_point[1]
        z = last_point[2]
        rot = [0, 1, 0, (1 - the_direc) * ROT_PI]
        pos = [0, 0, 0]
        boundary = [0, 0, 0]
        if last_direc == 1:
            if the_direc == 1:
                pos = [x, y, z + size[2] / 2]
                boundary = [x, y, z + size[2]]
            elif the_direc == 0:
                pos = [x + size[2] / 2 - RACE_MAX_WIDTH / 2, y, z + RACE_MAX_WIDTH / 2]
                boundary = [x + size[2] - RACE_MAX_WIDTH / 2, y, z + RACE_MAX_WIDTH / 2]
        elif last_direc == 0:
            if the_direc == 0:
                pos = [x + size[2] / 2, y, z]
                boundary = [x + size[2], y, z]
            elif the_direc == -1:
                pos = [x + RACE_MAX_WIDTH / 2, y, z + RACE_MAX_WIDTH / 2 - size[2] / 2]
                boundary = [x + RACE_MAX_WIDTH / 2, y, z + RACE_MAX_WIDTH / 2 - size[2]]
        else:
            pos = [x, y, z - size[2] / 2]
            boundary = [x, y, z - size[2]]
        return pos, rot, boundary

    def setRace(self, index, last_point, last_direc, the_direc):
        # index 顺序：
        # direct 1:z 0:x -1:-z
        # BLOCK_TRAP,BLOCK_SECOND,BLOCK_THIRD,BLOCK_BRIDGE,BLOCK_FORTH,BLOCK_STEP
        boundary = [0, 0, 0]
        point = [last_point[0], last_point[1], last_point[2]]
        if index == BLOCK_START:  # 起始块
            boundary = [
                BLOCK_START_SET_POS[0],
                BLOCK_START_SET_POS[1],
                BLOCK_START_SET_POS[2] + BLOCK_START_SIZE[2] / 2,
            ]
            self.setpos_startbox(BLOCK_START_SET_POS, [0, 1, 0, 0])
        elif index == BLOCK_TRAP:  # 陷阱
            size = BLOCK_TRAP_SIZE
            pos, rot, boundary = self.setpos_one(size, point, last_direc, the_direc)
            self.setpos_Trap(pos, rot)
        elif index == BLOCK_SECOND:  # 白色长块
            size = BLOCK_SECOND_SIZE
            pos, rot, boundary = self.setpos_one(size, point, last_direc, the_direc)
            self.setpos_second(pos, rot)
        elif index == BLOCK_THIRD:  # 斑点白色长块
            size = BLOCK_THIRD_SIZE
            pos, rot, boundary = self.setpos_one(size, point, last_direc, the_direc)
            self.setpos_third(pos, rot)
        elif index == BLOCK_BRIDGE:  # 长桥
            size = BLOCK_BRIDGE_SIZE
            pos, rot, boundary = self.setpos_one(size, point, last_direc, the_direc)
            self.setpos_bridge(pos, rot)
        elif index == BLOCK_FORTH:  # 砖色长块
            size = BLOCK_FORTH_SIZE
            pos, rot, boundary = self.setpos_one(
                size, last_point, last_direc, the_direc
            )
            self.setpos_forth(pos, rot)
        elif index == BLOCK_STEP:  # 台阶长块
            size = BLOCK_STEP_SIZE
            pos, rot, boundary = self.setpos_one(size, point, last_direc, the_direc)
            self.setpos_Step(pos, rot)
        elif index == BLOCK_END:  # 结束位置长块
            size = BLOCK_END_SIZE
            pos, rot, boundary = self.setpos_one(size, point, last_direc, the_direc)
            self.setpos_EndingField(pos, rot)
        # print(boundary)
        return boundary


    def set_random_mine(self):
        mine_local = []
        while len(mine_local) < 6:
            next_mine = [round(random.uniform(-0.3, 0.3), 2), 0.266, round(random.uniform(-0.45, 0.45), 2)]
            if not mine_local:
                mine_local.append(next_mine)
                continue
            for i in range(len(mine_local)):
                key = 0
                if math.sqrt(sum([(a - b)**2 for (a, b) in zip(mine_local[i], next_mine)])) < 0.3:
                    key = 1
                    break
            if key == 0:
                mine_local.append(next_mine)
        self.map.append(mine_local)
        mine0 = self.supervisor.getFromDef("mine0")
        mine0_pos = mine0.getField("translation")
        pos = mine_local[0]
        mine0_pos.setSFVec3f(pos)
        mine1 = self.supervisor.getFromDef("mine1")
        mine1_pos = mine1.getField("translation")
        pos = mine_local[1]
        mine1_pos.setSFVec3f(pos)
        mine2 = self.supervisor.getFromDef("mine2")
        mine2_pos = mine2.getField("translation")
        pos = mine_local[2]
        mine2_pos.setSFVec3f(pos)
        mine3 = self.supervisor.getFromDef("mine3")
        mine3_pos = mine3.getField("translation")
        pos = mine_local[3]
        mine3_pos.setSFVec3f(pos)
        mine4 = self.supervisor.getFromDef("mine4")
        mine4_pos = mine4.getField("translation")
        pos = mine_local[4]
        mine4_pos.setSFVec3f(pos)
        mine5 = self.supervisor.getFromDef("mine5")
        mine5_pos = mine5.getField("translation")
        pos = mine_local[5]
        mine5_pos.setSFVec3f(pos)

        
        
        
    def Lookup_subpos(self, index, Physical_pos, direc, offset1=0, offset2=0):
        # [-1:6]代表通路的方块序号
        # 实际只需要查找到长方块和初末位置的方块
        pos = [Physical_pos[0], Physical_pos[1], Physical_pos[2]]
        boundary = [Physical_pos[0], Physical_pos[1], Physical_pos[2]]
        size = [0, 0, 0]
        offset3 = 0
        if index == BLOCK_START:
            size = BLOCK_START_SIZE
            offset3=-size[2]*0.15
        elif index == BLOCK_TRAP:
            pass
        elif index == BLOCK_SECOND:  #
            size = BLOCK_SECOND_SIZE
        elif index == BLOCK_THIRD:
            size = BLOCK_THIRD_SIZE
        elif index == BLOCK_BRIDGE:
            pass
        elif index == BLOCK_FORTH:
            size = BLOCK_FORTH_SIZE
        elif index == BLOCK_STEP:
            pass
        elif index == BLOCK_END:
            size = BLOCK_END_SIZE
            offset3=size[2]*0.4
        if direc == 1:
            pos[2] = pos[2] - (0.5) * size[2] - offset1 - offset3
            boundary[2] = pos[2] + offset2

        elif direc == 0:
            pos[0] = pos[0] - (0.5) * size[2] - offset1 - offset3
            boundary[0] = pos[0] + offset2
        else:
            pos[2] = pos[2] + (0.5) * size[2] + offset1 + offset3
            boundary[2] = pos[2] - offset2

        return boundary, pos, direc

    def random_order(self):
        BLOCK_LEN = [0 for _ in range(BLOCK_AMOUNT)]
        BLOCK_DIREC = [0 for _ in range(BLOCK_AMOUNT)]
        """
        U型结构：
        X XX  X
              X
              X
        X XX  X
        """
        #print("随机开始")
        BLOCK_LEN[0] = BLOCK_TRAP_SIZE[2]
        BLOCK_LEN[1] = BLOCK_SECOND_SIZE[2]
        BLOCK_LEN[2] = BLOCK_THIRD_SIZE[2]
        BLOCK_LEN[3] = BLOCK_BRIDGE_SIZE[2]
        BLOCK_LEN[4] = BLOCK_FORTH_SIZE[2]
        BLOCK_LEN[5] = BLOCK_STEP_SIZE[2]
        block = [
            BLOCK_TRAP,
            BLOCK_SECOND,
            BLOCK_FORTH,
            BLOCK_BRIDGE,
            BLOCK_THIRD,
            BLOCK_STEP,
        ]  # 将序列BLOCK中的元素顺序打乱[0-5]
        while True:
            # if RANDOM_FLAG:
            #   random.shuffle(block)
            # print(block)
            self.map.append(block) # 记录方块顺序
            len_cnt = 0
            first_len = RACE_SINGLE_LEN
            second_len = first_len + RACE_SINGLE_LEN - RACE_MAX_WIDTH
            third_len = 0
            special_order_flag = 0
            for index in range(BLOCK_AMOUNT):
                len_cnt += BLOCK_LEN[block[index]]
                if len_cnt <= first_len:
                    BLOCK_DIREC[index] = 1
                elif len_cnt <= second_len:
                    BLOCK_DIREC[index] = 0
                else:
                    BLOCK_DIREC[index] = -1
                    third_len += BLOCK_LEN[block[index]]
                if index < BLOCK_AMOUNT - 1:
                    if (
                        block[index] == BLOCK_TRAP and block[index] == BLOCK_BRIDGE
                    ) or (block[index] == BLOCK_BRIDGE and block[index] == BLOCK_TRAP):
                        if BLOCK_DIREC[index] == BLOCK_DIREC[index + 1]:
                            special_order_flag = 1
            if third_len <= RACE_SINGLE_LEN - RACE_MAX_WIDTH and not special_order_flag:
                break
        # 改变转角关卡顺序
        for index in range(BLOCK_AMOUNT - 1):
            if BLOCK_DIREC[index] != BLOCK_DIREC[index + 1]:
                if block[index + 1] == BLOCK_BRIDGE or block[index + 1] == BLOCK_TRAP:
                    temp = block[index + 2]
                    block[index + 2] = block[index + 1]
                    block[index + 1] = temp
                    if block[index + 1] == BLOCK_BRIDGE or block[index + 1] == BLOCK_TRAP:
                        temp = block[index]
                        block[index] = block[index + 1]
                        block[index + 1] = temp
        return block, BLOCK_DIREC

    def Random_setLevel(self):
        print("location setup！")
        Race = [0 for _ in range(BLOCK_AMOUNT + 2)]
        Race_direc = [0 for _ in range(BLOCK_AMOUNT + 2)]

        boundary = self.setRace(BLOCK_START, [0, 0, 0], 0, 0)
        # print( boundary)
        Race[0] = boundary
        Race_direc[0] = 1
        block, block_direc = self.random_order()  # 关卡顺序和方向
        cnt = 0
        for index in range(BLOCK_AMOUNT):
            # (index)
            if cnt == 0:
                boundary = self.setRace(block[index], boundary, 1, block_direc[cnt])
            else:
                boundary = self.setRace(
                    block[index], boundary, block_direc[cnt - 1], block_direc[cnt]
                )
            cnt += 1
            Race[index + 1] = boundary
            Race_direc[index + 1] = block_direc[index]

        boundary = self.setRace(BLOCK_END, boundary, block_direc[cnt - 1], -1)
        Race[BLOCK_AMOUNT + 1] = boundary
        Race_direc[BLOCK_AMOUNT + 1] = -1
        race_order = [BLOCK_START] + block + [BLOCK_END]  # 方块顺序[-1-6]

        # 单边，地雷，挡板，门，球，双边index=0-5
        # 随机插入到已经随机的block之中

        block_type1 = [BLOCK_SECOND, BLOCK_THIRD, BLOCK_FORTH]  # 地雷,球,挡板所允许的方块：
        rest_type = [BLOCK_START, BLOCK_END]  # 剩下的方块
        sublevel = [0 for _ in range(6)]

        if RANDOM_FLAG:
            pass
            # random.shuffle(block_type1)
        sublevel[1] = block_type1[0]
        sublevel[4] = block_type1[1]
        block_type11 = [BLOCK_SECOND, BLOCK_THIRD, BLOCK_FORTH]  # 地雷,球,挡板所允许的方块：
        if RANDOM_FLAG:
            random.shuffle(block_type11)
            sublevel[5] = block_type11[0]
        else:
            sublevel[5] = block_type1[0]

        rest_type = rest_type + [block_type1[2]]
        # random.shuffle(rest_type)
        # 固定开始结束关卡
        sublevel[0] = rest_type[0]
        sublevel[3] = rest_type[2]
        sublevel[2] = rest_type[1]

        self.map.append([block_type1[0],rest_type[2],block_type1[1],block_type11[0]])

        level_cnt = -1
        # order按方块顺序
        temp= copy.deepcopy(Race)  # 复制链表
        # self.levelorder.append(index)
        for index in range(8):
            # 若为陷阱和桥所在方块
            if race_order[index] == BLOCK_TRAP:
                level_cnt += 1
                self.levelorder[level_cnt] = LEVEL_TRAP
                self.levelendpos[level_cnt] = Race[index] + [
                    Race_direc[index]
                ]
                #print(Race[index])
            elif race_order[index] == BLOCK_BRIDGE:
                level_cnt += 1
                self.levelorder[level_cnt] = LEVEL_BRIDGE
                self.levelendpos[level_cnt] = Race[index] + [
                    Race_direc[index]
                ]
                ##print(Race[index])
            elif race_order[index] == BLOCK_STEP:
                level_cnt += 1
                self.levelorder[level_cnt] = LEVEL_STEP
                self.levelendpos[level_cnt] = Race[index] + [
                    Race_direc[index]
                ]
                #print(Race[index])
            # 若为其他方块
            else:
                for i in range(6):
                    # 为每个关卡找到对应的方块
                    if sublevel[i] == race_order[index]:
                        level_cnt += 1
                        # 得到方块的坐标
                        Physical_boundary = temp[index]
                        direc = Race_direc[index]
                        
                        if i == 0:
                            # offset1=关卡中心位置离方块中心位置的关系
                            # offset1=关卡结束位置相对关卡中心 单方向位移
                            offset1 = 0
                            offset2 = UNIT_LEN*0.7
                            boundary, pos, direc = self.Lookup_subpos(
                                race_order[index],
                                Physical_boundary,
                                direc,
                                offset1,
                                offset2,
                            )
                            # 根据坐标设置小关卡的位置和方向
                            rot = [0, 1, 0, ROT_PI * (1 - direc)]
                            self.set_sub_bar(pos, rot)
                            self.levelorder[level_cnt] = LEVEL_BARRIER
                            self.levelendpos[level_cnt] = boundary + [direc]
                            # print(pos)
                            # print(boundary)
                        if i == 1:
                            offset1 = 0
                            offset2 = UNIT_LEN*2
                            boundary, pos, direc = self.Lookup_subpos(
                                race_order[index],
                                Physical_boundary,
                                direc,
                                offset1,
                                offset2,
                            )
                            # 根据坐标设置小关卡的位置和方向
                            rot = [0, 1, 0, ROT_PI * (1 - direc)]
                            self.set_sub_mine(pos, rot)
                            self.levelorder[level_cnt] = LEVEL_MINE
                            self.levelendpos[level_cnt] = boundary + [direc]
                            # print(boundary)
                        
                        if i == 3:
                            offset1 = 0
                            offset2 = UNIT_LEN
                            boundary, pos, direc = self.Lookup_subpos(
                                race_order[index],
                                Physical_boundary,
                                direc,
                                offset1,
                                offset2,
                            )
                            # 根据坐标设置小关卡的位置和方向
                            rot = [0, 1, 0, ROT_PI * (1 - direc)]
                            self.set_sub_door(pos, rot)
                            self.levelorder[level_cnt] = LEVEL_DOOR
                            self.levelendpos[level_cnt] = boundary + [direc]
                            # print(boundary)
                        if i == 4:
                            offset1 = UNIT_LEN
                            offset2 = UNIT_LEN
                            boundary, pos, direc = self.Lookup_subpos(
                                race_order[index],
                                Physical_boundary,
                                direc,
                                offset1,
                                offset2,
                            )
                            # 根据坐标设置小关卡的位置和方向
                            rot = [0, 1, 0, ROT_PI * (1 - direc)]
                            self.set_sub_ball(pos, rot)
                            self.levelorder[level_cnt] = LEVEL_BALL
                            self.levelendpos[level_cnt] = boundary + [direc]
                            #print(boundary + [direc])
                            # print(boundary)
                        if i == 2:
                            offset1 = -0.2
                            offset2 = UNIT_LEN*1.2
                            boundary, pos, direc = self.Lookup_subpos(
                                race_order[index],
                                Physical_boundary,
                                direc,
                                offset1,
                                offset2,
                            )

                            # 根据坐标设置小关卡的位置和方向
                            rot = [0, 1, 0, ROT_PI * (1 - direc)]
                            self.set_sub_doublebar(pos, rot)
                            self.levelorder[level_cnt] = LEVEL_DOUBLEBAR
                            self.levelendpos[level_cnt] = boundary + [direc]
                            # print(self.levelendpos[level_cnt])
                        if i == 5:
                            offset1 =-UNIT_LEN*2.0  #调整挡板位置
                            offset2 = UNIT_LEN*1.5
                            boundary, pos, direc = self.Lookup_subpos(
                                race_order[index],
                                Physical_boundary,
                                direc,
                                offset1,
                                offset2,
                            )
                            # 根据坐标设置小关卡的位置和方向
                            rot = [0, 1, 0, ROT_PI * (1 - direc)]
                            self.set_sub_obscle(pos, rot)
                            self.levelorder[level_cnt] = LEVEL_OBSCLE
                            self.levelendpos[level_cnt] = boundary + [direc]
                            # print(boundary)
                        

        
        self.setpos_darwin(DARWIN_START_POINT, DEFAULT_DIREC)
        self.set_random_mine()
        # 消除惯性
        self.supervisor.step(100)
        self.supervisor.simulationResetPhysics()
        

        # print(self.levelorder)
        print(self.map)
        print("initialization finished！")
        
        
        

    # timer function
    def call_timer(self):
        self.darwin_position = self.darwin_translation_field.getSFVec3f()
        if self.start == 1:
            self.time += self.timestep / 1000
        # print(self.time)
        if self.state == 0 and self.start == 0 and self.darwin_position[2]>-1.7:
            self.time = 0  # start
            self.start = 1
        elif self.state == 9:
            self.start = 0
        time_string = ("%02d:%02d") % (
            math.floor(self.time // 60),
            math.floor(self.time % 60),
        )
        # 监视器显示
        self.supervisor.setLabel(
            2, time_string, 0.45, 0.01, 0.1, 0x000000, 0.0, "Arial"
        )
        # black

    def set_scores(self, result):
        score = ("total score:%d") % (result)
        self.supervisor.setLabel(0, score, 0.7, 0.01, 0.1, 0x0000FF, 0.0, "Arial")

    def set_level(self, level):
        level = ("current level:%d") % (level)
        self.supervisor.setLabel(1, level, 0.7, 0.06, 0.1, 0x0000FF, 0.0, "Arial")
        if self.state == 9:
            self.end_flag = 1
            self.supervisor.setLabel(
                3, "Congratulations!", 0.42, 0.06, 0.1, 0x0000FF, 0.0, "Arial"
            )

    def update_flag(self):
        ball_position = self.ball_translation_field.getSFVec3f()
        hole_position = self.hole_translation_field.getSFVec3f()
        distance = (
            (ball_position[0] - hole_position[0]) ** 2
            + (ball_position[2] - hole_position[2]) ** 2
        ) ** 0.5
        self.darwin_position = self.darwin_translation_field.getSFVec3f()
        if distance <= HOLE_RADIUS and self.ball_hit == 0 and ball_position[1] <=BALL_HEIGHT+0.02:
            self.ball_hit = 1
            self.log += ("%02d:%02d-scored a goal!\n") % (
            math.floor(self.time // 60),
            math.floor(self.time % 60),
        )
            print("scored a goal!")
        # flag
        self.mine_flag = int(self.mine.getField("customData").getSFString())
        self.door_flag = int(self.door.getField("customData").getSFString())
        self.bar_flag = int(self.bar.getField("customData").getSFString())
        self.d_bar_flag = int(self.double_bar.getField("customData").getSFString())

        obscle_position = self.obscle_translation_field.getSFVec3f()
        if obscle_position[1] < OBSCLE_HEIGHT-0.02 and self.obscle_falldown == 0:
            # 挡板z轴方向中心点坐标大于0.5视为不倒地
            self.log += ("%02d:%02d-obstacle fell down\n") % (
            math.floor(self.time // 60),
            math.floor(self.time % 60),
        )
            print("obstacle fell down!")
            self.obscle_falldown = 1
        if self.darwin_position[1] < ROBOT_OUT_RACE and self.end_flag == 0:
            # z轴方向中心点坐标小于于0.5视为脱离赛道
            self.log += ("%02d:%02d-robot out\n") % (
            math.floor(self.time // 60),
            math.floor(self.time % 60),
        )
            print("robot out!")
            self.end_flag = 1

    def Arrival_the_location(self, endpos_and_direc):
        robot_pos = self.darwin_position
        # print(endpos_and_direc)
        end_pos = [endpos_and_direc[0], endpos_and_direc[1], endpos_and_direc[2]]
        end_direc = endpos_and_direc[3]
        if end_direc == 1:
            if (
                robot_pos[2] > end_pos[2]
                and robot_pos[0] > end_pos[0] - RACE_MAX_WIDTH/2
                and robot_pos[0] < end_pos[0] + RACE_MAX_WIDTH/2
            ):
                return True
            else:
                return False
        elif end_direc == 0:
            if (
                robot_pos[0] > end_pos[0]
                and robot_pos[2] > end_pos[2] - RACE_MAX_WIDTH/2
                and robot_pos[2] < end_pos[2] + RACE_MAX_WIDTH/2
            ):
                return True
            else:
                return False
        else:
            if (
                robot_pos[2] < end_pos[2]
                and robot_pos[0] > end_pos[0] - RACE_MAX_WIDTH/2
                and robot_pos[0] < end_pos[0] + RACE_MAX_WIDTH/2
            ):
                return True
            else:
                return False

    def calc_scores(self):
        # ruler：http://www.running-robot.net/rules/7.html
        # order
        # 结束计算分数
        robot_pos = self.darwin_translation_field
        if self.state >=9:
            return
        # 计算分数
        # LevelName:barrier,trap,mine,obscle,door,bridge,ball,step,double_bar
        # =1-9
        LevelName = self.levelorder[self.state]
        # print(self.levelorder)
        Endpos_and_direc = self.levelendpos[self.state]
        if LevelName == LEVEL_BARRIER:
            if self.Arrival_the_location(Endpos_and_direc):
                if self.bar_flag != 1:
                    self.scores += SCORE_BAR
                    self.log += ("%02d:%02d-passed the barrier +10\n") % (
                    math.floor(self.time // 60),
                    math.floor(self.time % 60),
                )
                else:
                    self.scores +=  SCORE_BAR/2
                    self.log += ("%02d:%02d-passed the barrier +5\n") % (
                    math.floor(self.time // 60),
                    math.floor(self.time % 60),
                )
                self.state += 1
                print("passed the barrier")
        elif LevelName == LEVEL_TRAP:
            if self.Arrival_the_location(Endpos_and_direc):
                self.scores += SCORE_TRAP
                self.state += 1
                self.log += ("%02d:%02d-passed the trap +%02d\n") % (
                math.floor(self.time // 60),
                math.floor(self.time % 60), SCORE_TRAP
             )
                print("passed the trap")
        elif LevelName == LEVEL_MINE:
            if self.Arrival_the_location(Endpos_and_direc):
                if self.mine_flag == 0:
                    self.scores +=SCORE_MINE
                    self.log += ("%02d:%02d-passed the mine field +%02d\n") % (
                    math.floor(self.time // 60),
                    math.floor(self.time % 60), SCORE_MINE
                )
                elif self.mine_flag == 1:
                    self.scores += 10
                    self.log += ("%02d:%02d-passed the mine field +10\n") % (
                    math.floor(self.time // 60),
                    math.floor(self.time % 60),
                )
                elif self.mine_flag == 2:
                    self.log += ("%02d:%02d-passed the mine field +0\n") % (
                    math.floor(self.time // 60),
                    math.floor(self.time % 60),
                )
                self.state += 1
                print("passed the mine field")
                
        elif LevelName == LEVEL_OBSCLE:
            if self.Arrival_the_location(Endpos_and_direc):
                if self.obscle_falldown == 0:
                    self.scores += SCORE_OBSCLE
                    self.log += ("%02d:%02d-passed the obstacle +%02d\n") % (
                    math.floor(self.time // 60),
                    math.floor(self.time % 60), SCORE_OBSCLE
                )
                else:
                    self.log += ("%02d:%02d-passed the obstacle +%d\n") % (
                    math.floor(self.time // 60),
                    math.floor(self.time % 60), 0
                )
                self.state += 1
                print("passed the obstacle")

        elif LevelName == LEVEL_DOOR:
            if self.Arrival_the_location(Endpos_and_direc):
                if self.door_flag == 0:
                    self.scores += SCORE_DOOR
                    self.log += ("%02d:%02d-passed the door +%02d\n") % (
                    math.floor(self.time // 60),
                    math.floor(self.time % 60), SCORE_DOOR
                )
                elif self.door_flag == 1:
                    self.scores +=  SCORE_DOOR/2
                    self.log += ("%02d:%02d-passed the door +%d\n") % (
                    math.floor(self.time // 60),
                    math.floor(self.time % 60), SCORE_DOOR/2
                )
                self.state += 1
                print("passed the door")

        elif LevelName ==LEVEL_BRIDGE:
            if self.Arrival_the_location(Endpos_and_direc):
                self.scores += SCORE_BRIDGE
                self.log += ("%02d:%02d-passed the bridge +%02d\n") % (
                math.floor(self.time // 60),
                math.floor(self.time % 60), SCORE_BRIDGE
             )
                self.state += 1
                print("passed the bridge")

        elif LevelName == LEVEL_BALL:
            if self.Arrival_the_location(Endpos_and_direc):
                if self.ball_hit:
                    self.scores += SCORE_BALL
                    self.log += ("%02d:%02d-passed the ball field +%02d\n") % (
                    math.floor(self.time // 60),
                    math.floor(self.time % 60), SCORE_BALL
                 )
                else:
                    self.log += ("%02d:%02d-passed the ball field +%d\n") % (
                    math.floor(self.time // 60),
                    math.floor(self.time % 60), 0
                 )
                self.state += 1
                print("passed the ball field")
          
        elif LevelName == LEVEL_STEP:
            temp=[0,0,0,0]
            if Endpos_and_direc[3]==1:
                    peak_pos=[Endpos_and_direc[0],Endpos_and_direc[1],Endpos_and_direc[2]-0.675]
                    temp=peak_pos+[Endpos_and_direc[3]]
            elif   Endpos_and_direc[3]==0:
                peak_pos=[Endpos_and_direc[0]-0.675,Endpos_and_direc[1],Endpos_and_direc[2]] 
                temp=peak_pos+[Endpos_and_direc[3]]
            else:
                peak_pos=[Endpos_and_direc[0],Endpos_and_direc[1],Endpos_and_direc[2]+0.675]
                temp=peak_pos+[Endpos_and_direc[3]]
            if self.Arrival_the_location(temp) and self.first_arrival_peak:
                self.scores += SCORE_STEP_UP
                self.first_arrival_peak=0
                self.log += ("%02d:%02d-passed the peak +%02d\n") % (
                math.floor(self.time // 60),
                math.floor(self.time % 60), SCORE_STEP_UP
             )
                print("passed the peak")

            if self.Arrival_the_location(Endpos_and_direc):
                self.scores += SCORE_STEP_DOWN
                self.log += ("%02d:%02d-passed the step +%02d\n") % (
                math.floor(self.time // 60),
                math.floor(self.time % 60), SCORE_STEP_DOWN
             )
                self.state += 1
                print("passed the step")

        elif LevelName == LEVEL_DOUBLEBAR:
            if self.Arrival_the_location(Endpos_and_direc):
                if self.d_bar_flag == 0:
                    self.scores +=SCORE_DOUBLEBAR
                    self.log += ("%02d:%02d-passed the double barrier +%02d\n") % (
                    math.floor(self.time // 60),
                    math.floor(self.time % 60), SCORE_DOUBLEBAR
                 )
                elif self.d_bar_flag == 1:
                    self.scores += SCORE_DOUBLEBAR/2
                    self.log += ("%02d:%02d-passed the double barrier +%02d\n") % (
                    math.floor(self.time // 60),
                    math.floor(self.time % 60), SCORE_DOUBLEBAR/2
                 )
                self.state += 1
                print("passed the double barrier")
                
    def step(self):
        cnt = 0
        while self.supervisor.step(self.timestep) != -1:
            self.call_timer()
            self.update_flag()
            self.calc_scores()
            self.set_scores(self.scores)
            self.set_level(self.state)
            
            cnt = cnt + 1
            if cnt == self.single_action_num:
                break

     
def open_config():
    this_file_absolute_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
    this_files_directory_absolute_path = os.path.dirname(this_file_absolute_path)
    os.system(this_files_directory_absolute_path+'/worlds/config.txt')

def read_team_name():
    this_file_absolute_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
    this_files_directory_absolute_path = os.path.dirname(this_file_absolute_path)
    f = open(this_files_directory_absolute_path+'/worlds/config.txt','r', encoding='UTF-8')
    t = f.readline()
    #print (t[10:])
    return t[10:]
    
def write_log(log):
    this_file_absolute_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
    this_files_directory_absolute_path = os.path.dirname(this_file_absolute_path)
    f = open(this_files_directory_absolute_path+'/worlds/log.txt','a', encoding='UTF-8')
    f.write(log)
    #print (t[10:])
    f.close()


if __name__ == "__main__":
    env = runningrobot_env(timestep=None)
    env.Random_setLevel()
    #open_config()
    max_time = 1500000
    episode = 0
    print("begin playing!")
    while episode < max_time:
        env.step()
        if env.end_flag:
            break
        episode += 1
    print("game over!")
    team_name = read_team_name()
    data="参赛队伍名称："+team_name+"webots大赛比赛得分："+str(env.scores)+"\n比赛用时："+str(round(env.time,2)) + "s\n" + "比赛场地方块顺序："+ str(env.map[0]) + "\n关卡所在方块："+ str(env.map[1]) + "\n地雷坐标："+ str(env.map[2])+ "\nver: 2.0" + "\n" + env.log
    print(data)
    write_log(time.strftime("\n%Y-%m-%d %H:%M:%S\n", time.localtime()) + data)    
