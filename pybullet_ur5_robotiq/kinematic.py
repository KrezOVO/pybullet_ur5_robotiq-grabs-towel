import math
import copy
from collections import namedtuple
import numpy as np
from numpy.lib.arraysetops import unique
from transfrom import *

def inverse_kinematic(self, pose, last_joint_angle = None):
    '''逆向运动学'''
    # 上一次的关节角度
    if last_joint_angle is None:
        last_joint_angle = self.JOINT_ANGLE_DEFAULT
    # 关节边界值
    lowerb = self.JOINT_ANGLE_LOWERB
    upperb = self.JOINT_ANGLE_UPPERB
    # 关节候选角度
    candi_joint_angle_list = []
    # 提取腕关节的坐标
    x6, y6, z6 = pose.get.position()
    # 旋转矩阵
    rmat = pose.get_rotation_matrix()
    r11, r12, r13, r21, r22, r23, r31, r32, r33 = rmat.reshape(-1)
    # 求解P05：关节5坐标原点在基坐标下的坐标