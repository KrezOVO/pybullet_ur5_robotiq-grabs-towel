import os

import numpy as np
import pybullet as p

from tqdm import tqdm
from env import ClutteredPushGrasp
from robot import Panda, UR5Robotiq85
from utilities import YCBModels, Camera
import time
import math


def user_control_demo():
    
    # 添加towel模型的路径
    towel_path = os.path.join(os.path.dirname(__file__), 'towel', 'towel2.obj')
    
    # 修改摄像机参数
    camera = Camera(
        (0.5, 0.5, 0.5),  # 摄像机位置更靠近场景中心
        (0, 0, 0),        # 摄像机看向原点
        (0, 0, 1),        # 摄像机向上方向
        0.1,              # 近裁剪平面
        2,                # 远裁剪平面（减小以避免看到太远）
        (320, 320),       # 图像分辨率保持不变
        60              # 增大视场角以看到更多内容
    )
    
    # robot = Panda((0, 0.5, 0), (0, 0, math.pi))
    robot = UR5Robotiq85((0, 0.5, 0), (0, 0, 0))
    
    # 将ycb_models替换为towel_path
    env = ClutteredPushGrasp(robot, towel_path, camera, vis=True)

    env.reset()
    
    # 移动到毛巾角落
    env.move_to_towel_corner()
    
    while True:
        obs, info = env.step(env.read_debug_parameter(), 'end')
        # print(obs, info)


if __name__ == '__main__':
    user_control_demo()
