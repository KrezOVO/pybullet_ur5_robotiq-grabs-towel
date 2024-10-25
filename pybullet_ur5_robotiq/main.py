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
    
    camera = Camera((1, 1, 1),
                    (0, 0, 0),
                    (0, 0, 1),
                    0.1, 5, (320, 320), 40)
    camera = None
    # robot = Panda((0, 0.5, 0), (0, 0, math.pi))
    robot = UR5Robotiq85((0, 0.5, 0), (0, 0, 0))
    
    # 将ycb_models替换为towel_path
    env = ClutteredPushGrasp(robot, towel_path, camera, vis=True)

    env.reset()
    # env.SIMULATION_STEP_DELAY = 0
    while True:
        obs, reward, done, info = env.step(env.read_debug_parameter(), 'end')
        # print(obs, reward, done, info)


if __name__ == '__main__':
    user_control_demo()
