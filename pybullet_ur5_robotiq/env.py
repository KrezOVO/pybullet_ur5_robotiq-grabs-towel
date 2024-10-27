import time
import math
import random

import numpy as np
import pybullet as p
import pybullet_data

from utilities import Models, Camera
from collections import namedtuple
from attrdict import AttrDict
from tqdm import tqdm


class FailToReachTargetError(RuntimeError):
    pass


class ClutteredPushGrasp:

    SIMULATION_STEP_DELAY = 1 / 240.

    def __init__(self, robot, towel_path, camera=None, vis=False) -> None:
        self.robot = robot
        self.vis = vis
        if self.vis:
            self.p_bar = tqdm(ncols=0, disable=False)
        self.camera = camera

        # define environment
        self.physicsClient = p.connect(p.GUI if self.vis else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        self.planeID = p.loadURDF("plane.urdf")

        self.robot.load()
        self.robot.step_simulation = self.step_simulation

        # 加载毛巾模型
        self.towel_id = p.loadSoftBody(
            fileName=towel_path,
            basePosition=[0, 0, 0.01],  # 将毛巾放置在地面上方一点点的位置
            baseOrientation=[0, 0, 0, 1],
            scale=0.5,  # 减小毛巾的尺寸
            mass=0.5,   # 相应减小质量
            useNeoHookean=0,
            useBendingSprings=0,
            useMassSpring=0,
            springElasticStiffness=1.0,
            springDampingStiffness=0.1,
            springDampingAllDirections=1,
            useSelfCollision=0,
            frictionCoeff=.5,
            useFaceContact=0,
            collisionMargin=0.001
        )

        print(f"毛巾加载成功,ID为: {self.towel_id}")

        # custom sliders to tune parameters (name of the parameter,range,initial value)
        self.xin = p.addUserDebugParameter("x", -0.224, 0.224, 0)
        self.yin = p.addUserDebugParameter("y", -0.224, 0.224, 0)
        self.zin = p.addUserDebugParameter("z", 0, 1., 0.5)
        self.rollId = p.addUserDebugParameter("roll", -3.14, 3.14, 0)
        self.pitchId = p.addUserDebugParameter("pitch", -3.14, 3.14, np.pi/2)
        self.yawId = p.addUserDebugParameter("yaw", -np.pi/2, np.pi/2, np.pi/2)
        self.gripper_opening_length_control = p.addUserDebugParameter("gripper_opening_length", 0, 0.085, 0.04)

    def step_simulation(self):
        """
        Hook p.stepSimulation()
        """
        p.stepSimulation()
        if self.vis:
            time.sleep(self.SIMULATION_STEP_DELAY)
            self.p_bar.update(1)

    def read_debug_parameter(self):
        # read the value of task parameter
        x = p.readUserDebugParameter(self.xin)
        y = p.readUserDebugParameter(self.yin)
        z = p.readUserDebugParameter(self.zin)
        roll = p.readUserDebugParameter(self.rollId)
        pitch = p.readUserDebugParameter(self.pitchId)
        yaw = p.readUserDebugParameter(self.yawId)
        gripper_opening_length = p.readUserDebugParameter(self.gripper_opening_length_control)

        return x, y, z, roll, pitch, yaw, gripper_opening_length

    def step(self, action, control_method='joint'):
        """
        action: (x, y, z, roll, pitch, yaw, gripper_opening_length) for End Effector Position Control
                (a1, a2, a3, a4, a5, a6, a7, gripper_opening_length) for Joint Position Control
        control_method:  'end' for end effector position control
                         'joint' for joint position control
        """
        assert control_method in ('joint', 'end')
        self.robot.move_ee(action[:-1], control_method)
        self.robot.move_gripper(action[-1])
        for _ in range(120):  # Wait for a few steps
            self.step_simulation()

        reward = self.update_reward()
        done = True if reward == 1 else False
        info = dict(box_opened=self.box_opened, btn_pressed=self.btn_pressed, box_closed=self.box_closed)
        return self.get_observation(), reward, done, info

    def update_reward(self):
        # 由于删除了盒子,我们需要修改奖励计算逻辑
        # 这里暂时返回一个固定值,您可以根据需要自定义奖励函数
        return 0

    def get_observation(self):
        obs = dict()
        if isinstance(self.camera, Camera):
            rgb, depth, seg = self.camera.shot()
            obs.update(dict(rgb=rgb, depth=depth, seg=seg))
        else:
            assert self.camera is None
        obs.update(self.robot.get_joint_obs())

        return obs

    def reset(self):
        self.robot.reset()
        
        # 重置毛巾位置
        p.resetBasePositionAndOrientation(self.towel_id, [0, 0, 0.01], [0, 0, 0, 1])
        
        return self.get_observation()

    def close(self):
        p.disconnect(self.physicsClient)
