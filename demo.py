import pybullet as p
import pybullet_data
import time
import os
import math
import numpy as np
from collections import namedtuple
from tqdm import tqdm

class FailToReachTargetError(RuntimeError):
    pass

class RobotBase(object):
    def __init__(self, pos, ori):
        self.base_pos = pos
        self.base_ori = p.getQuaternionFromEuler(ori)

    def load(self):
        self.__init_robot__()
        self.__parse_joint_info__()
        self.__post_load__()
        print(self.joints)

    def step_simulation(self):
        raise RuntimeError('`step_simulation` method of RobotBase Class should be hooked by the environment.')

    def __parse_joint_info__(self):
        numJoints = p.getNumJoints(self.id)
        jointInfo = namedtuple('jointInfo', 
            ['id','name','type','damping','friction','lowerLimit','upperLimit','maxForce','maxVelocity','controllable'])
        self.joints = []
        self.controllable_joints = []
        for i in range(numJoints):
            info = p.getJointInfo(self.id, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = info[2]
            jointDamping = info[6]
            jointFriction = info[7]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            controllable = (jointType != p.JOINT_FIXED)
            if controllable:
                self.controllable_joints.append(jointID)
                p.setJointMotorControl2(self.id, jointID, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
            info = jointInfo(jointID,jointName,jointType,jointDamping,jointFriction,jointLowerLimit,
                            jointUpperLimit,jointMaxForce,jointMaxVelocity,controllable)
            self.joints.append(info)

        assert len(self.controllable_joints) >= self.arm_num_dofs
        self.arm_controllable_joints = self.controllable_joints[:self.arm_num_dofs]

        self.arm_lower_limits = [info.lowerLimit for info in self.joints if info.controllable][:self.arm_num_dofs]
        self.arm_upper_limits = [info.upperLimit for info in self.joints if info.controllable][:self.arm_num_dofs]
        self.arm_joint_ranges = [info.upperLimit - info.lowerLimit for info in self.joints if info.controllable][:self.arm_num_dofs]

    def __init_robot__(self):
        raise NotImplementedError
    
    def __post_load__(self):
        pass

    def reset(self):
        self.reset_arm()
        self.reset_gripper()

    def reset_arm(self):
        for rest_pose, joint_id in zip(self.arm_rest_poses, self.arm_controllable_joints):
            p.resetJointState(self.id, joint_id, rest_pose)
        for _ in range(10):
            self.step_simulation()

    def reset_gripper(self):
        self.open_gripper()

    def open_gripper(self):
        self.move_gripper(self.gripper_range[1])

    def close_gripper(self):
        self.move_gripper(self.gripper_range[0])

    def move_ee(self, action, control_method):
        assert control_method in ('joint', 'end')
        if control_method == 'end':
            x, y, z, roll, pitch, yaw = action
            pos = (x, y, z)
            orn = p.getQuaternionFromEuler((roll, pitch, yaw))
            joint_poses = p.calculateInverseKinematics(self.id, self.eef_id, pos, orn,
                                                       self.arm_lower_limits, self.arm_upper_limits, self.arm_joint_ranges, self.arm_rest_poses,
                                                       maxNumIterations=20)
        elif control_method == 'joint':
            assert len(action) == self.arm_num_dofs
            joint_poses = action
        for i, joint_id in enumerate(self.arm_controllable_joints):
            p.setJointMotorControl2(self.id, joint_id, p.POSITION_CONTROL, joint_poses[i],
                                    force=self.joints[joint_id].maxForce, maxVelocity=self.joints[joint_id].maxVelocity)

    def move_gripper(self, open_length):
        raise NotImplementedError

    def get_joint_obs(self):
        positions = []
        velocities = []
        for joint_id in self.controllable_joints:
            pos, vel, _, _ = p.getJointState(self.id, joint_id)
            positions.append(pos)
            velocities.append(vel)
        ee_pos = p.getLinkState(self.id, self.eef_id)[0]
        return dict(positions=positions, velocities=velocities, ee_pos=ee_pos)

class UR5Robotiq85(RobotBase):
    def __init_robot__(self):
        self.eef_id = 7
        self.arm_num_dofs = 6
        self.arm_rest_poses = [-1.5690622952052096, -1.5446774605904932, 1.343946009733127, -1.3708613585093699,
                               -1.5707970583733368, 0.0009377758247187636]
        self.id = p.loadURDF('./ur/urdf/ur5_robotiq_85.urdf', self.base_pos, self.base_ori,
                             useFixedBase=True, flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
        self.gripper_range = [0, 0.085]
    
    def __post_load__(self):
        mimic_parent_name = 'finger_joint'
        mimic_children_names = {'right_outer_knuckle_joint': 1,
                                'left_inner_knuckle_joint': 1,
                                'right_inner_knuckle_joint': 1,
                                'left_inner_finger_joint': -1,
                                'right_inner_finger_joint': -1}
        self.__setup_mimic_joints__(mimic_parent_name, mimic_children_names)

    def __setup_mimic_joints__(self, mimic_parent_name, mimic_children_names):
        self.mimic_parent_id = [joint.id for joint in self.joints if joint.name == mimic_parent_name][0]
        self.mimic_child_multiplier = {joint.id: mimic_children_names[joint.name] for joint in self.joints if joint.name in mimic_children_names}

        for joint_id, multiplier in self.mimic_child_multiplier.items():
            c = p.createConstraint(self.id, self.mimic_parent_id,
                                   self.id, joint_id,
                                   jointType=p.JOINT_GEAR,
                                   jointAxis=[0, 1, 0],
                                   parentFramePosition=[0, 0, 0],
                                   childFramePosition=[0, 0, 0])
            p.changeConstraint(c, gearRatio=-multiplier, maxForce=100, erp=1)

    def move_gripper(self, open_length):
        open_angle = 0.715 - math.asin((open_length - 0.010) / 0.1143)
        p.setJointMotorControl2(self.id, self.mimic_parent_id, p.POSITION_CONTROL, targetPosition=open_angle,
                                force=self.joints[self.mimic_parent_id].maxForce, maxVelocity=self.joints[self.mimic_parent_id].maxVelocity)

# 使用示例
class ClutteredPushGrasp:
    def __init__(self, robot, vis=True):
        self.robot = robot
        self.vis = vis
        self.physicsClient = p.connect(p.GUI if self.vis else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        self.planeID = p.loadURDF("plane.urdf")
        self.robot.load()
        self.robot.step_simulation = self.step_simulation

        # 加载毛巾
        self.load_towel()

        # 获取机械臂参数
        self.parse_robot_joints()

        # 设置摄像头
        self.setup_camera()

    def setup_camera(self):
        # 设置摄像头的目标位置（场景中心）
        target_position = [0.5, 0, 0]
        
        # 设置摄像头的距离、偏航角和俯仰角
        distance = 1.5
        yaw = 150
        pitch = -30
        
        # 应用摄像头设置
        p.resetDebugVisualizerCamera(distance, yaw, pitch, target_position)

    def parse_robot_joints(self):
        numJoints = p.getNumJoints(self.robot.id)
        jointInfo = namedtuple('jointInfo',
                    ['id','name','type','damping','friction','lowerLimit','upperLimit','maxForce','maxVelocity','controllable'])
        self.joints = []
        self.controllable_joints = []
        for i in range(numJoints):
            info = p.getJointInfo(self.robot.id, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = info[2]
            print(f"The {i} joint type is: {jointType}")
            jointDamping = info[6]
            jointFriction = info[7]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            controllable = (jointType != p.JOINT_FIXED)
            if controllable:
                self.controllable_joints.append(jointID)
                p.setJointMotorControl2(self.robot.id, jointID, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
            info = jointInfo(jointID,jointName,jointType,jointDamping,jointFriction,jointLowerLimit,
                            jointUpperLimit,jointMaxForce,jointMaxVelocity,controllable)
            self.joints.append(info)
        print(f"The controllable_joints are: {self.controllable_joints}")

    def load_towel(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.towel_position = [0.5, 0, 0.01]
        towelStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
        towel_path = os.path.join(current_dir, "towel", "towel2.obj")
        self.towelId = p.loadSoftBody(
            fileName=towel_path,
            basePosition=self.towel_position,
            baseOrientation=towelStartOrientation,
            scale=0.5,
            mass=1.0,
            useNeoHookean=0,
            useBendingSprings=1,
            useMassSpring=1,
            springElasticStiffness=40,
            springDampingStiffness=0.1,
            springDampingAllDirections=1,
            useSelfCollision=1,
            frictionCoeff=0.5,
            useFaceContact=1
        )

    def move_arm_to_target(self, target_position, target_orientation):
        joint_positions = p.calculateInverseKinematics(self.robot.id, 7, target_position, target_orientation, maxNumIterations=40)
        for joint_index, joint_id in enumerate(self.controllable_joints[:6]):
            p.setJointMotorControl2(self.robot.id, joint_id, p.POSITION_CONTROL, joint_positions[joint_index], force=3)

    def step_simulation(self):
        p.stepSimulation()

    def reset(self):
        self.robot.reset()
        p.resetBasePositionAndOrientation(self.towelId, self.towel_position, p.getQuaternionFromEuler([0, 0, 0]))
        return self.get_observation()

    def get_observation(self):
        robot_obs = self.robot.get_joint_obs()
        towel_pos, towel_orn = p.getBasePositionAndOrientation(self.towelId)
        robot_obs['towel_pos'] = towel_pos
        robot_obs['towel_orn'] = towel_orn
        return robot_obs

    def step(self, action, control_method='end'):
        self.move_arm_to_target(action[:3], p.getQuaternionFromEuler(action[3:6]))
        self.robot.move_gripper(action[6])
        for _ in range(120):
            self.step_simulation()
        return self.get_observation(), 0, False, {}

    def close(self):
        p.disconnect(self.physicsClient)

# 主程序
if __name__ == "__main__":
    robot = UR5Robotiq85([0, 0, 0], [0, 0, 0])
    env = ClutteredPushGrasp(robot)

    # 定义关键位置
    initial_pos = [0.3, -0.2, 0.5]
    towel_corner_pos = [0.5, 0, 0.02]  # 毛巾的一个角的位置
    lift_pos = [0.5, 0, 0.2]
    fold_pos = [0.7, 0.2, 0.02]

    # 执行抓取和折叠动作
    actions = [
        initial_pos + [0, -math.pi/2, 0, 0.085],  # 移动到初始位置，夹爪打开
        towel_corner_pos + [0, -math.pi/2, 0, 0.085],  # 移动到毛巾角落
        towel_corner_pos + [0, -math.pi/2, 0, 0.02],  # 闭合夹爪
        lift_pos + [0, -math.pi/2, 0, 0.02],  # 提起毛巾
        fold_pos + [0, -math.pi/2, 0, 0.02],  # 移动到折叠位置
        fold_pos + [0, -math.pi/2, 0, 0.085],  # 释放毛巾
        lift_pos + [0, -math.pi/2, 0, 0.085],  # 抬起机械臂
    ]

    for i, action in enumerate(actions):
        print(f"执行动作 {i+1}")
        obs, reward, done, info = env.step(action)
        print(f"末端执行器位置: {obs['ee_pos']}")
        print(f"毛巾位置: {obs['towel_pos']}")
        time.sleep(1)  # 暂停1秒，便于观察

    env.close()
