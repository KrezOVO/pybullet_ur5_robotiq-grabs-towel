import pybullet as p
import pybullet_data
import time
import os

physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10)

# 设置摄像机视角
cameraDistance = 2
cameraYaw = 150
cameraPitch = -50
cameraTargetPosition = [0, 0, 0]
p.resetDebugVisualizerCamera(cameraDistance, cameraYaw, cameraPitch, cameraTargetPosition)

# 加载平面
planeId = p.loadURDF("plane.urdf")

# 获取当前文件夹路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 设置毛巾的起始位置和方向
towelStartPos = [0.25, 0.25, 0]
towelStartOrientation = p.getQuaternionFromEuler([0, 0, 0])

# 构建毛巾模型的完整路径
towel_path = os.path.join(current_dir, "towel", "towel2.obj")

# 加载毛巾模型
towelId = p.loadSoftBody(
    fileName=towel_path,
    basePosition=towelStartPos,
    baseOrientation=towelStartOrientation,
    scale=0.5,
    mass=0.5,
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

# 加载UR5机械臂
ur5_path = os.path.join(current_dir, "ur", "urdf", "ur5_robotiq_85.urdf")
ur5StartPos = [0, 0, 0.5]
ur5StartOrientation = p.getQuaternionFromEuler([0, 0, 0])
ur5Id = p.loadURDF(ur5_path, ur5StartPos, ur5StartOrientation)

# 模拟循环
for i in range(10000):
    p.stepSimulation()
    time.sleep(1./240.)

# 获取毛巾的位置和方向
towelPos, towelOrn = p.getBasePositionAndOrientation(towelId)

# 获取UR5机械臂的位置和方向
ur5Pos, ur5Orn = p.getBasePositionAndOrientation(ur5Id)

p.disconnect()
