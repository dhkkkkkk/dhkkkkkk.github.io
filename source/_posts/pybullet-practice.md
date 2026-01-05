---
title: 基于机械臂的pybullet实战
date: 2025-11-12 18:37:52
tags: 机器人学
---

本章使用pubullet完成常规机器人学相关计算，使用六自由度机械臂

# 配置与初始化

```python
import pybullet as p
import time
import pybullet_data
import numpy as np

physicsCilent = p.connect(p.DIRECT)

p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
planeId = p.loadURDF("plane.urdf")
robotId = p.loadURDF('E:/robo_prj/pybullet/examples/urdf/ur5.urdf',useFixedBase=True)  #记得固定基座
p.setGravity(0, 0, -9.8)
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

q0 = np.array([0, 0, 0, 0, 0, 0])
q1 = np.array([-1.5, -1.0, 1.0, -1.57, -1.57, -1.57])

controllable_joints = [i for i in range(p.getNumJoints(robotId)) if p.getJointInfo(robotId, i)[2] != p.JOINT_FIXED]

#################初始化角度##############################
zero_vec = [0.0] * len(controllable_joints)
kp = 1.0
kv = 1.0
p.setJointMotorControlArray(
    robotId,
    controllable_joints,
    p.POSITION_CONTROL,
    targetPositions=q1,
    targetVelocities=zero_vec,
    positionGains=[kp] * len(controllable_joints),
    velocityGains=[kv] * len(controllable_joints)
)
for _ in range(100):  # to settle the robot to its position
    p.stepSimulation()

joint_state = p.getJointStates(robotId,controllable_joints)
joint_theta = [i[0] for i in joint_state]
joint_velocity = [i[1] for i in joint_state]
joint_torque = [i[3] for i in joint_state]
#########################################################
```

该部分代码完成了模型加载、重力设置、遍历可运动关节、设置当前角度为q1并计算关节角及其一阶二阶导

* `controllable_joints = [i for i in range(p.getNumJoints(robotId)) if p.getJointInfo(robotId, i)[2] != p.JOINT_FIXED]`是常用的遍历可运动关节的操作
* 对于当前情况，机器人是静止的，因此关节速度和加速度都为0

# 运动学

```python
###############运动学正解#################################
link_state = p.getLinkState(robotId, controllable_joints[-1])
link_pos = link_state[0]
link_orn = link_state[1] #笛卡尔坐标姿态
link_R = np.array(p.getMatrixFromQuaternion(link_orn)).reshape(3, 3)
link_T = np.eye(4) #齐次变换矩阵
link_T[:3, :3] = link_R
link_T[:3, 3] = link_pos
print(f"q1下末端质心齐次矩阵为：{link_T}")
########################################################


###############运动学逆解#################################
joint_angle_solve = p.calculateInverseKinematics(robotId,
                                                 controllable_joints[-1],
                                                 targetPosition = link_pos,
                                                 targetOrientation = link_orn)
print(f"q1下运动学逆解为：{joint_angle_solve}")
########################################################
```

* `p.getLinkState`会返回所查询关节**对应子连杆**的位姿等信息

  在pybullet中，

  ```
  base → joint0 → link1 → joint1 → link2 → joint2 → link3
  ```

  对于joint2，其子连杆为link3，父连杆为link2

* 默认的姿态表示是基于笛卡尔坐标系的，pybullet提供其与矩阵、欧拉坐标的转换api

# 雅可比矩阵相关

```python
#################雅可比计算##############################
J_v, J_w = p.calculateJacobian(robotId,
                               controllable_joints[-1],
                               link_pos,
                               joint_theta,
                               zero_vec, zero_vec)
J = np.concatenate((np.asarray(J_v),np.asarray(J_w)),axis=0)
print(f"q1下末端的雅可比矩阵为{J}")
#######################################################
```

pybullet提供的雅可比函数默认分开返回线速度雅可比和角速度雅可比，需要输入所计算连杆的位置和所有关节的转角，关节速度和加速度默认0即可（形状需与机器人自由度DOF匹配）

# 机械臂控制(逆动力学)

大致流程：

* 规划轨迹，得到q，qd，qdd
* 逆动力学计算（pybullet使用牛顿欧拉法）
* 控制电机

首先定义如下轨迹规划函数：

```python
def getTrajectory(thi, thf, tf, dt):
    desired_position, desired_velocity, desired_acceleration = [], [], []
    t = 0
    while t <= tf:
        th = thi + ((thf - thi) / tf) * (t - (tf / (2 * np.pi)) * np.sin((2 * np.pi / tf) * t))
        dth = ((thf - thi) / tf) * (1 - np.cos((2 * np.pi / tf) * t))
        ddth = (2 * np.pi * (thf - thi) / (tf * tf)) * np.sin((2 * np.pi / tf) * t)
        desired_position.append(th)
        desired_velocity.append(dth)
        desired_acceleration.append(ddth)
        t += dt
    desired_position = np.array(desired_position)
    desired_velocity = np.array(desired_velocity)
    desired_acceleration = np.array(desired_acceleration)
    return desired_position, desired_velocity, desired_acceleration
```

该函数根据始末关节角规划n个关节空间点

## 位置控制

```python
sim_time = 2
dt = 1e-3
q,qd,qdd = getTrajectory(q0,q1,tf=sim_time,dt=dt)

#不设置阻尼
for link_idx in range(9):
    p.changeDynamics(robotId, link_idx, linearDamping=0.0, angularDamping=0.0, 								jointDamping=0.0)
    p.changeDynamics(robotId, link_idx, maxJointVelocity=200)
    
    
n = 0
kp = 1
kv = 1
while n < q.shape[0]:
	#位置控制
    p.setJointMotorControlArray(robotId,controllable_joints,p.POSITION_CONTROL,
                                targetPositions = list(q[n]),
                                targetVelocities = list(qd[n]),
                                positionGains=[kp] * len(controllable_joints),
                                velocityGains=[kv] * len(controllable_joints)
                                )
    p.stepSimulation()
    print(tau)
    time.sleep(dt)
    n += 1
p.disconnect()
```

查阅官方手册可知，输入模式为`p.POSITION_CONTROL`时，误差函数如图：

{% asset_img 1.png This is an image %} 

因此，为达到最佳控制效果，我们最好额外输入角速度项，也就是qd

## 速度控制

只需将控制电机函数改为：

```python
    p.setJointMotorControlArray(robotId,controllable_joints,p.VELOCITY_CONTROL,
                                targetVelocities = list(qd[n]),
                                forces = [200]*len(controllable_joints),
                                )
```

由误差函数可知，单速度项控制非常简单，因此其效果也不好，一般在机械臂中不使用

## 力控制

### 简单版

```python
p.setPhysicsEngineParameter(fixedTimeStep=dt, numSolverIterations=100, numSubSteps=10)
#重要，此处fixedTimeStep应与getTrajectory和p.setTimeStep(dt)一致！！！！
...
...
...
#初始化力控制
p.setJointMotorControlArray(robotId,controllable_joints,p.VELOCITY_CONTROL,
                            forces = [0]*len(controllable_joints))
n = 0
while n < q.shape[0]:

    tau = p.calculateInverseDynamics(robotId,list(q[n]),list(qd[n]),list(qdd[n]))
    p.setJointMotorControlArray(robotId,controllable_joints,p.TORQUE_CONTROL,
                                forces = tau,
                                )
    p.stepSimulation()
    print(tau)
    time.sleep(dt)
    n += 1
p.disconnect()
```

如果在此之前使用过位置控制，需要先进行一次初始化力控制（不知道为什么），然后就是一定要注意在创建环境时使用如下代码保证物理引擎计算稳定性：

```python
p.setTimeStep(dt)
p.setPhysicsEngineParameter(fixedTimeStep=dt, numSolverIterations=100, numSubSteps=10)
```

### 带反馈版

```python
while n < q.shape[0]:
    joint_states = p.getJointStates(robotId,controllable_joints)
    q_actual = np.array([state[0] for state in joint_states])
    qd_actual = np.array([state[1] for state in joint_states])


    q_e = q[n] - q_actual
    qd_e = qd[n] - qd_actual

    aq = qdd[n] + 400 * q_e + 40 * qd_e

    tau = p.calculateInverseDynamics(robotId,list(q_actual),list(qd_actual),list(aq))
    p.setJointMotorControlArray(robotId,controllable_joints,p.TORQUE_CONTROL,
                                forces = tau,
                                )
    p.stepSimulation()
    print(tau)
    time.sleep(dt)
    n += 1
```

实际上，对于逆动力学或者`p.calculateInverseDynamics`函数来说，我们在构建动力学方程时，都应该使用系统**当前**位置、**当前**速度和**期望**加速度，因为在机器人控制系统中，**<u>加速度项才是真正的唯一前馈项</u>**，而要构建动力学方程中的M，C，G矩阵则需要**当前**系统的位置和速度（即q_actual，qd_actual决定动力学方程内容，aq根据当前动力学方程输出**在我们期望的加速度下对应的关节力**，以车为例，即我们要控制车速，最终只能通过改变加速度来控制，而要达到期望加速度，我们就需要构建动力学方程将期望加速度映射到电机输出力上）

因此我们若要引入反馈控制，也应该**仅对**期望加速度项进行调整，因此真正的动力学方程才是：

```python
tau = p.calculateInverseDynamics(robotId,list(q_actual),list(qd_actual),list(aq))
```

