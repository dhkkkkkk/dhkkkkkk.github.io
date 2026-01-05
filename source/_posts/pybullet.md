---
title: pybullet基础
date: 2025-11-11 10:19:08
tags: 机器人学
---

来源：[(PyBullet笔记（二）从hello world开始的杂谈（引擎连接，URDF模型加载，查看信息） - 知乎](https://zhuanlan.zhihu.com/p/347177629)，开源好人一生平安！

# 连接引擎

使用pybullet的第一件事就是连接物理引擎，整个pybullet的结构可以理解为客户端和服务端，客户端发送指令，服务端来执行。为了<u>让我们在客户端编写的脚本能够被解释</u>，并在物理引擎运行整个环境，需要使用pybullet的connect方法。

```python
import pybullet as p
import time
import pybullet_data

# 连接物理引擎
physicsCilent = p.connect(p.GUI)
```

connect函数接受一个参数，代表用户选择连接的物理引擎服务器（**physics server**），可选的有`pybullet.GUI`和`pybullet.DIRECT` ，返回一个数字代表服务器的ID。这两个物理引擎执行的内容，返回的结果等方面完全一致，唯一区别是，<u>GUI可以实时渲染场景到gui上，而DIRECT则不会且不允许用户调用内置的渲染器</u>，（即不进行画面渲染等可视化，在RL训练时，这些不必要操作会使训练变慢）也不允许用户调用外部的openGL，VR之类的硬件特征。

## 关闭服务器（引擎）

与gym操作相同，存在一个断开与服务器连接的函数：`p.disconnect()`

# 调试配置

## ui界面配置

在使用GUI引擎时，可以通过一些操作对渲染ui界面进行配置（如不显示界面的控件、禁用cpu核显渲染等）这些操作通常都通过`p.configureDebugVisualizer(..., 0 or 1)`实现，例如：

```python
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)
```

### 绘制辅助线

```python
froms = [[1, 1, 0], [-1, 1, 0], [-1, 1, 3], [1, 1, 3]]
tos = [[-1, 1, 0], [-1, 1, 3], [1, 1, 3], [1, 1, 0]]
for f, t in zip(froms, tos):
    p.addUserDebugLine(
        lineFromXYZ=f,
        lineToXYZ=t,
        lineColorRGB=[0, 1, 0],
        lineWidth=2
    )
```

### 添加文字

```pyhton
p.addUserDebugText(
    text="Destination",
    textPosition=[0, 1, 3],
    textColorRGB=[0, 1, 0],
    textSize=1.2,
)
```

### 添加控件

此处为在debug界面添加机器人某关节的运动控制，`p.addUserDebugParameter`会返回控件id，用`p.readUserDebugParameter`即可读取该id对应具体值

注：当rangeMin>rangeMax时，控件会从默认滑块变为按钮，按按钮的次数会放反映为**累加值**

```python
v_id = p.addUserDebugParameter(
    paramName="V",
    rangeMin=-50,
    rangeMax=50,
    startValue=0
)
f_id = p.addUserDebugParameter(
    paramName="F",
    rangeMin=-10,
    rangeMax=10,
    startValue=0
)

p.setJointMotorControl2(    bodyUniqueId=robot_id,
                            jointIndices=joint_id,
                            controlMode=p.VELOCITY_CONTROL,
                            targetVelocities=p.readUserDebugParameter(v_id),
                            forces=p.readUserDebugParameter(f_id))

```

### 添加合成摄像机视角

使用` p.getCameraImage`获取ui界面左侧三个窗口的视角，不具体展开了，用到了再说

### 移除debug配置

- removeAllUserParameters：移除所有的滑块和按钮类控件。
- removeUserDebugItem：接受一个代表**debug text**或**debug line**的ID，并移除该ID的debug text或者debug line
- removeAllUserDebugItems：移除所有的debug text和debug line

## 获取键盘、鼠标事件

* getKeyboardEvents：默认无输入即可，能够返回一个**字典**，字典中为当前时刻被按下去的按键的ID（key）以及它的状态（value）。其中，一般的按键的ID（key）就是它的小写字母Unicode码，而value则固定为三种状态：KEY_IS_DOWN, KEY_WAS_TRIGGERED 和 KEY_WAS_RELEASED。KEY_WAS_TRIGGERED 会在该按键刚刚被按下去后触发，并将按钮状态设为KEY_IS_DOWN；只要按键被一直按着，那么KEY_IS_DOWN就会一直触发；按钮松开，KEY_WAS_RELEASED会被触发。

  当无键盘事件时，该函数返回空字典（也就是说当有按键触发时，返回一个`“按键”:p.KEY_WAS_TRIGGERED`

  对于特殊按键的key，如下：

* getMouseEvents：获取鼠标事件，包括移动和点击，具体用到了再展开

| 说明                                                 | 按键ID常量                                                   |
| ---------------------------------------------------- | ------------------------------------------------------------ |
| F1到F12                                              | B3G_F1 … B3G_F12                                             |
| 上下左右方向键                                       | B3G_LEFT_ARROW, B3G_RIGHT_ARROW, B3G_UP_ARROW, B3G_DOWN_ARROW |
| 同一页向上/下，页尾，起始页                          | B3G_PAGE_UP, B3G_PAGE_DOWN, B3G_PAGE_END, B3G_HOME           |
| 删除，插入，Alt，Shift，Ctrl，Enter，Backspace，空格 | B3G_DELETE, B3G_INSERT, B3G_ALT, B3G_SHIFT, B3G_CONTROL, B3G_RETURN, B3G_BACKSPACE, B3G_SPACE |



# 加载模型

## 直接加载urdf模型

```python
# 设置环境重力加速度
p.setGravity(0, 0, -10)

# 加载URDF模型，此处是加载蓝白相间的陆地
planeId = p.loadURDF("plane.urdf")

# 加载机器人，并设置加载的机器人的位姿
startPos = [0, 0, 1]
startOrientation = p.getQuaternionFromEuler([0, 0, 0])
boxId = p.loadURDF("r2d2.urdf", startPos, startOrientation)
```

pybullet提供了非常方便的函数loadURDF来加载外部的urdf文件，返回值是**创建的模型对象的ID**（每个加载的模型在服务器中都使用唯一的ID），接受的参数有8个，只有第一个是必填参数（urdf文件绝对路径），第二个参数为机器人起始位置，第三个为起始姿态，对于姿态，可以调用`p.getQuaternionFromEuler`使用欧拉角定义其姿态

另外，也可以使用`p.resetBasePositionAndOrientation`重置位姿

## 通过3D文件创建模型

`createVisualShape`负责创建视觉模型，`createCollisionShape`负责创建碰撞箱模型，而`createMultiBody`则是负责将视觉模型和碰撞箱模型整合在一起形成一个完整的物理模型对象，并可以加入一些额外的参数，比如质量，转动惯量。

```python
# 创建过程中不渲染
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

# 创建视觉模型和碰撞箱模型时共用的两个参数
shift = [0, -0.02, 0]
scale = [1, 1, 1]

# 创建视觉形状
visual_shape_id = p.createVisualShape(
    shapeType=p.GEOM_MESH,
    fileName="duck.obj",
    rgbaColor=[1, 1, 1, 1],
    specularColor=[0.4, 0.4, 0],
    visualFramePosition=shift,
    meshScale=scale
)

#碰撞模型
collision_shape_id = p.createCollisionShape(
    shapeType=p.GEOM_MESH,
    fileName="duck_vhacd.obj",
    collisionFramePosition=shift,
    meshScale=scale
)

#使用createMultiBody将两者结合在一起
p.createMultiBody(
    baseMass=1,
    baseCollisionShapeIndex=collision_shape_id,
    baseVisualShapeIndex=visual_shape_id,
    basePosition=[0, 0, 2],
    useMaximalCoordinates=True
)

# 创建结束，重新开启渲染
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
```

对于创建的所有模型id，均可重复调用，因此可以多次使用`p.createMultiBody`创建多个一样的鸭子

# 开始模拟

## 步进模拟

```python
# 开始一千次迭代，也就是一千次交互，每次交互后停顿1/240
for i in range(1000):
    p.stepSimulation()
    time.sleep(1 / 240)
```

利用正向动力学进行步进模拟。由于计算中模拟物理过程还是离散得模拟的，因此，使用`stepSimulation`可以看成是进行一次迭代步。可以理解为gym中的`env.step(action)`，使用`time.sleep`可以方便观察

## 实时模拟

`setRealTimeSimulation`函数直接将物理引擎渲染的时间和RTC(real time clock)同步，这样做，就不需要使用`stepSimualtion`显式地执行模拟步了。引擎会根据RTC自动执行模拟步。这对于实时展示很有利:

```python
p.setRealTimeSimulation(1)
p.setTimeStep(1/240)
while 1:
    pass
```

# 查看机器人信息

## 位姿查看

```python
# 获取位置与方向四元数
cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
```

## 关节信息查看

```python
joint_num = p.getNumJoints(robot_id)
print("r2d2的节点数量为：", joint_num)

print("r2d2的信息：")
for joint_index in range(joint_num):
    info_tuple = p.getJointInfo(robot_id, joint_index)
    print(f"关节序号：{info_tuple[0]}\n\
            关节名称：{info_tuple[1]}\n\
            关节类型：{info_tuple[2]}\n\  #4表示该关节为固定关节
            机器人第一个位置的变量索引：{info_tuple[3]}\n\
            机器人第一个速度的变量索引：{info_tuple[4]}\n\
            保留参数：{info_tuple[5]}\n\
            关节的阻尼大小：{info_tuple[6]}\n\
            关节的摩擦系数：{info_tuple[7]}\n\
            slider和revolute(hinge)类型的位移最小值：{info_tuple[8]}\n\
            slider和revolute(hinge)类型的位移最大值：{info_tuple[9]}\n\
            关节驱动的最大值：{info_tuple[10]}\n\
            关节的最大速度：{info_tuple[11]}\n\
            节点名称：{info_tuple[12]}\n\
            局部框架中的关节轴系：{info_tuple[13]}\n\
            父节点frame的关节位置：{info_tuple[14]}\n\
            父节点frame的关节方向：{info_tuple[15]}\n\
            父节点的索引，若是基座返回-1：{info_tuple[16]}\n\n")
```

`p.getNumJoints`返回关节数量，`p.getJointInfo`返回指定关节的信息

# 控制关节电机

pybullet中控制机器人关节电机的API主要有两个：`setJointMotorControl2`与`setJointMotorControlArray`，这两个的用法基本一样，不同的是前者调用一次只能设置一台关节电机的参数，后者调用一次则可以设置一组关节电机的参数。

* `setJointMotorControl2`有三个必选参数:（`setJointMotorControlArray`将对应参数替换为列表）
  * 被控机器人id
  * 被控关节id（即关节序号索引info_tuple[0]）
  * 控制模式（可选POSITION_CONTROL, VELOCITY_CONTROL, TORQUE_CONTROL and PD_CONTROL）

```python
import pybullet as p
import time
import pybullet_data

# 连接物理引擎
physicsCilent = p.connect(p.GUI)

# 添加资源路径
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# 设置环境重力加速度
p.setGravity(0, 0, -10)

# 加载URDF模型，此处是加载蓝白相间的陆地
planeId = p.loadURDF("plane.urdf")
plane_Pos,_ = p.getBasePositionAndOrientation(planeId)
# 加载机器人，并设置加载的机器人的位姿
startPos = [0, 0, plane_Pos[2]+0.5]
startOrientation = p.getQuaternionFromEuler([0, 0, 0])
boxId = p.loadURDF("r2d2.urdf", startPos, startOrientation)

available_joints_indexes = [i for i in range(p.getNumJoints(boxId)) if p.getJointInfo(boxId, i)[2] != p.JOINT_FIXED]
wheel_joints_indexes = [i for i in available_joints_indexes if "wheel" in str(p.getJointInfo(boxId, i)[1])]

target_v = 10                   # 电机达到的预定角速度（rad/s）
max_force = 10                  # 电机能够提供的力，这个值决定了机器人运动时的加速度，想禁用电机给0即可


for i in range(1000):
    p.stepSimulation()
    p.setJointMotorControlArray(
        bodyUniqueId=boxId,
        jointIndices=wheel_joints_indexes,
        controlMode=p.VELOCITY_CONTROL,
        targetVelocities=[target_v for _ in wheel_joints_indexes],
        forces=[max_force for _ in wheel_joints_indexes]
    )
    time.sleep(1 / 240)         # 模拟器一秒模拟迭代240步


# 断开连接
p.disconnect()
```

需要注意的是，当**需要电机反转时，应该设定速度为负值，而力仍为正值**

# 相机跟踪

`p.resetDebugVisualizerCamera`通过设置相机欧拉角，和实时坐标，可以使其跟踪机器人一起运动：

```python
    location, _ = p.getBasePositionAndOrientation(boxId)
    p.resetDebugVisualizerCamera(
        cameraDistance=3,
        cameraYaw=110,
        cameraPitch=-30,
        cameraTargetPosition=location
    )
```

如果想实现更复杂的跟踪（如机器人第一视角），可以结合`p.getJointState(),p.getLinkState()`通过获取多种坐标实现

# 状态保存与加载

* saveState：将目前模拟器的状态保存到内存中，让这段程序后面可以随时读取内存中的这个模拟器状态，然后载入这个存档，因此saveState只需要指定模拟器环境ID，返回一个状态ID
* saveBullet：将状态保存到磁盘上，需要接受模拟器ID和路径
* restoreState：读取状态，上述两种都用此方法读，具体用法搜官方文档

# 碰撞检测

为了简化碰撞检测，通常使用规则的几何物体代替机器人实际碰撞模型用于检测碰撞，在pybullet中，则使用AABB包围盒作为一个碰撞检测的长方体。<u>AABB包围盒的各条边都与坐标轴平行，那么我们只需要选取两个位于体对角线上的点就可以确定这个长方体</u>，并且当一个点位于这两点之间时，则可以视作发生碰撞

* `getAABB`:默认返回基于世界坐标系的，指定机器人的AABB对角点（两个tuple）

* `getOverlappingObjects`:输入AABB对角点，返回与这个AABB模型发生碰撞的模型id和对应具体link的id

  注：其返回的是n个二元tuple，当未发生碰撞时，始终返回((1,-1),)，代表模型自己与自己重合，基于这个机制，可以通过以下方法判断是否碰撞：

  ```python
  while True:
      p.stepSimulation()
      P_min, P_max = p.getAABB(robot_id)
      id_tuple = p.getOverlappingObjects(P_min, P_max)
      if len(id_tuple) > 1:
          for ID, _ in id_tuple:
              if ID == robot_id:
                  continue
              else:
                  print(f"hit happen! hit object is {p.getBodyInfo(ID)}")
      sleep(1 / 240)
  ```

其余相关函数：

* `getContactPoints`:返回与一个物体接触的所有接触点
* `getClosestPoints`:返回两个物体距离最近的点
* `setCollisionFilterGroupMask`:忽略模型与模型之间碰撞
* `setCollisionFilterPair`:忽略关节之间碰撞



