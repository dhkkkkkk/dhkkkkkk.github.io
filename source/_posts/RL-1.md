---
title: 强化学习
date: 2025-11-05 16:28:49
tags: 深度学习
mathjax: true
---

# 强化学习概念

强化学习是一种解决控制任务（也称为决策问题）的框架，通过构建智能体，这些智能体通过与环境互动、试错并接收奖励（正面或负面）作为独特的反馈来从环境中学习。

## 强化学习框架

在下图中，以机器人为例，agent则为机器人的控制中心，environment则可以为机器人的各个关节电机

{% asset_img 1.png This is an image %} 

当控制中心（agent）得到关节电机（environment）0时刻的角度（state）时，会控制其进行转动（action），在转动过后，t1时刻会产生一个新的角度，如果该转角达到了预期，则会产生一个奖励（reward）

因此，强化学习的目标应该是**最大化累积奖励**，称为**<u>预期回报</u>**的最大化

### 状态和观测空间

* 状态（state）是对世界状态的完整描述，没有隐藏信息
* 观测（observation）是对状态的**部分描述**

### 行动空间

行动空间是环境中所有可能行动（action）的合集，分为离散和连续行动。

* 离散空间：可能的行动数量有限（例如游戏的移动只有上左下右）
* 连续空间：可能的行动数量无限（例如汽车的移动方向）

### 奖励与折扣

强化学习中的唯一反馈是累积奖励，而在较早时间步上奖励更有可能发生，为了表达对不同时间步奖励的关心程度，在积累奖励中引入折扣率gamma：
$$
R(\tau)=r_{t}+\gamma r_{t+1}+ \gamma^2 r_{t+2}+ \gamma^3r_{t+3}+...
$$
gamma大多数情况介于0.95与0.99之间，越大代表折扣越小，即**更关心长期奖励**，反之成立。

### 任务类型

* 情景式任务：存在起始与终结点的任务
* 持续式任务：没有终止状态的任务

### 探索与利用

* 利用：利用已知信息来最大化奖励
* 探索：通过随机行动探索环境，获取更多的信息

简单来说，探索就是风险更大的获取奖励的方式，相比利用，可能获得更大奖励，但也可能**获得更大惩罚（负奖励值）**，因此，必须定义一个有助于**权衡二者的规则**。

## 强化学习的目标

我们需要得到一个函数，该函数在得到当前环境状态（state）时，会给出最优的行动（action）

* 基于策略的方法：直接学习一个策略函数，该函数将定义**每个状态到最佳动作的直接映射（或概率分布）**
* 基于价值的方法：学习一个价值函数，该函数将每个状态映射到对应的一个预期价值，因此行动策略即“走向价值最高的状态”

# Gymnasium

Gymnasium是一个用于强化学习创建环境的库

Gymnasium 的核心是 `Env`，一个表示强化学习理论中马尔可夫决策过程（MDP）的高级 Python 类（注意：这不是一个完美的重构，缺少 MDP 的几个组件）。该类为用户提供了开始新情节、采取行动和可视化智能体当前状态的能力。

```python
import gymnasium as gym

env = gym.make('CartPole-v1') #通过调用make函数返回一个env类，此处为倒立摆环境
observation, info = env.reset() #初始化环境
```

上述代码创建了一个倒立摆模型，现在我们可以通过`env.step()`对环境执行动作（action）：

```python
while not episode_over:
    #随机动作，对于倒立摆来说，其动作空间只有左0或右1，也就是说其动作空间为离散空间 
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    episode_over = terminated or truncated
```

对于step的返回如下，对于每个环境，其返回的各元素具体内容都不同：

- `observation`：新状态 (st+1)，取决于环境，对于倒立摆，其为(4,)，分别为小车位置、速度、杆角度、角速度
- `reward`：执行动作后获得的奖励
- `terminated`：指奖励是否到达阈值
- `truncated`：指环境是否因超出边界而结束，对于倒立摆，杆角度过大、小车位置超出屏幕或时间到达上限都会结束
- `info`：一个提供额外信息的字典（取决于环境）。

## 月球车实践

首先创建对应环境，对于月球车，其奖励方案较为复杂，具体可查阅[月球着陆器 - Gymnasium 文档 - Gymnasium 文档](https://gymnasium.org.cn/environments/box2d/lunar_lander/)

```python
import gymnasium as gym

lunar = gym.make("LunarLander-v3")
lunar.reset()
print("Observation Space Shape", lunar.observation_space.shape) #环境形状
print("Sample observation", lunar.observation_space.sample())#随机取一个环境观测值
print("Action Space Shape", lunar.action_space) #动作空间形状
print("Action Space Sample", lunar.action_space.sample()) #随机取动作空间
```

现在我们已经有了环境、可执行的动作以及奖励，就差引入agent进行强化学习，这里我们引入stable_baselines3库中的PPO深度强化学习方法：

```python
model = sb3.PPO('MlpPolicy', lunar, verbose=1,device='cpu')
#PPO算法在cpu的表现上更好
model.learn(total_timesteps=1000000)
model.save("ppo")
```

对于强化学习，其总学习时长不再由epoch决定，有以下几个关键的训练参数：

* total_timesteps：指**整个训练过程中**模型一共会与环境交互多少个时间步

* n_step：每个rollout的长度，可以将其**理解为训练数据集的大小**，只不过在强化学习中数据集在不断更新，数据集更新次数为`total_timesteps\n_step`

  （注：当在rollout的过程中如果环境终止而时间步未达到`n_step`，则会立即reset环境继续采样）

* batchsize：每次**梯度更新**的数据长度，即将`n_step`切分为`n_step/batchsize`份，与常规深度学习同理

* n_epoch：对于每个数据集，其都会重复利用`n_epoch`次，但每次数据集都会被打乱

综上所述，深度强化学习的训练相比于一般深度学习，多了一个类似“更新数据集的操作”（即rollout），除此之外其他的操作是类似的

**<u>由于笔者暂时未接触其他强化学习算法，因此以上结论均只针对PPO</u>**

### 可视化

在训练完成后，我们可以通过动画观察模型训练效果：**（设置`render_mode="human"`）**

```python
import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("LunarLander-v3", render_mode="human")
model = PPO.load("ppo", env=env,device="cpu")

obs, info = env.reset()
total_reward = 0

for _ in range(1000):	#随便取的值，可能会执行多次episode
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    if terminated or truncated:
        obs, info = env.reset()
        print(f"episode reward:{total_reward}")
        total_reward = 0
env.close()
```

