---
title: ROS中的常用组件
date: 2024-04-18 15:50:30
tags: ROS
---

# 1.launch启动文件

## 基本元素

启动文件Launch File是ROS中一种同时启动多个节点的途径。可以自动启动ROS Master并实现多个节点的各种配置，为多个节点的操作提供了很大便利。

先来看一个简单的.launch文件：

```xml
 <launch>
    <!-- Turtlesim Node，注意注释的格式-->
    <node pkg="turtlesim" type="turtlesim_node" name="sim"/>

    <node pkg="turtlesim" type="turtle_teleop_key" name="teleop" output="screen"/>
    <!-- Axes -->
    <param name="scale_linear" value="2" type="double"/>
    <param name="scale_angular" value="2" type="double"/>

    <node pkg="learning_tf" type="turtle_tf_broadcaster"
          args="/turtle1" name="turtle1_tf_broadcaster" />
    <node pkg="learning_tf" type="turtle_tf_broadcaster"
          args="/turtle2" name="turtle2_tf_broadcaster" />

  </launch>
```

其中：

* `<launch>...</launch>`为XML语法中的根元素，XML语法要求每个文件必须包含一个根元素，文件中其他内容都必须包含在该标签内。

* `<node.../>`为启动ROS节点的标签元素，由上述定义规则可知，在launch文件中启动一个节点需要三个属性：

  * pkg：该节点所在**功能包**的名字
  * type：该节点的**可执行文件**名字，可执行文件是在为CMake_Lists中通过add_executable定义的
  * name：用于定义节点运行中的名字，**会覆盖掉init()中定义的节点名字**

  此外，还有可能用到：

  * output="screen"：将节点的标准输出打印到终端屏幕
  * respawn="ture"：当节点停止时会自动重启，默认false
  * require="ture"：必要节点，当该节点终止时launch中的其他节点也终止
  * ns="namespace"：为节点内的相对名称添加命名空间前缀
  * args="arguments"：节点需要的输入参数

## 参数设置

变量声明，由param和arg两种标签元素，它们的作用是不同的

* `<param.../>`：设置ros中运行的参数：

  ```xml
  <param name="scale_linear" value="2" type="double"/>
  ```

  作用显而易见：名为scale_linear的参数（parameter）的值被设置为了2，类型为double

* `<arg.../>`设置仅限于launch内部使用的局部变量

  ```xml
  <arg name="aca" default="value"/>
  ```

## 重映射机制（重要）

ROS提供一种重映射机制，便于将社区中其他人的功能包提供的接口进行重映射，而不是直接修改。

例如乌龟控制节点发布的控制话题可能是/turtlebot/cmd_vel，但是我们机器人订阅的话题是/cmd_vel，此时只需要重映射，我们的机器人就能收到话题消息了

```xml
<remap from="/turtlebot/cmd_vel" to="/com_vel"/>
```

# 2.TF变换

TF功能包使用树形数据结构，可以让用户跟踪多个坐标系，以时间为轴跟踪这些坐标系（默认10s内），并允许开发者请求如下数据：

* 5s前，机器人头部坐标系相对于全局坐标系是怎样的？
* 机器人夹取的物体相对于机器人中心坐标系的位置在哪？
* 机器人中心坐标系相对于全局坐标系的位置在哪？

想要使用TF功能包，总体来说要以下两个步骤：

* 监听TF变换，接收并缓存系统中发布的所有坐标数据，并查询需要的坐标变换关系
* 广播TF变换，向系统广播坐标变换关系

## 乌龟例程中的TF

该环境若出现问题可看[linux、ROS杂 | 小董的BLOG](https://dhkkk.gitee.io/2024/02/19/linux杂/)

- 准备库环境：`sudo apt-get install ros-noetic-turtle-tf`
- 启动launch文件进行实验：`roslaunch turtle_tf turtle_tf_dmeo.launch`
- 开启键盘控制：`rosrun turtlesim turtle_teleop_key`

可得到两只乌龟，一只会跟踪另一只

* `rosrun tf view_frames`后可在用户根目录下找到frames.pdf。可得当前系统中存在三个坐标系：world、turtle1、turtle2，其中world为TF树的根节点。
* `rosrun tf tf_echo turtle1 turtle2`可实时查看当前turtle2跟随turtle1所需的坐标变换

得到坐标变换后，便可以计算两乌龟间的距离和角度。

## 创建TF广播器

```cpp
#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include <turtlesim/Pose.h>

std::string turtle_name;

void poseCallback(const turtlesim::PoseConstPtr& msg)
{
    // tf广播器
    static tf::TransformBroadcaster br;

    // 根据乌龟当前的位姿，设置相对于世界坐标系的坐标变换
    tf::Transform transform;
    transform.setOrigin( tf::Vector3(msg->x, msg->y, 0.0) );//平移矩阵，因为为平面移动，第三项z为0
    tf::Quaternion q;
    q.setRPY(0, 0, msg->theta);//旋转矩阵，平面上只有yaw有值
    transform.setRotation(q);

    // 发布坐标变换
    br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "world", turtle_name));
}

int main(int argc, char** argv)
{
    // 初始化节点
    ros::init(argc, argv, "my_tf_broadcaster");
    if (argc != 2)//argc为输入字符串个数
    {
        ROS_ERROR("need turtle name as argument"); 
        return -1;
    };
    turtle_name = argv[1];//输入的参数从argv[1]开始存放

    // 订阅乌龟的pose信息
    ros::NodeHandle node;
    ros::Subscriber sub = node.subscribe(turtle_name+"/pose", 10, &poseCallback);

    ros::spin();

    return 0;
};

```

## 创建TF监听器

```cpp
#include <ros/ros.h>
#include <tf/transform_listener.h>
#include <geometry_msgs/Twist.h>
#include <turtlesim/Spawn.h>

int main(int argc, char** argv)
{
    // 初始化节点
    ros::init(argc, argv, "my_tf_listener");

    ros::NodeHandle node;

    // 通过服务调用，产生第二只乌龟turtle2
    ros::service::waitForService("spawn");
    ros::ServiceClient add_turtle =
    node.serviceClient<turtlesim::Spawn>("spawn");
    turtlesim::Spawn srv;
    add_turtle.call(srv);

    // 定义turtle2的速度控制发布器
    ros::Publisher turtle_vel =
    node.advertise<geometry_msgs::Twist>("turtle2/cmd_vel", 10);

    // tf监听器,创建后自动接收TF树的消息并缓存10s
    tf::TransformListener listener;

    ros::Rate rate(10.0);
    while (node.ok())
    {
        tf::StampedTransform transform;
        try
        {
            // 查找turtle2与turtle1的坐标变换
            listener.waitForTransform("/turtle2", "/turtle1", ros::Time(0), ros::Duration(3.0));//给定目标坐标系和源坐标系，等待两个坐标系之间一定周期下的变换关系，第四个参数为超时时间（该函数会产生堵塞）
            listener.lookupTransform("/turtle2", "/turtle1", ros::Time(0), transform);//给定目标坐标系和源坐标系，得到两个坐标系之间一定周期下的坐标变换（存入第四个参数中），ros::Time(0)表示想要的是最新的一次坐标变换
        }
        catch (tf::TransformException &ex) 
        {
            ROS_ERROR("%s",ex.what());
            ros::Duration(1.0).sleep();
            continue;
        }

        // 根据turtle1和turtle2之间的坐标变换，计算turtle2需要运动的线速度和角速度
        // 并发布速度控制指令，使turtle2向turtle1移动
        geometry_msgs::Twist vel_msg;
        vel_msg.angular.z = 4.0 * atan2(transform.getOrigin().y(),
                                        transform.getOrigin().x());
        vel_msg.linear.x = 0.5 * sqrt(pow(transform.getOrigin().x(), 2) +
                                      pow(transform.getOrigin().y(), 2));
        turtle_vel.publish(vel_msg);

        rate.sleep();
    }
    return 0;
};
```

## 通过launch文件运行这些节点

```xml
 <launch>
    <!-- 海龟仿真器 -->
    <node pkg="turtlesim" type="turtlesim_node" name="sim"/>

    <!-- 键盘控制 -->
    <node pkg="turtlesim" type="turtle_teleop_key" name="teleop" output="screen"/>

    <!-- 两只海龟的tf广播,用同一可执行文件创建两个广播节点 -->
    <node pkg="learning_tf" type="turtle_tf_broadcaster"
          args="/turtle1" name="turtle1_tf_broadcaster" />
    <node pkg="learning_tf" type="turtle_tf_broadcaster"
          args="/turtle2" name="turtle2_tf_broadcaster" />

    <!-- 监听tf广播，并且控制turtle2移动 -->
    <node pkg="learning_tf" type="turtle_tf_listener"
          name="listener" />

 </launch>
```

