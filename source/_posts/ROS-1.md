---
title: ROS的一些基础概念与配置
date: 2024-04-10 15:17:26
tags: ROS
---

# 1.关键概念

## 节点Node

执行运算任务的进程，一个系统一般由多个节点组成。

## 消息Message

节点之间的通信机制就是基于发布/订阅模型的消息通信，消息有多种数据结构。

## 话题Topic

消息的传递方式。

一个节点可以针对一个给定的话题发布消息，也可以关注某个话题并订阅特定类型的数据。

## 服务Service

双向同步传输模式，两节点一个用于请求，一个用于应答。ROS只允许一个节点提供指定命名的服务。

## 节点管理器ROS Master

管理节点

# 2.文件系统

## 功能包

ROS软件的基本单元，包含节点、库、配置文件等

### 对应文件夹内容

* config：功能包的配置文件

* include：功能包需要用到的头文件

* scripts：可以直接运行的py脚本

* src：需要编译的cpp代码

* launch：所有启动文件

* msg：功能包自定义的消息类型

* srv：自定义的服务类型

* action：自定义的动作指令

* CMakeLists.txt：编译器编一功能包的规则

* package.xml：功能包清单，包含该功能包名称、版本号等信息。

  <build_depend>定义了代码编译所依赖的其他功能包

  <run_depend>定义了功能包中可执行程序运行时所依赖的其他功能包

```
针对功能包的常用命令：
catkin_create_pkg 创建功能包
rospack 获取功能包信息
catkin_make 编译工作空间中的功能包
rosdep 自动安装功能包依赖的其他包
roscd 功能包目录跳转
roscp 拷贝功能包中的文件
rosed 编辑功能包中的文件
rosrun 运行功能包中的可执行文件
roslaunch 运行启动文件
```

## 元功能包

只包含一个package.xml，将多个功能包整合成一个逻辑上的独立功能包

与功能包中的类似，需额外包含一个

```
<export>
	<metapackage/>
</export>
```

# 3.ROS的通信机制

## 话题通信

* Talker/Listener注册
* ROS Master进行信息匹配，根据Listener的订阅信息从注册列表找Talker，没找到则等
* Listener发送连接请求
* Talker确认连接请求
* Listener尝试与Talker建立网络连接
* Talker向Listener发布数据

## 服务通信

与话题相比减少了RPC通信，即匹配后直接进行网络连接

服务是一种带应答的通信，最后一步为Talker接收到Listener的请求和参数后开始执行服务功能，完成后Talker发送应答数据。

## 区别

* 异步；同步
* 无反馈；有反馈
* 有缓冲区；无缓冲区
* 多对多；一对一

话题适用于不断更新的数据通信；服务适用于逻辑处理复杂的数据同步交换

# 4.小乌龟仿真

* roscore为运行ROS Master

* rosrun ...   ...    **启动...功能包中的...节点**

  ```
  rosrun turtlesim ...	启动turtlesim功能包中的某个节点
  rosrun turtlesim 	turtlesim_node	启动turtlesim仿真器节点
  rosrun turtlesim turtle_teleop_key	运行键盘控制节点
  ```

# 5.创建工作空间和功能包

工作空间是存放工程开发相关文件的文件夹，现默认使用Catkin编译系统。

一个典型的工作空间包含以下目录空间：

* src：代码空间，储存所有ROS功能包的源码
* build：编译空间，用于存储工作空间编译过程中产生的缓存信息和中间文件
* devel：开发空间，放置编译生成的可执行文件
* install：安装空间，编译成功后，可以使用make install命令将可执行文件安装到当前工作空间。运行该空间中的环境变量脚本即可在终端中运行这些可执行文件。该空间非必要。

## 创建工作空间

```
mkdir catkin_ws/src
进入src后
catkin_init_workspace	创建工作空间
cd ..	回到工作空间
catkin_make	编译
```

编译成功后，自动生成build和devel。devel中生成几个setup.*sh形式的环境变量设置脚本，可使用source运行。

运行后该工作空间环境变量生效。可在工作空间外使用？

```
source devel/setup.bash
该命令设置的环境变量只能在当前终端中剩下
```

{% asset_img 1.jpg This is an image %} 

## 创建功能包

```
catkin_create_pkg <package_name> [depend1] [depend2]...
```

* <package_name>为功能包名字
* depend为当前创建的功能包编译所依赖的其他功能包c'd
* ROS不允许功能包嵌套，所有功能包**平行放置在<u>src</u>中**
* 任何添加操作完后都应回到根目录source添加其环境变量

## 工作空间覆盖

ROS允许多个工作空间并存，当遇到工作空间中**名字相同**（内容不一定相同）的功能包时，新设置的路径会自动放到最前端，在运行时，ROS也会优先查找最前端的工作空间是否存在指定的功能包。

```
rospack find ... 查找...功能包的位置
/opt/ros为ROS的默认工作空间
```

如果一个工作空间下的b功能包**依赖**同空间的a功能包，而a功能包又被另一工作空间下的新a功能包覆盖，**该新a功能包名字与a功能包相同，但内容可能不同，**因此可能导致b功能包存在潜在风险。

# 6.vocode配置ROS环境

## 1打开工作空间

在工作空间的根目录下输入：

```
code .
```

因为安装了ROS插件，VScode会直接识别catkin环境，并且自动生成.vscode文件夹，里面保含c_cpp_properties.json、settings.json 两个文件。

## 2创建功能包

在vscode资源管理中右键src选择create catkin package

## 3配置相关文件

1.在.vscoce下的task.json（记得加逗号和引号）

```
			"args": [
				"--directory",
				"/home/dhk/catkin_ws",
				"-DCMAKE_BUILD_TYPE=RelWithDebInfo", 
				"-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
			],
```

2.在c_cpp_properties.json中添加 （记得逗号和引号）

"compileCommands": "${workspaceFolder}/build/compile_commands.json" 

```
 "configurations": [
    {
      "browse": {
        "databaseFilename": "${default}",
        "limitSymbolsToIncludedHeaders": false
      },
      "includePath": [
        "/opt/ros/noetic/include/**",
        "/home/dhk/catkin_ws/src/learning_communication/include/**",
        "/usr/include/**"
      ],
      "name": "ROS",
      "intelliSenseMode": "gcc-x64",
      "compilerPath": "/usr/bin/gcc",
      "cStandard": "gnu11",
      "cppStandard": "c++14",
      "compileCommands": "${workspaceFolder}/build/compile_commands.json"   
    }
```

**若后续对节点提示找不到ros头文件且确实无法编译，应该与这两项有关**

## 4编写测试节点

在新功能包的src中创建helloworld.cpp，编写如下

```cpp
#include "ros/ros.h"

int main(int argc, char *argv[])
{
    //执行 ros 节点初始化
    ros::init(argc,argv,"hello");//节点名为hello
    //创建 ros 节点句柄(非必须)
    ros::NodeHandle n;
    //控制台输出 hello world
    ROS_INFO("hello world!");

    return 0;
}
```

在**<u>新功能包</u>**中的的CmakeLsit.txt添加：

```
add_executable(hello src/helloworld.cpp)
target_link_libraries(hello ${catkin_LIBRARIES})


其中
add_executable(节点名 src/文件名)
target_link_libraries(节点名  ${catkin_LIBRARIES})
```

vscode编译，效果同catkin_make

```
执行快捷键：ctrl+shift+b
```

## 5运行ROS Master

建议还是用roscore

vscode中 c+s+p

ros:start和ros:stop对应开关，但是运行后没有提示？

## 6运行节点

执行快捷键ctrl + shfit + p输入ROS：Run a Ros executable， 依次输入你创建的功能包的名称以及节点名称（即编译成功后二进制文件的名称）

## 问题

当工作空间可以正常编译但却找不到功能包中的某个节点时：

将第4节中的

```
add_executable(hello src/helloworld.cpp)
target_link_libraries(hello ${catkin_LIBRARIES})
```

一定要放在该功能包中的CmakeLsit.txt的**<u>末尾处！！</u>**

## 快捷键

ctrl+shift+p：调出用于执行命令的输入框
ctrl+shift+b：编译