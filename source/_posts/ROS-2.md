---
title: ROS的话题与服务
date: 2024-04-15 15:01:43
tags: ROS
---

# 1.话题中的发布者与订阅者

* 使用`rqt_graph`可以查看当前的节点关系图，如图为乌龟例程的键盘输入控制节点图

  {% asset_img 1.jpg This is an image %} 

  其中teleop_turtle节点创建了一个发布者，turtlesim节点创建了一个订阅者；一个发布键盘控制的命令，一个订阅命令实现🐢的移动，此时的话题是/turtlel/cmd_vel。

## 创建Publisher（发布者）

```cpp
 #include <sstream>
#include "ros/ros.h"
#include "std_msgs/String.h"

int main(int argc, char **argv)
{
    // ROS节点初始化，一个cpp对应一个节点，节点名为talker，在当前运行的ROS中独一无二
    ros::init(argc, argv, "talker");

    // 创建节点句柄，方便对talker节点的使用
    ros::NodeHandle n;

 /* 创建一个Publisher，发布名为chatter（chatter才是话题名字！！）的话题，消息类型为std_msgs::String
ros::声明命名空间，chatter_pub为话题变量（？自己的理解），其发布的内容为String类型消息，1000为队列大小，advertise类似cpp容器之类的东西？*/
   
    ros::Publisher chatter_pub = n.advertise<std_msgs::String>("chatter", 1000);
	
    // 设置循环的频率，单位Hz
    ros::Rate loop_rate(10);

    int count = 0;
    while (ros::ok())//节点未发送异常则持续循环
    {
        // 初始化std_msgs::String类型的消息
        std_msgs::String msg;
        std::stringstream ss;
        ss << "hello world " << count;
        msg.data = ss.str();//将流的内容全部返回到msg的data中，std_msgs::String对象只有data这一个成员

        // 
        ROS_INFO("%s", msg.data.c_str());//打印内容，只是打印而已
        chatter_pub.publish(msg);//发布消息，发布后Master会找订阅该话题的节点

        // 循环等待回调函数
        ros::spinOnce();

        // 按照循环频率延时
        loop_rate.sleep();//节点休眠，时长与前设置的循环频率有关
        ++count;
    }

    return 0;
}
```

## 创建Subscriber（订阅者）

```cpp
/**
 * 该例程将订阅chatter话题，消息类型String
 */
 
#include "ros/ros.h"
#include "std_msgs/String.h"

// 接收到订阅的消息后，会进入消息回调函数，传入的参数为一个消息指针（记住吧，感觉形式怪怪的）
void chatterCallback(const std_msgs::String::ConstPtr& msg)
{
    // 将接收到的消息打印出来
    ROS_INFO("I heard: [%s]", msg->data.c_str());//注意形式
}

int main(int argc, char **argv)
{
    // 初始化ROS节点
    ros::init(argc, argv, "listener");

    // 创建节点句柄
    ros::NodeHandle n;

    // 创建一个Subscriber，订阅名为chatter的topic，注册回调函数chatterCallback，注意这里要注册回调函数，消息类型与订阅者无关
    ros::Subscriber sub = n.subscribe("chatter", 1000, chatterCallback);

    // 循环等待回调函数
    ros::spin();

    return 0;
}
```

然后在该**功能包**中的CmakeLsit.txt中：

```
add_executable(talker src/talker.cpp)
target_link_libraries(talker ${catkin_LIBRARIES})
##add_dependencies(talker ${PROJECT_NAME}_generate_messages_cpp)不需

add_executable(listener src/talker.cpp)
target_link_libraries(listener ${catkin_LIBRARIES})
##add_dependencies(listener ${PROJECT_NAME}_generate_messages_cpp)不需
```

其中：

* `add_ex...`：为设置需编译的代码和可执行文件。第一个参数为期望生成可执行文件的**名字**，一般与节点名相同，方便使用；第二个参数为要编译的文件。
* `target...`：设置链接库。第一个参数为需链接的可执行文件名（同上的名字）；第二个为要链接的库
* `add_de...`：设置依赖。为可执行文件添加能动态产生消息代码的依赖。**但是先版本好像已不需要添加这个**

## 自定义消息类型

前两节中使用的消息类型为ROS元功能包定义的std_msgs（标准数据类型）中预定义的String类型，除此之外，用户可以自定义msg文件，使用自定义的类型，流程如下：

### 1.编写msg文件

在本功能包中创建msg文件夹（与src平行），并创建person.msg

```
string name
uint8 sex
uint8 age

uint8 unknown=0
uint8 male=1
uint8 female=2
```

### 2.修改本功能包下的package.xml

添加：

```
 <build_depend>message_generation</build_depend>
 <exec_depend>message_runtime</exec_depend>
```

保证msg文件能转化为cpp,py等语言的源文件，第一行为编译依赖；第二行为执行依赖

### 3.修改本功能包下的**CMakeLists.txt**

* 在**<u>已有</u>**的find_package中：

  ```
  find_package(catkin REQUIRED COMPONENTS
     roscpp
     rospy
     std_msgs
     message_generation
  )
  ```

* 在**<u>已有</u>**的catkin_package中增加：

  ```
  catkin_package(
    ...
  	CATKIN_DEPENDS roscpp rospy std_msgs message_runtime
    ...)
  ```

* 找到被注释或已有的add_message_files()语句：

  ```
  add_message_files(
    FILES
    person.msg
   )
  ```

* 找到被注释或已有的genreate_message()语句：

  ```
  generate_messages(
  DEPENDENCIES
  std_msgs
  )
  ```

### 3.编译

编译成功后，对于C++而言，编译器帮我们自动编写一个头文件：`sensor.h`，文件位于`workspace/devel/include`中，通过引用头文件就可以使用这个自定义数据了。

* 引用格式：`#include "learning_communication/person.h"`/前为该msg文件（不是.h文件）所在的功能包的名字

* 使用格式：`learning_communication::person`

  ```cpp
  learning_communication::person msg;
  std::stringstream ss;
  ss << "dhk";
  msg.name=ss.str();
  ```

## 将自定义消息用于刚刚的话题中

### 发布者

```cpp
#include <sstream>
#include "ros/ros.h"
#include "std_msgs/String.h"
#include "learning_communication/person.h"
int main(int argc, char **argv)
{
    // ROS节点初始化
    ros::init(argc, argv, "talker");

    // 创建节点句柄
    ros::NodeHandle n;

    // 创建一个Publisher，发布名为chatter的topic，消息类型为自定义消息类型
    ros::Publisher chatter_pub = n.advertise<learning_communication::person>("chatter", 1000);

    // 设置循环的频率
    ros::Rate loop_rate(10);

    int count = 0;
    while (ros::ok())
    {
        // 初始化std_msgs::String类型的消息
        learning_communication::person msg;
        std::stringstream ss;
        ss << "dhk" << count;
        msg.name = ss.str();

        // 发布消息
        ROS_INFO("%s", msg.name.c_str());//c_str()通用于string类型
        chatter_pub.publish(msg);

        // 循环等待回调函数
        ros::spinOnce();

        // 按照循环频率延时
        loop_rate.sleep();
        ++count;
    }

    return 0;
}
```

### 订阅者

```cpp
#include "ros/ros.h"
#include "std_msgs/String.h"
#include "learning_communication/person.h"
// 接收到订阅的消息后，会进入消息回调函数
//ConstPtr&为通用类型，直接cv就行
void chatterCallback(const learning_communication::person::ConstPtr& msg)
{
    // 将接收到的消息打印出来
    ROS_INFO("I heard: [%s]", msg->name.c_str());
}

int main(int argc, char **argv)
{
    // 初始化ROS节点
    ros::init(argc, argv, "listener");

    // 创建节点句柄
    ros::NodeHandle n;

    // 创建一个Subscriber，订阅名为chatter的topic，注册回调函数chatterCallback
    ros::Subscriber sub = n.subscribe("chatter", 1000, chatterCallback);

    // 循环等待回调函数
    ros::spin();

    return 0;
}
```

{% asset_img 2.jpg This is an image %} 

如图，talker节点发布了名为chatter的话题，被listener节点订阅。

# 2.服务中的服务端与客户端

服务（service）为节点之间通信的一种方式，由客户端（Client）发布请求（Request），服务端（Server）处理后返回应答（Response）

## 自定义服务数据

### 1.编写srv文件

同话题中的自定义消息一样，服务数据可以通过**srv文件**进行定义，且同msg文件一样放置在具体功能包文件夹中，但由于文件包含**请求**与**应答两个数据域**，因此要特别分割一下：

```
int64 a
int64 b
---
int64 sum
```

上为请求域，下为应答域

### 2.修改package.xml

内容同自定义订阅消息中的操作

```
 <build_depend>message_generation</build_depend>
 <exec_depend>message_runtime</exec_depend>
```

### 3.修改CMakeLists.txt

找到被注释或已有的add_service_files()语句：

```
add_service_files(
  FILES
  AddTwoInts.srv
)
```

其他操作同自定义订阅消息

## 创建Server

```cpp
#include "ros/ros.h"
#include "learning_communication/AddTwoInts.h"//记得添加头文件

// service回调函数，第一参数为请求域req，第二参数为应答域res，不同域要分别声明
bool add(learning_communication::AddTwoInts::Request  &req,
         learning_communication::AddTwoInts::Response &res)
{
    // 将输入参数中的请求数据相加，结果放到应答变量中
    res.sum = req.a + req.b;
    ROS_INFO("request: x=%ld, y=%ld", (long int)req.a, (long int)req.b);
    ROS_INFO("sending back response: [%ld]", (long int)res.sum);

    return true;
}

int main(int argc, char **argv)
{
    // ROS节点初始化
    ros::init(argc, argv, "add_two_ints_server");

    // 创建节点句柄
    ros::NodeHandle n;

    // 创建一个名为add_two_ints的server，注册回调函数add()
    ros::ServiceServer service = n.advertiseService("add_two_ints", add);

    // 等待回调函数
    ROS_INFO("Ready to add two ints.");
    ros::spin();

    return 0;
}

```

## 创建Client

```c++
#include <cstdlib>
#include "ros/ros.h"
#include "learning_communication/AddTwoInts.h"

int main(int argc, char **argv)
{
    // ROS节点初始化
    ros::init(argc, argv, "add_two_ints_client");

    // 从终端命令行获取两个加数，为main的输入参数
    if (argc != 3)
    {
        ROS_INFO("usage: add_two_ints_client X Y");
        return 1;
    }

    // 创建节点句柄
    ros::NodeHandle n;

    // 创建一个client，请求add_two_int的service，两文件在此处联系
    // service消息类型是learning_communication::AddTwoInts
    ros::ServiceClient client = n.serviceClient<learning_communication::AddTwoInts>("add_two_ints");

    // 创建learning_communication::AddTwoInts类型的service消息
    learning_communication::AddTwoInts srv;
    srv.request.a = atoll(argv[1]);
    srv.request.b = atoll(argv[2]);

    // 发布service请求，等待加法运算的应答结果
   //这里输入参数为1个，但是回调是两个，回调的写法应该是固定格式？
    if (client.call(srv))
    {
        ROS_INFO("Sum: %ld", (long int)srv.response.sum);
    }
    else
    {
        ROS_ERROR("Failed to call service add_two_ints");
        return 1;
    }

    return 0;
}

```

## 编译功能包

在CMakeLists.txt中添加相关内容

```
add_executable(add_two_ints_server src/server.cpp)
target_link_libraries(add_two_ints_server ${catkin_LIBRARIES})

add_executable(add_two_ints_client src/client.cpp)
target_link_libraries(add_two_ints_client ${catkin_LIBRARIES})
```

这里要注意add_executable第一个参数为期望的可执行文件名，可以不与节点名相同，但我喜欢相同（哼）

## 运行

{% asset_img 3.jpg This is an image %} 

这里要注意

* 运行节点时不要运行成cpp文件
* Client节点需添加2个初始参数

# 话题与服务的区别

* 话题中建立两节点关系的阶段是在listener中订阅相关话题并注册回调函数；而服务中建立两节点关系的阶段是在client中请求名为xx的服务。因此顺序应该为：
  * 话题：先编写talker，再写listener
  * 服务：先编写server，再写client
* 两种通信的具体顺序：
* 话题
  * 创建发布者，话题，并规定话题的消息形式
  * 发布消息
  * 创建订阅者，并订阅话题，注册回调函数，等待回调函数运行
* 服务
  * 创建服务端，服务名，并注册服务回调函数
  * 创建客户端，并请求相关服务名的服务，此时规定服务的消息形式
  * 客户端请求服务并输入相关内容，服务端回调函数进行处理时客户端堵塞，等待服务完成，服务完成后客户端内与回调函数相关的参数已改变
* **需要注意的是，话题的回调函数只在订阅文件生效，不是反馈；而服务的回调函数会生效于客户端，是真正的反馈**

# ROS命名规则

* 基础名称：dhk
* 全局名称：/xx/dhk
* 相对名称：xx/dhk
* 私有名称：~xx/dhk

起始为/的都为全局名称