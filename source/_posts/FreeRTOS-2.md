---
title: FreeRTOS的任务相关
date: 2022-08-23 18:16:08
tags: 嵌入式学习
---

# 任务相关

## 1、RTOS调度器

在任何时刻，只有一个任务得到运行，RTOS调度器决定运行哪个任务。调度器会不断的启动、停止每一个任务，宏观看上去就像整个应用程序都在执行。作为任务，不需要对调度器的活动有所了解，在任务切入切出时保存上下文环境（寄存器值、堆栈内容）是调度器主要的职责。为了实现这点，每个任务都需要有自己的堆栈。当任务切出时，它的执行环境会被保存在该任务的堆栈中，这样当再次运行时，就能从堆栈中正确的恢复上次的运行环境。

## 2、任务状态

{% asset_img 任务.jpg This is an image %} 

## 3、任务例子（用户）

```c
void vATaskFunction( void *pvParameters )
{
    while(1)
    {
        /*-- 应用程序代码放在这里. --*/
    }
 
    /* 任务不可以从这个函数返回或退出。在较新的FreeRTOS移植包中，如果
    试图从一个任务中返回，将会调用configASSERT()（如果定义的话）。
    如果一个任务确实要退出函数，那么这个任务应调用vTaskDelete(NULL)
    函数，以便处理一些清理工作。*/
    vTaskDelete( NULL );
}
```

例如步兵v2.6中底盘任务

```c
void chassis_task(void *pvParameters)
{
    //空闲一段时间
    vTaskDelay(CHASSIS_TASK_INIT_TIME);
    //底盘初始化
    chassis_init(&chassis_move);
	//判断遥控器在线状态
    while (toe_is_error(DBUSTOE))
	{
      vTaskDelay(CHASSIS_CONTROL_TIME_MS);
	}
	//底盘循环代码
    while (1)
    {
	  //底盘模式设置
      chassis_set_mode(&chassis_move);
      //状态切换数据保存与处理
      chassis_mode_change_control_transit(&chassis_move);
      //底盘相关反馈数据更新
      chassis_feedback_update(&chassis_move);
        。。。。。。。。
	  //系统延时
	  vTaskDelay(CHASSIS_CONTROL_TIME_MS);
	}
}	
```

* 任务函数决不应该返回，因此通常任务函数都是一个死循环。

* 任务由xTaskCreate()函数创建，由vTaskDelete()函数删除。（后面会详细讲到）

## 4、空闲任务

 空闲任务是启动RTOS调度器时由内核自动创建的任务，这样可以确保至少有一个任务在运行。空闲任务具有最低任务优先级，这样如果有其它更高优先级的任务进入就绪态就可以立刻让出CPU。

### 空闲任务钩子函数

空闲任务钩子是一个函数，每一个空闲任务周期被调用一次。

* **因为FreeRTOS必须至少有一个任务处于就绪或运行状态，因此钩子函数不可以调用可能引起空闲任务阻塞的API函数（比如vTaskDelay()或者带有超时事件的队列或信号量函数）。**

创建一个空闲钩子步骤如下：

-  在FreeRTOSConfig.h头文件中设置configUSE_IDLE_HOOK为1；

-  定义一个函数，名字和参数原型如下所示：

  ```c
  void vApplicationIdleHook( void );
  ```

  通常，**使用这个空闲钩子函数设置CPU进入低功耗模式**。

## 5、任务的创建

任务创建和删除API函数位于文件task.c中，需要包含task.h头文件

具体函数定义如下：

```
BaseType_t xTaskCreate(
           TaskFunction_t pvTaskCode,
           const char * const pcName,
           unsigned short usStackDepth,
           void *pvParameters,
           UBaseType_t uxPriority,
           TaskHandle_t * pvCreatedTask
            );
  例：          
    xTaskCreate( (TaskFunction_t)chassis_task,//底盘任务
                 (const char *)"chassis_task",
                 (uint16_t)CHASSIS_STK_SIZE,
                 (void *)NULL,
                 (UBaseType_t)CHASSIS_TASK_PRIO,
                 (TaskHandle_t *)&CHASSISTask_Handler
               );
```

### 参数描述

* pvTaskCode：指针，指向任务函数的入口。

  追溯到TaskFunction_t的定义可以找到

  ```c
  typedef void (*TaskFunction_t)( void * );
  ```

  此处涉及C语言中typedef的一个使用方法

  ​	void (*TaskFunction_t)( void * )意为定义了一个名为TaskFunction_t，入口参数类型为void*型，返回值为void型的函数指针变量；而在使用typedef后再进行TaskFunction_t  pvTaskCode的操作，则意为TaskFunction_t在这里作为类的别名，而 pvTaskCode作为类的变量。也就是说，在使用typedef后，可以将此处的TaskFunction_t理解为一个类似于int，void之类的类型

  做个总结：此处

  ```c
  typedef void (*TaskFunction_t)( void * );
  TaskFunction_t pvTaskCode;
  ```

  等同于：

  ```
  void (*pvTaskCode)( void * );
  ```

  **也就是说，pvTaskCode其实就是一个函数指针**

  （此用法同步收录于“C语言学习-一些关键字-typedef-指针"一栏）

* pcName：任务描述，名字。字符串的最大长度由宏configMAX_TASK_NAME_LEN指定

* usStackDepth：指定任务堆栈大小，能够支持的堆栈变量数量，而**不是字节数**。

- pvParameters：指针，当任务创建时，作为一个参数传递给任务。

* uxPriority：任务的优先级。

* pvCreatedTask：用于回传一个句柄（ID），创建任务后可以使用这个句柄引用任务

### 返回值

 如果任务成功创建并加入就绪列表函数返回pdPASS，否则函数返回错误码，具体参见projdefs.h。

### 用法（个人）

* 创建任务具体内容，例如上面的chassis_task
* 创建针对freertos的任务（xTaskCreate）
* 使用freertos调度任务

## 6.任务的删除

函数定义：

```c
void vTaskDelete( TaskHandle_t xTask );
```

从RTOS内核管理器中删除一个任务。任务删除后将会从就绪、阻塞、暂停和事件列表中移除。在配置头文件中，必须定义宏INCLUDE_vTaskDelete 为1，本函数才有效。被删除的任务，其在任务创建时由内核分配的存储空间，会由空闲任务释放。如果有应用程序调用xTaskDelete()，必须保证空闲任务获取一定的微控制器处理时间。任务代码自己分配的内存是不会自动释放的，因此删除任务前，应该将这些内存释放。

# 任务控制

## 延时（阻塞）

### 1.相对延时

* void vTaskDelay( portTickType xTicksToDelay )；

  参数描述：xTicksToDelay：延时时间总数，单位是系统时钟节拍周期

vTaskDelay()指定的延时时间是从调用vTaskDelay()后开始计算的相对时间。比如vTaskDelay(100)，那么从调用vTaskDelay()后，任务进入阻塞状态，经过100个系统时钟节拍周期，任务解除阻塞。因此，vTaskDelay()并不适用与周期性执行任务的场合。因为调用vTaskDelay()到任务解除阻塞的时间不总是固定的并且该任务下一次调用vTaskDelay()函数的时间也不总是固定的（两次执行同一任务的时间间隔本身就不固定，中断或高优先级任务抢占也可能会改变每一次执行时间）。

在文件FreeRTOSConfig.h中，宏INCLUDE_vTaskDelay 必须设置成1，此函数才能有效。

### 2.绝对延时

*  void vTaskDelayUntil( TickType_t *pxPreviousWakeTime, const TickType_tx TimeIncrement );

  参数描述：

  * pxPreviousWakeTime：指针，指向一个变量，该变量保存任务最后一次解除阻塞的时间。第一次使用前，**该变量必须初始化为当前时间**。之后这个变量会在vTaskDelayUntil()函数内**自动更新**。
  * xTimeIncrement：周期循环时间。当**当前时间**等于pxPreviousWakeTime+xTimeIncrement时任务会解除阻塞。

应当指出的是，如果指定的唤醒时间已经达到，vTaskDelayUntil()立刻返回（不会有阻塞）。因此，使用vTaskDelayUntil()周期性执行的任务，无论任何原因（比如，任务临时进入挂起状态）停止了周期性执行，使得任务少运行了一个或多个执行周期，**那么需要重新计算所需要的唤醒时间**。（例如任务挂起态结束后，当前的绝对时间已经超过指定的唤醒时间，vTaskDelayUntil()会立刻返回，且此时需要重新获得当前时间）这可以通过传递给函数的指针参数pxPreviousWake指向的值与当前系统时钟计数值比较来检测。
例：

```c
//每10次系统节拍执行一次
 void vTaskFunction( void * pvParameters )
 {
     static portTickType xLastWakeTime;
     const portTickType xFrequency = 10;
 
     // 使用当前时间初始化变量xLastWakeTime
     xLastWakeTime = xTaskGetTickCount();
 
     while(1)
     {
     	// 需要周期性执行代码放在这里
     	.........
     	//等待下一个周期
         vTaskDelayUntil( &xLastWakeTime,xFrequency ); 
     }
 }
```

在文件FreeRTOSConfig.h中，宏INCLUDE_vTaskDelayUntil 必须设置成1，此函数才有效。

## 获取优先级

* UBaseType_t uxTaskPriorityGet(TaskHandle_t xTask )；

  参数描述：

  * xTask：任务句柄。NULL表示获取当前任务的优先级。

例：

```c
void vAFunction( void )
 {
    xTaskHandle xHandle;
    // 创建另一个任务，并保存任务句柄
    xTaskCreate( vTaskCode, "NAME",STACK_SIZE, NULL, tskIDLE_PRIORITY, &xHandle );
    // 当前任务优先级比新创建的任务优先级高？
    if( uxTaskPriorityGet( xHandle ) <uxTaskPriorityGet( NULL ) )
    {
         // 当前优先级较高
    }
    else
    {
      ......
    }
 }
```

在文件FreeRTOSConfig.h中，宏INCLUDE_vTaskPrioritySet 必须设置成1，此函数才有效。

## 任务挂起

* void vTaskSuspend( TaskHandle_txTaskToSuspend );

  **被挂起的任务绝不会得到处理器时间，不管该任务具有什么优先级。**

  参数描述：

  * xTask：任务句柄。NULL表示表示挂起当前任务。

在文件FreeRTOSConfig.h中，宏INCLUDE_vTaskSuspend必须设置成1，此函数才有效。

## 解除任务挂起状态

*  void vTaskResume( TaskHandle_txTaskToResume );

宏INCLUDE_vTaskSuspend必须置1，此函数才有效。

## 解除任务挂起状态（用于中断服务函数）

还在学习中，暂不记录。

# 任务辅助调试类函数

这类函数主要用于调试信息输出、获取任务句柄、获取任务状态、操作任务标签值等等，种类繁多，不再单独总结，具体可跳至[FreeRTOS任务应用函数_研究是为了理解的博客-CSDN博客](https://freertos.blog.csdn.net/article/details/50498173)中讲解的很详细。

