---
title: FreeRTOS在CubeMX中的使用
date: 2022-08-25 19:08:45
tags: 嵌入式学习
---

# 说在前面

**此篇所有有关函数均为CubeMX 提供的os函数（特殊地方会专门说明），freertos相关用户调用函数也分类似标准库和hal库的区别！但大体用法也与**[FreeRTOS的任务相关 | 小董的BLOG (gitee.io)](https://dhkkk.gitee.io/2022/08/23/FreeRTOS-2/)**中介绍的相似，可以参考**

**2022.8.26： 阅读了os函数内容并验证后，发现os函数与vtask函数兼容，（其实os函数也是调用的vtask函数）所以理论来说两边的函数可以混用，但是为了可读性等方面尽量还是不要混用**

**2022.8.28：在自己用了很多功能后，越发觉得库函数比os函数好用，大家自己取舍吧，反正二者是可以混用的，先用cubemx配置，再用库函数编写具体内容也是完全🆗的。**

# 1、配置界面

|          配置项           |                            功能                             |
| :-----------------------: | :---------------------------------------------------------: |
|   **Tasks and Queues**    |           任务与队列，用于配置任务体以及消息队列            |
| **Timers and Semaphores** | 软件定时器与信号量，用于配置内核对象 （软件定时器和信号量） |
|        **Mutexes**        |             互斥量，用于配置内核对象（互斥量）              |
|        **Events**         |                 事件，配置内核对象（事件）                  |
|  **FreeRTOS Heap Usage**  |               查看用户任务和系统任务的堆占用                |
|   **Config Parameters**   |                       系统的参数配置                        |
|  **Include Parameters**   |                       系统的功能裁剪                        |
|   **Advanced Settings**   |                   CubeMX 生成代码预配置项                   |
|    **User Constants**     |                        用户常量定义                         |

## 版本选择

{% asset_img 版本.jpg This is an image %} 

CMSIS是一种接口标准，目的是屏蔽软硬件差异以提高软件的兼容性。RTOS v1使得软件能够在不同的实时操作系统下运行（屏蔽不同RTOS提供的API的差别），而RTOS v2则是拓展了RTOS v1，兼容更多的CPU架构和实时操作系统。因此我们在使用时可以根据实际情况选择，如果学习过程中使用STM32F1、F4等单片机时没必要选择RTOS v2，更高的兼容性背后时更加冗余的代码，理解起来比较困难。

# 2、各配置详细设置

## **一、Config Parameters——系统的参数配置**

### 1.Kernel Setting——FreeRTOS 调度内核设置

由于一些重要配置已在[FreeRTOS_基本框架 | 小董的BLOG (gitee.io)](https://dhkkk.gitee.io/2022/08/18/FreeRTOS-1/)中讲到，因此本文只会记录一些容易搞混的地方

* **USE_PREEMPTION**

  USE_PREEMPTION 是 RTOS 的调度方式选择，为 1 时使用抢占式调度器，为 0 时使用协程。

  当使用协程的话会在如下地方进行任务切换：

  1. 一个任务调用了函数 taskYIELD()。
  2. 一个任务调用了可以使任务进入阻塞态的 API 函数。
  3. 应用程序明确定义了在中断中执行上下文切换。

- **USE_MUTEXES、USE_RECURSIVE_MUTEXES、USE_COUNTING_SEMAPHORES**

  为 1 则开启系统构建过程中的互斥量、递归互斥量和信号量,默认开启就可。

- **QUEUE_REGISTRY_SIZE**

  队列注册表的大小，可以用于管理队列名称和队列实体，方便运行中进行查看与管理，默认为8

- **ENABLE_BACKWARD_COMPATIBILITY**

  一个兼容性使能，使能后， FreeRTOS 8.0.0 之后的版本可以通过宏定义使用 8.0.0 版本之前的函数接口，默认使能

- **USE_TASK_NOTIFICATIONS**

  任务通知使能，每个RTOS任务都有一个32位的通知值，RTOS任务通知是一个直接发送给任务的事件，它可以解除接收任务的阻塞，并可选地更新接收任务的通知值，为1开启，为0关闭，关闭可以为每个任务节省8个字节的内存空间

- **RECORD_STACK_HIGH_ADDRESS**

  记录任务的堆栈入口地址到TCB，为1使能，为0关闭

### **2.Memory management setting**——内存管理设置

视情况调整即可，内存分配方式默认heap_4，方便完善。

### 3.Hook function relateed difinitions——钩子函数配置

**钩子函数是一种回调函数，用于在任务执行一次之后或者某些事件发生后执行的函数**，该配置项里面有五个选项，控制5种不同功能的钩子函数开启，当然用户也可以在代码中自己定义

- **USE_IDLE_HOOK**

  每当空闲任务执行一次，钩子函数都会被执行一次

* **USE_TICK_HOOK**

  每个TICK周期，钩子函数都会执行一次

* **USE_MALLOC_FAILED_HOOK**

  当申请动态内存失败时，钩子函数会执行一次(需要额外配置申请动态内存失败的相关事项)

- **USE_DAEMON_TASK_STARTUP_HOOK**

  任务刚启动时，钩子函数会执行一次

- **CHECK_FOR_STACK_OVERFLOW**

  任务栈溢出时，钩子函数会执行一次，传入任务 TCB 和任务名称（也需要额外配置）

### 4.Run time and task stats... ——任务运行追踪配置

- **GENERATE_RUN_TIME_STATS**

  开启时间统计功能，在调用 vTaskGetRunTimeStats() 函数时，**将任务运行时间信息保存到可读列表中**

- **USE_TRACE_FACILITY**

  使能后会包含额外的结构成员和函数以帮助执行可视化和跟踪，**默认开启，方便 MDK 软件工具调试使用**

- **USE_STATS_FORMATTING_FUNCTIONS**

  使能后会**生成 vTaskList() 和 vTaskGetRunTimeStats() 函数用于获取任务运行状态**

### 5.Co-routine ralated definitions——协程配置

默认不开启，协程现在几乎很少使用且FreeRTOS不再更新和维护协程。

### 6.Software timer definitions——软件定时器配置

默认即可。

### 7.Interrupt nesting behavior configuration——优先级配置

具体参数在[FreeRTOS_基本框架 | 小董的BLOG (gitee.io)](https://dhkkk.gitee.io/2022/08/18/FreeRTOS-1/)已有详细描述，不再赘述。

## **二、Include Parameters**——系统的功能裁剪

Include Parameters 下的选项应用于内核裁剪，裁剪不必要的功能，精简系统功能，减少资源占用

|          **选项**           |                           **功能**                           |
| :-------------------------: | :----------------------------------------------------------: |
|      vTaskPrioritySet       |                   改变某个任务的任务优先级                   |
|      uxTaskPriorityGet      |                     查询某个任务的优先级                     |
|         vTaskDelete         |                           删除任务                           |
|    vTaskCleanUpResources    |                回收任务删除后的资源如RAM等等                 |
|        vTaskSuspend         |                           挂起任务                           |
|       vTaskDelayUntil       |                     阻塞延时一段绝对时间                     |
|         vTaskDelay          |                     阻塞延时一段相对时间                     |
|   xTaskGetSchedulerState    |              获取任务调度器的状态，开启或未开启              |
|     xTaskResumeFromISR      |              在中断服务函数中恢复一个任务的运行              |
|    xQueueGetMutexHolder     |        获取信号量的队列拥有者，返回拥有此信号量的队列        |
|  xSemaphoreGetMutexHolder   |             查询拥有互斥锁的任务，返回任务控制块             |
|      pcTaskGetTaskName      |                         获取任务名称                         |
| uxTaskGetStackHighWaterMark |                获取任务的堆栈的历史剩余最小值                |
|  xTaskGetCurrentTaskHandle  |               此函数用于获取当前任务的任务句柄               |
|        eTaskGetState        |               此函数用于查询某个任务的运行壮态               |
|  xEventGroupSetBitFromISR   |              在中断服务函数中将指定的事件位清零              |
|   xTimerPendFunctionCall    | 定时器守护任务的回调函数（定时器守护任务使用到一个命令队列，只要向队列发送信号就可以执行相应代码，可以实现“中断推迟处理”功能） |
|       xTaskAbortDelay       | 中止延时函数，该函数能立即解除任务的阻塞状态，将任务插入就绪列表中 |
|       xTaskGetHandle        |               此函数根据任务名字获取的任务句柄               |

## **三、Tasks and Queues**——创建任务与队列

### 1.任务创建

{% asset_img freertos_taskadd.jpg This is an image %} 

具体内容已在[FreeRTOS的任务相关 | 小董的BLOG (gitee.io)](https://dhkkk.gitee.io/2022/08/23/FreeRTOS-2/)中讲到，需要注意的是：

* Task Name为任务**名称**，Entry Function才是任务实体，即运行的**任务函数入口**（函数名）

* 任务的优先级细分有各等级优先级，详见文章最后对于优先级问题的理解

* Code Generation为代码生成模式，细分为

  * As weak： 产生一个用 __weak 修饰的弱定义任务函数，用户可自己再进行定义
  * As external： 产生一个外部引用的任务函数，用户需要自己定义该函数
  * Default： 产生一个默认格式的任务函数，用户需要在该函数内实现自己的功能

  一般使用默认格式即可

* Allocation为定义内存分配方式，分为静态与动态，与c语言中static用法类似

#### CubeMX 提供的一些用户调用函数

|       **函数**        |             **功能**             |
| :-------------------: | :------------------------------: |
|      osThreadNew      |            创建新任务            |
|    osThreadGetName    |           获取任务名称           |
|     osThreadGetId     |   获取当前任务的控制块（TCB）    |
|   osThreadGetState    |      获取当前任务的运行状态      |
| osThreadGetStackSize  |        获取任务的堆栈大小        |
| osThreadGetStackSpace |      获取任务剩余的堆栈大小      |
|  osThreadSetPriority  |          设定任务优先级          |
|  osThreadGetPriority  |          获取任务优先级          |
|     osThreadYield     |      切换控制权给下一个任务      |
|    osThreadSuspend    |             挂起任务             |
|    osThreadResume     | 恢复任务（挂起多少次恢复多少次） |
|    osThreadDetach     |  分离任务，方便任务结束进行回收  |
|     osThreadJoin      |        等待指定的任务停止        |
|     osThreadExit      |           停止当前任务           |
|   osThreadTerminate   |           停止指定任务           |
|   osThreadGetCount    |        获取激活的任务数量        |
|   osThreadEnumerate   |          列举激活的任务          |

### 2.队列创建

FreeRTOS中的队列是一种用于实现【任务与任务】，【任务与中断】以及【中断与任务】之间的通信机制。此外，任务从队列读数据或者写入数据到队列时，都可能被阻塞。这个特性使得任务可以被设计成基于事件驱动的运行模式，大大提高了CPU的执行效率。队列是实现FreeRTOS中其他特性的基础组件，像软件定时器，信号量，互斥锁都是基于队列而实现的。

在操作系统里面，直接使用全局变量传输数据十分危险，看似正常运行，但不知道啥时候就会因为寄存器或者内存等等原因引起崩溃，所以引入消息，队列的概念。

#### 队列的基本特性

队列是一种FIFO操作的数据结构，入队操作就是把一个新的元素放进队尾(tail)，出队操作就是从队头(front)取出一个元素。FreeRTOS中也支持把一个元素放到队头的操作，这个操作会**覆盖**之前队头的元素。

#### 入队策略

队列在设计的时候，主要有两种元素入队存储策略：Queue by copy 和 Queue by reference。FreeRTOS的队列使用的是Queue by copy存储策略，考虑到这种策略实现起来更加简单，灵活，且安全。

- Queue by copy：数据入队的时候，队列中存储的是此数据的一份拷贝
- Queue by reference：数据队列的时候，队列中存储的是数据的指针，而非数据本身

#### 入队堵塞

一个任务在尝试写入数据到队列时，可以指定一个阻塞时间，即任务在**等待队列有空余空间（非满）可写前最长等待阻塞时间**。当队列中有空间可写入时，任务会自动从阻塞态转换为就绪态。如果在等待时间内队列一直是满的，则等待时间到期后，任务也会自动从阻塞态转换为就绪态，但是它会返回一个写入失败的结果。

队列可能有多个writerTask，所以在等待队列有空闲空间时，可能会有多个任务阻塞。**当队列有空闲空间可写时，调度器会从所有阻塞的任务中选取优先级最高的那个任务让其进入到就绪态**。**如果最高优先级的不止一个，则让等待最久的那个进入到就绪态。**

#### **出队堵塞**

一个任务在尝试从队列中读取数据时，可以指定一个阻塞时间，即任务在等待队列有元素可读前最长等待阻塞时间。当队列中有元素可读时，任务会自动从阻塞态转换为就绪态。如果在等待时间内队列一直没有数据可读，则等待时间到期后，任务也会自动从阻塞态转换为就绪态，但是它会返回一个读取失败的结果。

队列可能有多个reader Task，所以在等待队列有数据可读时，可能会有多个任务阻塞。**当队列有数据可读时，调度器会从所有阻塞的任务中选取优先级最高的那个任务让其进入到就绪态。如果最高优先级的不止一个，则让等待最久的那个进入到就绪态。**

#### 相关API函数（非os函数，原博客中提供的os函数不适用于v1版本）（已测试，可用）

* 队列创建

  使用xQueueCreate()内核函数来创建一个队列。队列的1存储空间从FreeRTOS heap中分配。在使用xQueueCreate()创建队列时，如果FreeRTOS heap中没有足够的存储空间分配给当前队列，则函数返回NULL。如果创建成功，则返回队列的句柄(QueueHandle_t类型)

|  函数  | QueueHandle_t xQueueCreate( UBaseType_t uxQueueLength, UBaseType_t uxItemSize ); |
| :----: | :----------------------------------------------------------: |
|  参数  | **uxQueueLength**：指定队列的长度，即最多可以存放的元素个数<br/>**uxItemSize**：队列中存储的元素的大小（占用的字节数） |
| 返回值 | 返回NULL代表创建失败，没有足够的堆空间来创建当前队列；创建成功则返回队列的句柄 |

* 元素入队

  使用xQueueSendToBack()函数来向队尾存放一个元素，使用xQueueSend()函数来向队首存放一个元素，两个函数用法一样

|  函数  | BaseType_t xQueueSendToBack( QueueHandle_t xQueue,<br/>                                                                        const void * pvItemToQueue,<br/>                                                       TickType_t xTicksToWait ); |
| :----: | :----------------------------------------------------------: |
|  参数  | **xQueue**：目标队列的句柄<br/>**pvItemToQueue**：入队元素的指针。队列将存储此指针指向的数据的备份<br/>**xTicksToWait**：指定等待队列有空间可以容纳新元素入队的最长等待（阻塞）时间 |
| 返回值 | 当元素成功入队时返回pdPASS，因为队列满而无法入队时返回errQUEUE_FULL（超时后） |

* 元素出队

  使用xQueueReceive()内核函数来从队列中读取队头元素，**读取成功后队列将删除这个元素**，实现元素出队操作。

|  函数  | BaseType_t xQueueReceive( QueueHandle_t xQueue,<br/>                                                  void * const pvBuffer,<br/>                                              TickType_t xTicksToWait ); |
| :----: | :----------------------------------------------------------: |
|  参数  | **xQueue**：目标队列的句柄<br/>**pvBuffer**：用于存放读取到的队列元素的缓冲区，队列将把出队的元素拷贝到此缓冲区中<br/>**xTicksToWait**：指定等待队列有数据可读的最长等待（阻塞）时间 |
| 返回值 | 当成功从队列读取到元素时返回pdPASS，因为队列空而无法出队时返回errQUEUE_EMPTY |

* 查询队列元素个数

  使用内核函数uxQueueMessagesWaiting()来获取队列中有多少个元素

|  函数  | UBaseType_t uxQueueMessagesWaiting( QueueHandle_t xQueue ); |
| :----: | :---------------------------------------------------------: |
|  参数  |                   **xQueue**：队列的句柄                    |
| 返回值 |             队列中的元素个数，返回0代表队列为空             |

老规矩，来个按键例程（可以正常运行）：

```c
void task1(void const * argument) //低优先级任务
{
  BaseType_t result; 
  uint8_t dat[]="666\r\n";
  /* Infinite loop */
  for(;;)
  {
	if(HAL_GPIO_ReadPin(key_1_GPIO_Port,key_1_Pin == GPIO_PIN_SET))
	{
		result= xQueueSendToBack(myQueue01Handle,dat,portMAX_DELAY);
		if(result==pdPASS)
		{
			HAL_GPIO_WritePin(led_g_GPIO_Port,led_g_Pin,GPIO_PIN_RESET);
		}
	}
   	osDelay(5);
  }
} 

void task2(void const * argument) //高优先级任务
{
  BaseType_t result;
  uint8_t datbuf[10]; //随便定义个缓冲区就可以
  /* Infinite loop */
  for(;;)
  {
	result=xQueueReceive(myQueue01Handle,datbuf,portMAX_DELAY);
	if(result==pdPASS)
	{
		HAL_GPIO_WritePin(led_r_GPIO_Port,led_r_Pin,GPIO_PIN_RESET);
	}
	osDelay(10);
  }    
}

```

懒得去找os函数了，库函数挺好用的是吧😋

## 四、事件(组)

### 写在前面2022.8.27

**目前网上对于os函数的使用教程很少，而且大部分都有问题，本小节讲到的os函数使用方法感觉BUG是真的多，搞了整整两天也无法完全实现想要的功能，而且总是一些莫名其妙的问题，感觉是与任务调度有关，但是始终没找到真正原因，人都麻了QAQ。所以本小节（事件）仅提供思路，无法保证能正常运行。（新手建议直接跳过，高手可以找找原因）**

**在尝试两天发现效果极差后，果断换任务通知的方法，使用起来可以说很丝滑了，第一次用就可以正常实现功能（从未如此美妙的开局！！）也基本能代替事件通知的功能，建议直接使用任务通知（见下文）**

**没有试过库函数，因为发现任务通知够用了，感兴趣的同学可以试一下**

***************************************************************************************************************分割线**************************************************************************************************************************************

* 先列出cubemx提供的事件组函数：

|       **函数**       |        **功能**        |
| :------------------: | :--------------------: |
|   osEventFlagsNew    |     创建事件标志组     |
| *osEventFlagsGetName |   获取事件标志组名称   |
|   osEventFlagsSet    |     设置事件标志组     |
|  osEventFlagsClear   |     清除事件标志组     |
|   osEventFlagsGet    | 获取当前事件组标志信息 |
|   osEventFlagsWait   |   等待事件标志组触发   |
|  osEventFlagsDelete  |     删除事件标志组     |

### 具体配置

*对于网络上的解释感觉有点晦涩难懂，自己总结了容易理解的方法，仅供参考*

事件的作用主要是实现任务间通信的机制，主要用于实现多任务间的同步，但是事件类型通信只能是**事件类型的通信**，**没有数据传输**

使用流程：

* 在cubemx中配置相关事件句柄等
* 在程序中定义触发事件位

```c
#define event1 1<<1 	//事件1
#define event2 1<<2 	//事件2
```

事件标志组中的所有事件位都存储在一个无符号的 EventBits_t 类型的变量（**事件组**）中（例如下文中的result），当 configUSE_16_BIT_TICKS 为 1 的时候事件标志组可以存储 8 个事件位，当 configUSE_16_BIT_TICKS 为 0 的时候事件标志组存储 24个事件位，每个事件位其实就是一个0或者1数字

* 定义事件组（在触发任务和使用到等待事件函数的任务中均需定义，定义成局部变量感觉方便点）

```c
osStatus_t result;//也可以用 EventBits_t 不过过程会麻烦一点
```

以下是对osStatus_t的定义，方便查阅

```c
typedef enum {
  osOK                      =  0,         ///< Operation completed successfully.
  osError                   = -1,         ///< Unspecified RTOS error: run-time error but no other error message fits.
  osErrorTimeout            = -2,         ///< Operation not completed within the timeout period.
  osErrorResource           = -3,         ///< Resource not available.
  osErrorParameter          = -4,         ///< Parameter error.
  osErrorNoMemory           = -5,         ///< System is out of memory: it was impossible to allocate or reserve memory for the operation.
  osErrorISR                = -6,         ///< Not allowed in ISR context: the function cannot be called from interrupt service routines.
  osStatusReserved          = 0x7FFFFFFF  ///< Prevents enum down-size compiler optimization.
} osStatus_t;
```

* 设置事件标志（触发条件）

  以按键触发为例：（可以正常运行）

```c
void event(void *argument)
{
	osStatus_t result;
    for(;;)
    {
        if(HAL_GPIO_ReadPin(key_1_GPIO_Port,key_1_Pin)!=0)
        {
            result = osEventFlagsSet(key_pHandle,event1);
            if(result == osOK)
            {
                printf("事件1触发成功");
            }
            else
            {
                printf("事件1触发失败");
            }
        }
        if(HAL_GPIO_ReadPin(key_2_GPIO_Port,key_2_Pin)!=0)
        {
            result = osEventFlagsSet(key_pHandle,event2);
            if(result == osOK)
            {
                printf("事件2触发成功");
            }
            else
            {
                printf("事件2触发失败");
            }
        }
    }    
}
```

* 等待事件标志

等待函数参数介绍：

`uint32_t osEventFlagsWait (osEventFlagsId_t ef_id, uint32_t flags, uint32_t options, uint32_t timeout);`

| **参数** |                           **功能**                           |
| :------: | :----------------------------------------------------------: |
|  ef_id   |                      事件标志组控制句柄                      |
|  flags   |                         等待的事件位                         |
| options  | 等待事件位的操作<br/>osFlagsWaitAny ：等待的事件位有任意一个等到就恢复任务<br/>osFlagsWaitAll：等待的事件位全部等到才恢复任务<br/>osFlagsNoClear：等待成功后不清楚所等待的标志位（默认清除） |
| timeout  | 等待事件组的等待时间（等待期间任务挂起在内核对象的挂起队列） |

使用例子：同时等待事件1和事件2，且等待到不清除

**（该任务由于未知原因不能正常运行，仅提供思路）**

```c
void wait_task(void *argument)
{
	osStatus_t result;
	for(;;)
	{
		result = osEventFlagsWait(myEvent01Handle,
								  event1|event2,
								  osFlagsWaitAll,
								  10);
		if(result == osOK)
		{
		//可使用阻塞、挂起等方式进入其他任务或直接在此处进行其他操作
		}
		else
		{
		//未触发等待
		}
	}
}
```

## 五、任务通知

### 写在前面 2022.8.27

**本文最开始参考的是[ CubeMX使用FreeRTOS编程指南_Top嵌入式的博客-CSDN博客_cubemx freertos](https://blog.csdn.net/qq_45396672/article/details/120877303?ops_request_misc={"request_id":"166155998716782395375007","scm":"20140713.130102334.."}&request_id=166155998716782395375007&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-1-120877303-null-null.142^v42^control,185^v2^tag_show&utm_term=freertos)但是在实际使用的过程中可以说有很多漏洞甚至错误，本小节在参考博客中所提供的flag函数好像根本不是任务通知的函数（？）并且其所说的使用方法中连函数的入口参数都写错了，（不晓得他自己跑过自己写的程序没有😅）所以关于flag相关函数在这里仅提供函数内容，用法就不探索了，后面会讲到我实际跑过并可以使用的任务通知函数。**

### flag相关函数介绍

FreeRTOS 的每个任务都有一个 32 位的通知值，任务控制块中的成员变量 ulNotifiedValue 就是这个通知值。任务通知是一个事件，假如某个任务通知的接收任务因为等待任务通知而阻塞的话，向这个接收任务发送任务通知以后就会解除这个任务的阻塞状态，CubeMX内没有提供相关的配置项，但在其生成的 FreeRTOS 接口里面有相关函数进行配置。

|      **函数**      |      **功能**      |
| :----------------: | :----------------: |
|  osThreadFlagsSet  | 设置任务的通知标志 |
| osThreadFlagsClear |    清除任务通知    |
|  osThreadFlagsGet  |    获取任务标志    |
| osThreadFlagsWait  | 等待特定的任务标志 |

**常用osThreadFlagsSet，osThreadFlagsWait，使用方式与事件触发相同**，不再细说，仅介绍一下函数参数

`uint32_t osThreadFlagsWait (uint32_t flags, uint32_t options, uint32_t timeout);`

其中options的参数：

| **参数**       | **功能**                                     |
| -------------- | -------------------------------------------- |
| osFlagsWaitAny | 等待32位通知值任意一位触发后恢复任务（默认） |
| osFlagsWaitAll | 等待指定的任务通知值全部触发后再恢复任务     |
| osFlagsNoClear | 恢复任务后不清除任务标志（默认清除）         |

**任务通知其实个任务事件标志组使用上没有多大的区别，但他们两个的实现原理不同，同时任务通知对资源的占用更少**

**根据 FreeRTOS 官方的统计，使用任务通知替代二值信号量的时候任务解除阻 塞的时间要快 45%，并且需要的 RAM 也更少**

### 可以实际使用的任务通知

使用前记得在cubemx配置中开启`USE_TASK_NOTIFICATIONS`和在include parameters里开启`eTaskGetState`

FreeRTOS 提供以下几种方式发送通知给任务 ：

- 发送通知给任务， 如果有通知未读，不覆盖通知值。
- 发送通知给任务，直接覆盖通知值。
- 发送通知给任务，设置通知值的一个或者多个位 ，可以当做事件组来使用。
- 发送通知给任务，递增通知值，可以当做计数信号量使用。

通过对以上任务通知方式的合理使用，可以在一定场合下替代 FreeRTOS 的信号量，队列、事件组等。

#### 相关api函数：

* 向指定任务发送任务通知

| **函数** | **int32_t osSignalSet (osThreadId thread_id, int32_t signal)** |
| :------: | :----------------------------------------------------------: |
|   参数   | **thread_id：** 接收通知的任务ID<br/>**signal：**任务通知值（按位操作数字） |
|  返回值  |                            错误码                            |

* 等待任务通知

| **函数** | **osEvent osSignalWait (int32_t signals, uint32_t millisec)** |
| :------: | :----------------------------------------------------------: |
|   参数   | signals： 接收完成后等待被清零的数据位(0x0001\|0x0002=0x003)<br/>**millisec：** 等待超时时间，单位为系统节拍周期 |
|  返回值  |                            错误码                            |

#### 实例（按键触发任务通知，已经过验证）

```c
#define event1 0x0001
void task1(void const * argument) //低优先级任务
{
  /* USER CODE BEGIN led_G */

  /* Infinite loop */
  for(;;)
  {
		if(HAL_GPIO_ReadPin(key_1_GPIO_Port,key_1_Pin == GPIO_PIN_SET))
		{
			osSignalSet(task2Handle,event1);
		}	
		osDelay(10);
  }
  /* USER CODE END led_G */
}

void task2(void const * argument) //高优先级任务
{
  /* USER CODE BEGIN led_B */
	osEvent event;	
  /* Infinite loop */
  for(;;)
  {
	event = osSignalWait(event1,osWaitForever);  
	if(event.status == osEventSignal)  
	{
		HAL_GPIO_WritePin(led_g_GPIO_Port,led_g_Pin,GPIO_PIN_RESET);
	}
  }
}
```

这里task2可以不用额外在循环中添加堵塞，因为wait函数本身自带堵塞，**其堵塞的解除是在osSignalSet函数的作用之后的瞬间，也就是说在执行完osSignalSet(task2Handle,event1)后task2会直接抢占task1**

**而xTask函数则是在task1执行完后才会进入task2，这点请务必记住，二者通知原理不同，使用方法因此也不同**

## 其他的内容以后再来探索吧（指还没学到）XD

# 遇见的一些问题

## 一、FreeRTOS的时钟源配置

当使用了FreeRtos的时候，建议HAL库使用除了Systick以外的时钟源。也就是说当不使用FreeRtos的时候，HAL使用的是systick作为时钟源，现在使用了rtos，不建议hal库和rtos一起使用systick作为时钟源

## 二、优先级问题

{% asset_img 2.jpg This is an image %} 

这里我自己也没太搞懂，姑且谈谈自己的理解：

按道理来说cortex内核对于优先级数值是越小其逻辑上的优先度就越高，其在cubemx上的大部分配置也同样符合这一原则，但是在对于任务优先级的分配上cubemx采用了上图这种low、normal等的分配方式，并且在程序中的定义数值则是优先级数值越大，逻辑优先度越高；这可能是cubemx为了方便理解做出的优化(？)

## 三、阻塞问题

**在程序中的阻塞函数≠正常的延时函数**！！！

当程序执行到vTaskDelay或osdelay后会将当前任务添加到延时任务列表里，并强行切换任务。**也就是说，在这段阻塞期内，只要有任务处于就绪态，就会进入该任务，若多个任务就绪，则按照优先级进行抢占。当堵塞态的任务延时到期过后会重新进入就绪态等待。**并且在重新进入就绪态后任务会从阻塞之后的位置开始。

* 任务函数内可以正常调用延时函数
* 挂起规律与阻塞规律基本一致

## 四、任务抢占问题

原先一直以为抢占式工作流程为：先运行优先级最高的任务，直至其被阻塞、挂起等；这期间运行优先级第二高的任务，直到其被阻塞、挂起等；若最高优先级此时已处于就绪态，则会在第二优先级任务进入阻塞等非运行态时进入运行态，**但其实这是不对的**

例如：

```c
void task1(void *argument)//低优先级
{
  /* USER CODE BEGIN led_R */
	HAL_GPIO_WritePin(led_r_GPIO_Port,led_r_Pin,GPIO_PIN_SET);
  /* Infinite loop */
  for(;;)
  {
	HAL_GPIO_TogglePin(led_b_GPIO_Port,led_b_Pin);
	HAL_Delay(5000);
    osDelay(1);
  }
  /* USER CODE END led_R */
}



void task2(void *argument)//高优先级
{
  /* USER CODE BEGIN led_B */
	osStatus_t result;
	
  /* Infinite loop */
  for(;;)
  {
		result=osEventFlagsSet(key1_pHandle,event1);
		if(result == osOK)
		{
			HAL_GPIO_WritePin(led_r_GPIO_Port,led_r_Pin,GPIO_PIN_RESET);
			HAL_GPIO_WritePin(led_b_GPIO_Port,led_b_Pin,GPIO_PIN_SET);
			HAL_GPIO_WritePin(led_g_GPIO_Port,led_g_Pin,GPIO_PIN_SET);

		}
		else
		{
			HAL_GPIO_WritePin(led_b_GPIO_Port,led_b_Pin,GPIO_PIN_RESET);
			HAL_GPIO_WritePin(led_g_GPIO_Port,led_g_Pin,GPIO_PIN_RESET);
		}
		osDelay(5);
}
```

按照原先的思路，程序开始，task2会优先运行，若事件设定正常，此时红灯会亮起；然后task2进入阻塞态5ms，task1进入运行态，此时按道理红灯会熄灭，蓝灯亮起5秒后，task1进入阻塞态，task2重新运行，此时红灯再亮，蓝灯灭。

**但是现实是：红灯亮起后，蓝灯只会极快地闪烁一下后红灯继续亮起，并且蓝灯闪烁的频率大概为5s左右。**

所以，真正的抢占流程应该是：程序开始，task2会优先运行，若事件设定正常，此时红灯会亮起；然后task2进入阻塞态5ms，**task1进入运行态，当task1运行5ms后，task2的阻塞结束，进入就绪态，此时task1刚刚开始运行HAL_Delay(5000)，由于task2优先级更高且已经就绪，task2直接抢占并运行（此时蓝灯就会熄灭，因此蓝灯一次只亮了5ms），而不是等到task1运行osDelay(10)才会抢占**；当task2再一次进入堵塞时，task1从上次被抢占的位置继续运行5ms然后被task2抢占，这样一直循环，直到HAL_Delay(5000)结束，蓝灯才会再次闪烁一次。

总结：高优先级的任务**一旦就绪后会立刻抢占**优先级低的任务,哪怕当前低优先级正在运行，而不是等到优先级低的任务进入非运行态时才抢占。

