---
title: FreeRTOS任务间的交互方法
date: 2022-08-30 15:56:18
tags: 嵌入式学习
---

# 说在前面

本来是打算在[FreeRTOS在CubeMX中的使用](https://www.dhksblog.top/2022/08/25/FreeRTOS-3/)就把这些内容给讲了的，但是学习的过程中发现网络上很多教程都存在着一些差异，所以为了更系统地学习，同时也方便日后查阅，就单独开一篇文章总结一下相关内容

**注：本文所有例程均经过测试且可以正常运行**

# 1、队列

FreeRTOS中的队列**是一种用于实现【任务与任务】，【任务与中断】以及【中断与任务】之间的通信机制。**此外，任务从队列读数据或者写入数据到队列时，都可能被阻塞。这个特性使得任务可以被设计成基于事件驱动的运行模式，大大提高了CPU的执行效率。队列是实现FreeRTOS中其他特性的基础组件，像软件定时器，信号量，互斥锁都是基于队列而实现的。

在操作系统里面，直接使用全局变量传输数据十分危险，看似正常运行，但不知道啥时候就会因为寄存器或者内存等等原因引起崩溃，所以引入消息，队列的概念。

## 队列的基本特性

队列是一种FIFO操作的数据结构，入队操作就是把一个新的元素放进队尾(tail)，出队操作就是从队头(front)取出一个元素。FreeRTOS中也支持把一个元素放到队头的操作，这个操作会**覆盖**之前队头的元素。

## 入队策略

队列在设计的时候，主要有两种元素入队存储策略：Queue by copy 和 Queue by reference。FreeRTOS的队列使用的是Queue by copy存储策略，考虑到这种策略实现起来更加简单，灵活，且安全。

- Queue by copy：数据入队的时候，队列中存储的是此数据的一份拷贝
- Queue by reference：数据队列的时候，队列中存储的是数据的指针，而非数据本身

## 入队堵塞

一个任务在尝试写入数据到队列时，可以指定一个阻塞时间，即任务在**等待队列有空余空间（非满）可写前最长等待阻塞时间**。当队列中有空间可写入时，任务会自动从阻塞态转换为就绪态。如果在等待时间内队列一直是满的，则等待时间到期后，任务也会自动从阻塞态转换为就绪态，但是它会返回一个写入失败的结果。

队列可能有多个writerTask，所以在等待队列有空闲空间时，可能会有多个任务阻塞。**当队列有空闲空间可写时，调度器会从所有阻塞的任务中选取优先级最高的那个任务让其进入到就绪态**。**如果最高优先级的不止一个，则让等待最久的那个进入到就绪态。**

## **出队堵塞**

一个任务在尝试从队列中读取数据时，可以指定一个阻塞时间，即任务在等待队列有元素可读前最长等待阻塞时间。当队列中有元素可读时，任务会自动从阻塞态转换为就绪态。**如果在等待时间内队列一直没有数据可读，则等待时间到期后，任务也会自动从阻塞态转换为就绪态，但是它会返回一个读取失败的结果**。

队列可能有多个reader Task，所以在等待队列有数据可读时，可能会有多个任务阻塞。**当队列有数据可读时，调度器会从所有阻塞的任务中选取优先级最高的那个任务让其进入到就绪态。如果最高优先级的不止一个，则让等待最久的那个进入到就绪态。**

## 相关API函数

* 队列创建（若使用cubemx则只用点点点，不用再用函数创建，所有创建类函数同理）

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

* 查询队列元素个数

  使用内核函数uxQueueMessagesWaiting()来获取队列中有多少个元素

|  函数  | UBaseType_t uxQueueMessagesWaiting( QueueHandle_t xQueue ); |
| :----: | :---------------------------------------------------------: |
|  参数  |                   **xQueue**：队列的句柄                    |
| 返回值 |             队列中的元素个数，返回0代表队列为空             |

按键例程：

```c
void task1(void const * argument) //低优先级任务
{
  BaseType_t result; 
  uint8_t dat[]="666\r\n";
  /* Infinite loop */
  for(;;)
  {
	if(HAL_GPIO_ReadPin(key_1_GPIO_Port,key_1_Pin == GPIO_PIN_SET)) //检测按键
	{
		result= xQueueSendToBack(myQueue01Handle,dat,portMAX_DELAY); //将元素dat放至队尾
		if(result==pdPASS) //放置成功返回pdPASS
		{
			HAL_GPIO_WritePin(led_g_GPIO_Port,led_g_Pin,GPIO_PIN_RESET);//点亮绿灯
		}
	}
   	osDelay(5);
  }
} 

void task2(void const * argument) //高优先级任务
{
  BaseType_t result;
  uint8_t datbuf[10]; //随便定义个缓冲区就可以，用于存放取出的元素
  /* Infinite loop */
  for(;;)
  {
	result=xQueueReceive(myQueue01Handle,datbuf,portMAX_DELAY);//取出元素放入datbuf中
	if(result==pdPASS)//读取成功返回pdPASS
	{
		HAL_GPIO_WritePin(led_r_GPIO_Port,led_r_Pin,GPIO_PIN_RESET);//点亮红灯
	}
	osDelay(10);
  }    
}
```

# 2、任务通知

*在网上找了很多博客对于任务通知介绍的函数都不一样，我自己也实验了很多，就把我试验过的可行的函数都总结一遍，可以根据具体的需要选择相关函数（其实函数功能基本差不多，但是原理不一样）*

每个RTOS任务都有一个32位的通知值，任务创建时，这个值被初始化为0。RTOS任务通知相当于直接向任务发送一个事件，接收到通知的任务可以解除阻塞状态，**前提是这个阻塞事件是因等待通知而引起的**。发送通知的同时，也可以可选的改变接收任务的通知值。

相对于用前必须分别创建队列、二进制信号量、计数信号量或事件组的情况，使用任务通知显然更灵活。更好的是，相比于使用信号量解除任务阻塞，使用任务通知可以快45%、使用更少的RAM。

**下面介绍的函数大类都可以实现任务通知的功能，可以根据需要选择**

## 一、xTask函数

### 发送函数（1）

|  函数  | BaseType_t xTaskNotify( TaskHandle_txTaskToNotify,<br/>                          uint32_t ulValue,<br/>                           eNotifyAction eAction); |
| :----: | :----------------------------------------------------------: |
|  参数  | xTaskToNotify：被通知的任务句柄<br/>ulValue： 通知更新值<br/> eAction:枚举类型，指明更新通知值的方法 |
| 返回值 |                       成功则返回pdPASS                       |

eAction具体说明：

|         枚举成员          |                             作用                             |
| :-----------------------: | :----------------------------------------------------------: |
|         eNoAction         |         发送通知但不更新通知值，意味着ulValue未使用          |
|         eSetBits          |              被通知任务值赋值按位或赋值ulValue               |
|        elncrement         |                     被通知任务的通知值++                     |
|  eSetValueWithOverwrite   |                  被通知任务值赋值为ulValue                   |
| eSetValueWithoutOverwrite | 如果被通知的任务还没取走上一个通知的情况下，又向被通知任务发送一个新的通知，则新通知值被丢弃，且xTaskNotify()会返回pdFALSE |

{% asset_img eaction.jpg This is an image %} 

### 发送函数（2）

|  函数  | BaseType_t xTaskNotifyGive(TaskHandle_t xTaskToNotify ); |
| :----: | :------------------------------------------------------: |
|  参数  |             xTaskToNotify：被通知的任务句柄              |
| 返回值 |                     成功则返回pdPASS                     |

其实这是一个宏，本质上相当于xTaskNotify( ( xTaskToNotify ), ( 0 ), **eIncrement** )在这种情况下，应该使用API函数ulTaskNotifyTake()来等待通知，而不应该使用API函数xTaskNotifyWait()

此函数不可以在中断服务例程中调用，中断保护等价函数为vTaskNotifyGiveFromISR()

### 接收函数（1）

<u>**注意，此处讲到的接收函数应与上面讲到的发送函数配对，例如：使用发送函数（1），接收通知也要使用接收函数（1）**</u>

|  函数  | BaseType_t xTaskNotifyWait( uint32_tulBits  ToClearOnEntry,<br/>                                                      uint32_tulBits  ToClearOnExit,<br/>                                                          uint32_t*  pulNotificationValue,<br/>                                    TickType_t  xTicksToWait ); |
| :----: | :----------------------------------------------------------: |
|  参数  |                            见下方                            |
| 返回值 |                       成功则返回pdPASS                       |

关于参数：

* ulBitsToClearOnEntry：在使用通知之前，先将任务的通知值与参数ulBitsToClearOnEntry的按位取反值按位与操作。设置参数ulBitsToClearOnEntry为0xFFFFFFFF(ULONG_MAX)，表示清零任务通知值。
  * **但是在实际调试过程中感觉给0x00和0xFFFFFFFF在使用上没有区别，所以就给0xFFFFFFFF就可以**

{% asset_img qufan1.jpg This is an image %} 

* *ulBitsToClearOnExit：在函数xTaskNotifyWait()退出前，将任务的通知值与参数ulBitsToClearOnExit的按位取反值按位与操作。设置参数ulBitsToClearOnExit为0xFFFFFFFF(ULONG_MAX)，表示清零任务通知值。
  * **使用同上，默认0xFFFFFFFF就可以**

{% asset_img qufan2.jpg This is an image %} 

* pulNotificationValue：用于向外回传任务的通知值。这个通知值在参数ulBitsToClearOnExit起作用前将通知值拷贝到*pulNotificationValue中（**需额外定义一个缓冲区**）。如果不需要返回任务的通知值，这里设置成NULL。
* xTicksToWait：因等待通知而进入阻塞状态的最大时间。时间单位为系统节拍周期。宏pdMS_TO_TICKS用于将指定的毫秒时间转化为相应的系统节拍数。

### 接收函数（2）

|  函数  | uint32_t ulTaskNotifyTake( BaseType_t  xClearCountOnExit,<br/>                                TickType_t   xTicksToWait ); |
| :----: | :----------------------------------------------------------: |
|  参数  | xClearCountOnExit：如果该参数为pdFALSE，则API函数xTaskNotifyTake()退出前，将任务的通知值减1；如果该参数设置为pdTRUE，则API函数xTaskNotifyTake()退出前，将任务通知值清零。<br/>xTicksToWait：因等待通知而进入阻塞状态的最大时间 |
| 返回值 |                       成功则返回pdPASS                       |

### 上例程！老规矩按键+led

发送函数（1）+接收函数（1）

```c
#define event1 0x0001<<1
#define event2 0x0001<<2
void task1(void const * argument) //低优先级
{
	BaseType_t result;
  for(;;)
  {
		if(HAL_GPIO_ReadPin(key_1_GPIO_Port,key_1_Pin) == GPIO_PIN_SET)//按键检测
		{
			result=  xTaskNotify(task2Handle,event1,4);//直接数字代替枚举成员
			if(result==pdPASS)
			{
				HAL_GPIO_WritePin(led_g_GPIO_Port,led_g_Pin,GPIO_PIN_RESET);//绿灯
			}
		}
		if(HAL_GPIO_ReadPin(key_2_GPIO_Port,key_2_Pin) == GPIO_PIN_SET)//按键检测
		{
			result=  xTaskNotify(task2Handle,event2,4);//直接数字代替枚举成员
			if(result==pdPASS)
			{
				HAL_GPIO_WritePin(led_b_GPIO_Port,led_b_Pin,GPIO_PIN_RESET);//蓝灯
			}
		}
		osDelay(100);
  }
}

void task2(void const * argument)//高优先级
{
	BaseType_t result;
	uint32_t ulNotifiedValue; //存放任务通知值的缓冲区
  for(;;)
  {
		result=xTaskNotifyWait(0xFFFFFFFF, 0xFFFFFFFF,&ulNotifiedValue, portMAX_DELAY);
		if(result==pdPASS)
		{
			HAL_GPIO_WritePin(led_r_GPIO_Port,led_r_Pin,GPIO_PIN_RESET);//红灯
		}
      	if(ulNotifiedValue&event1)
		{
			HAL_GPIO_WritePin(led_r_GPIO_Port,led_r_Pin,GPIO_PIN_SET);
			HAL_GPIO_WritePin(led_b_GPIO_Port,led_b_Pin,GPIO_PIN_SET);
			HAL_GPIO_WritePin(led_g_GPIO_Port,led_g_Pin,GPIO_PIN_SET);//全灭
		}
        if(ulNotifiedValue&event2)
		{
			HAL_GPIO_WritePin(led_r_GPIO_Port,led_r_Pin,GPIO_PIN_RESET);
			HAL_GPIO_WritePin(led_b_GPIO_Port,led_b_Pin,GPIO_PIN_RESET);
			HAL_GPIO_WritePin(led_g_GPIO_Port,led_g_Pin,GPIO_PIN_RESET);//全亮
		}
 /* 
 整体逻辑：
 按键1按下->绿灯亮->红灯亮->全灭
 按键2按下->蓝灯亮->红灯亮->全灭
 
 这里对于通知值的判断可以让用户知道是哪个任务通知函数触发了这次任务接收，感觉非常好用！！！
 */
		osDelay(10);
  }
}
```

总结：

发送函数（2）+接收函数（2）就不单独写了，基本同理，只是接收函数（2）无法对通知值进行判断，但是整体更方便快捷

## 二、os函数

### 向指定任务发送任务通知

| **函数** | **int32_t osSignalSet (osThreadId thread_id, int32_t signal)** |
| :------: | :----------------------------------------------------------: |
|   参数   | **thread_id：** 接收通知的任务ID<br/>**signal：**任务通知值（按位操作数字） |
|  返回值  |                            错误码                            |

### 等待任务通知

| **函数** | **osEvent osSignalWait (int32_t signals, uint32_t millisec)** |
| :------: | :----------------------------------------------------------: |
|   参数   | signals： 接收完成后等待被清零的数据位(0x0001\|0x0002=0x003)<br/>**millisec：** 等待超时时间，单位为系统节拍周期 |
|  返回值  |                            错误码                            |

### 例程，按键+led

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

**而xTask函数不同于os函数，xTask函数是在task1执行完后才会进入task2，这点请务必记住，二者通知原理不同，所以根据使用情况选择使用xTask函数还是os函数**

# 3、信号量

信号量同样是一种任务之间交互的函数，但是就我按照网上教程使用过后感觉。。。不如队列和任务通知，可能是我目前的一些工程量还没到使用到信号量的程度？而且在使用过程中也遇到了一些迷惑的地方，所以目前不准备深究，等到实在要用的时候再说吧😋，感兴趣的同学可以去[(19条消息) FreeRTOS系列第19篇---FreeRTOS信号量_研究是为了理解的博客-CSDN博客_freertos 信号](https://freertos.blog.csdn.net/article/details/50835613)看看，讲的很详细了可以说，我大部分内容也是跟着这个学的。