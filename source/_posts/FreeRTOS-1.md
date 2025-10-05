---
title: FreeRTOS_基本框架
date: 2022-08-18 18:12:33
tags: 嵌入式学习
---

# 1、基础内容

## 状态

* 运行态:任务正在运行的状态。如果使用的是核处理器的话那么不管在任何时刻永远都只有一个任务处于运行态。
* 就绪态：处于就绪态的任务是那些已经准备就绪（这些任务没有被阻塞或者挂起 可以运行的任务，但是处于就绪态的任务还没有运行，因为有一个同优先级或者更高优先级的任务正在运行）
* 阻塞态：如果一个任务当前正在等待某个外部事件的话就说它处于阻塞态,比如说如果某个任务调用了函数 vTaskDelay()的话就会进入阻塞态， 直到延时周期完成。（任务在等待队列、信号量、事件组、通知或互斥信号量的时候也会进入阻塞态 。**任务进入阻塞态会有一个超时时间，当超过这个超时时间任务就会退出阻塞态，即使所等待的事件还没有来临**）
* 挂起态：任务进入挂起态以后也不能被调度器调用进入运行态，但是进入挂起态的任务没有超时时间。

需要注意的是，当任务从阻塞/挂起态退出时是进入的就绪态而非直接进入运行态。

## 优先级

* 优先级数字越**低**表示任务优先级越**低 （除使用Cortex内核的情况！！！！！！！！！）**

  例如当一个任务A正在运行，另外一个任务B(优先级高于A)阻塞时间到或者事件触发处于就绪态，那么B会从A那抢占处理器，B开始运行，A停止运行

  **※※※※※当使用cortex内核时相反（例如STM32)※※※※※**

* 时间片轮转

  当宏`configUSE_TIME_SLICING`定义为1 的时候多个任务可以共用一个优先级，数量不限。此时处于就绪态的优先级相同的任务就会使用时间片轮转调度器获取运行时间

## 任务堆栈

FreeRTOS之所以能正确的恢复一个任务的运行就是因为有任务堆栈在保驾护航，任务调度器在进行任务切换的时候会将当前任务的现场 (CPU寄存器值等 )保存在此任务的任务堆栈中，等到此任务下次运行的时候就会先用堆栈中保存的值来恢复现场 ，恢复现场以后任务就会接着从上次中断的地方开始运行。


# 2、机制

## 任务通知

任务通知来代替信号量、消息队列、事件标志组等这些东西。使用任务通知的话效率会更高。

## STM32支持的低功耗模式

* sleep睡眠模式

  在 SLEEP 模式下，只有内核停止了工作，而外设仍然在运行。
  在进入 SLEEP 模式后，所有中断（‘外部中断、串口中断、定时器中断等’）均可唤醒 MCU，从而退出 SLEEP 模式。

* stop停止模式

  在进入 STOP 模式后，此时 SYSTICK 也会被停掉，只有外部中断（EXTI）才能唤醒 MCU由于 RTC 中断挂在外部中断线上，所以 RTC 中断也能唤醒 MCU）

* standby待机模式

  在 STANDBY 模式下，内核、所有的时钟、以及后备 1.2V 电源全部停止工作。

  从 STANDBY 模式中唤醒后，系统相当于执行了一次复位操作，程序会从头来过。

## Tickless 模式

FreeRTOS系统提供的低功耗模式，当处理器进入空闲任务周期以后就关闭系统节拍中断(滴答定时器中断)，只有当其他中断发生或者其他任务需要处理的时侯处理器才会从低功耗模式中唤醒。

# 3、一些内核配置

* **configUSE_PREEMPTION**

  为1时RTOS使用抢占式调度器，为0时RTOS使用协作式调度器（时间片）。（协作式操作系统是任务主动释放CPU后，切换到下一个任务。任务切换的时机完全取决于正在运行的任务）

* **configUSE_PORT_OPTIMISED_TASK_SELECTION**

  某些运行FreeRTOS的硬件有两种方法选择下一个要执行的任务：通用方法和特定于硬件（硬件计算前导零指令）的方法

  * 设置为0 通用办法，可以用于所有FreeRTOS支持的硬件
  * 设置为1 硬件计算前导零指令，并非所有硬件都支持

* **configUSE_IDLE_HOOK**

  设置为1使用空闲钩子（Idle Hook类似于回调函数），0忽略空闲钩子。

  空闲任务钩子是一个函数，这个函数由用户来实现，RTOS规定了函数的名字和参数，这个函数在每个空闲任务周期都会被调用。

  ```c
  void vApplicationIdleHook(void);
  //这个钩子函数不可以调用会引起空闲任务阻塞的API函数（例如：vTaskDelay()、带有阻塞时间的队列和信号量函数
  ```

  使用空闲钩子函数设置CPU进入省电模式是很常见的。

* configUSE_MALLOC_FAILED_HOOK

  如果定义并正确配置malloc()失败钩子函数，则这个函数会在pvPortMalloc()函数返回NULL时被调用。只有FreeRTOS在响应内存分配请求时发现堆内存不足才会返回NULL

  ```c
  void vApplicationMallocFailedHook( void);
  ```

* **configUSE_TICK_HOOK**

  设置为1使用时间片钩子（Tick Hook），0忽略时间片钩子.

  时间片中断可以周期性的调用一个被称为钩子函数（回调函数）的应用程序。时间片钩子函数可以很方便的实现一个定时器功能。

  ```c
  void vApplicationTickHook( void );
  //vApplicationTickHook()函数在中断服务程序中执行，因此这个函数必须非常短小，不能大量使用堆栈，只能调用以”FromISR" 或 "FROM_ISR”结尾的API函数。
  ```

* 一些顾名思义的配置

  ```c
  #define configCPU_CLOCK_HZ				( SystemCoreClock ) //CPU频率
  #define configTICK_RATE_HZ				( ( TickType_t ) 1000 ) //时钟节拍频率，这里设置为1000，周期就是1ms
  #define configMAX_PRIORITIES			( 32 )  //可使用的最大优先级
  #define configMINIMAL_STACK_SIZE		( ( unsigned short ) 128 )//空闲任务使用的堆栈大小
  #define configTOTAL_HEAP_SIZE			( ( size_t ) ( 64 * 1024 ) )//系统所有总的堆大小
  #define configMAX_TASK_NAME_LEN			( 16 )  //任务名字字符串长度
  
  #define configUSE_16_BIT_TICKS			0   //系统节拍计数器变量数据类型，
                                              //1表示为16位无符号整形，0表示为32位无符号整形
  ```

* **configIDLE_SHOULD_YIELD**

  这个参数控制任务在空闲优先级中的行为。仅在满足下列条件后，才会起作用。

  * 使用抢占式内核调度（见第一点）
  * 用户任务使用空闲优先级

  使用同一优先级的多个任务，且该优先级大于空闲优先级时，这些任务反映在时间片上会获得相同的处理器时间。单当多个任务（不止空闲任务，还包括用户任务）共享空闲优先级时，如果configIDLE_SHOULD_YIELD为1，当用户任务运行时，空闲任务立刻让出CPU，但是空闲任务仍然会占据时间片中的一段时间，**就会导致空闲任务与接下来的用户任务会共享一个时间片，即该用户任务占有时间片少于正常分配的时间片**

  **设置configIDLE_SHOULD_YIELD为0将阻止空闲任务为用户任务让出CPU，直到空闲任务的时间片结束。这确保所有处在空闲优先级的任务分配到相同多的处理器时间，但是，这是以分配给空闲任务更高比例的处理器时间为代价的。**

* **configUSE_TASK_NOTIFICATIONS**（很有用）

  置1将会开启任务通知功能，每个RTOS任务具有一个32位的通知值，RTOS任务通知相当于直接向任务发送一个事件，接收到通知的任务可以解除任务的阻塞状态（因等待任务通知而进入阻塞状态）。相对于以前必须分别创建队列、二进制信号量、计数信号量或事件组的情况，使用任务通知显然更灵活。更好的是，相比于使用信号量解除任务阻塞，使用任务通知可以快45%

* **configGENERATE_RUN_TIME_STATS**

  设置宏configGENERATE_RUN_TIME_STATS为1使能运行时间统计功能。一旦设置为1，则下面两个宏必须被定义：

  * portCONFIGURE_TIMER_FOR_RUN_TIME_STATS()

    使用一个比运行时间更精准的基准定时器使统计更加精确，基准定时器中断频率要比统节拍中断快10~100倍。基准定时器中断频率越快，统计越精准，但能统计的运行时间也越短（比如，基准定时器10ms中断一次，8位无符号整形变量可以计到2.55秒，但如果是1秒中断一次，8位无符号整形变量可以统计到255秒）

  * portGET_RUN_TIME_COUNTER_VALUE()

    返回基准时钟的值以供计数（在定时器中使使长整形变量ulHighFrequencyTimerTicks自增）

  当我们配置了一个定时器中断且要使用时间统计时，需要在config.h中添加

  ```c
      extern volatile unsigned long ulHighFrequencyTimerTicks;
      #define portCONFIGURE_TIMER_FOR_RUN_TIME_STATS() ( ulHighFrequencyTimerTicks = 0UL )
      #define portGET_RUN_TIME_COUNTER_VALUE() ulHighFrequencyTimerTicks
  ```

  ### C语言中的0UL和1UL

  * 0UL ：无符号长整型0
  * 1UL ：无符号长整型1

* **LIBRARY_LOWEST_INTERRUPT_PRIORITY** & **LIBRARY_MAX_SYSCALL_INTERRUPT_PRIORITY**  

  由于在使用cortex内核的硬件设备中优先级数值为越小，逻辑优先级越高，所以这里的最小优先级即为硬件的最低逻辑优先级，表现在数值上即为最大数值；而此处的configMAX_SYSCALL_INTERRUPT_PRIORITY是用来设置可以在中断服务程序中安全调用FreeRTOS API函数的最高中断优先级。优先级小于等于这个宏所代表的优先级时，程序可以在中断服务程序中安全的调用FreeRTOS API函数；如果优先级大于这个宏所代表的优先级，表示FreeRTOS无法禁止这个中断，在这个中断服务程序中绝不可以调用任何API函数。

  {% asset_img freertos_1.jpg This is an image %} 

  运行在大于configMAX_SYSCALL_INTERRUPT_PRIORITY的优先级中断是不会被RTOS内核所屏蔽的，因此也不受RTOS内核功能影响。这主要用于非常高的实时需求中。比如执行电机转向。但是，这类中断的中断服务例程中绝不可以调用FreeRTOS的API函数。

  # 4.一些报错

  configASSERT( ( portNVIC_INT_CTRL_REG & portVECTACTIVE_MASK ) == 0 )报错。

  此处报错，第一种是因为高于configMAX_SYSCALL_INTERRUPT_PRIORITY优先级的中断调用了RTOS的API导致的，解决办法，将中断优先级调低，比configMAX_SYSCALL_INTERRUPT_PRIORITY要低，就可以调用RTOS的API了。

  第二种是因为中断发送消息队列，发送信号量等操作使用了不带ISR结尾的API，而是调用了普通不带ISR的API导致的，解决办法，将API替换为带ISR结尾的API便可以解决问题。
  
