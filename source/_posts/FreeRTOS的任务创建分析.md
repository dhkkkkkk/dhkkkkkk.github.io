---
title: FreeRTOS的任务创建分析
date: 2022-12-12 15:33:00
tags: 嵌入式学习
---

本文的学习参考了[FreeRTOS高级篇2---FreeRTOS任务创建分析_研究是为了理解的博客-CSDN博客_](https://freertos.blog.csdn.net/article/details/51303639)

# 任务TCB分析

在[基于FreeRTOS的列表学习 | 小董的BLOG (gitee.io)](https://dhkkk.gitee.io/2022/12/09/基于FreeRTOS的列表学习/)中对列表项分析时，在列表项的结构中有一个无类型指针：

```c
void * pvOwner;                                     /*指向一个任务TCB*/
```

该指针用于指向任务的TCB（Task Control Block）。顾名思义，该结构作为一个用于控制任务的集合。它用于存储任务的状态信息，包括任务运行时的环境。**每个任务都有自己的任务TCB。**

## 结构解读

可以在task.c中找到定义TCB的结构体：

```c
typedef struct tskTaskControlBlock
{
	volatile StackType_t	*pxTopOfStack;		//栈顶指针

	#if ( portUSING_MPU_WRAPPERS == 1 )		//是否使用MPU
		xMPU_SETTINGS	xMPUSettings;		
	#endif

	ListItem_t			xStateListItem;	//状态列表项
	ListItem_t			xEventListItem;	//事件列表项	
	UBaseType_t			uxPriority;			//优先级
	StackType_t			*pxStack;			//指向堆栈起始位置
	char				pcTaskName[ configMAX_TASK_NAME_LEN ];		//任务名字最大长度

	#if ( portSTACK_GROWTH > 0 )		//判断堆栈生长方向,>0为向上生长
		StackType_t		*pxEndOfStack;		 //需要额外的一个指针来判断堆栈是否溢出
	#endif

	#if ( portCRITICAL_NESTING_IN_TCB == 1 )
		UBaseType_t		uxCriticalNesting;	
	#endif

	#if ( configUSE_TRACE_FACILITY == 1 )
		UBaseType_t		uxTCBNumber;		//一个数值，每个任务都有一个唯一的值
		UBaseType_t		uxTaskNumber;		//存储一个特定数值
	#endif

	#if ( configUSE_MUTEXES == 1 )	//是否使用互斥量
		UBaseType_t		uxBasePriority;		//用于保存任务的基础的基础优先级
		UBaseType_t		uxMutexesHeld;
	#endif

	#if ( configUSE_APPLICATION_TASK_TAG == 1 )
		TaskHookFunction_t pxTaskTag;
	#endif

	#if( configNUM_THREAD_LOCAL_STORAGE_POINTERS > 0 )
		void *pvThreadLocalStoragePointers[ configNUM_THREAD_LOCAL_STORAGE_POINTERS ];
	#endif

	#if( configGENERATE_RUN_TIME_STATS == 1 )
		uint32_t		ulRunTimeCounter;		//记录任务在运行状态下执行的总时间
	#endif

	#if ( configUSE_NEWLIB_REENTRANT == 1 )
    /* 为任务分配一个Newlibreent结构体变量。Newlib是一个C库函数，并非FreeRTOS维护，FreeRTOS也不对使用结果负责。如果用户使用Newlib，必须熟知Newlib的细节*/
		struct	_reent xNewLib_reent;
	#endif

	#if( configUSE_TASK_NOTIFICATIONS == 1 ) 		//与任务通知相关
		volatile uint32_t ulNotifiedValue;
		volatile uint8_t ucNotifyState;
	#endif

	#if( tskSTATIC_AND_DYNAMIC_ALLOCATION_POSSIBLE != 0 )
		uint8_t	ucStaticallyAllocated; 		
	#endif

	#if( INCLUDE_xTaskAbortDelay == 1 )
		uint8_t ucDelayAborted;
	#endif

} tskTCB;
typedef tskTCB TCB_t;
```

### 状态列表项与事件列表项

从TCB的结构中可以发现两个定义的列表项，分别是**状态列表项和事件列表项**，在上一章也略微提到过，列表与列表项用于调度任务、跟踪任务状态。例如：在task.c中，定义了一些静态列表变量，其中有就绪、阻塞、挂起列表，例如当某个任务处于就绪态时，调度器就将这个任务TCB的**状态列表项**挂接到就绪列表。事件列表项也与之类似，当队列满的情况下，任务因入队操作而阻塞时，就会将**事件列表项**挂接到队列的等待入队列表上。具体实现方法和规则会在之后学习任务调度时讲到。

### uxTCBNumber与uxTaskNumber

这两个变量主要用于调试与可视化追踪。仅当宏configUSE_TRACE_FACILITY（位于FreeRTOSConfig.h中）为1时有效。变量uxTCBNumber在创建任务时**由内核自动分配**，每个任务对应一个值，**后续不再改变**。变量uxTaskNumber由**API函数vTaskSetTaskNumber()来设置的**，数值由函数参数指定。

其定义在task.c中如下，参数为对应任务的引用句柄和要设置的数值

```c
void vTaskSetTaskNumber( TaskHandle_t xTask, const UBaseType_t uxHandle )
{
	TCB_t *pxTCB;

		if( xTask != NULL )
		{
			pxTCB = ( TCB_t * ) xTask;
			pxTCB->uxTaskNumber = uxHandle;
		}
}
```

# 任务创建的内部流程

假设当前创建了一个任务如下：

```c
static TaskHandle_t xHandle；
xTaskCreate(vTask_A,”Task A”,120,NULL,1,&xHandle);
```

当这个语句执行后，任务A被创建并加入就绪任务列表，我们这章的主要目的，就是看看这个语句在执行过程中，发生了什么事情。

## 1、创建任务堆栈和TCB所用空间

注：原博客中也就是接下来讲到的函数在我使用的基于STM32的FreeRTOS中并不存在，但是可以在对xTaskCreate的定义下找到基本类似的操作（可能是版本原因）**，不过其并不是用函数封装的，但是功能基本一致**。为了方便学习就使用原博客中的函数来理解。

```c
static TCB_t *prvAllocateTCBAndStack( const uint16_t usStackDepth, StackType_t * const puxStackBuffer, TCB_t * const pxTaskBuffer )
{
TCB_t *pxNewTCB;
StackType_t *pxStack;
 
    /* 分配堆栈空间*/
    pxStack = ( StackType_t * ) pvPortMallocAligned( ( ( ( size_t ) usStackDepth ) * sizeof( StackType_t ) ), puxStackBuffer );
    if( pxStack != NULL )
    {
        /* 分配TCB空间 */
        pxNewTCB = ( TCB_t * ) pvPortMallocAligned( sizeof( TCB_t ), pxTaskBuffer );
 
        if( pxNewTCB != NULL )
        {
            /* 将堆栈起始位置存入TCB*/
            pxNewTCB->pxStack = pxStack;
        }
        else
        {
            /* 如果TCB分配失败，释放之前申请的堆栈空间 */
            if( puxStackBuffer == NULL )
            {
                vPortFree( pxStack );
            }
        }
    }
    else
    {
        pxNewTCB = NULL;
    }
 
    if( pxNewTCB != NULL )
    {
        /* 如果需要，使用固定值填充堆栈 */
        #if( ( configCHECK_FOR_STACK_OVERFLOW> 1 ) || ( configUSE_TRACE_FACILITY == 1 ) || ( INCLUDE_uxTaskGetStackHighWaterMark== 1 ) )
        {
            /* 仅用于调试 */
            ( void ) memset( pxNewTCB->pxStack, ( int ) tskSTACK_FILL_BYTE, ( size_t ) usStackDepth * sizeof( StackType_t ) );
        }
        #endif
    }
 
    return pxNewTCB;
}
```

基本就是一些分配内存的逻辑与操作，还是比较好理解的。

## 2、初始化TCB必要的字段

这些操作在我的版本中都是用预编译判断执行的，没有使用函数封装，要找到对应操作实在是有点眼花，姑且看到大部分操作都是在task.c中对xTaskCreate的定义的后面，这里为了**方便理解**还是使用原博主的函数：

```c
static void prvInitialiseTCBVariables( 
TCB_t * const pxTCB,
const char * const pcName, 
UBaseType_t uxPriority,   
const MemoryRegion_t * const xRegions, 
const uint16_t usStackDepth 
)
{
	UBaseType_t x;
 
    /* 将任务描述存入TCB */
    for( x = ( UBaseType_t ) 0; x < ( UBaseType_t ) configMAX_TASK_NAME_LEN; x++ )
    {
        pxTCB->pcTaskName[ x ] = pcName[ x ];
        if( pcName[ x ] == 0x00 )
        {
            break;
        }
    }
    /* 确保字符串有结束 */
    pxTCB->pcTaskName[ configMAX_TASK_NAME_LEN - 1 ] = '\0';
 
    /* 调整优先级，宏configMAX_PRIORITIES的值在FreeRTOSConfig.h中设置 */
    if( uxPriority >= ( UBaseType_t ) configMAX_PRIORITIES )
    {
        uxPriority = ( UBaseType_t ) configMAX_PRIORITIES - ( UBaseType_t ) 1U;
    }
 
    pxTCB->uxPriority = uxPriority;
    #if ( configUSE_MUTEXES == 1 )              /*使用互斥量*/
    {  
        pxTCB->uxBasePriority = uxPriority;
        pxTCB->uxMutexesHeld = 0;
    }
    #endif /* configUSE_MUTEXES */
   
    /*初始化列表项*/
    vListInitialiseItem( &( pxTCB->xStateListItem ) );
    vListInitialiseItem( &( pxTCB->xEventListItem ) );
 
    /* 设置列表项xStateListItem的成员pvOwner指向当前任务控制块 */
    listSET_LIST_ITEM_OWNER( &( pxTCB->xStateListItem ), pxTCB );
 
    /* 设置列表项xEventListItem的成员xItemValue*/
    listSET_LIST_ITEM_VALUE( &( pxTCB->xEventListItem ), ( TickType_t ) configMAX_PRIORITIES - ( TickType_t ) uxPriority );
    /* 设置列表项xEventListItem的成员pvOwner指向当前任务控制块 */
    listSET_LIST_ITEM_OWNER( &( pxTCB->xEventListItem ), pxTCB );
 
    #if ( portCRITICAL_NESTING_IN_TCB ==1 )    /*使能临界区嵌套功能*/
    {  
        pxTCB->uxCriticalNesting = ( UBaseType_t ) 0U;
    }
    #endif /* portCRITICAL_NESTING_IN_TCB */
 
    #if ( configUSE_APPLICATION_TASK_TAG == 1 ) /*使能任务标签功能*/
    {  
        pxTCB->pxTaskTag = NULL;
    }
    #endif /* configUSE_APPLICATION_TASK_TAG */
 
    #if ( configGENERATE_RUN_TIME_STATS == 1 )  /*使能事件统计功能*/
    {
        pxTCB->ulRunTimeCounter = 0UL;
    }
    #endif /* configGENERATE_RUN_TIME_STATS */
 
    #if ( portUSING_MPU_WRAPPERS == 1 )         /*使用MPU功能*/
    {
        vPortStoreTaskMPUSettings( &( pxTCB->xMPUSettings ), xRegions, pxTCB->pxStack, usStackDepth );
    }
    #else /* portUSING_MPU_WRAPPERS */
    {
        ( void ) xRegions;
        ( void ) usStackDepth;
    }
    #endif /* portUSING_MPU_WRAPPERS */
 
    #if( configNUM_THREAD_LOCAL_STORAGE_POINTERS != 0 )/*使能线程本地存储指针*/
    {
        for( x = 0; x < ( UBaseType_t )configNUM_THREAD_LOCAL_STORAGE_POINTERS; x++ )
        {
            pxTCB->pvThreadLocalStoragePointers[ x ] = NULL;
        }
    }
    #endif
 
    #if ( configUSE_TASK_NOTIFICATIONS == 1 )   /*使能任务通知功能*/
    {
        pxTCB->ulNotifiedValue = 0;
        pxTCB->ucNotifyState = taskNOT_WAITING_NOTIFICATION;
    }
    #endif
 
    #if ( configUSE_NEWLIB_REENTRANT == 1 )     /*使用Newlib*/
    {
        _REENT_INIT_PTR( ( &( pxTCB->xNewLib_reent ) ) );
    }
    #endif
 
    #if( INCLUDE_xTaskAbortDelay == 1 )
    {
        pxTCB->ucDelayAborted = pdFALSE;
    }
    #endif
}
```

其实大部分也就是对应TCB结构体中各个成员的初始化，其中有几点需要着重讲一下：

### 对两个列表项的初始化：

* 对状态列表项的初始化

  使成员pvOwner指向对应的任务TCB，其余成员不对其操作
  
* **对事件列表项的初始化**

  之前讲到过事件列表项值会存储任务的优先级相关信息，其存储方式比较特别：

  ```c
  /* 设置列表项xEventListItem的成员xItemValue*/
  listSET_LIST_ITEM_VALUE( 
  &( pxTCB->xEventListItem ), 
  ( TickType_t ) configMAX_PRIORITIES - ( TickType_t ) uxPriority );
  ```

  可以看出，事件列表项的xItemValue中存储的值为**<u>用户设置的最大优先级-当前任务的优先级</u>**，也就是说，ItemValue的值越小，当前任务的优先级**越高**

  再回想一下列表项插入列表的函数vListInsert()：

  {% asset_img 15.jpg This is an image %} 

  可以发现xItemValue越小的值越是排在列表的前面，这也就解释了这样做的原因，**其实也就是优先级越高的任务越在列表前面**

### 任务通知

```c
    #if ( configUSE_TASK_NOTIFICATIONS == 1 )   /*使能任务通知功能*/
    {
        pxTCB->ulNotifiedValue = 0;
        pxTCB->ucNotifyState = taskNOT_WAITING_NOTIFICATION;
```

在[FreeRTOS任务间的交互方法 | 小董的BLOG (gitee.io)](https://dhkkk.gitee.io/2022/08/30/freertos任务通知与信号量/)中将任务通知的流程时也使用到了使用该值的函数：

```c
BaseType_t xTaskNotifyWait( uint32_tulBits ToClearOnEntry,
									 uint32_tulBits ToClearOnExit,
									 uint32_t* pulNotificationValue,
									 TickType_t xTicksToWait );
```

其实pulNotificationValue就是拷贝自TCB中的ulNotifiedValue值，该值在任务通知发送函数将ulValue的值拷贝进ulNotifiedValue，在接收时用户可以通过pulNotificationValue判断任务通知发送的值。在初始化时，其为0，因此发送的任务值不能为0

## 3、初始化任务堆栈

调用函数pxPortInitialiseStack()初始化任务堆栈，并将最新的栈顶指针赋值给任务TCB的pxTopOfStack字段。

具体与寄存器的调用有关，目前不深读。

## 4、进入临界区

 调用taskENTER_CRITICAL()进入临界区，进入临界区后的代码段将不能被打断，比如有的外设的初始化需要严格的时序，初始化过程中不能被打断。FreeRTOS 在进入临界区代码的时候需要关闭中断，当处理完临界区代码以后再打开中断。

FreeRTOS 与 临界区代 码 保 护 有 关 的 函 数 有 4 个 ： taskENTER_CRITICAL() 、taskEXIT_CRITICAL() 、taskENTER_CRITICAL_FROM_ISR() 和taskEXIT_CRITICAL_FROM_ISR()，这四个函数其实是宏定义，在 task.h 文件中有定义。 这四个函数的区别是**前两个是任务级的**临界段代码保护，**后两个是中断级**的临界段代码保护。

## 5、改变跟踪任务的变量

在task.c中定义了一些静态私有变量用于跟踪任务的数量、状态等信息，每当有任务发生变化时这些对应的变量也会发生变化。

其中变量uxCurrentNumberOfTasks表示当前任务的总数量，每创建一个任务，这个变量都会增加1。

## 6、第一次运行的必要初始化

如果当前创建的任务为第一个任务，则会调用函数prvInitialiseTaskLists()对列表进行**初始化**，在这之前，task.c中定义了这些静态类型的列表变量：(List_t类型变量在上一章已经讲过)

```c
PRIVILEGED_DATAstatic List_t pxReadyTasksLists[ configMAX_PRIORITIES ];/*按照优先级排序的就绪态任务*/
PRIVILEGED_DATAstatic List_t xDelayedTaskList1;                    /*延时的任务 */
PRIVILEGED_DATAstatic List_t xDelayedTaskList2;                    /*延时的任务 */
PRIVILEGED_DATAstatic List_t xPendingReadyList;             /*任务已就绪,但调度器被挂起 */
 
#if (INCLUDE_vTaskDelete == 1 )
    PRIVILEGED_DATA static List_t xTasksWaitingTermination;/*任务已经被删除,但内存尚未释放*/
#endif
 
#if (INCLUDE_vTaskSuspend == 1 )
    PRIVILEGED_DATA static List_t xSuspendedTaskList;            /*当前挂起的任务*/
#endif
```

初始化：

```c
static void prvInitialiseTaskLists( void )
{
UBaseType_tuxPriority;
 
    for( uxPriority = ( UBaseType_t ) 0U; uxPriority < ( UBaseType_t ) configMAX_PRIORITIES; uxPriority++ )
    {
        vListInitialise( &( pxReadyTasksLists[ uxPriority ] ) );
    }
 
    vListInitialise( &xDelayedTaskList1 );
    vListInitialise( &xDelayedTaskList2 );
    vListInitialise( &xPendingReadyList );
 
    #if ( INCLUDE_vTaskDelete == 1 )
    {
        vListInitialise( &xTasksWaitingTermination );
    }
    #endif /* INCLUDE_vTaskDelete */
 
    #if ( INCLUDE_vTaskSuspend == 1 )
    {
        vListInitialise( &xSuspendedTaskList );
    }
    #endif /* INCLUDE_vTaskSuspend */
 
    /* Start with pxDelayedTaskList using list1 and the pxOverflowDelayedTaskListusing list2. */
    pxDelayedTaskList = &xDelayedTaskList1;
    pxOverflowDelayedTaskList = &xDelayedTaskList2;
}
```

其中主要调用的函数vListInitialise()在讲列表时已经讲过，其作用就是初始化列表成员的一些指针的初始指向，为后续接入列表项做准备。

## 7、更新任务TCB指针

tasks.c中定义了一个任务TCB指针型变量：

* PRIVILEGED_DATA TCB_t * volatile pxCurrentTCB= NULL;

 这是一个全局变量，在tasks.c中只定义了这一个全局变量。这个变量用来指向当前正在运行的任务TCB。

当调度器还没有开启时（程序刚开始运行时，可能会先创建几个任务，之后才会启动调度器），如果新创建的任务优先级大于pxCurrentTCB指向的任务优先级，则设置pxCurrentTCB指向当前新创建的任务TCB。

```c
if( xSchedulerRunning == pdFALSE )
{
    if( pxCurrentTCB->uxPriority <= uxPriority )
    {
        pxCurrentTCB = pxNewTCB;
    }
    else
    {
        mtCOVERAGE_TEST_MARKER();
    }
}
```

这项操作可以**确保pxCurrentTCB始终指向优先级最高的就绪任务。**

## 8、将新创建的任务加入就绪列表数组

新创建的任务会被直接加入到就绪列表中，因为它不会一被创建就被阻塞或挂起。观察在task.c中定义的几个静态类型的列表可以发现就绪列表是一个结构体数组：

```c
PRIVILEGED_DATAstatic List_t pxReadyTasksLists[ configMAX_PRIORITIES ];
/*按照优先级排序的就绪态任务*/
```

因此当前新创建的任务会根据该任务TCB中存储的优先级**决定存入的列表下标**。比如我们新创建的任务优先级为1，则这个任务被加入到列表pxReadyTasksLists[1]中。

并且由于同一个列表中任务的优先级是相同的，因此每次加入列表时会直接将当前任务放入列表尾：

```c
#define prvAddTaskToReadyList( pxTCB )                        
    taskRECORD_READY_PRIORITY( ( pxTCB)->uxPriority );       
    vListInsertEnd( &( pxReadyTasksLists[ (pxTCB )->uxPriority ] ), 
                    &( ( pxTCB )->xStateListItem ) );
```

调用一个宏将任务放入列表。可以看到，加入列表的为任务的**状态列表项**，但最终调用宏的**参数直接是任务TCB**，因此可能用户不用关心何时操作状态列表项，何时操作事件列表项？（我的猜测，因为目前还不太清楚二者的明显区别）

## 9、退出临界区

到此，一个任务的创建基本结束，也就可以退出临界区了（本身进入临界区的原因就是让任务的创建过程不被打断）

## 10、新任务进入调度

如果调度器已开启，新任务便会根据其优先级由调度器进行相关调度。

# 任务创建过程总结

* 为当前任务**分配内存**创建堆栈，为任务TCB创建空间
* **初始化TCB**，主要是使TCB成员中的两个列表项指向当前TCB和改变**事件列表项的列表项值**
* **初始化**任务的**堆栈**
* ***进入临界区，之后的进程不能被打断***
* 改变一些存储任务信息的**变量**（主要是全局的信息，并不是指某一个任务的信息。比如任务的总数量等）
* 如果当前为第一个任务，则**初始化各列表**
* **更新**当前的任务TCB指针（该指针专用于指向当前TCB）
* 将新任务加入就绪列表数组中，**根据优先级决定加入的数组下标**
* ***退出临界区***
* 新任务也加入调度过程中