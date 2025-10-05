---
title: FreeRTOS任务延时分析
date: 2022-12-14 10:49:06
tags: 嵌入式学习
---

前几章中都有涉及FreeRTOS的列表相关内容，在学习过程中，我们也能看出来列表与任务的状态（就绪、延时、堵塞）与任务调度息息相关，本文就针对任务延时进行**列表运行机制**的进一步学习

参考：[FreeRTOS高级篇9---FreeRTOS系统延时分析_研究是为了理解的博客-CSDN博客_pdms_to_ticks](https://freertos.blog.csdn.net/article/details/51705148)

# 相对延时

|                    函数名                    |     参数     | 返回值 |
| :------------------------------------------: | :----------: | :----: |
| vTaskDelay( const TickType_t xTicksToDelay ) | 系统节拍个数 |   无   |

再次提醒一下，FreeRTOS中的延时参数都是以FreeRTOS**自身的系统节拍**为单位，系统节拍可以在在配置文件FreeRTOSConfig.h，改变宏configTICK_RATE_HZ的值；cubemx中的配置项也有相关的配置处，单位为Hz

## 函数体

```c
void vTaskDelay( const TickType_t xTicksToDelay )
{
BaseType_t xAlreadyYielded = pdFALSE;
 
    /* 如果延时时间为0,则不会将当前任务加入延时列表 */
    if( xTicksToDelay > ( TickType_t ) 0U )
    {
        configASSERT( uxSchedulerSuspended == 0 );//禁用中断
        vTaskSuspendAll();//将调度器挂起
        {
        /* 将当前任务从就绪列表中移除,并根据当前系统节拍计数器值计算唤醒时间,然后将任务加入延时列表 */
            prvAddCurrentTaskToDelayedList( xTicksToDelay, pdFALSE );
        }
        xAlreadyYielded = xTaskResumeAll();//恢复调度器运行
    }
 
    /* 强制执行一次上下文切换*/
    if( xAlreadyYielded == pdFALSE )
    {
        portYIELD_WITHIN_API();
    }
}
```

其中，关键函数为prvAddCurrentTaskToDelayedList()：

```c
static void prvAddCurrentTaskToDelayedList( TickType_t xTicksToWait, const BaseType_t xCanBlockIndefinitely )
{
TickType_t xTimeToWake;
const TickType_t xConstTickCount = xTickCount;

	#if( INCLUDE_xTaskAbortDelay == 1 )
	{
		/* 延时相关标志位清零 */
		pxCurrentTCB->ucDelayAborted = pdFALSE;
	}
	#endif

	/* 将列表项移出当前列表（就绪列表），pxCurrentTCB作为全局变量指向当前任务TCB，uxListRemove()返回列表的成员个数 */
	if( uxListRemove( &( pxCurrentTCB->xStateListItem ) ) == ( UBaseType_t ) 0 )
	{
		/* The current task must be in a ready list, so there is no need to
		check, and the port reset macro can be called directly. */
		portRESET_READY_PRIORITY( pxCurrentTCB->uxPriority, uxTopReadyPriority );
	}
	else
	{
		mtCOVERAGE_TEST_MARKER();
	}

	#if ( INCLUDE_vTaskSuspend == 1 )
	{
		if( ( xTicksToWait == portMAX_DELAY ) && ( xCanBlockIndefinitely != pdFALSE ) )
		{
			/*如果设置的延时参数等于portMAX_DELAY(0xffffffff)，则视为堵塞而不是延时 */
			vListInsertEnd( &xSuspendedTaskList, &( pxCurrentTCB->xStateListItem ) );
		}
		else
		{
			/* 计算唤醒时间 */
			xTimeToWake = xConstTickCount + xTicksToWait;

			/* 将当前任务TCB的状态列表项值设置为唤醒时间 */
			listSET_LIST_ITEM_VALUE( &( pxCurrentTCB->xStateListItem ), xTimeToWake );

			if( xTimeToWake < xConstTickCount )
			{
				/* Wake time has overflowed.  Place this item in the overflow
				list. */
				vListInsert( pxOverflowDelayedTaskList, &( pxCurrentTCB->xStateListItem ) );
			}
			else
			{
				/* 使用的是插入函数，因此延时列表中列表项会根据唤醒时间的大小进行排序 */
				vListInsert( pxDelayedTaskList, &( pxCurrentTCB->xStateListItem ) );

				/* If the task entering the blocked state was placed at the
				head of the list of blocked tasks then xNextTaskUnblockTime
				needs to be updated too. */
				if( xTimeToWake < xNextTaskUnblockTime )
				{
					xNextTaskUnblockTime = xTimeToWake;
				}
				else
				{
					mtCOVERAGE_TEST_MARKER();
				}
			}
		}
	}
   
   
		/* Avoid compiler warning when INCLUDE_vTaskSuspend is not 1. */
		( void ) xCanBlockIndefinitely;
	}
	#endif /* INCLUDE_vTaskSuspend */
}
```

通过对上述代码解释了列表的具体使用方法：

* 若要将当前任务延时或挂起，可以通过全局变量pxCurrentTCB找到该任务TCB，再操作该TCB的**状态**列表项

* 当任务延时或挂起时，状态列表项的列表项值xItemValue会用于存放唤醒（延时结束）的时间，由于列表插入函数会自动根据xItemValue的值进行排序，**所以唤醒时间越短的列表项会越排在列表前面**

* FreeRTOS使用了两个延时列表：xDelayedTaskList1和xDelayedTaskList2，并使用两个列表指针类型变量pxDelayedTaskList和pxOverflowDelayedTaskList分别指向上面的延时列表1和延时列表2

  pxOverflowDelayedTaskList用于解决计时溢出的问题：

  tasks.c中定义了很多局部静态变量，其中有一个变量xTickCount用于记录系统节拍中断次数（可以理解为记录当前时间），当xTicksToDelay达到4294967295后再增加，就会溢出变成0。 如果内核判断出xTickCount+ xTicksToDelay溢出，就将当前任务挂接到列表指针pxOverflowDelayedTaskList指向的列表中

## 相对延时的缺点

由于相对延时是在每次调用vTaskDelay之后才开启延时，如果调用延时的任务在运行过程中发生中断，那通过相对延时进行的周期任务就会因为该中断而被影响导致不能发生中断。

因此，为了能最大程度的使相对延时达到精准的效果，最好将使用相对延时的任务优先级设置为最高

# 绝对延时

|                            函数名                            |                参数                | 返回值 |
| :----------------------------------------------------------: | :--------------------------------: | :----: |
| vTaskDelayUntil( TickType_t * const pxPreviousWakeTime, <br>                   const TickType_t xTimeIncrement ) | 上一次记录的时间；<br>周期循环时间 |   无   |

## 函数体

```c
void vTaskDelayUntil( TickType_t * const pxPreviousWakeTime, const TickType_t xTimeIncrement )
{
TickType_t xTimeToWake;
BaseType_t xAlreadyYielded, xShouldDelay = pdFALSE;
 
    vTaskSuspendAll();
    {
        /* 保存系统节拍中断次数计数器 */
        const TickType_t xConstTickCount = xTickCount;
 
        /* 计算任务下次唤醒时间(以系统节拍中断次数表示)   */
        xTimeToWake = *pxPreviousWakeTime + xTimeIncrement;
        
        /* *pxPreviousWakeTime中保存的是上次唤醒时间,唤醒后需要一定时间执行任务主体代码,如果上次唤醒时间大于当前时间,说明节拍计数器溢出了 */
        if( xConstTickCount < *pxPreviousWakeTime )
        {
            /*只有当周期性延时时间大于任务主体代码执行时间,才会将任务挂接到延时列表.*/
            if( ( xTimeToWake < *pxPreviousWakeTime ) && ( xTimeToWake > xConstTickCount ) )
            {
                xShouldDelay = pdTRUE;
            }
        }
        else
        {
            /* 也都是保证周期性延时时间大于任务主体代码执行时间 */
            if( ( xTimeToWake < *pxPreviousWakeTime ) || ( xTimeToWake > xConstTickCount ) )
            {
                xShouldDelay = pdTRUE;
            }
        }
 
        /* 更新唤醒时间,为下一次调用本函数做准备. */
        *pxPreviousWakeTime = xTimeToWake;
 
        if( xShouldDelay != pdFALSE )
        {
            /* 将本任务加入延时列表,注意阻塞时间并不是以当前时间为参考,因此减去了当前系统节拍中断计数器值*/
            prvAddCurrentTaskToDelayedList( xTimeToWake - xConstTickCount, pdFALSE );
        }
    }
    xAlreadyYielded = xTaskResumeAll();
 
    /* 强制执行一次上下文切换 */
    if( xAlreadyYielded == pdFALSE )
    {
        portYIELD_WITHIN_API();
    }
}
```

写法和基本思路和相对延时区别不大，唯一需要注意的是。绝对延时之所以叫绝对延时，**是因为其第一个参数pxPreviousWakeTime除开第一次使用时需获取当前系统时间外，每次调用绝对延时函数时pxPreviousWakeTime内保存的值为上一次调用时计算出的唤醒时间**

也就是说，绝对延时的**每一次延时的起点都是上一次延时的终点**，这也就保证了任务主体即使被中断也不会影响到下一次执行任务主体（前提是任务主体+中断执行的时间小于延时长度，如果超过延时时间，需要重新获取pxPreviousWakeTime为当前系统的最新时间）

