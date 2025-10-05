---
title: 基于FreeRTOS的列表学习
date: 2022-12-09 20:03:34
tags: [数据结构,嵌入式学习]
---

*本文参考了[FreeRTOS高级篇1---FreeRTOS列表和列表项研究是为了理解的博客-CSDN博客freertos list](https://freertos.blog.csdn.net/article/details/51190095)进行学习*

# 列表与列表项

FreeRTOS内核为了调度任务之间的关系，大量地使用了列表和列表项的数据结果来跟踪任务当前的状态。例如，处于挂起、延时、就绪的任务都会被挂接**到各自的列表中**

FreeRTOS中使用指针实现列表的功能，类似于链式列表。**一个列表下有多个列表项**，且每个列表项中都有一个指针指向列表。

{% asset_img 1.jpg This is an image %} 

**需要注意的是，<u>列表</u>和之前讲到任务交互时使用的<u>队列</u>是两个东西！**

# FreeRTOS列表项的构成

## 全功能版列表项

我们可以在list.h中找到freertos对列表项的定义

```c
struct xLIST_ITEM
{
     listFIRST_LIST_ITEM_INTEGRITY_CHECK_VALUE           /*用于检测列表项数据是否完整*/
     configLIST_VOLATILE TickType_t xItemValue;           /*列表项值*/
     struct xLIST_ITEM * configLIST_VOLATILE pxNext;      /*指向列表中下一个列表项*/
     struct xLIST_ITEM * configLIST_VOLATILE pxPrevious;  /*指向列表中上一个列表项*/
     void * pvOwner;                                     /*指向一个任务TCB*/
     void * configLIST_VOLATILE pvContainer;             /*指向包含该列表项的列表 */
     listSECOND_LIST_ITEM_INTEGRITY_CHECK_VALUE          /*用于检测列表项数据是否完整*/
};
typedef struct xLIST_ITEM ListItem_t;
```

* listFIRST_LIST_ITEM_INTEGRITY_CHECK_VALUE 和listSECOND_LIST_ITEM_INTEGRITY_CHECK_VALUE 这两个宏定义用于**检查列表项的数据是否完整**。

  当使能projdefs.h中的宏configUSE_LIST_DATA_INTEGRITY_CHECK_BYTES时，这两个宏会被替换成两个已知的数值用于检测完整性。

* xItemValue为**列表项值**，通常是一个被跟踪的任务优先级或是一个调度事件的计数器值。

  例如在队列的使用中，如果一个任务因为在等待从队列中读取数据而进入阻塞态，则该任务的**事件列表项中的列表项值**会用于保存该任务的优先级有关信息；该任务的**状态列表项的列表项值**会用于保存阻塞时间的有关信息（之前讲过，当有多个任务等待读取队列时，读取顺序与优先级和阻塞时间相关）。至于什么是事件列表项和状态列表项，之后的文章会讲到。

  由于列表项值应是随时都可能更新的，因此该变量被configLIST_VOLATILE修饰，**该宏被映射为C的关键字volatile**。

* pxNext和pxPrevious为两个结构体指针，用于指向上下列表项，实现类似于双向链表的结构，对结构体指针还不太懂的话可以参考[数据结构_链表 | 小董的BLOG (gitee.io)](https://dhkkk.gitee.io/2022/11/18/数据结构-线性表/)中的一些解释。

* pvOwner用于指向一个任务的TCB（Task Control Block），之后的文章会讲到。

* pvContainer用于指向**包含该列表项的列表**。

## 迷你版列表项

list.h中还有一个迷你版列表项的定义：

```c
struct xMINI_LIST_ITEM
{
	listFIRST_LIST_ITEM_INTEGRITY_CHECK_VALUE			//检测完整性
	configLIST_VOLATILE TickType_t xItemValue;	//列表项值
	struct xLIST_ITEM * configLIST_VOLATILE pxNext;		//指向下一列表项
	struct xLIST_ITEM * configLIST_VOLATILE pxPrevious;	//指向前一列表项
};
typedef struct xMINI_LIST_ITEM MiniListItem_t;
```

迷你版列表项的作用下一节讲列表时会讲到。

## 初始化列表项

  列表项的初始比较简单，只要确保列表项不在任何列表中即可。

```c
void vListInitialiseItem( ListItem_t * const pxItem )
{
     pxItem->pvContainer = NULL;
 
     /*设置为已知值，用于检测列表项数据是否完整*/
     listSET_FIRST_LIST_ITEM_INTEGRITY_CHECK_VALUE(pxItem );
     listSET_SECOND_LIST_ITEM_INTEGRITY_CHECK_VALUE(pxItem );
}
```

# FreeRTOS的列表构成

## 列表的定义

```c
typedef struct xLIST
{
	listFIRST_LIST_INTEGRITY_CHECK_VALUE				//检测完整性
	configLIST_VOLATILE UBaseType_t uxNumberOfItems;
	ListItem_t * configLIST_VOLATILE pxIndex;			
	MiniListItem_t xListEnd;							
	listSECOND_LIST_INTEGRITY_CHECK_VALUE				//检测完整性
} List_t;
```

* uxNumberOfItems用于表示该列表中挂接的列表项数目，0表示列表为空。

* pxIndex为一个类型为完全体列表项的结构体指针，用于**遍历列表项。**

* 列表项xListEnd用于标记列表结束，**因为该列表项只用于标记列表的结束，因此不需要完全体列表项的所有功能**，因此使用了迷你版列表项的类型。

  xListEnd.xItemValue被**初始化为一个常数**，其值与硬件架构相关，为0xFFFF（16位架构）或0xFFFFFFFF（32位架构）。

## 列表的初始化

可以在list.c中找到初始化函数vListInitialise()

```c
void vListInitialise( List_t * const pxList )
{
     /*列表索引指向列表项xListEnd*/
     pxList->pxIndex = ( ListItem_t * )&( pxList->xListEnd );                  
     /* 设置为最大可能值为0xffff */
     pxList->xListEnd.xItemValue =portMAX_DELAY;
 
     /* 列表项xListEnd的pxNext和pxPrevious指针指向了它自己 */
     pxList->xListEnd.pxNext = (ListItem_t * ) &( pxList->xListEnd );
     pxList->xListEnd.pxPrevious= ( ListItem_t * ) &( pxList->xListEnd );
     pxList->uxNumberOfItems = ( UBaseType_t) 0U;
 
     /* 设置为已知值，用于检测列表数据是否完整*/
     listSET_LIST_INTEGRITY_CHECK_1_VALUE(pxList );
     listSET_LIST_INTEGRITY_CHECK_2_VALUE(pxList );
}
```

* 如果宏configUSE_LIST_DATA_INTEGRITY_CHECK_BYTES设置为1，则使能列表项数据完整性检查，则宏listSET_LIST_INTEGRITY_CHECK_1_VALUE()和listSET_LIST_INTEGRITY_CHECK_2_VALUE被一个已知值代替，默认为0x5a5a（16位架构）或者0x5a5a5a5a（32位架构）。

* 按照上述初始化后就应该是下图的情况：

  {% asset_img 2.jpg This is an image %} 

  此时pxIndex、pxList->xListEnd.pxNext、pxList->xListEnd.pxPrevious都指向了 pxList->xListEnd（也可以说是xListEnd这个结构体的首元素）

# 将列表项插入列表中

```c
void vListInsert( List_t * const pxList, ListItem_t * const pxNewListItem )
{
ListItem_t *pxIterator;		//函数内定义一个列表项指针，通过列表项值的比较，找到当前输入列表项需插入的位置
const TickType_t xValueOfInsertion = pxNewListItem->xItemValue;	//定义一个变量存放当前输入列表项的列表项值

	listTEST_LIST_INTEGRITY( pxList );		//检测完整性
	listTEST_LIST_ITEM_INTEGRITY( pxNewListItem );		//检测完整性

   /*开始为输入列表项寻找合适的插入位置*/
	if( xValueOfInsertion == portMAX_DELAY )
	{
		pxIterator = pxList->xListEnd.pxPrevious;
	}
	else
	{
		for( pxIterator = ( ListItem_t * ) &( pxList->xListEnd ); 
								  pxIterator->pxNext->xItemValue <= xValueOfInsertion; 
								  pxIterator = pxIterator->pxNext ) 
		{
			/* There is nothing to do here, just iterating to the wanted
			insertion position. */
		}
	}
   /******************************/
   
   /*此时pxIterator为当前pxNewListItem要插入的位置*/
	pxNewListItem->pxNext = pxIterator->pxNext;
	pxNewListItem->pxNext->pxPrevious = pxNewListItem;
	pxNewListItem->pxPrevious = pxIterator;
	pxIterator->pxNext = pxNewListItem;
   
	/*将列表项接入列表，列表项数目+1*/
	pxNewListItem->pvContainer = ( void * ) pxList;
	( pxList->uxNumberOfItems )++;
}
```

这个函数中比较难以理解的地方就是关于寻找列表项插入位置的判断和循环部分，接下来将一步一步讲解：

## 当列表内无列表项时（当前插入的列表项为第一个列表项）

假设该列表项值为50

### if判断

不满足条件，跳过

### for循环

* 初始化`pxIterator = ( ListItem_t * ) &( pxList->xListEnd );`后，此时情况为：

  {% asset_img 3.jpg This is an image %} 

  此时pxIterator成员与pxList->xListEnd的成员一一对应

* 条件判断`pxIterator->pxNext->xItemValue <= xValueOfInsertion;`

  这里需要注意：**列表初始化时，xListEnd的成员pxNext，pxPrevious是指向xListEnd的**

  因此pxIterator->pxNext->xItemValue的值就是0xFFFF，是大于50的，因此不满足条件，跳出循环

### 赋值

* **pxNewListItem->pxNext = pxIterator->pxNext;**

  pxIterator->pxNext指向pxList->xListEnd.pxnext，而pxList->xListEnd.pxnext又指向本身xListEnd；因此，<u>**最终pxNewListItem->pxNext指向了pxList->xListEnd**</u>

  {% asset_img 4.jpg cysgbj %} 

* **pxNewListItem->pxNext->pxPrevious = pxNewListItem;**

  上一步中pxNewListItem->pxNext指向了pxList->xListEnd，因此pxNewListItem->pxNext->pxPrevious等同于pxList->xListEnd.pxPrevious，而pxList->xListEnd.pxPrevious之前是指向pxList->xListEnd本身的，因此<u>**这一步最终的结果相当于将pxList->xListEnd.pxPrevious改变指向，指向pxNewListItem**</u>

  {% asset_img 5.jpg cysgbj %} 

* **pxNewListItem->pxPrevious = pxIterator;**

  由于pxIterator本身就是指向pxList->xListEnd的，因此这一步相当于：

  <u>**pxNewListItem->pxPrevious = pxList->xListEnd**</u>

  {% asset_img 6.jpg cysgbj %} 

* **pxIterator->pxNext = pxNewListItem;**

  pxIterator->pxNext指向pxList->xListEnd.pxnext，此时pxList->xListEnd.pxnext本来是指向自己（pxList->xListEnd）的，因此这一步最终是：

  **<u>将pxList->xListEnd.pxnext改变指向，指向pxNewListItem</u>**

  {% asset_img 7.jpg cysgbj %} 

* 综上，最终的结果为：

  {% asset_img 8.jpg cysgbj %} 

  *上图省略了pvContainer和pxIndex的指向和uxNumberOfItems的变化*

## 第二次，当列表中已有列表项时，且该列表项值大于上一个

假设第二个列表项值为60（第一个为50）

### if判断

不满足条件，跳过

### for循环

* 初始化`pxIterator = ( ListItem_t * ) &( pxList->xListEnd );`后，由于在上一个列表项插入后，pxList.xListEnd中的pxNext和pxPrevious已经改变了指向，因此当前情况为：

  {% asset_img 9.jpg cysgbj %} 

* 条件判断`pxIterator->pxNext->xItemValue <= xValueOfInsertion;`

  由上图可得知，此时pxIterator->pxNext->xItemValue为50，小于60，条件成立

* 执行`pxIterator = pxIterator->pxNext`

  由于pxIterator->pxNext指向pxNewListItem(1)，因此当前情况如下：

  {% asset_img 10.jpg cysgbj %} 

* 再次进行条件判断`pxIterator->pxNext->xItemValue <= xValueOfInsertion;`

  此时pxIterator->pxNext->xItemValue为0xffff，不满足条件，循环结束。

**综上，通过该for循环，可以根据每个列表项值的大小进行列表中的位置排序**

### 赋值

这里的所有pxNewListItem在未特别说明的情况下都是指pxNewListItem(2）

* **pxNewListItem->pxNext = pxIterator->pxNext;**

  同第一次插入列表，<u>**最终pxNewListItem->pxNext指向了pxList->xListEnd**</u>

* **pxNewListItem->pxNext->pxPrevious = pxNewListItem;**

  同第一次插入列表，<u>**将pxList->xListEnd.pxPrevious改变指向，指向pxNewListItem**</u>

* **pxNewListItem->pxPrevious = pxIterator;**

  pxIterator在for循环中已经指向了pxNewListItem(1)，因此这里：

  <u>**pxNewListItem->pxPrevious指向pxNewListItem(1)**</u>

* **pxIterator->pxNext = pxNewListItem;**

  pxIterator->pxNext指向pxNewListItem(1)->pxnext，而pxNewListItem(1)->next最终指向了pxlist.xListEnd，因此这里相当于:

  <u>**改变pxNewListItem(1)->pxnext的指向，使其指向pxNewListItem(2)**</u>

综上，当前结果为：

{% asset_img 11.jpg cysgbj %} 

有点乱，也可以参考下图，一样的

{% asset_img 12.jpg cysgbj %}

不难看出，赋值的最后两步将各个列表项相互链接

## 第二次，当列表中已有列表项时，但该列表项值小于上一个

假设第二个列表值为30，小于第一个列表值50

### if判断

不满足条件，跳过

### for循环

* 初始化`pxIterator = ( ListItem_t * ) &( pxList->xListEnd );`后，由于在上一个列表项插入后，pxList.xListEnd中的pxNext和pxPrevious已经改变了指向，因此当前情况为：

  {% asset_img 9.jpg cysgbj %} 

* 条件判断`pxIterator->pxNext->xItemValue <= xValueOfInsertion;`

  由上图可得知，此时pxIterator->pxNext->xItemValue为50，大于30，条件不成立，循环结束

### 赋值

这里的所有pxNewListItem在未特别说明的情况下都是指pxNewListItem(2）

* **pxNewListItem->pxNext = pxIterator->pxNext;**

  <u>**不同第一次插入列表！此时pxNewListItem->pxNext最终指向了xNewListItem(1)**</u>

* **pxNewListItem->pxNext->pxPrevious = pxNewListItem;**

  **不同第一次插入列表！**此时pxNewListItem->pxNext->pxPrevious 即是：

  <u>**pxNewListItem(1)->xPrevious指向了pxNewListItem(2)**</u>

* **pxNewListItem->pxPrevious = pxIterator;**

  pxIterator是直接指向pxList->xListEnd的，因此：

  <u>**pxNewListItem->pxPrevious 直接指向pxList->xListEnd**</u>

* **pxIterator->pxNext = pxNewListItem;**

  在执行这一步之前，情况如下：

  {% asset_img 13.jpg cysgbj %} 

  这里我产生了一个疑问，pxIterator->pxNext是指向pxNewListItem(1)，那让pxNewListItem(1)=pxNewListItem是什么意思呢？让上一个列表项指向当前列表项？这明显是不符合列表逻辑的。

  后来在找寻问题的过程中，才发现自己又犯了老错误，pxNewListItem(1)其实也是一个指针变量，他是直接指向列表项1的内存单元的，**因此pxIterator->pxNext是直接指向了列表项1的内存单元**，因此改变pxIterator->pxNext是直接改变了这个指针的指向，也就是说结果是：

  **<u>将pxList->xListEnd.pxNext改变指向，使其指向pxNewListItem(2)</u>**

综上，最后结果为：

{% asset_img 14.jpg cysgbj %} 

## 对比两种情况

第二个列表项值大于第一个的情况：

{% asset_img 11.jpg cysgbj %} 

对比一下，可以看出：

* **pxList->xListEnd.pxNext总是指向列表项值最小的列表项**
* **pxList->xListEnd.pxPrevious总是指向列表项值最大的列表项**
* **列表项值小的列表项的pxNext总是指向列表项值大的列表项**
* **列表项值大的列表项的PxPrevious总是指向列表项值小的列表项**
* **首列表项的pxPrevious和尾列表项的pxNext都指向pxList->xListEnd**

根据上述的规律，就可以清晰地明白FreeRTOS对列表的插入方式了：

{% asset_img 15.jpg cysgbj %} 

因此，若要访问列表项，要通过pxList->xListEnd.pxNext访问手列表项，通过pxList->xListEnd.pxPrevious访问尾列表项。

实现插入的方法看了之后感觉真的很巧妙，我感觉也只能停留在理解的阶段，要自己写出来感觉难度还是很大。。。**但是规律还是很好记住的😋**

# 直接将列表项插入列表尾

原理跟上面的是一样的，如果能理解插入的方法，相信这个理解起来就很ez了，就不多说了

```c
void vListInsertEnd( List_t * const pxList, ListItem_t * const pxNewListItem )
{
ListItem_t* const pxIndex = pxList->pxIndex;
 
         /*检查列表和列表项数据的完整性，仅当configASSERT()定义时有效。*/
         listTEST_LIST_INTEGRITY( pxList );
         listTEST_LIST_ITEM_INTEGRITY(pxNewListItem );
 
         /*向列表中插入新的列表项*/
         pxNewListItem->pxNext = pxIndex;
         pxNewListItem->pxPrevious =pxIndex->pxPrevious;
 
         mtCOVERAGE_TEST_DELAY();
 
         pxIndex->pxPrevious->pxNext =pxNewListItem;
         pxIndex->pxPrevious = pxNewListItem;
 
         pxNewListItem->pvContainer = ( void* ) pxList;
 
         ( pxList->uxNumberOfItems )++;
}
```

# 总结

在FreeRTOS中，任务的调度与列表息息相关，因此要想学好任务调度的原理，列表是第一大难关！