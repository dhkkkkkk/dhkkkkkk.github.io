---
title: 结构体复习
date: 2022-08-01 10:09:45
tags: C语言学习
---

# 1.基本内容

## 几种初始化

```c
struct Student 
{
     char *name;
     int age;
 };
struct Student stu = {“lnj", 27}; //定义的同时按顺序初始化
/**************************************************************************/
struct Student stu = {.age = 35, .name = “lnj"};//定义的同时不按顺序初始化
/**************************************************************************/
struct Student stu;
stu.name = "lnj";
stu.age = 35;  //先定义后逐个初始化
/**************************************************************************/
struct Student stu;
stu = (struct Student){"lnj", 35};    // 先定义后,再一次性初始化   
```

## 结构体数组

```
struct Student 
{
    char *name;
    int age;
};
struct Student stu[2]; 
stu[0] = {"lnj", 35};
stu[1] = {"zs", 18};
```

## 结构体指针

```c
      // 定义一个结构体类型
      struct Student 
      {
          char *name;
          int age;
      };

     // 定义一个结构体变量
     struct Student stu = {“lnj", 18};

     // 定义一个指向结构体的指针变量
     struct Student *p;

    // 指向结构体变量stu
    p = &stu;

     /*
      这时候可以用3种方式访问结构体的成员
      */
     // 方式1：结构体变量名.成员名
     printf("name=%s, age = %d \n", stu.name, stu.age);

     // 方式2：(*指针变量名).成员名
     printf("name=%s, age = %d \n", (*p).name, (*p).age);

     // 方式3：指针变量名->成员名
     printf("name=%s, age = %d \n", p->name, p->age);
	  //->运算符只用于指针结构体
     return 0;
 }
```

## 结构体嵌套

```c
struct Date{
     int month;
     int day;
     int year;
}
struct  stu{
     int num;
    char *name;
    char sex;
    struct Date birthday;
    Float score;
}
```

* 成员访问

  如果某个成员也是结构体变量，可以连续使用成员运算符"."访问最低一级成员。

# 2.内存分析

- 结构体变量占用的内存空间永远是所有成员中占用内存最大成员的倍数(对齐问题)

  这种强制的要求一来简化了处理器与内存之间传输系统的设计,二来可以提升读取数据的速度。比如这么一种处理器,它每次读写内存的时候都从某个8倍数的地址开始,一次读出或写入8个字节的数据,假如软件能 保证double类型的数据都从8倍数地址开始,那么读或写一个double类型数据就只需要一次内存操作。否则,我们就可能需要两次内存操作才能完成这个动作,因为数据或许恰好横跨在两个符合对齐要求的8字节。

  ```c
  struct Person
  {
     int age; // 4
     char ch; // 1
     double score; // 8
  };
  struct Person p;
  printf("sizeof = %i\n", sizeof(p)); // 16
  ```

  - 占用内存最大属性是score, 占8个字节, 所以第一次会分配8个字节
  - 将第一次分配的8个字节分配给age4个,分配给ch1个, 还剩下3个字节
  - 当需要分配给score时, 发现只剩下3个字节, 所以会再次开辟8个字节存储空间
  - 一共开辟了两次8个字节空间, 所以最终p占用16个字节

  ```c
  struct Person
  {
    int age; // 4
    double score; // 8
    char ch; // 1
  };
  struct Person p;
  printf("sizeof = %i\n", sizeof(p)); // 24
  ```

  * 占用内存最大属性是score, 占8个字节, 所以第一次会分配8个字节
  * 将第一次分配的8个字节分配给age4个,还剩下4个字节
  * 当需要分配给score时, 发现只剩下4个字节, 所以会再次开辟8个字节存储空间
    将新分配的8个字节分配给score, 还剩下0个字节
  * 当需要分配给ch时, 发现上一次分配的已经没有了, 所以会再次开辟8个字节存储空间
    一共开辟了3次8个字节空间, 所以最终p占用24个字节

**需要注意的是，对占用内存最大的变量的判断仅限基本类型，同时不包括数组，也就是说当一个结构体同时包括**

**`chara[5];int b;`时仍然用int的大小计算。**

# 3.结构体之间赋值

- 结构体虽然是构造类型, **但是结构体之间赋值是值拷贝, 而不是地址传递**。所以结构体变量作为函数形参时也是值传递, **在函数内修改形参, 不会影响外界实参**。

  ```c
  #include <stdio.h>
  struct Person
  {
      char *name;
      int age;
  };
  void test(struct Person per);
  int main()
  {
      struct Person p1 = {"lnj", 35};
      printf("p1.name = %s\n", p1.name); // lnj
      test(p1);
      printf("p1.name = %s\n", p1.name); // lnj
      return 0;
  }
  void test(struct Person per)
  {
      per.name = "zs";
  }
  ```

# 共用体

- 和结构体不同的是, 结构体的每个成员都是占用一块独立的存储空间, 而共用体**所有的成员都占用同一块存储空间，因此所有成员的地址都是相同的**

- 特点: 由于所有属性共享同一块内存空间, **所以只要其中一个属性发生了改变, 其它的属性都会受到影响**。

  ```c
      union Test{
          int age;
          char ch;
      };
      union Test t;
      printf("sizeof(p) = %i\n", sizeof(t)); //4
  
      t.age = 33;
      printf("t.age = %i\n", t.age); // 33
      t.ch = 'a';
      printf("t.ch = %c\n", t.ch); // a
      printf("t.age = %i\n", t.age); // 97
  ```

# 枚举

- 枚举使用的注意
  - C语言编译器会将枚举元素(spring、summer等)作为**整型常量**处理，称为枚举常量。
  - 枚举元素的值取决于定义时各枚举元素排列的先后顺序。默认情况下，第一个枚举元素的值为0，第二个为1，依次顺序加1。
  - 也可以在定义枚举类型时改变枚举元素的值

```c
enum Season {
    Spring = 9,
    Summer,
    Autumn = 20,
    Winter
};
// 也就是说spring的值为9，summer的值为10，autumn的值为20，winter的值为21
```

* 关于枚举类型的内存大小：
  * 若为没有在其中赋值的操作，默认4字节
  * 若有赋值，大小取决于赋值数的类型

# 利用结构体实现动态链表

```c
#include"stdio.h"
#include "stdlib.h"
#include"windows.h"
typedef struct list
{
	int num;
	char name[30];
	struct list *next;
}test;

test *creat()
{
	test *linkhead,*linkend,*pt;
	int i;
	for(i=0;i<3;i++)
	{
		pt=(test*)malloc(sizeof(test));//分配动态内存
		scanf("%d %s",&pt->num,pt->name);
		if(i==0)
		{
			linkhead=pt;//第一次先将当前节点赋给链头链尾
			linkend=pt;
		}
		else//若不是第一次，先将当前节点存入上一个节点的next中，再将linkend更新为当前节点以供下一次访问
		{                                                                //当前节点的next
			linkend->next=pt;
			linkend=pt;
		}
	}
	linkend->next=NULL;
	return linkhead;
}
void destroy(test *linkHead)
{
	test *p;
	p=linkHead; 
	while(p!=NULL)
	{
		linkHead=linkHead->next;
		free(p);
		printf("clear\n");
		p=linkHead;
	}
}
int main()
{	
	test *LH，*Free_item;
	LH=creat();
   Free_item=LH;
	while(LH!=NULL)
	{
		printf("NUM:%d   NAME:%s\n",LH->num,LH->name);
		LH=LH->next;
	}
	destroy(Free_item);
	system("pause");
	return 0;
}
```

**以下解释存在一定问题，我修改后的内容以及写在下一节！！！**

* *注：其实这里最开始打印也应该单独写个函数，但是由于我懒得去写了，因此直接在main中操作参数，并且在操作完后我意识到好像无法再以LH为参数去使用destroy函数，按照一般地址与指针的理解，在打印完成过后应该已经改变了在creat()函数中分配的内存所对应的值，但是我抱着尝试的心理用main中的Free_item作为参数使用destroy函数竟然成功了，这一度让我以为我在指针的学习中是不是疏忽了什么重要内容！之后自己以简单的函数间的指针交互写了一些程序发现自己的理解好像也并没有什么问题，最后，在以"形参"为关键字搜索我的个人博客后终于找到如下一段话：*

  *“结构体虽然是构造类型, 但是结构体之间赋值是值拷贝, 而不是地址传递。所以结构体变量作为函数形参时也是值传递, 在函数内修改形参, 不会影响外界实参。”（完全忘记了捏😋，所以要多多复习原来的内容！！）*

  *因此LH也只是拷贝了creat函数所创建的linkhead的值而已，对LH的任何操作都不会影响creat函数所创建的linkhead的值，而Free_item拷贝的也是LH改变前的值，因此LH的改变也不会影响Free_item的内容。*

## 2022.11.17改

因为最近又在学习FreeRTOS的相关内容，再次复习了一下链表的相关原理，顺便又读了一下上面写的内容，发现对于上一节最后这里的解释好像并不正确，因为当时是按照结构体赋值解释的，**但是在这个链表程序中，传递的内容明显是地址而不是值！**于是我再次查阅了结构指针的相关文献，解释如下：

**我们一步一步来捋一下**：

### ①creat函数

首先我们可以确定，在这个链表程序中的所有test类型变量均为指针变量，也就是说，在creat()函数中:

```c
test *creat()
{
	test *linkhead,*linkend,*pt;
	int i;
	for(i=0;i<3;i++)
	{
		pt=(test*)malloc(sizeof(test));//分配动态内存
		scanf("%d %s",&pt->num,pt->name);
		if(i==0)
		{
			linkhead=pt;//第一次先将当前节点赋给链头链尾
			linkend=pt;
		}
		else//若不是第一次，先将当前节点存入上一个节点的next中，再将linkend更新为当前节点以供下一次访问
		{                                                                //当前节点的next
			linkend->next=pt;
			linkend=pt;
		}
	}
	linkend->next=NULL;
	return linkhead;
}
```



* 在`scanf("%d %s",&pt->num,pt->name);`中，其实是让pt这个结构体指针指向了一个整型数字和一个字符串，按照C++的说法就是**直接使指针指向了两个字符值常量。**
* 在三次循环中，分别开辟了三个动态内存区域。
* creat函数返回的也是一个结构体指针

### ②main()函数

```c
int main()
{	
	test *LH，*Free_item;
	LH=creat();
   Free_item=LH;
	while(LH!=NULL)
	{
		printf("NUM:%d   NAME:%s\n",LH->num,LH->name);
		LH=LH->next;
	}
	destroy(Free_item);
	system("pause");
	return 0;
}
```

* creat函数返回的指针**赋值**给结构体指针LH，并且在这里，所有的指针都不是二重指针，因此，creat的返回值linkhead的成员num和name都是直接指向的字符值常量，**因此LH也通过linkhead直接指向这两个成员，而不是LH指向linkhead这个指针变量的地址**，可以理解为linkhead和LH是并列的，没有先后顺序的。
* 下一步的`Free_item=LH;`也是同理，字面意义是Free_item指向了LH的地址，但是实际上Free_item也是直接指向的动态内存中的字符值常量，因为Free_item是一个一重指针，**它不具有保存另一个指针变量地址的功能，所以它会通过LH直接指向动态内存**
* 综上，实际上Free_item和LH也是并列关系，因此循环中LH的指向的改变不会影响到Free_item的指向，即使LH已经指向null，Free_item也还是指向的链头。

### 总结

其实说到底，是我自己对指针理解一直存在的误区，例如如下例程：

```c
int main()
{
	int a=10;
	int *p1,*p2;
	p1=&a;
	p2=p1;
	printf("%d %d %d %d",p1,&p1,p2,&a); //6422036 6422024 6422036 6422036
	return 0;
}
```

我之前一直认为一个指针指向另一个指针，该指针的便会指向另一个指针变量自己的地址，**但实际上，一重指针并不具有保存另一个指针变量地址的功能，所以该指针会直接指向另一个指针指向的内容。**

就像上述例程中,p1指向整型变量a的地址，再将p1赋值给p2，但打印结果中p2指向的地址是a的地址(&a)，而不是p1本身的地址（&p1)。

这也从另一方面说明了二重指针的一些重要性吧。