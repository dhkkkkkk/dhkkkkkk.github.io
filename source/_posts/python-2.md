---
title: python_2
date: 2022-09-16 10:29:36
tags: python学习
---

# 一些基础语法

## 1、条件控制

一般形式：

```python
if condition_1:
    statement_block_1
elif condition_2:
    statement_block_2
else:
    statement_block_3
```

* 记得在条件后加`:`
* Python 中用 **elif** 代替了 **else if**，所以if语句的关键字为：**if – elif – else**
* python中没有switch语句，python3.10后更新了类似语法，由于我是3.9，就不写在这里了

## 2、循环语句

### while语句

可以与else配合使用：

```python
while <expr>: #记得冒号
    <statement(s)>
else:
    <additional_statement(s)>
```

while中的break和continue用法同C

### for语句

一般格式

```python
a = 'dhk'
for x in a: 
	print(x,end='') #dhk
else:	#else用于在循环结束后执行的内容
   print("error!!")
  
sites = ["Baidu", "Google","Runoob","Taobao"]
for site in sites:
    print(site)
  
questions = {'name': 'lancelot', 'quest':'the holy grail', 'favorite color':'blue'}
for q,b in questions.items(): #字典会稍微复杂一点
    print(f'{q}  {b}')
```

不同于C，python的for语句**常用来遍历某个可迭代对象**，如列表、字符串，如上的循环中，x每次只取遍历对象的**最基本单位并自动递增**。

### range函数

生成一个数字序列，格式：

```python
for x in range(1,10,2):
   print(x,end='')#13579
   

>>>a = ['Google', 'Baidu', 'Runoob', 'Taobao', 'QQ']
>>> for i in range(len(a)):
...     print(i, a[i])
... 
0 Google
1 Baidu
2 Runoob
3 Taobao
4 QQ
>>>
```

range函数的指定区间同样**遵循左闭右开**的原则

### pass语句

空语句，表示空

## 3、迭代器

迭代是Python最强大的功能之一，是**访问集合元素的一种方式**。

大概作用和C的指针有点像？

迭代器是一个**可以记住遍历的位置**的对象。

迭代器对象从集合的第一个元素开始访问，直到所有的元素被访问完结束。迭代器只能往前不会后退。

```python
    list = [1, 2, 3, 4]
    it = iter(list)  # 创建迭代器对象
    it2 = iter(list)  # 创建迭代器对象
    for x in it:
        if x != 4:
            print(x, end=" ")
        else:
            print(x)
    print(next(it2))
    print(next(it2))
    print(next(it2))
    print(next(it2))
'''
1 2 3 4
1
2
3
4
'''
```

### 把一个类作为迭代器使用：

若想把一个类作为迭代器使用，需要在类中实现两个方法：

* `__iter__()`:返回一个特殊的迭代器对象， 这个迭代器对象实现了 __next__() 方法并通过 StopIteration 异常标识迭代的完成。

* `__next__()`:返回下一个迭代器对象

  ```python
      class MyNumbers:
          def __iter__(self):
              self.a = 1
              return self
  
          def __next__(self):
              if self.a <= 20:
                  x = self.a
                  self.a += 1
                  return x
              else:
                  raise StopIteration #挂起StopIteration标志迭代完成，不能用return
      a = MyNumbers()
      i = iter(a)
      for x in i:
          print(x, end=' ')
   #1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 
  ```

## 4、生成器

 在 Python 中，使用了 yield 的函数被称为生成器（generator）。

在调用生成器运行的过程中，**每次遇到 yield 时函数会暂停并保存当前所有的运行信息，返回 yield 的值**, 并在下一次执行 next() 方法时从当前位置继续运行。

* 生成器函数返回的是一个**迭代器对象**

```python
import sys

def fibonacci(n): # 生成器函数 - 斐波那契
    a, b, counter = 0, 1, 0
    while True:
        if (counter > n): 
            return
        yield a
        a, b = b, a + b
        counter += 1
f = fibonacci(10) # f 是一个迭代器，由生成器返回生成
 
while True:
    try:
        print (next(f), end=" ")
    except StopIteration:
        sys.exit()
```

## 5、函数

规则：

- 函数代码块以 **def** 关键词开头，后接函数标识符名称和圆括号 **()**。
- 函数内容以冒号 **:** 起始，并且缩进。**（记得冒号！！）**
- 与C不同，py的函数不需要确定返回值类型，直接return就可以

```
def 函数名（参数列表）:
    函数体
```

### 关键字参数

在调用函数时，实参可以是key=value的形式，称为关键字参数。

凡是按照这种形式定义的实参，可以不按照从左到右的顺序定义，但仍能为指定的形参赋值。

* **关键字参数在位置参数后面**，因为python函数在解析参数时，是按照顺序来的，**位置参数**是必须先满足，才考虑其他可变参数。

  ```python
      def printinfo(age, name):
          print("名字: ", name)
          print("年龄: ", age)
          return
  
      printinfo(50, name="runoob") #age为位置参数，name为关键字参数
  ```

  需要注意：

  * **位置参数的输入必须和定义时的位置相同**
  * **关键字参数的输入必须在位置参数之后**

### 当参数不定长时

```py
def printinfo( arg1, *vartuple ):
```

加了星号 ***** 的参数会以元组(tuple)的形式导入，存放所有未命名的变量参数

当参数添加双星号******时，传入的参数将以字典的形式保存

```python
def func(**dic)
	print(dic)
func(**{'a':1,'b':2})
#or
func(a=1,b=2)
```

### 匿名函数

Python 使用 **lambda** 来创建匿名函数

- lambda 的主体是一个表达式，而不是一个代码块。仅仅能在 lambda 表达式中封装有限的逻辑进去。

```python
# 可写函数说明
sum = lambda arg1, arg2: arg1 + arg2
 
# 调用sum函数
print ("相加后的值为 : ", sum( 10, 20 ))
print ("相加后的值为 : ", sum( 20, 20 ))
'''
30
40
'''
```

### 强制位置参数＆命名关键字参数

```python
# a, b, c成为限定位置形参
def func(a, b, c, /, d):
    pass
```

```python
#这时参数kw1, kw2必须传入关键字参数
def func(其他形参, *, kw1, kw2):
    pass
```

## 6、类

Python中的类提供了面向对象编程的所有基本功能：类的继承机制允许多个基类，派生类可以覆盖基类中的任何方法，方法中可以调用基类中的同名方法。

定义：

```python
class MyClass:
    i = 12345
    a = 1
    def f(self):#类中的函数称作类的方法
        return 'hello world'
```

类的初始化（构造方法），**这里注意类的方法的第一个参数**

```python
class Complex:
    name = ''
    age = 0
    def __init__(self, n, a): #类的方法必须包含self，且其必须为第一个参数
        self.name = n
        self.age = a
```

创建一个类的实例：

```python
x = MyClass()
```

访问类的属性、方法：

```python
print("MyClass 类的属性 i 为：", x.i)
print("MyClass 类的方法 f 输出为：", x.f())
```

#### 类的继承：

```python
#类定义
class people:
    #定义基本属性
    name = ''
    age = 0
    #定义私有属性,私有属性在类外部无法直接进行访问
    __weight = 0	
    #定义构造方法
    def __init__(self,n,a,w):
        self.name = n
        self.age = a
        self.__weight = w
    def speak(self):
        print("%s 说: 我 %d 岁。" %(self.name,self.age))
 
#单继承示例
class student(people):
    grade = ''
    def __init__(self,n,a,w,g):
        #调用父类的构函
        people.__init__(self,n,a,w)
        self.grade = g
    #重写父类的方法
    def speak(self):
        print("%s 说: 我 %d 岁了，我在读 %d 年级"%(self.name,self.age,self.grade))
 
 
 
s = student('ken',10,60,3)
s.speak()
```

* 子类中若出现了父类同名的方法，则在调用该子类方法时使用的内容也是子类中定义的内容

私有属性与方法：

在**方法**或属性的开头加两个下划线，**声明**（不包含在变量名中）该属性为私有，不能在类的外部被使用或直接访问

# 文件间的引用

## 1、import语句

想使用一个 Python 源文件，只需在另一个源文件里执行 import 语句导入该模块（就是后缀为.py的文件）

* 一个模块只会被导入一次，不管你执行了多少次 **import**。这样可以防止导入模块被一遍又一遍地执行。

## 2、from..import语句

从模块中导入指定的部分，例如:

```python
from test_file import test_fuction1,test_fuction2
#从test_file.py中导入两个函数
#将某个模块中的全部函数导入，格式为from somemodule import *
```

## 3、使用模块中的变量、函数

通过modname.itemname 这样的表示法来访问模块内的函数

## 4、_name__属性

一个模块被另一个程序第一次引入时，其主程序将运行。每个模块都有一个__name__属性，当其值是'__main__'时，表明该模块自身在运行，否则是被引入。

```python
if __name__ == '__main__':
   print('程序自身在运行')
else:
   print('其他程序在运行')
```

可以通过name属性使某个文件**仅在自身运行而非被引用时**运行某些函数

## 5、包

包是一种管理 Python 模块命名空间的形式，采用"点模块名称"。

```
sound/                          顶层包
      __init__.py               初始化 sound 包
      formats/                  文件格式转换子包
              __init__.py
              wavread.py
              wavwrite.py
              aiffread.py
              aiffwrite.py
              auread.py
              auwrite.py
              ...
      effects/                  声音效果子包
              __init__.py
              echo.py
              surround.py
              reverse.py
              ...
      filters/                  filters 子包
              __init__.py
              equalizer.py
              vocoder.py
              karaoke.py
              ...

```

**目录只有包含一个叫做 _ _init__.py 的文件才会被认作是一个包**

* 导入方法

  导入模块

  ```python
  from sound.effects import echo
  ```

  导入函数、变量

  ```python
  from sound.effects.echo import echofilter
  ```

* 导入多个子模块

  在sound/下的init文件中：

  ```
  __all__ = ["echo", "surround", "reverse"]
  ```

  由于windows不区分大小写，因此可以通过这种方式在当使用导入*时区分大小写地导入子模块

# 读、写文件

```python
f=open(name,mode)
```

mode可选：

| 模式 | 描述                                                         |
| :--- | :----------------------------------------------------------- |
| r    | 以只读方式打开文件。文件的指针将会放在文件的开头。这是默认模式。 |
| rb   | 以二进制格式打开一个文件用于只读。文件指针将会放在文件的开头。 |
| r+   | 打开一个文件用于读写。文件指针将会放在文件的开头。           |
| rb+  | 以二进制格式打开一个文件用于读写。文件指针将会放在文件的开头。 |
| w    | 打开一个文件只用于写入。如果该文件已存在则打开文件，并从开头开始编辑，即原有内容会被删除。如果该文件不存在，创建新文件。 |
| wb   | 以二进制格式打开一个文件只用于写入。如果该文件已存在则打开文件，并从开头开始编辑，即原有内容会被删除。如果该文件不存在，创建新文件。 |
| w+   | 打开一个文件用于读写。如果该文件已存在则打开文件，并从开头开始编辑，即原有内容会被删除。如果该文件不存在，创建新文件。 |
| wb+  | 以二进制格式打开一个文件用于读写。如果该文件已存在则打开文件，并从开头开始编辑，即原有内容会被删除。如果该文件不存在，创建新文件。 |
| a    | 打开一个文件用于追加。如果该文件已存在，文件指针将会放在文件的结尾。也就是说，新的内容将会被写入到已有内容之后。如果该文件不存在，创建新文件进行写入。 |
| ab   | 以二进制格式打开一个文件用于追加。如果该文件已存在，文件指针将会放在文件的结尾。也就是说，新的内容将会被写入到已有内容之后。如果该文件不存在，创建新文件进行写入。 |
| a+   | 打开一个文件用于读写。如果该文件已存在，文件指针将会放在文件的结尾。文件打开时会是追加模式。如果该文件不存在，创建新文件用于读写。 |
| ab+  | 以二进制格式打开一个文件用于追加。如果该文件已存在，文件指针将会放在文件的结尾。如果该文件不存在，创建新文件用于读写。 |

open的其他方法见：[Python3 File 方法 | 菜鸟教程 (runoob.com)](https://www.runoob.com/python3/python3-file-methods.html)

文件对象的其他方法见[Python3 输入和输出 | 菜鸟教程 (runoob.com)](https://www.runoob.com/python3/python3-inputoutput.html)

# 局部变量和全局变量

在局部作用域修改全局变量，感觉。。还是我们C方便。。。

```python
num = 1
def fun1():
    global num  # 需要使用 global 关键字声明
    print(num) 
    num = 123
    print(num)
fun1()
print(num)
```

# Numpy库

## array操作

```python
a = np.zeros((1, 5))                                       
print(f"a shape = {a.shape}, a = {a}")                     

a = np.zeros((2, 1))                                                                 
print(f"a shape = {a.shape}, a = {a}") 

a = np.random.random_sample((1, 1))  
print(f"a shape = {a.shape}, a = {a}") 

'''
a shape = (1, 5), a = [[0. 0. 0. 0. 0.]]
a shape = (2, 1), a = [[0.]
 [0.]] #此处为2个行向量
a shape = (1, 1), a = [[0.78320514]]
'''


a = np.array([[5], [4], [3]])  
print(f" a shape = {a.shape}, np.array: a = {a}")

'''
 a shape = (3, 1), np.array: a = [[5]
 [4]
 [3]]
'''
```

## 索引

```python
a = np.arange(6).reshape(-1, 2)   #将0~5转化为？行，2列的格式
print(f"a.shape: {a.shape}, \na= {a}")
'''
a.shape: (3, 2), 
a= [[0 1]
 [2 3]
 [4 5]]
'''

#索引某个元素（索引从0开始,因此索引的为第3排第1列）
print(f"\na[2,0].shape: {a[2, 0].shape}, a[2,0] = {a[2, 0]}, type(a[2,0]) = {type(a[2, 0])} Accessing an element returns a scalar\n")
'''
a[2,0].shape: (), a[2,0] = 4, type(a[2,0]) = <class 'numpy.int32'> Accessing an element returns a scalar
'''

#索引某一行
print(f"a[2].shape:   {a[2].shape}, a[2]   = {a[2]}, type(a[2])   = {type(a[2])}")
'''
a[2].shape:   (2,), a[2]   = [4 5], type(a[2])   = <class 'numpy.ndarray'>
'''
```

## np.expand_dims

我也还不太清楚具体用法，先做个记录，可以仿照着用

```python
A = np.arange(1,11).reshape(1,-1) #只有1行的二维
print(f'{A}\n')
A = np.expand_dims(A,axis=2)
print(f'{A}\n{A.shape}')
A = A[:,:,0]
print(f'{A}\n')

'''
[[ 1  2  3  4  5  6  7  8  9 10]]

[[[ 1]
  [ 2]
  [ 3]
  [ 4]
  [ 5]
  [ 6]
  [ 7]
  [ 8]
  [ 9]
  [10]]]
(1, 10, 1)
[[ 1  2  3  4  5  6  7  8  9 10]]
'''
```

## np.pad

对数组在不同维度填充值，默认为0

```python
import numpy as np

A = np.arange(1,11).reshape(2,-1)
A = np.expand_dims(A,axis=2)	#变为3维
A = np.pad(A,((1,2),(0,0),(0,0)))
print(f'{A}\n')

'''
[[[ 0]
  [ 0]
  [ 0]
  [ 0]
  [ 0]]

 [[ 1]
  [ 2]
  [ 3]
  [ 4]
  [ 5]]

 [[ 6]
  [ 7]
  [ 8]
  [ 9]
  [10]]

 [[ 0]
  [ 0]
  [ 0]
  [ 0]
  [ 0]]

 [[ 0]
  [ 0]
  [ 0]
  [ 0]
  [ 0]]]

'''
```

```python
A = np.pad(A,((0,0),(1,2),(0,0)))
print(f'{A}\n')

'''
[[[ 0]
  [ 1]
  [ 2]
  [ 3]
  [ 4]
  [ 5]
  [ 0]
  [ 0]]

 [[ 0]
  [ 6]
  [ 7]
  [ 8]
  [ 9]
  [10]
  [ 0]
  [ 0]]]

'''
```

```python
A = np.pad(A,((0,0),(0,0),(1,2)))
print(f'{A}\n')

'''
[[[ 0  1  0  0]
  [ 0  2  0  0]
  [ 0  3  0  0]
  [ 0  4  0  0]
  [ 0  5  0  0]]

 [[ 0  6  0  0]
  [ 0  7  0  0]
  [ 0  8  0  0]
  [ 0  9  0  0]
  [ 0 10  0  0]]]
'''
```

# matplotlib.pyplot库

用法接近matlab

## subplots

```python
fig, axarr = plt.subplots(6)
axarr[0].plot(...)
axarr[1].imshow()
...
plt.show()
```

## subplot

```
plt.subplot(1,2,1)
plt.plot(...)
plt.subplot(1,2,2)
plt.plot(...)
plt.show()
```

# OS库

## os.listdir()

os.listdir() 方法用于返回指定的文件夹包含的**文件或文件夹**的**名字**的**列表**

## os.path()

[Python os.path 模块 | 菜鸟教程 (runoob.com)](https://www.runoob.com/python/python-os-path.html)

### os.path.join(path1[, path2[, ...]])

把path1path2合成一个路径