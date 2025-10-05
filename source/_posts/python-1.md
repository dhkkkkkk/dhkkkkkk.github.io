---
title: python_基础定义
date: 2022-09-14 18:30:37
tags: python学习
---

# 1、基本数据类型

* 变量不需要声明，也没有类型。
* python允许同时给多个变量赋值

## 标准数据类型：

* Number
* String
* List
* Tuple（元组）
* Set（集合）
* Dictionary（字典）

### 1、Number（数字）

Python3 支持 int、float、bool、**complex（复数）**。

* 可使用type(*变量* )函数查询变量类型
* 可用isinstance(*变量，类型* )判断变量是否为想要的类型，为真返回Ture

* del 语句可用删除对对象的引用

* 在除法的计算中，`/`返回浮点数，`//`返回整数，在混合计算时会把整数转化为浮点数

* `//`返回的不一定是整型的数，这与分母分子的类型有关

python可以使用`**`来进行幂运算，如5的平方：`5**2`

* **number为不可变类型，**例如变量赋值 **a=5** 后再赋值 **a=10**，这里实际是新生成一个 int 值对象 10，再让 a 指向它，而 5 被丢弃，不是改变 a 的值，相当于新生成了 a

* 数字类型转换

  ```python
  a=1.0
  int(a)
  float(a)
  complex(a)	#转换为复数，实数部分为a，虚数部分为0
  b=2
  complex(a,b)	#转换为复数，实数部分为a，虚数部分为b
  ```

* 其他常用数字函数

  [Python3 数字(Number) | 菜鸟教程 (runoob.com)](https://www.runoob.com/python3/python3-number.html)

### 2、String（字符串）

Python 没有单独的字符类型，一个字符就是长度为 1 的字符串

python中的字符串可用单引号或双引号括起来，使用完全相同

字符串的截取：

```python
#变量[头下标:尾下标:截取步长]
#若截取步长为负数则逆向读取
#第一个字符索引值为0，末尾字符为-1

print("dhk"*2)
print("dhk"+"dhk")
#dhkdhk
```

* 字符索引：变量[索引下标]
* 屏蔽转义字符：在字符串前（引号外）添加一个r
* **python字符串不能被改变，也就是说不能给单个索引位置赋值**

字符串的格式化：`print("%s %d" % ('abc', 10))`

#### input

```python
input([prompt]) #prompt为用户写的提示信息


>>>a = input("input:")
input:123                  # 输入整数
>>> type(a)
<type 'int'>               # 整型


>>> a = input("input:")    
input:"runoob"           # 正确，字符串表达式
>>> type(a)
<type 'str'>             # 字符串，python3 里 input() 默认接收到的是 str 类型


>>> a = input("input:")
input:runoob               # 报错，不是表达式
```

#### f-string字符串格式化

 格式化字符串以 **f** 开头，后面跟着字符串，字符串中的表达式用大括号 {} 包起来，它会将变量或表达式计算后的值替换进去，括号中的变量不仅限于字符串

```python
name = 'dhk'
print(f'hello {name}')
#hello dhk
```

* print中的转义字符用法同C

#### print

python的输出不同于c，直接把想输出的内容通过逗号连接即可

```python
a=11
b='dhk'
print(a,b)
```

也可以通过类似C的语法：

但是感觉这种方法不如f-string和上述用法方便简单

```python
print ("我叫 %s 今年 %d 岁!" % ('小明', 10))
```

格式化符号详见[Python3 字符串 | 菜鸟教程 (runoob.com)](https://www.runoob.com/python3/python3-string.html)

* end关键字，用于将结果输出到同一行

  ```python
  print(b, end=',')
  ```

#### 转义字符

[Python3 字符串 | 菜鸟教程 (runoob.com)](https://www.runoob.com/python3/python3-string.html)

其中进制的转换为：

* bin(a)：转二进制
* oct(a)：转八进制
* hex(a)：转十六进制

#### 字符串相关函数

[Python3 字符串 | 菜鸟教程 (runoob.com)](https://www.runoob.com/python3/python3-string.html)

### 3、List（列表），中括号

定义方式：

```python
a=[1,10.1,'abcd']
```

* 基本使用同字符串，也可以索引、截取，使用步长截取（包括逆向）等。
* **列表内的元素可改变！**，直接像c的数组一样改就行
* 逆向读取不会将元素内容逆向

#### 更新列表：

```python
list1 = ['Google', 'Runoob', 'Taobao']
list1.append('Baidu')
print ("更新后的列表 : ", list1)	#更新后的列表 :  ['Google', 'Runoob', 'Taobao', 'Baidu']
```

#### 删除列表元素

```python
list1 = ['Google', 'Runoob', 'Taobao']
del list[2]
```

#### 索引

变量[头下标:尾下标] or 变量[头下标:尾下标:**步长**]步长为-1时逆向读取

其中-1代表末尾（用于不知道实际长度时的索引），同时-2、-3对应倒数第二、第三...

正数和负数的使用**不相互冲突**

#### 列表操作函数

[Python3 列表 | 菜鸟教程 (runoob.com)](https://www.runoob.com/python3/python3-list.html)

#### 创建指定长度空列表

```python
n = 5
empty_list = [None] * n
```

### 4、Tuple（元组），小括号

定义方式：

```python
a=(1,10.1,'abcd')
```

* 基本使用同字符串，也可以索引、截取，使用步长截取（包括逆向）等。

* 元组在输入时可能没有括号，例如`t = 12345, 54321, 'hello!'`

* **元组内元素不可改变，但可以包含可变的对象，比如列表**，同时也可以对元组之间进行连接组合成新元组：

  ```python
  tup1 = (12, 34.56)
  tup2 = ('abc', 'xyz')
   
  # 以下修改元组元素操作是非法的。
  # tup1[0] = 100
   
  # 创建一个新的元组
  tup3 = tup1 + tup2
  print (tup3)
  ```

* 包含0和1个的元组有额外的语法规则：

  ```python
  tup1 = ()    # 空元组
  tup2 = (20,) # 一个元素，需要在元素后添加逗号,否则括号会被当作运算符使用
  ```

* **<u>元组中只包含一个元素时，需要在元素后面添加逗号  ，否则括号会被当作运算符使用</u>**

* 内置函数

  [Python3 元组 | 菜鸟教程 (runoob.com)](https://www.runoob.com/python3/python3-tuple.html)

### 5、Set（集合），大括号

定义方式：

```python
a=() #空集合需用小括号
b={1, 2, 3}
c=set('abcde') #使用set函数={'a','b','c','d','e'}
#必须用 set() 而不是 { }，因为 { } 是用来创建一个空字典。
b.add(4)
b.update(4)	#添加元素上述两种都可以
b.remove(4)
b.discard(4)	#remove不存在元素会报错，discard则不会
```

特点：

* **输出时会自动去掉重复的元素（从第二次重复的元素开始删除）**

* 可以进行集合运算

* **不可像前面几个类型一样进行索引**，集合是无序的

  ```python
  print(a-b) #将a中与b相同的元素去掉后输出a
  print(a|b) # 并集
  print(a&b) #交集
  print(a^b) #非交集
  ```

其余内置函数

[Python3 集合 | 菜鸟教程 (runoob.com)](https://www.runoob.com/python3/python3-set.html)

### 6、Dictionary（字典）

字典是一种映射类型，它是一个**无序**的 **键(key) : 值(value)** 的集合。

定义方式：

```python
a = {'a' : 1, 'b': 2, 'c' :3}  #':'前为“键”，后为“值”
#or
a= {}
a['a'] = 1
a['b'] = 2

print(a) #{'a': 'abcd', 'b': '123'}
print(a.keys()) #dict_keys(['a', 'b'])
print(a.values()) #dict_values(['abcd', '123'])
print(a['b'])	#2    访问字典里的值
```

键的特性：

* 同一个键出现两次时只有后一个值会被记住

* 键**必须是不可变**的，因此可以是数字、字符串、元组，而不能是列表等可以改变的变量

* 修改字典

  ```python
  tinydict = {'Name': 'Runoob', 'Age': 7, 'Class': 'First'}
   
  tinydict['Age'] = 8               # 更新 Age
  tinydict['School'] = "菜鸟教程"  # 添加信息
   
   
  print ("tinydict['Age']: ", tinydict['Age'])
  print ("tinydict['School']: ", tinydict['School'])
  ```

* 同时遍历多个序列和用单个字典表示：

```python
questions = {'name': 'lancelot', 'quest':'the holy grail', 'favorite color':'blue'}
for q, a in questions.items():#q和a会分别遍历键和值
	print(f'What is your {q}?  It is {a}.')
   
####################or#####################

questions = ['name', 'quest', 'favorite color']
answers = ['lancelot', 'the holy grail', 'blue']
for q, a in zip(questions, answers): #索引多个序列
   print(f'What is your {q}?  It is {a}.')

'''
What is your name?  It is lancelot.
What is your quest?  It is the holy grail.
What is your favorite color?  It is blue.
'''
```

* 其余内置函数和方法

  [Python3 字典 | 菜鸟教程 (runoob.com)](https://www.runoob.com/python3/python3-dictionary.html)

## 综上

* list元素可改变，有序
* tuple元素不可改变，有序
* set无序，不可索引

```python
a=[1,10.1,'abcd']  #列表
a=(1,10.1,'abcd')	#元组
a={1, 2, 3}	#集合
a = {'a' : 1, 'b': 2, 'c' :3} #字典
```

# 2、推导式

Python 推导式是一种独特的数据处理方式，**可以从一个数据序列构建另一个新的数据序列的结构体**。

Python 支持各种数据结构的推导式：

- 列表(list)推导式
- 字典(dict)推导式
- 集合(set)推导式
- 元组(tuple)推导式

## 列表推导式

格式：

```python
[表达式 for 变量 in 列表 if 条件]
```

* 表达式：需要对输出元素的操作，可以是有返回值的函数，**其格式即推导式最终输出的格式**
* 变量＆列表：我的理解是定义一个新变量，通过这个新变量将现有列表元素传入**表达式**中

* 条件：对变量进行筛选

例：

```python
a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
b = [i/2 for i in a if i % 2 == 0]
#定义变量i，i来自列表a中的可以整除2的元素，最后返回i/2的值作为新列表b的元素
print(b)
#[1.0, 2.0, 3.0, 4.0, 5.0]
```

同时定义多个变量存放值：

```
vec1 = [2, 4, 6]
vec2 = [4, 3, -9]
[x*y for x in vec1 for y in vec2] 
[x+y for x in vec1 for y in vec2]
[vec1[i]*vec2[i] for i in range(len(vec1))]
```

在矩阵列表中使用：

```
a = [[1, 2, 3], [4, 5, 6]]
b = [row[0] for row in a]
print(b)
```

## 字典推导式

格式与列表推导式几乎相同，只是将表达式换成了`key:value`的格式，具体见下：

```python
a = [97, 98, 99, 100, 101] # 传入推导式的为一个列表
b = {k: chr(k) for k in a}
print(b)	#{97: 'a', 98: 'b', 99: 'c', 100: 'd', 101: 'e'}
```

```python
a = {97: 'a', 98: 'b', 99: 'c', 100: 'd', 101: 'e'}# 传入推导式的为一个字典
b = {k: chr(k) for k in a.keys()}
print(b)	#{97: 'a', 98: 'b', 99: 'c', 100: 'd', 101: 'e'}

#or

a = {97: 'a', 98: 'b', 99: 'c', 100: 'd', 101: 'e'}
b = {k: i for k,i in a.items()}
print(b)
```

但如果如下操作：

```python
a = {97: 'a', 98: 'b', 99: 'c', 100: 'd', 101: 'e'}
b = {k: i for i in a.values() for k in a.keys()}
print(b)
#{97: 'e', 98: 'e', 99: 'e', 100: 'e', 101: 'e'}
```

目前还不知道原因。

## 集合推导式

规则格式与列表推导式相同

例：

```python
a = {x for x in 'abracadabra' if x not in 'abc'} #判断不是abc的字母
print(a) #{'d', 'r'}
print(type(a)) #<class 'set'>
```

## 元组推导式（生成器表达式）

元组推导式和列表推导式的用法也完全相同，但元组推导式返回的结果是一个生成器对象

```python
>>> a = (x for x in range(1,10))
>>> a
<generator object <genexpr> at 0x7faf6ee20a50>  # 返回的是生成器对象

>>> tuple(a)       # 使用 tuple() 函数，可以直接将生成器对象转换成元组
(1, 2, 3, 4, 5, 6, 7, 8, 9)
```

# 3、运算符

## 赋值运算符

* `:=`   海象运算符，在表达式内部为变量赋值：

  ```python
  if (n := len(a)) > 10  #可以避免调用两次len函数
  ```

## python逻辑运算符

| 运算符 | 表达式  |                作用                |
| :----: | :-----: | :--------------------------------: |
|  and   | x and y | x为**假**则返回x，x为**真**则返回y |
|   or   | a or y  | x为**真**则返回x，x为**假**则返回y |
|  not   |  not x  |      x为真返回假，为假返回真       |

## python成员运算符

可用于字符串、列表、元组、集合、字典（貌似是判断键）

| 运算符 |    表达式     |
| :----: | :-----------: |
|   in   |   x in list   |
| not in | x not in list |

## python身份运算符

| 运算符 |   表达式   |
| :----: | :--------: |
|   is   |   a is b   |
| is not | a is not b |

# 4、其他

* end关键字，用于将结果输出到同一行，同时将end=的内容添加在每次输出的末尾

  ```python
  b=1
  while b<10：
  	print(b,end=',')
  	b+=1
  ```

* 多行语句

  ```python
  total = item_one + \
          item_two + \
          item_three
  #在 [], {}, 或 () 中的多行语句，不需要使用反斜杠
  ```

  

