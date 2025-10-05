---
title: linux、ROS杂
date: 2024-02-19 17:06:26
tags: ROS
---

# Ubuntu网络图标消失

```
sudo service network-manager stop
sudo rm /var/lib/NetworkManager/NetworkManager.state 
sudo service network-manager start
```

# 常用命令

| ls               | 查看当前目录                                                 |
| ---------------- | ------------------------------------------------------------ |
| cd               | 进入目标路径                                                 |
| pwd              | 查看当前路径                                                 |
| mkdir            | 创建目录                                                     |
| rmdir/rm         | 删除目录                                                     |
| touch            | 创建空文件                                                   |
| echo             | 将内容回显到标准输出，可重定向                               |
| file             | 查看文件类型                                                 |
| more/less        | 查看文件内容，空格翻页，q退出。（less更灵活，/..搜索字符串） |
| head/tail        | 查看文件头部/尾部                                            |
| cat              | 将一个或多个文件输出到标准输出，可重定向                     |
| cat -n           | 从1开始对输出行进行编号，-s：将多行空白合并                  |
| tar              | -c文件打包；-x解包；-A合并；-f指定文件，加在最后；-C到指定目录 |
| *cvf xvf ..-C..* | -j调用bzip2；-z调用gzip；-Z调用compress；-t列出列表;-v详细信息 |
| mv               | 文件改名、移动。路径相同则改名，不同则移动                   |
| cp               | 文件复制                                                     |
| clear            | 清屏                                                         |
|                  |                                                              |
|                  |                                                              |
|                  |                                                              |

* find [路径] -name ""

  ```
  无路径默认桌面
  find -name "*1*"	名字含1的文件
  find -name "*.txt"	为后缀的文件
  find -name "1*"	以1为开头的文件
  ```

* tar

  打/解包文件，解包后文件存放路径为当且文件夹

  ```
  tar -cvf test.tar 1 2
  tar -xvf test.tar
  ```

  tar下的压缩与解压

  ```
  tar -zcvf test.gz 1 2
  tar -zxvf test.gz
  tar -zxvf test.gz -C 指定路径（必须存在）
  
  tar -jcvf test.bz2 1 2
  tar -jxvf test.bz2
  tar -jxvf test.bz2 -C 指定路径（必须存在）
  ```

* 软件安装

  ```
  # 1. 安装软件
  sudo apt install 软件包
  # 2. 卸载软件
  sudo apt remove 软件名
  # 3. 更新已安装的包
  sudo apt upgrade
  ```

* which 提示文件/命令所在位置

## 杀死进程

* xkill	强制关闭窗口
* kill (pid号)     pid号用ps -aux查询
* pkill (程序名)

# ROS无法定位软件包

[E：无法定位软件包 | 鱼香ROS (fishros.org.cn)](https://fishros.org.cn/forum/topic/149/e-无法定位软件包/3)

版本对应or源的问题，ubuntu20对应noetic

# ROS TF乌龟程序报错

`sudo apt install python-is-python3`升级python版本

# **view_frames**意外停止

* `sudo apt-get install vim`
* `sudo vim /opt/ros/noetic/lib/tf/view_frames` 
* 89行处，修改为 `m = r.search(vstr.decode('utf-8'))` 

## Vim的输入模式

在输入模式下，Vim 可以对文件执行写操作，类似于在 Windows 系统的文档中输入内容。

使 Vim 进行输入模式的方式是在命令模式状态下输入 i、I、a、A、o、O 等插入命令（各指令的具体功能如表 3 所示），**当编辑文件完成后按 Esc 键即可返回命令模式。**

| 快捷键 | 功能描述                                                     |
| ------ | ------------------------------------------------------------ |
| i      | **在当前光标所在位置插入随后输入的文本，光标后的文本相应向右移动** |
| I      | 在光标所在行的行首插入随后输入的文本，行首是该行的第一个非空白字符，相当于光标移动到行首执行 i 命令 |
| o      | 在光标所在行的下面插入新的一行。光标停在空行首，等待输入文本 |
| O      | 在光标所在行的上面插入新的一行。光标停在空行的行首，等待输入文本 |
| a      | 在当前光标所在位置之后插入随后输入的文本                     |
| A      | 在光标所在行的行尾插入随后输入的文本，相当于光标移动到行尾再执行a命令 |

## 退出vim

**按esc**后，输入如下，一般`:wq`就行

```
:q  
//退出

:q! 
//退出且不保存（:quit!的缩写）

:wq
//保存并退出

:wq!
//保存并退出即使文件没有写入权限（强制保存退出）

:x
//保存并退出（类似:wq，但是只有在有更改的情况下才保存）

:exit
//保存并退出（和:x相同）

:qa
//退出所有(:quitall的缩写)

:cq
//退出且不保存（即便有错误）

```

