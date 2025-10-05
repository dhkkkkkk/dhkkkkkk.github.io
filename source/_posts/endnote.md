---
title: EndNote
date: 2024-07-09 16:12:39
tags: 软件&工具的学习笔记
---

本文学习基于[endnote中科大慕课_bilibili](https://www.bilibili.com/video/BV1W4411b7GM/?p=3&spm_id_from=pageDriver&vd_source=063d3dd412d275078eaf18c62b5f21f6)

# 基本使用

{% asset_img 1.jpg This is an image %} 

* 在线搜索可在不同数据库中按需检索文献
* Retrieve result可选检索结果显示数量
* ➕可将选中文献复制到本地数据库
* <u>查找全文</u>可搜索原文PDF；也可通过<u>打开网址</u>进行检索。PDF最好用endnote内置阅读器打开
* 可在左侧栏对数据库文献进行分组，直接拖拽就可

# 在word中使用

## endnote关联word

* 在endnote安装路径下运行configure endnote并勾选第一个选项

* 文件-选项-加载项-管理COM加载项 转到-勾选Endnote cite while you write，如果出现**报错**：

  * 检查window系统位数（一般为64），在endnote安装路径下的Product-Support/CWYW中将..x64.dat改为压缩包格式后缀并解压，将解压后的5个文件复制到word的endnote加载项地址进行替换

  * 如果还是不行，检查**word**是否为64位，如果win系统为64位而word为32位建议**重装一个64位office（**本人原为32位office2019，全网的方法都试过了还是不行，重装为64位office2021后就正常了，使用的是endnote 21）

  * 其他问题可参考：[Word 找不到 Endnote选项_word加载项里找不到endnote-CSDN博客](https://blog.csdn.net/weixin_45936544/article/details/134223088?spm=1001.2014.3001.5506)

    [ word2019与endnote如何关联？ - 知乎 (zhihu.com)](https://www.zhihu.com/question/340094831/answer/1057369528)

    如果上述方法都不行真的建议立刻重装一个64位office，反正我是被折磨了一整天的😥

    office卸载建议使用微软官方的彻底卸载（整个过程大概20min左右）：[卸载 Office -Microsoft 支持](https://support.microsoft.com/zh-cn/office/从-pc-卸载-office-9dd49b83-264a-477a-8fcc-2fdf5dbf61d8?ui=zh-cn&rs=zh-cn&ad=cn)

## 在word中插入参考文献

{% asset_img 2.jpg This is an image %} 

* 可在**word中插入**endnote中选中的文献
* 也可以**在endnote中**有个引号图标在指定文档中插入文献
* 也可以直接在endnote中把相关文献**拖**进word（需打开word中的即时格式化）
* 在样式中可以根据要求选择不同的**引用格式**
* 在插入引文中可以插入**作者名**等信息（需endnote账号）
* 在“编辑＆管理引文”中可以**移除引文**
* 投稿前需将论文在“转换引文和参考文献”中转换为纯文本

## 修改引文样式

endnote21 中 工具-输出样式-编辑xx or新样式

## 使用论文模板

endnote21 中 工具-格式化论文-格式论文

## 插图

使用endnote插图可以自动根据选择的样式进行排版

* 新建文献-reference type设置为figure
* title设置该文献名字
* caption设置批注
* figure选择图片

# 将文献导入endnote

## 1、直接检索

在endnote中在不同数据库中进行在线搜索，需注意几点

* web of science的年份搜索可通过20xx-20xx进行区间搜索；Pub为20xx:20xx
* 可以点击➕选择更多数据库

## 2、网站输出

* web of science

  通过网站的save to endnote desktop（需根据网站具体情况） 

  web of science可先将选中文献Add to Marked List，再导出至endnote

* Scopus

  export-输出为RIS格式

* 知网

  *导出的txt在endnote中进行导入，导入选项为endnote import*（现在知网可以直接导出endnote可识别的数据库格式）

## 3、PDF导入

需在有网络时使用

如果没有成功识别pdf信息可能是文献无doi号，可以尝试改名+右键 查找文献更新

可在endnote首选项（preference）中设置pdf自动导入文件夹

## 4、手动输入和软件之间的交互

* 手动输入

  ctrl+N（引用按钮旁边的按钮） 

* endnote数据库导出到其他软件，格式一般为RIS

  一点就闪退捏😋，这盗版用不了一点

# endnote的一些功能

* **查找重复文献**

  endnote21：数据库-查找重复项

* **分组/Tags/Label**

  * 分组：手动拖动；创建可筛选的智能组；根据已有组再次筛选新组；组集下可以再创组

    可以通过组集(group set)+组(group)的方式对文献进行**分级管理**

  * tag：创建后可在每篇文献的summary下管理tag，主要作为分组的补充

    可以将某文献的具体类型、期刊等元素通过tag进行标记，**tag的颜色有限**

  * Label:编辑文献中可在Label中添加对改文献的一些**标注**

* **分享**

  用格式化复制；邮件发送（用的office）；可导出为txt、网页格式

* **文献分析**

  工具-学科文献，按照作者、年份、期刊进行数量分析

# 高效阅读文献

* 先阅读标题，通过rating先大概筛选一遍
* 浏览摘要，再通过rating筛选一遍

# 文献检索技巧

* 先通过各网站找到相关专业名词，一种东西可能对应了**多个词组**[CNKI翻译助手](https://dict.cnki.net/index)
* 可通过单词加*检索该单词所有形式（复数、大小写...）
* 可以从标题->主题进行检索

