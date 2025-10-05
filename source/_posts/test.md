---
title: hexo d 无法更新
date: 2022-07-27 18:28:32
tags: 杂项
---

# 解决方案

删除 `hexo/.deploy_git` 文件, 然后重新尝试 `hexo d` , 就可以成功更新了.

（注：代理最好开全局模式，不然容易卡住）

具体步骤：

1.先hexo d 一遍

2.hexo cl + hexo g 

3.删除文件

4.hexo d

