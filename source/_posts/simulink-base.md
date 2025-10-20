---
title: simulink_base
date: 2025-10-13 14:49:28
tags: 软件&工具的学习笔记
---

# 常规模块

## 信号

* 正弦波 sine wave
* 阶跃信号 step
* 斜坡信号 ramp
* 常量 constant

## 运算

* 平方根：square root
* 相减：subtract
* 相加：add
* 增益 Gain

## 逻辑

* 与0比较：compare to zero
  * 更通用：compre to contrast
* 逻辑门：and or not
* 条件：switch

## 信号导入

* signal editor

## 函数模块

Matlab function模块

```matlab
function [power,v] = fcn(deltaH) %首先先定义输出与输入，可定义多个输入输出，多个输入：fcn(x1,x2)

Cp = 0.35; 
rho = 1000; 
A = pi*10^2; 
g = 9.81; 

%具体函数，左侧为输出，右侧为包含输入变量的计算
power = 0.5*Cp*rho*A*(2*g*abs(deltaH))^(3/2);
v = sqrt(2*g*abs(deltaH));
```

## 离散系统

* 延迟：unit delay，接受一个输入信号并在指定的采样时间内保持其值

  当步长为-1或等于输入信号采样时间时，则模块在t时刻输出t-1时刻信号

  当步长为其他固定值时，对信号进行重采样，并在t步输出原信号的t-1

## 连续系统



## 其他



* 打印信号值 display
* 信号可视化（示波器） scope

# 操作

* 创建分支：按住ctrl
