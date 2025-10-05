---
title: 麦克风阵列与延迟求和波束形成
date: 2025-10-05 15:10:49
tags: 杂项
mathjax: true
---

# 导向矢量

{% asset_img 1.jpg This is an image %} 

选定参考阵元后，后面阵元接收到的信号比参考阵元晚：
$$
\Delta t = \left[0,\ \frac{\delta \cos(\theta_d)}{c},\ \frac{2\delta \cos(\theta_d)}{c},\ \cdots,\ \frac{(M-1)\delta \cos(\theta_d)}{c}\right]
$$
其中c为声速，δ为后续阵元相对参考阵元的间距（针对复杂阵元δ不为定值），θ为信号入射角度，将其反应到信号的相位变化上（假设输入为单频信号）：

*对于使用减法的理解：由于后续信号接收都晚于参考信号，对于参考信号的波形，在绝对时间轴上需要<u>右移（x轴减法）</u>才会露出更早部分的信号，而在同一时刻，后续麦克风受到的波形就是包含更早部分的参考波形（不包含更新、更晚的波形）*
$$
y(n) = \left[ x(n),\ x(n)e^{-j2\pi f_0\frac{\delta\cos(\theta_d)}{c}},\ x(n)e^{-j2\pi f_0\frac{2\delta\cos(\theta_d)}{c}},\ \cdots,\ x(n)e^{-j2\pi f_0\frac{(M-1)\delta\cos(\theta_d)}{c}} \right]  \\ = \left[ 1,\ e^{-j2\pi f_0\frac{\delta\cos(\theta_d)}{c}},\ e^{-j2\pi f_0\frac{2\delta\cos(\theta_d)}{c}},\ \cdots,\ e^{-j2\pi f_0\frac{(M-1)\delta\cos(\theta_d)}{c}} \right]x(n) \\
=a(\theta_d) \cdot x(n)
$$
其中a反应了不同a阵元的空间相位差，即导向矢量，受到以下变量影响：

* 阵元布置结构，影响阵元间距
* 来波方向，影响入射角
* 参考阵元位置，影响其他阵元计算间距，不影响阵元间相对相位差

若多个声源从不同角度入射，则导向矢量变为一个MxN的矩阵，M为阵元个数，N为信号数量

## 利用导向矢量估计声源入射方向

当入射角未知时，当阵元布置和参考阵元确定时，导向矢量的形式是确定的，且只受到入射角影响，因此可以构造一个方向为α的导向矢量形式的向量（共轭转置）与输入信号做**内积**计算：
$$
a^H(\alpha) \cdot a(\theta_d) \cdot x(n) \\= \left[ 1 + e^{-j2\pi f_0\delta \frac{[\cos(\alpha) - \cos(\theta_d)]}{c}} + e^{-j2\pi f_0\delta \frac{2[\cos(\alpha) - \cos(\theta_d)]}{c}} + \cdots + e^{-j2\pi f_0\delta \frac{(M-1)[\cos(\alpha) - \cos(\theta_d)]}{c}} \right] \\\leq M \cdot x(n)
$$
当角度重合时，内积最大

# 延迟求和

由上一章得知麦克风信号为对参考信号相位分别进行了减法的信号，因此只要对麦克风信号的相位分别进行相反的加法（加权）即可得到原始的信号波形：
$$
z(n)=\frac{1}{M}(1,e^{j2\pi f \tau_1},e^{j2\pi f \tau_2}...) \cdot y(n)^T
\\= \frac{1}{M}\omega(\theta) \cdot y(n)^T
$$
因此，在进行某个方向的波束求和时，频率f为变量（f依次变化计算出每个频率上的响应）

（其实原理和上一小节的估计声源方向一样）
