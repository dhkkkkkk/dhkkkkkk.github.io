---
title: 麦克风阵列与成像算法
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

*早到的信号相位为**正**，表示相位超前，晚到的则为**负**（即下述场景）*
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

# 波束形成延迟求和成像

由上一章得知麦克风信号为对参考信号相位分别进行了减法的信号，因此只要对麦克风信号的相位分别进行相反的加法（加权）即可得到原始的信号波形：
$$
z(n)=\frac{1}{M}(1,e^{j2\pi f \tau_1},e^{j2\pi f \tau_2}...) \cdot y(n)^T
\\= \frac{1}{M}\omega(\theta) \cdot y(n)^T
$$
因此，在进行某个方向的波束求和时，频率f为变量（f依次变化计算出每个频率上的响应）

（其实原理和上一小节的估计声源方向一样）

综上，对于(θ，φ)方向上，f频率下的功率为：（Y为f下各通道的复信号向量）
$$
P(f)  = |\mathbf{w}(f) \mathbf{Y}(f)|^2=|a^H(\theta,f)\mathbf{Y}(f)|^2
$$

*为了在今后计算中加以区分，以w代表延时权重，a代表导向矢量，这与clean-sc论文中不同，论文中的w为导向矢量（矩阵），因此本文中w的相位与论文是相反的。*

# 使用交叉谱矩阵

在实际工业场景中，往往不使用上述的单快拍FFT的方法，因为该方法中的Y(f)通常只来源于一段极短的时域信号（DFT的原理，详见[数学基础 | 小董的BLOG](https://dhkkkkkk.github.io/2025/10/17/math-base/)），也就是单快拍数据。这种约等于瞬时成像的方法容易受到随机噪声的干扰，从而产生一些虚假的声源信息。

因此，我们引入了多快拍计算方法，也就是在一次成像中使用多段时域信号进行计算，我们用一个矩阵表示运算结果：
$$
\mathbf{C}_{f_0} = E[\mathbf{Y}\mathbf{Y}^H] = \begin{bmatrix} E[Y_1 Y_1^*] & E[Y_1 X_2^*] & \dots \\ E[Y_2 Y_1^*] & E[Y_2 Y_2^*] & \dots \\ \vdots & \vdots & \ddots \end{bmatrix}
$$
这个矩阵就是交叉谱矩阵CSM，每个值代表了两个麦克风的相关程度，对角线则为当前麦克风的能量信息（无相位信息，因此在计算中可考虑不使用对角线元素）。其中X_n代表第n个麦克风在某个频率f0下的FFT 复数值。

在原来的单快拍计算中：
$$
P_{single} = |\mathbf{w} \mathbf{Y}|^2 = (\mathbf{w} \mathbf{Y})(\mathbf{w} \mathbf{Y})^H= \mathbf{w} \mathbf{Y} \mathbf{Y}^H \mathbf{w}^H
$$
当引入CSM后并加入对能量取平均，则有：
$$
P_{multi} = \mathbf{w} E[\mathbf{Y} \mathbf{Y}^H] \mathbf{w}^H = \mathbf{w} \mathbf{C} \mathbf{w}^H
$$
因此对于延迟权重向量（矩阵）W，其与单快拍计算中的完全一致，连形状都不用变。

# 声学橡皮擦（HDR）

当环境中存在多个声源时，较弱的声源容易直接被强声源覆盖，若想显示该弱声源，则可以通过波束形成单独复原强声源波形，再将该波形从麦克风信号中减去。对于该波形的波束形成，有：
$$
\hat f(s_1) = \frac 1 M \sum^M_{i=1} w_i(\theta,\phi)y_i
$$
*需要注意的是，这里与波束形成**成像**略有不同。此处在对每个通道的麦克风信号分别进行相位对其后，直接将所有通道信号相加再取平均，是最标准的波束形成。相加之后，在(θ，φ)方向上的信号会汇聚（因为对它们的相位进行了对齐），其他方向上的信号则因为相位不一致，最终被弱化。*

由于延迟权重w是根据参考阵元计算得到的，**因此最终复原的s1波形相位也与参考阵元相同**。所有我们如果想在每个通道中都减去s1，就要也把每个通道数据的相位先对齐至参考阵元（其实也还是完全一样的延时操作）：
$$
\mathbf{Y}'_{\text{w/o s1}} = \mathbf{w}(\theta,\phi) \mathbf{Y} -\hat f(s_1)
$$
减去s1后再把每个通道的麦克风数据按照**反方向**进行相位调整即可得到去除s1的各通道数据：
$$
\mathbf{Y}_{\text{w/o s1}} = \mathbf{w}^H(\theta,\phi)\mathbf{Y}'_{\text{w/o s1}}
$$

后续再进行前文中的成像即可，或者也可以直接使用以对齐的Y'进行成像（理论上可以，我没有试过，我是先用Y计算CSM后再计算的功率）

同时，我们也可以单独的对s1进行更清晰的成像，排除弱声源的干扰：
$$
P_{s_1} = |\mathbf{w}(\theta,\phi)(\mathbf{Y}-\mathbf{Y}_{\text{w/o s1}})|^2
$$
虽然感觉减来减去的有点多此一举，但是论文中确实也是这样写的。理论上对\hat f(s_1)根据延迟情况复制M个直接成像也可以？我懒得试了。总之最重要的是在进行波束形成时务必要**通过求和**，才可以获得估计的s1，也就是本小节第一个公式，其他的大家可以自由发挥。
