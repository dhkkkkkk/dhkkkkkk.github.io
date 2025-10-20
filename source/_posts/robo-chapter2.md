---
title: 空间描述和变换
date: 2025-10-14 09:29:01
tags: 机器人学
mathjax: true
---

# 位姿描述与变换

## 姿态描述

对于空间中一个坐标系{B}相对于世界坐标系{W}的姿态，可以通过旋转矩阵R进行表示：
$$
^W_B\mathbf{R} = \begin{bmatrix}
r_{11} & r_{12} & r_{13} \\
r_{21} & r_{22} & r_{23} \\
r_{31} & r_{32} & r_{33}
\end{bmatrix}
$$


旋转矩阵仅描述当前坐标系相对于世界坐标系的**旋转姿态**，<u>不包含平移信息</u>，因此可以将旋转矩阵的**每一列**想象成将原点重合后，当前坐标系X、Y、Z**单位矢量**的坐标投影。因此也有：
$$
^W_B\mathbf{R} = \begin{bmatrix}
\hat{X}_B \cdot \hat{X}_W & \hat{Y}_B \cdot \hat{X}_W & \hat{Z}_B \cdot \hat{X}_W \\
\hat{X}_B \cdot \hat{Y}_W & \hat{Y}_B \cdot \hat{Y}_W & \hat{Z}_B \cdot \hat{Y}_W \\
\hat{X}_B \cdot \hat{Z}_W & \hat{Y}_B \cdot \hat{Z}_W & \hat{Z}_B \cdot \hat{Z}_W
\end{bmatrix}
$$
由于这些矢量均为单位向量，因此它们的点积则直接代表了两个矢量夹角的余弦值（矢量点积的性质），但需要注意的是，**对于两个坐标系而已，它们之间的互相形成的两个旋转矩阵不是相同的（互为转置）**

由于旋转是相对的、可逆的，因此旋转矩阵为一个正交阵（逆等于转置）

## 坐标系描述

当已知旋转矩阵后，只需知道坐标系{B}原点相对于参考坐标系（还是假设为世界坐标系{W}）的坐标，即可描述坐标系{B}的所有信息：
$$
\{B\}=\{^W_B\mathbf{R},^W\mathbf{P_{B_{org}}}\}
$$
其中P是坐标系{B}原点**在{W}中的坐标向量**

因此，一个坐标系可以通过任意另一坐标系表示出来

## 变换映射

对于{B}中的一个矢量P，若想用参考坐标系{A}将其表示，对于单纯平移，有：
$$
^AP=^BP+^W\mathbf{P_{B_{org}}}
$$
对于单纯旋转，则有：
$$
^AP=^A_B\mathbf{R}^BP
$$
因此对于一般情况，则有：
$$
\begin{bmatrix}^AP \\ 1\end{bmatrix}=\begin{bmatrix}^A_B\mathbf{R} & ^AP_{B_{org}}\\
000 & 1\end{bmatrix}\begin{bmatrix}^BP \\ 1\end{bmatrix}
$$
其中4x4的T矩阵就是**齐次变换矩阵**，当只进行旋转或平移时，只需令旋转矩阵为E或平移矩阵为0，0，0即可

## 注意点

当存在多个坐标系转换时：（注意T的合成是右乘）
$$
^AP=^A_B\mathbf{T}^B_C\mathbf{T}^CP=^A_CT^CP
$$
其中：
$$
^A_CT=\begin{bmatrix}^A_B\mathbf{R}^B_C\mathbf{R} & ^AP_{B_{org}}+^A_B\mathbf{R}^BP_{C_{org} }
\\
000 & 1\end{bmatrix}
$$
对于旋转映射，有：（注意顺序）
$$
^A_C\mathbf{R}=^A_B\mathbf{R}^B_C\mathbf{R}
$$
**其中需要注意的是，在A齐次变换矩阵中代表平移的映射：**
$$
^AP_{C_{org}}=^AP_{B_{org}}+^A_B\mathbf{R}^BP_{C_{org}} \neq ^AP_{B_{org}}+^BP_{C_{org}}
$$
虽然在空间上来看{C}原点在{A}中的坐标矢量感觉就是三个坐标系原点间矢量的首尾相加，**<u>但其实对C原点在{A}中的矢量坐标都应该统一到{A}坐标系下进行计算</u>**，而
$$
^BP_{C_{org}}
$$
是C原点在{B}下的坐标矢量，其在{A}中的表达还需要乘一个{B}相对于{A}的**旋转映射**

综上，可以推导出，在多个坐标系变换时，平移映射中只有{A}的下一个坐标系平移映射不需要乘旋转映射，其余所有平移映射都需要乘该坐标系相对于{A}的旋转映射，即：
$$
^AP_{N_{org}}=^AP_{B_{org}}+^A_B\mathbf{R}^BP_{C_{org}}+^A_C\mathbf{R}^CP_{D_{org}}+...+^A_{N-1}\mathbf{R}^{N-1}P_{N_{org}}
$$
但一般而言，想要直接得到各坐标系相对参考坐标系{A}的旋转矩阵是比较困难的，所以更常见的写法是：
$$
{}^A P_{N_{org}} 
= {}^A P_{B_{org}}
+ {}^A_{B}\mathbf{R}
\Big(
{}^B P_{C_{org}}
+ {}^B_{C}\mathbf{R}
\big(
{}^C P_{D_{org}}
+ {}^C_{D}\mathbf{R}
\big(
{}^D P_{E_{org}}
+ \cdots
+ {}^{N-1}_{N}\mathbf{R}\,{}^{N}P_{N_{org}}
\big)
\big)
\Big)
$$
也就是齐次变换矩阵相乘后**平移项**的表达

## 逆变换

对于求其次矩阵的逆变换，对于**旋转矩阵，由于其为正交阵，因此直接取转置**即可；对于平移算映射，有：
$$
^BP_{A_{org}}=-^A_BR^T{^AP_{B_{org}}}
$$
即最终逆变换齐次变换矩阵为：
$$
^B_AT=\begin{bmatrix}^A_B\mathbf{R}^T & -^A_BR^T{^AP_{B_{org}}}\\
000 & 1\end{bmatrix}
$$
推广到N个坐标系中时，注意此时旋转算子乘积顺序是**反序的**

# 其次变换矩阵的其他定义

对于齐次变换矩阵：
$$
^A_BT
$$
其本身描述了相对于坐标系{A}的{B}，同时其也可以描述不同坐标系中同一个矢量的映射关系

齐次变换矩阵同时**也可用于描述矢量单纯的旋转、平移变换**，这时其被称为变换算子：
$$
\begin{bmatrix}^AP_1 \\ 1\end{bmatrix}=\begin{bmatrix}\mathbf{R} & Q\\
000 & 1\end{bmatrix}\begin{bmatrix}^AP_2 \\ 1\end{bmatrix}
$$
其中R为旋转算子，Q为平移算子；

在多次变换中，对于平移只需要进行加减即可；对于旋转变换，其顺序会影响R的值（虽然最终结果是一样的）：
$$
R_z(30)R_x(30) \neq R_x(30)R_z(30)
$$
