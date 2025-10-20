---
title: 深度学习基础杂项
date: 2025-10-05 15:14:37
tags: 深度学习
mathjax: true
---

# 常用激活函数

## sigmoid

指一类**<u>S型</u>的两端饱和函数**，常见的有logistic和Tanh及其变种

### logistic

$$
\sigma(x)=\frac{1}{1+e^{-x}}
$$

由于最大导数只有0.25，两端饱和严重，输出始终为正，导致梯度下降变慢，现在深度学习隐藏层中已很少使用

### Tanh

$$
\sigma(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}
$$

零中心化，优于logistic

## ReLU

$$
f(x) = \max(0, x)
$$

导数1缓解了梯度消失问题

0的部分导致了该神经元的输入参数死亡，永不更新（注：死亡ReLU和梯度消失不是一回事）

### LeakyReLU

$$
f(x) = \max(0, x)+\gamma\min(0,x)
$$

gamma为一个很小值（torch中默认e-2），代替ReLU的0

### PReLU

将gamma换作可学习的参数

### Swish

$$
f(x)=x\sigma(\beta x)
$$

自门控激活函数。其中sigma为logisitc函数，beta是一个可学习或超参数。

当logistic接近1时，swish输出接近x，logistic接近0时，输出接近0但不为0。

形状类似LeakyRelu

# 损失函数

## 均方误差（Mean Squared Error ）

$$
MSE = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2
$$

其中y^hat为模型输出（概率），y为标签值（真1假0）

在分类问题中，使用sigmoid/softmx得到概率，配合MSE损失函数时，采用梯度下降法进行学习时，会出现模型**一开始训练时，学习速度非常慢**的情况

## 交叉熵（Cross Entropy Loss Function）

$$
L = \frac{1}{N} \sum_{i=1}^{N} L_i = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{M} y_{ic} \log(p_{ic})
$$

其中N可以理解为batchsize（只是最后对所有样本的交叉熵求平均），M为类别数量（模型输出维度），y_ic为标签值（真1假0），p_ic为模型对每个类的输出概率

*<u>对于torch，可选择是否最终除以N</u>*

# 反向传播算法

对于某个神经元的输出，设为（假设无偏置项）：
$$
out^{(l)}=f(\omega x)=in^{(l+1)}
$$
其中，l上标指网络层数，f为激活函数。（其实ωx是一个矩阵，为方便理解，假设该神经元只有一个输入）对于梯度下降算法中权重的更新，有：
$$
\omega _{new}= \omega - \alpha \frac{\partial L}{\partial \omega}
$$
由链式法则可知：
$$
\frac{\partial L}{\partial \omega} = \frac{\partial L}{\partial f}
\cdot \frac{\partial f}{\partial (\omega x)}
\cdot \frac{\partial (\omega x)}{\partial \omega}=
\frac{\partial L}{\partial out^{(l)}} \cdot f' \cdot out^{(l-1)}
$$
一般称损失函数对l层该神经元输出的偏导为**误差项**δ，则损失函数对该权重的偏导则为：

误差项  x  **该神经元**激活函数的**梯度**  x  **该权重对应上一神经元**的输出

而误差项的计算：


$$
out^{l+1}=f_{l+1}(\omega^{(l+1)} out^{(l)})
\\
\frac{\partial L}{\partial out^{(l)}}= \frac{\partial L}{\partial out^{(l+1)}}
\cdot \frac{\partial out^{l+1}}{\partial f_{l+1}}
\cdot \frac{\partial f_{l+1}}{\partial out^{(l)}}= \delta^{(l+1)} \cdot f_{(l+1)}' \cdot \omega^{(l+1)}
$$
注意此处最后一导是对l层该神经元的输出偏导，所以结果是l+1层的权重。（我对激活函数的理解与标准公式中梯度的位置不太一样，但其实最终结果都是一样的。即：
$$
\frac{\partial L}{\partial \omega_{(l)}}=\frac{\partial L}{\partial out^{(last)}} \\ \cdot f_{(last)}'\cdot \omega^{(last)}\cdot...
\cdot  \cdot f_{(l+1)}' \cdot \omega^{(l+1)} \cdot
\\
 f_{(l)}' \cdot out^{(l-1)}
$$
最后一层的输出即模型的输出（概率），总结下来就是：

```
损失函数对l层权重的偏导 = 损失函数对模型输出的偏导 *
					  最后一层到l层的所有激活函数的梯度 *
					  最后一层到l+1层的权重和 *
					  l-1层的神经元输出
```

（题外话，由该公式可理解死亡ReLU造成的是在反向传播中该神经元之前所有相关线路中参数更新的直接死亡，而梯度消失问题则是由于某些激活函数饱和区导数接近0导致的网络参数更新缓慢）

# 自动微分与计算图

本节以pytorch操作为例

自动微分的存在是为了方便对参数梯度的计算，其基本原理是将所有的数值计算分解为基本数值运算和一些初等函数，最终通过链式法则自动计算该复合函数的梯度

```python
x = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
y = (torch.exp(-(2*x + 1) + 1) + 1)
y.retain_grad()
z = 1 / y
z.sum().backward()
print(x.grad) # tensor([0.2100, 0.0353, 0.0049, 0.0007])
print(y.grad) # tensor([-0.7758, -0.9644, -0.9951, -0.9993])
```

此时x到z的计算即为：
$$
z=\frac{1}{y}=\frac{1}{e^{(-(2x+1)+1)}+1}
$$
其计算图为：
$$
2x>2x+1>-(2x+1)>-(2x+1)+1>...>\frac{1}{e^{(-(2x+1)+1)}+1}
$$
因此若要计算z对x的导数，则可以通过链式法则+反向传播进行求解，因为每个子函数的导数都很简单，所以降低了运算量

若想单独得到某一步运算中tensor的梯度，则需要向上述代码中的y一样先单独保存该tensor并保存其梯度

## 在深度学习中

在训练神经网络时，backward会计算出loss计算图中所有节点的梯度，包括激活函数梯度等，然后后续再通过用户设置不同的优化器对参数进行优化计算，因此在代码中需要先backward然后再optimizer，backward只负责计算梯度，不负责优化（训练）参数。

因此，当网络中存在并联结构时（例如一个encoder后并联接入两个decoder），如果只想让一个decoder能影响encoder的更新（让另一个decoder自己单独训练），可以对这个单独训练的decoder输入进行.detach()操作以断掉其之前的计算图，这样对该decoder进行反向传播时梯度就只会计算至detach处（注：只要使用了detach，该条线路上之前的神经元就无论如何都无法被该线路的loss优化所训练了）

# 残差结构设计注意事项

对于残差结构最后相加处，主分支在最后一层网络后不应该再经过激活函数，而是应该直接与残差分支相加后再通过激活函数（但是主分支再相加前还是要进行归一化操作）
