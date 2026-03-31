---
title: 语音算法面试
date: 2026-01-20 14:41:40
tags: 杂项
---

# leetcode

* list有顺序相关或需要去除某些值可以用**.sort方法排序**，将不想要的赋大值排到后面

* list推导式：`list1=[x for x in list1 if x...]`

* 当索引经常变化时用while比for好

* 字符串本身不可改变，当使用.replace操作时需要重新赋给一个新的字符串

* switch用法:	

  ```python
  match x:
  	case x=1: ...
  	case _: ...
  ```

* 返回字典指定键的值：`dict.get(key)`

* 查询是否有指定键： `if key in dict`

* `collections.Counter`查看一个字符串中各字符的出现次数

* `s.split(" ")`以空格为间隔截取字符串

# 项目

* 相位loss：计算最小相位差后，计算相位loss，频率和时间方向差分loss
* 平衡多任务：预训练增强任务，根据各loss大小分配系数权重
* 波束形成：延迟求和为基础，clean-sc提升动态范围，HDR
* STFT缺陷：无法平衡时频分辨率，窗宽选择冲突
* 小波缺陷：依赖基小波选择，对数据要求较高

# Transformer/Attention相关

* 端到端语音识别的基本流程，训练和推理时的区别

  * *特征提取模块：提取MFCC或Fbank特征*
  * *Encoder：输入特征，经过自注意力，feedforward，提取时序特征与全局依赖*
  * *Decoder：输入文本，经过自注意力，feedforward，交叉注意力（Q为decoder输入，KV为encoder输出），feedforward，**将encoder得到的语音特征**根据decoder当前输入，映射为最终的翻译token*
  * *输出层，将预测token映射到**词汇表上每个词的概率分布***
  * *训练推理的区别：训练输入为完整的语音特征和完整的文本，在decoder中使用right shift和causal mask防止未来文本泄露**（一次性并行输入）**；推理时decoder的输入为该步之前的decoder输出**（类串行）***

* Transformer和LSTM的区别

  * *建模机制：LSTM为**递归式顺序处理**序列数据，具有硬编码时间结构；Transformer是一次性的**关系建模**，本身不包含时间结构，需要使用位置编码*
  * *长程依赖：LSTM靠门控缓解了梯度消失、爆炸，但仍受限；transformer不受长程影响*
  * *并行能力：LSTM必须串行计算，GPU利用率低*

* Conformer

  *在原encoder、decoder模块中引入卷积（depthwise和pointwise）以**优化transformer局部建模能力弱**的问题；在头尾都使用了feedforward，更强非线性*

# 模型

* 提高鲁棒性的方法：
  * *前端：信号处理层面，滤波处理，波束形成*
  * *后端：模型层面，使用语音增强模型（GAN、多任务学习等）分离噪声；在encoder引入卷积增强局部建模（conformer）；训练时使用数据增强；*

* CTC的缺点：

  *CTC仅使用encoder，输出的是语言帧到文本token的对齐概率，**没有语言建模能力，会出现重复与漏识别***

# Pytorch

* pytorch的优点：丰富的库，GPU支持和反向传播自动微分机制（将所有的数值计算分解为基本数值运算和一些初等函数）
* 重塑维度方法：reshape，view(直接对Tensor使用），permute
* 梯度相关运算：首先反向传播，之后可以对输入调用`.grad`属性求得`.backward()`对象对该输入的偏导（梯度），由于自动微分机制，对于计算中间变量需单独`.retain_grad()`保留其梯度（[深度学习基础杂项 | 小董的BLOG](https://dhkkkkkk.github.io/2025/10/05/DL-base/)）

* 张量乘法：`.matmul`支持不同维度张量的广播乘积，`mm`仅支持同维度矩阵乘积

* 参数调用：

  * 查询模型子模块并解包：`*model.children()`，获得不包含fc的resnet模型：`nn.Sequential(*list(resnet.children())[:-1])`

  * 返回模型某一层or全部参数：`model.layer.paramaters() model.layer.named_paramaters()`

    冻结参数：

    ```python
    for param in model.layer1.parameters():
        param.requires_grad = False
    #or
    for name, param in model.layer1.named_parameters():
        if 'conv' in name:
        	param.requires_grad = False
    ```

* 学习率调度器：`torch.optim.lr_scheduler.StepLR`(步进衰减)，在每个训练epoch后调用其step()方法，使用方法同优化器

* 量化配合Lora如何实现？

* 剪枝方式有几种？

# 嵌入式

## freertos

* 共有几种状态
* 两种调度模式
* 我常用的任务交互：队列和任务通知
