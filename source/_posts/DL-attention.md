---
title: 注意力机制原理与实现
date: 2025-10-05 15:18:37
tags: 深度学习
mathjax: true
---

# 注意力机制原理

在传统的序列处理模型中，如循环神经网络（RNN）和长短时记忆网络（LSTM），捕捉**长距离依赖关系是一个难题**。因为随着序列长度的增加，模型很容易丢失早期输入的信息。

注意力机制允许模型在序列的不同位置之间建立直接联系，无论这些位置相距多远，都能够有效地捕捉到它们之间的依赖关系。

## 最原始的注意力

对于输入X=(N,D)，其中D为每个token的embedding dimension，N为序列长度，为了得到某个特定任务的相关信息，引入一个和token同维度的查询向量q（q是一个和任务相关的表示，这里暂时理解为一个抽象的概念，后续会详细说明实际中的生成方式），通过计算每个token和q的相关性以得到注意力分布：
$$
\alpha_n=softmax(s(x_n,q))
$$
x_n为X中的第n个token；α为q对x_n的**关注程度**；s(.)为一个**打分函数**，用于计算token和q的相关性，最长用的是缩放点积模型：
$$
s(x,q)=\frac{qx^T}{\sqrt{D}}
$$
当token维度D较大时，如果不对点积结果进行缩放，会由于较大的方差导致softmax梯度较小

若想得到输入X与查询向量q的整体相关性，则可以通过加权平均获得，即直接将关注程度与token相乘再求和：
$$
att(X,q)=\sum_{n=1}^{N} \alpha_n x_n
$$
因此attention机制**可以单独使用**，但其更多地还是作为神经网络中的一个组件使用，并且现在主流的attention算法与上述还存在一定差别

## 自注意力模型

为了提高模型的表达能力，现在主流的attention模型使用查询-键-值(qkv)模式进行自注意力计算，对于上一小节中的打分函数，即缩放点积模型则变为一个矩阵：
$$
s(K,Q)=\frac{QK^T}{\sqrt{D_k}}
$$
其中Q和K为X分别经过一个线性变换得到的(D_k,N)的矩阵：
$$
Q(K)=W_{q(k)}X
$$
对于pytorch中则通常直接通过linear实现：

```python
Q = nn.Linear(embedding_dim, mid_dim, bias=False)
```

最终自注意模型输出仍为一个序列：
$$
Y=softmax(s(K,Q)) V=AV
$$
输出Y的embedding 维度取决于V，而V也是由X进行线性变换而来，因此输出的embedding维度等于V线性变换矩阵的行数；

**自注意机制的“自”就体现在Q，K，V都是由同一个X经过三个不同的线性变换得来**

对于关注度与V相乘的理解，我是这样理解的：

将Y的矩阵乘法展开，有：
$$
Y_i=\sum_{j=1}^nA_{ij}V_j
$$
因此对于Y_i，也就是每个输出的token，其是由注意力权重（关注程度）A对**所有输入token的加权<u>求和</u>**得来，**其包含了与所有输入token的相关程度**；对于A的理解，感觉不用过于较真，只需要知道A的每**一个**元素都代表了某个q和某个k之间的相关程度就行（这里的小写指向量而非矩阵）

(真要说的话，由公式可知，举个例子：A_12则为q_1对k_2的关注度)

## 多头注意力

多头注意力其实就是在计算注意力的过程中（线性变换之后），**将embedding dim拆为n份**，每份单独计算注意力，最后再进行拼接回完整输出。这样做的好处就是每个token之间存在的一些**局部相关性**会被更好地表示，论文原话是：

*Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions.多头注意力允许模型共同关注来自不同位置的不同表示子空间的信息。*

需要澄清的一点概念是，对于多头注意力在训练的过程中，**其权重（线性变换参数）仍然只训练了一次**，只是该次的训练综合了每个头的注意力结果，使训练后的模型能**同时捕捉多种类型的依赖关系**，这种依赖关系在物理意义上可以理解为两个词之间的语法关系、语义关系、位置关系、情感关系等

## 交叉注意力

对于自注意力，其Q，K，V都来自于同一个输入，始终是一个序列内的不同token在相互计算相关程度；而对于交叉注意力，K，V则来自于另一个序列；因此交叉注意力常出现在需要处理多种不同输入的场景（如机器翻译，语义生图等），以翻译为例，K，V为源语言，则Q为目标翻译语言

# 代码解读

该代码为多头注意力的实现：

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, dropout_rate, head_size=8):
        super(MultiHeadAttention, self).__init__()

        self.head_size = head_size	#多头数

        self.att_size = att_size = hidden_size // head_size
        self.scale = att_size ** -0.5 #缩放点积模型中的缩放值

        self.linear_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.linear_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.linear_v = nn.Linear(hidden_size, hidden_size, bias=False)

        self.att_dropout = nn.Dropout(dropout_rate)

        self.output_layer = nn.Linear(head_size * att_size, hidden_size,
                                      bias=False)


    def forward(self, q, k, v, mask=None, cache=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

		#在经过线性变换后，将(b,n,d) -> (b,n, 8, 原dim/8)
        q = self.linear_q(q).view(batch_size, -1, self.head_size, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.head_size, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.head_size, d_v)

        #将头数前置方便运算，k也在这里顺便进行了转置
        q = q.transpose(1, 2)                  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)                  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        #这里先进行了缩放，缩放先后无影响
        q.mul_(self.scale)
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
		
        #由之前的矩阵乘法展开可知，对于每一个最终输出的token，其使用的是A_ij(j=0,1,2...)，因此要对A的排维度进行softmax
        x = torch.softmax(x, dim=-1)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]
		
        #拼接token embedding 维度
        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.head_size * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x
```

