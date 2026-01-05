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
因此对于Y_i，也就是每个输出的token，其是由注意力权重（关注程度）A对**所有输入token的加权<u>求和</u>**得来，**其包含了与所有输入token的相关程度**；对于A的理解，其第n行代表了**第n个**q<u>向量</u>与**所有**key<u>向量</u>的相关程度

这里对A的理解很重要，涉及到attention中padding mask 和 causal mask的设计理解

(举个例子：A_12则为q_1对k_2的关注度)

## 多头注意力

多头注意力其实就是在计算注意力的过程中（线性变换之后），**将embedding dim拆为n份**，每份单独计算注意力，最后再进行拼接回完整输出。这样做的好处就是每个token之间存在的一些**局部相关性**会被更好地表示，论文原话是：

*Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions.多头注意力允许模型共同关注来自不同位置的不同表示子空间的信息。*

需要澄清的一点概念是，对于多头注意力在训练的过程中，**其权重（线性变换参数）仍然只训练了一次**，只是该次的训练综合了每个头的注意力结果，使训练后的模型能**同时捕捉多种类型的依赖关系**，这种依赖关系在物理意义上可以理解为两个词之间的语法关系、语义关系、位置关系、情感关系等

## 交叉注意力

对于自注意力，其Q，K，V都来自于同一个输入，始终是一个序列内的不同token在相互计算相关程度；而对于交叉注意力，K，V则来自于另一个序列；因此交叉注意力常出现在需要处理多种不同输入的场景（如机器翻译，语义生图等），以翻译为例，K，V为源语言，则Q为目标翻译语言

# 代码解读（此版本以过时，直接看下一节）

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

# 再战Transformer   2025.12.15

仔细想想自己好像还没用过标准的transformer，为了更清晰地理解LLM，自己手撕了一遍，主要看个思路，代码也许还存在一些bug，如果真要用还是建议去github找star多的

## Attention

```python
class MutiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_head, dropout_rate=0.):
        super(MutiHeadAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_head
        self.attn_size = hidden_size // num_head
        self.dropout = nn.Dropout(dropout_rate)

        self.wq = nn.Linear(self.hidden_size, self.hidden_size)  # 启用bias
        self.wk = nn.Linear(self.hidden_size, self.hidden_size)
        self.wv = nn.Linear(self.hidden_size, self.hidden_size)
        self.output_layer = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, q, k, v, padding_mask=None, causal_mask=None):
        batch_size = q.size(0)
        q_len, k_len = q.size(1), k.size(1)
        k_dim = k.size(2)
        v_dim = k_dim
        # 将feature dim分头,并将头数前置方便运算,(b,num_head,len,d)
        Q = self.wq(q).view(batch_size, -1, self.num_heads, self.attn_size).transpose(1, 2)
        K = self.wk(k).view(batch_size, -1, self.num_heads, self.attn_size).transpose(1, 2)
        V = self.wv(v).view(batch_size, -1, self.num_heads, self.attn_size).transpose(1, 2)

        x = torch.matmul(Q, K.transpose(2, 3)) / np.sqrt(self.attn_size)

        # 创建多头padding mask，因为在计算注意力时序列长度是不改变的，因此直接repeat即可
        # mask是作用于key的，因此其具体掩码内容仅由key的padding情况决定
        if padding_mask is not None:
            assert padding_mask.size() == (batch_size, q_len, k_len)
            padding_mask = padding_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            padding_mask = padding_mask.bool()
            x = x.masked_fill_(padding_mask, -1e9)

        # 我们不希望decoder自注意力阶段中中第n个token获得对未来token的注意力，因为实际预测中token是逐渐生成的，而训练时是全部存在的
        if causal_mask is not None:
            assert causal_mask.size() == (batch_size, q_len, k_len)
            causal_mask = causal_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            causal_mask = causal_mask.bool()
            x = x.masked_fill_(causal_mask, -1e9)

        # score的第n行代表第n个q向量与所有key向量的关联程度，因此该对每一行做softmax
        attn_score = torch.softmax(x, dim=-1)
        attn_score = self.dropout(attn_score)

        output = torch.matmul(attn_score, V)
        output = output.transpose(1, 2).contiguous().reshape(batch_size, -1, self.hidden_size)
        output = self.output_layer(output)

        return output
```

相关代码已经做了注释，这里再强调几个点：

### 关于掩码mask

在标准Transformer中，有两种mask：padding mask 和 causal mask，它们分别用于屏蔽将所有输入（通常是一个batch内）填充至同一长度的pad（因为我们不希望模型将这些pad视作语义的一部分），以及在训练中防止decoder在预测token时看到未来部分。

* Encoder：在encoder中，模型需要学习整个输入的全局语义信息，因此其仅使用self-attention模块，且仅使用padding mask
* Decoder：decoder包含了self-attention和cross-attention（关于他们的区别理解可查看[LLM学习 | 小董的BLOG](https://dhkkkkkk.github.io/2025/11/21/LLM/)）。
  * 在self-attention中，由于decoder的输入在训练时为整个语句，而实际预测时输入为逐个token，因此在训练时需要causal mask屏蔽掉未来信息。
  * 在cross-attention中，q为self-attention输出，而k，v则为encoder输出，因为要用q查询其与整个语句的关注度，因此不需要causal mask，仅使用padding mask。但需要注意的是，attention中score的形状为（q_len, k_len），因此此处的padding mask 和encoder 阶段的形状是不同的（q来源不同，但k形状是相同的，都来自encoder的输入）

关于mask的具体实现在后面会讲

### 维度的变化

attn_score的维度为（b,h, q_max_len, k_max_len)，max是指填充过后的长度，由于k和v的形状永远是相同的（但具体内容不同，因为它们经过了不同的投影层），因此attention最终输出的形状为(b,q_max_len,embedding_dim)

### 关于score的理解

第n行代表了第n个q向量对所有k向量的关注程度

## Feedforward

```python
class FeedForward(nn.Module):
    def __init__(self, hidden_size, scale_p, dropout_rate=0.):
        super(FeedForward, self).__init__()
        self.hidden_size = hidden_size
        self.ffn_size = int(scale_p) * hidden_size
        self.linear1 = nn.Linear(self.hidden_size, self.ffn_size)
        self.linear2 = nn.Linear(self.ffn_size, self.hidden_size)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.dropout(out)

        return out
```

由于attention更多的是在时间维度上进行信息融合，并没有在**特征维度上进行表达提升和非线性变换**，因此feedforward就用于补充这一不足而被使用，其本身其实就是特征维度的特征变换

## Encoder

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

    
class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, num_head, attn_dropout, ffn_scale, ffn_dropout):
        super(EncoderLayer, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_head
        self.ffn_scale = ffn_scale
        self.attn = MutiHeadAttention(self.hidden_size, self.num_heads, attn_dropout)
        self.FFN = FeedForward(self.hidden_size, self.ffn_scale, ffn_dropout)
        self.norm1 = nn.LayerNorm(self.hidden_size)  # 两个norm需分别初始化
        self.norm2 = nn.LayerNorm(self.hidden_size)

    def forward(self, x, padding_mask):
        enc_in = x
        x = self.norm1(x)  # pre-norm更稳定
        out = enc_in + self.attn(x, x, x, padding_mask)  # 残差要使用norm前的数据

        x = self.norm2(out)
        out = out + self.FFN(x)

        return out

    
class Encoder(nn.Module):
    def __init__(self, max_len, hidden_size, num_head, num_layer, ffn_scale,
                 emb_dropout, attn_dropout, ffn_dropout):
        super(Encoder, self).__init__()
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.pos_embedding = PositionalEncoding(self.hidden_size, dropout=emb_dropout, max_len=self.max_len)
        self.Layers = nn.ModuleList(
            [EncoderLayer(self.hidden_size, num_head, attn_dropout, ffn_scale,
             ffn_dropout) for _ in range(num_layer)])

    def forward(self, x, padding_mask):
        out = self.pos_embedding(x)  # embedding & add & dropout
        for layer in self.Layers:
            out = layer(out, padding_mask)

        return out
```

在Encoder layer中，现在主流的做法是**前置**layernorm，这样会获得更稳定的注意力结果；并且要注意的是，**残差一定使用的是归一化之前的值**

## Decoder

```python
class DecoderLayer(nn.Module):
    def __init__(self, hidden_size, num_head, attn_dropout, ffn_scale, ffn_dropout):
        super(DecoderLayer, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_head
        self.ffn_scale = ffn_scale
        self.self_attn = MutiHeadAttention(self.hidden_size, self.num_heads, attn_dropout)
        self.cross_attn = MutiHeadAttention(self.hidden_size, self.num_heads, attn_dropout)
        self.FFN = FeedForward(self.hidden_size, self.ffn_scale, ffn_dropout)
        self.norm1 = nn.LayerNorm(self.hidden_size)
        self.norm2 = nn.LayerNorm(self.hidden_size)
        self.norm3 = nn.LayerNorm(self.hidden_size)

    # 由于decoder的len_k和encoder的不同，因此padding mask 需要分开
    def forward(self, x, enc_out, enc_padding_mask, dec_padding_mask, causal_mask):
        dec_in = x
        x = self.norm1(x)
        out = dec_in + self.self_attn(x, x, x, dec_padding_mask, causal_mask)

        x = self.norm2(out)
        out = out + self.cross_attn(x, enc_out, enc_out, enc_padding_mask, None)

        x = self.norm3(out)
        out = out + self.FFN(x)

        return out


class Decoder(nn.Module):
    def __init__(self, max_len, hidden_size, num_head, num_layer, ffn_scale,
                 emb_dropout, attn_dropout, ffn_dropout):
        super(Decoder, self).__init__()
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.pos_embedding = PositionalEncoding(self.hidden_size, dropout=emb_dropout, max_len=self.max_len)
        self.Layers = nn.ModuleList(
            [DecoderLayer(self.hidden_size, num_head, attn_dropout, ffn_scale, 						ffn_dropout) for _ in range(num_layer)])

    def forward(self, x, enc_out, dec_padding_mask, causal_mask, enc_padding_mask):
        out = self.pos_embedding(x)
        for layer in self.Layers:
            out = layer(out, enc_out, enc_padding_mask, dec_padding_mask, causal_mask)

        return out

```

关于decoder中的padding mask 和 causal mask在后面统一讲

## Transformer

```python
class Transformer(nn.Module):
    def __init__(self,
                 hidden_size,
                 num_head,
                 enc_max_len,
                 dec_max_len,
                 num_enc_layer,
                 num_dec_layer,
                 ffn_scale,
                 emb_dropout,
                 attn_dropout,
                 ffn_dropout
                 ):
        super(Transformer, self).__init__()
        self.Encoder = Encoder(enc_max_len, hidden_size, num_head, num_enc_layer, ffn_scale,
                               emb_dropout, attn_dropout, ffn_dropout)
        self.Decoder = Decoder(dec_max_len, hidden_size, num_head, num_dec_layer, ffn_scale,
                               emb_dropout, attn_dropout, ffn_dropout)

    @staticmethod
    def get_enc_padding_mask(max_len: int, seq_len: torch.Tensor, batch_size: int, device: torch.device):
        enc_padding_mask = torch.zeros((batch_size, max_len, max_len), device=device)
        for i in range(batch_size):
            enc_padding_mask[i, :, seq_len[i]:] = 1

        return enc_padding_mask.bool()

    @staticmethod
    def get_dec_padding_mask(max_q_len: int, max_k_len: int, seq_len: torch.Tensor, batch_size: int,
                             device: torch.device):
        dec_padding_mask = torch.zeros((batch_size, max_q_len, max_k_len), device=device)
        for i in range(batch_size):
            dec_padding_mask[i, :, seq_len[i]:] = 1

        return dec_padding_mask.bool()

    @staticmethod
    def get_causal_mask(max_len: int, batch_size: int, device: torch.device):
        causal_mask = torch.triu(torch.ones((batch_size, max_len, max_len), device=device), diagonal=1)

        return causal_mask.bool()

    # enc_in_len 和 dec_in_len 是每个样本的实际长度，因此是一个(b,l)的tensor
    def forward(self, enc_in, enc_in_len, dec_in, dec_in_len):
        batch_size = enc_in.size(0)

        #取每个batch中的最长
        enc_in_maxlen = enc_in.size(1)
        dec_in_maxlen = dec_in.size(1)

        enc_padding_mask = self.get_enc_padding_mask(enc_in_maxlen, enc_in_len, batch_size, enc_in.device)
        enc_out = self.Encoder(enc_in, enc_padding_mask)

        # 自注意力中如果pad全在尾部则不需要padding mask
        causal_mask = self.get_causal_mask(dec_in_maxlen, batch_size, dec_in.device)
        self_padding_mask = None
        # cross attention mask
        enc_dec_padding_mask = self.get_dec_padding_mask(dec_in_maxlen, enc_in_maxlen,
                                                     dec_in_len, batch_size, dec_in.device)

        dec_out = self.Decoder(dec_in, enc_out, self_padding_mask, causal_mask, enc_dec_padding_mask)

        return dec_out #未添加最后的投影层
```

这里主要再讲一下mask的逻辑和具体实现：

### padding mask

首先我们需要知道的是，无论是padding mask 还是causal mask，其都是直接作用于attention中的注意力权重attn_score，其形状为(q_len,k_len)，每一行代表了第n个q向量对所有k向量的关注程度。

现在，由于我们的输入中存在pad，因此无论是Q还是K中都存在着pad，但我们**不想让Q关注到K中的pad**，同时，另一个重要的点就是**我们不关心<u>Q的pad对K的关注</u>**，具体原因不在这里展开，感兴趣的可以去搜一下。

假设`K=[a,b,c,<pad>,<pad>]`,则padding mask就该是：

```python
[[0,0,0,1,1],
[0,0,0,1,1],
[0,0,0,1,1],
[0,0,0,1,1],
[0,0,0,1,1]]
#如果Q=K，那么score中的第4、5行则是我们不关心的
```

此时我们的代码则为：

```python
    @staticmethod #如果不需要调用self.，可以使用这个略微提升性能与可读性
    def get_enc_padding_mask(max_len: int, seq_len: torch.Tensor, batch_size: int, device: torch.device):
        #先用0创建空白mask，再填1
        enc_padding_mask = torch.zeros((batch_size, max_len, max_len), device=device)
        for i in range(batch_size):
            enc_padding_mask[i, :, seq_len[i]:] = 1

        return enc_padding_mask.bool()
```

在当前batch中，我们会根据**每一个样本**的**有效长度**生成当前样本的padding mask。在我的代码中，1代表了需要mask的位置，因为attention代码中的`.masked_fill_`方法会将True的位置mask

**<u>另外，pad有可能不仅出现在尾部，但我的代码未考虑这种情况</u>**

### causal mask

causal mask**<u>仅出现</u>**在decoder的self-attention阶段，其目的是为了防止第n个q向量看到n之后的内容。我们已知attention score每一行代表了第n个q向量对所有k向量的关注程度，因此我们需要一个**上三角**mask：

```
[[0,1,1,1,1],
[0,0,1,1,1],
[0,0,0,1,1],
[0,0,0,0,1],
[0,0,0,0,0]]
```

这样第n个q就只能看到n之前的k向量了，相同地，1代表了需要mask的位置，通过调用`torch.triu`生成上三角矩阵，其大小仅与decoder的输入长度（padding后）有关

#### 关于decoder中的两个mask

也许你会产生这样一个疑问，在self-attention中，causal mask是不是天然的也起到了padding mask 的效果？因为对于有效的Q来说，其永远也看不到K中的pad。还是假设现在Q和K为`[a,b,c,<pad>,<pad>]`（自注意力中Q和K的pad是一样的），对于Q的最后一个有效token c，在经过causal mask后也只看得到a，b和自己；而对于a，b来说，它们连c都看不到，更别说pad了，那为什么在decoder的self-attention中还需要padding mask呢

答案其实很简单，也在之前就提到过，那就是**<u>pad不一定只出现在尾部</u>**。如果a被替换为pad，那causal mask就无法防止 b，c看到pad了

综上，我的代码中并没有解决该pad的问题，因此只用于理解transformer结构
