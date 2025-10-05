---
title: xLSTM结构梳理
date: 2024-11-13 16:50:54
tags: 深度学习
---

1源码中的sLSTM无法使用，本文只分析基于mLSTM的xLSTM结构梳理

# `xLSTMLMModel()`

实例化模型

```python
def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # x = self.token_embedding(idx)
        # x = self.emb_dropout(x)
        x = self.xlstm_block_stack(idx)
        logits = self.lm_head(x)#全连接层
        return logits
```



* <u>若用于语言模型</u>，则先通过`nn.Embedding(num_embeddings=.., embedding_dim=..)`进行一次词索引

  对于该函数，其作用是定义一个embedding模块，包含num_embeddings个张量，每个张量大小为embedding_dim。对于每个张量的索引则通过输入整型数字进行索引。

  ```python
  import torch
  import torch.nn as nn
  
  embedding = nn.Embedding(5, 3)
  input = torch.LongTensor([2])#对embeding模块中下标为2的张量进行索引
  e = embedding(input)
  
  print(embedding.weight)
  print(e)
  
  '''
  Parameter containing:
  tensor([[ 1.0561,  1.7934, -0.1304],
          [ 1.4425,  0.8412,  0.1474],
          [-0.0995,  0.1439, -0.7001],
          [ 0.3784, -1.0610,  0.0362],
          [ 0.5086, -0.5861, -0.4548]], requires_grad=True)
          
  tensor([[-0.0995,  0.1439, -0.7001]], grad_fn=<EmbeddingBackward0>)
  '''
  ```

  * `embedding()`的输入是任意维度的tensor格式，但tensor中的值必须为<=num_embeddings-1的数

  * 若输入的维度为[a,b,c]，则输出维度为[a,b,c,embedding_dim]
  * 语言模型中该方法可以使num_embeddings个词获得唯一的嵌入张量，张量大小为embedding_dim

* <u>若用于语言模型</u>，则将索引到的嵌入词向量通过`nn.Dropout()`进行正则化

  ```python
   m = nn.Dropout(0.2)
   input = torch.randn(20, 16)
   output = m(input)	#将input中随机20%的值变0，其他值则除以0.8
  ```

* **将要处理的tensor输入`xLSTMBlockStack(config=...)`模块中**

* 全连接层

## `xLSTMBlockStack()`

主要由**创建xlstm的block**和归一化组成，该类主要实现多个lstm块的实例化、拼接和归一化

blocks通过列表创建＋nn.ModuleList()拼接的方式进行实例化

[详解PyTorch中的ModuleList和Sequential - 知乎](https://zhuanlan.zhihu.com/p/75206669)

```python
self.blocks = self._create_blocks(config=config)
def _create_blocks(self, config: xLSTMBlockStackConfig):#先进行

    blocks = []#block通过列表形式创建
    for block_idx, block_type_int in enumerate(config.block_map):
        if block_type_int == 0:#根据该对象值判断创建哪种block
            config = deepcopy(self.config.mlstm_block)
            if hasattr(config, "_block_idx"):
                config._block_idx = block_idx
                config.__post_init__()
            blocks.append(mLSTMBlock(config=config))
        elif block_type_int == 1:
            config = deepcopy(self.config.slstm_block)
            if hasattr(config, "_block_idx"):
                config._block_idx = block_idx
                config.__post_init__()
            blocks.append(sLSTMBlock(config=config))
        else:
            raise ValueError(f"Invalid block type {block_type_int}")

    return nn.ModuleList(blocks)#拼接block

```

```python
def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:

    for block in self.blocks:
        x = block(x, **kwargs)

    x = self.post_blocks_norm(x)

    return x
```

`self.post_blocks_norm(x)`调用的`LayerNorm()`会在下一节说到

根据config.block_map中的值会创建n个block

### `mLSTMBlock()`

该类会继承父类`xLSTMBlock()`，该部分主要实现mLSTM的残差结构

```python
class xLSTMBlock(nn.Module):
    """An xLSTM block can be either an sLSTM Block or an mLSTM Block.

    It contains the pre-LayerNorms and the skip connections.
    """

    config_class = xLSTMBlockConfig

    def __init__(self, config: xLSTMBlockConfig) -> None:
        super().__init__()
        self.config = config
        embedding_dim = (
            self.config.mlstm.embedding_dim if self.config.mlstm is not None else self.config.slstm.embedding_dim
        )

        self.xlstm_norm = LayerNorm(ndim=embedding_dim, weight=True, bias=False)

        if self.config.mlstm is not None:
            self.xlstm = mLSTMLayer(config=self.config.mlstm)
        elif self.config.slstm is not None:
            self.xlstm = sLSTMLayer(config=self.config.slstm)
        else:
            raise ValueError("Either mlstm or slstm must be provided")

        if self.config.feedforward is not None:
            self.ffn_norm = LayerNorm(ndim=self.config.feedforward.embedding_dim, weight=True, bias=False)
            self.ffn = create_feedforward(config=self.config.feedforward)
        else:
            self.ffn_norm = None
            self.ffn = None

        self.reset_parameters()

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = x + self.xlstm(self.xlstm_norm(x), **kwargs)    #残差结构
        if self.ffn is not None:
            x = x + self.ffn(self.ffn_norm(x), **kwargs)
        return x

    def step(self, x: torch.Tensor, **kwargs) -> tuple[torch.Tensor, dict[str, tuple[torch.Tensor, ...]]]:
        x_xlstm, xlstm_state = self.xlstm.step(self.xlstm_norm(x), **kwargs)
        x = x + x_xlstm
        if self.ffn is not None:
            x = x + self.ffn(self.ffn_norm(x), **kwargs)
        return x, xlstm_state

    def reset_parameters(self) -> None:
        self.xlstm.reset_parameters()
        self.xlstm_norm.reset_parameters()
        if self.ffn is not None:
            self.ffn.reset_parameters()
            self.ffn_norm.reset_parameters()

```

* forward中可以看到，mLSTM使用了残差结构
* `self.xlstm_norm = LayerNorm`实现了带有可选偏置（`bias`）和残差权重（`residual_weight`）的 归一化处理。
* 通过`mLSTMLayer(config=self.config.mlstm)`建立mLSTM层

#### mLSTMLayer

mLSTM层的底层实现

{% asset_img 1.jpg This is an image %} 

```python
class mLSTMLayerConfig(UpProjConfigMixin):
    conv1d_kernel_size: int = 4
    qkv_proj_blocksize: int = 4
    num_heads: int = 4
    proj_factor: float = 2.0

    # will be set toplevel config
    embedding_dim: int = -1
    bias: bool = False
    dropout: float = 0.0
    context_length: int = -1

    _num_blocks: int = 1
    _inner_embedding_dim: int = None

    def __post_init__(self):
        self._set_proj_up_dim(embedding_dim=self.embedding_dim)
        self._inner_embedding_dim = self._proj_up_dim


class mLSTMLayer(nn.Module):
    config_class = mLSTMLayerConfig

    def __init__(self, config: mLSTMLayerConfig):
        super().__init__()
        self.config = config

        self.proj_up = nn.Linear(
            in_features=self.config.embedding_dim,
            out_features=2 * self.config._inner_embedding_dim,
            bias=self.config.bias,
        )

        num_proj_heads = round(self.config._inner_embedding_dim // self.config.qkv_proj_blocksize)
        self.q_proj = LinearHeadwiseExpand(
            config=LinearHeadwiseExpandConfig(
                in_features=self.config._inner_embedding_dim,
                num_heads=num_proj_heads,
                bias=self.config.bias,
            )
        )
        self.k_proj = LinearHeadwiseExpand(
            config=LinearHeadwiseExpandConfig(
                in_features=self.config._inner_embedding_dim,
                num_heads=num_proj_heads,
                bias=self.config.bias,
            )
        )
        self.v_proj = LinearHeadwiseExpand(
            config=LinearHeadwiseExpandConfig(
                in_features=self.config._inner_embedding_dim,
                num_heads=num_proj_heads,
                bias=self.config.bias,
            )
        )

        self.conv1d = CausalConv1d(
            config=CausalConv1dConfig(
                feature_dim=self.config._inner_embedding_dim,
                kernel_size=self.config.conv1d_kernel_size,
            )
        )
        self.conv_act_fn = nn.SiLU()
        self.mlstm_cell = mLSTMCell(
            config=mLSTMCellConfig(
                context_length=self.config.context_length,
                embedding_dim=self.config._inner_embedding_dim,
                num_heads=self.config.num_heads,
            )
        )
        self.ogate_act_fn = nn.SiLU()

        self.learnable_skip = nn.Parameter(torch.ones(self.config._inner_embedding_dim, requires_grad=True))

        self.proj_down = nn.Linear(
            in_features=self.config._inner_embedding_dim,
            out_features=self.config.embedding_dim,
            bias=self.config.bias,
        )
        self.dropout = nn.Dropout(self.config.dropout)
        self.reset_parameters()

    	def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        B, S, _ = x.shape   #输入为张量维度为(batch,time,feature)
        '''
        batch：同dataloader输出的batch维度
        time:时间序列
        feature：每个序列的值，对于频谱则为特征频率的幅值
        '''
        # up-projection
        x_inner = self.proj_up(x)   #全连接层，输出第三维尺寸*2
        x_mlstm, z = torch.split(x_inner,			                                          split_size_or_sections=self.config._inner_embedding_dim, dim=-1)
		  #
        # mlstm branch
        x_mlstm_conv = self.conv1d(x_mlstm)
        x_mlstm_conv_act = self.conv_act_fn(x_mlstm_conv)

        q = self.q_proj(x_mlstm_conv_act)
        k = self.k_proj(x_mlstm_conv_act)
        v = self.v_proj(x_mlstm)

        h_tilde_state = self.mlstm_cell(q=q, k=k, v=v)

        h_tilde_state_skip = h_tilde_state + (self.learnable_skip * x_mlstm_conv_act)

        # output / z branch
        h_state = h_tilde_state_skip * self.ogate_act_fn(z)

        # down-projection
        y = self.dropout(self.proj_down(h_state))
        return y

    def step(
        self,
        x: torch.Tensor,
        mlstm_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor] = None,
        conv_state: tuple[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, tuple[torch.Tensor, ...]]]:
        # B, S, _ = x.shape

        # up-projection
        x_inner = self.proj_up(x)
        x_mlstm, z = torch.split(x_inner, split_size_or_sections=self.config._inner_embedding_dim, dim=-1)

        # mlstm branch
        x_mlstm_conv, conv_state = self.conv1d.step(x_mlstm, conv_state=conv_state)
        x_mlstm_conv_act = self.conv_act_fn(x_mlstm_conv)

        q = self.q_proj(x_mlstm_conv_act)
        k = self.k_proj(x_mlstm_conv_act)
        v = self.v_proj(x_mlstm)

        h_tilde_state, mlstm_state = self.mlstm_cell.step(q=q, k=k, v=v, mlstm_state=mlstm_state)

        h_tilde_state_skip = h_tilde_state + (self.learnable_skip * x_mlstm_conv_act)

        # output / z branch
        h_state = h_tilde_state_skip * self.ogate_act_fn(z)

        # down-projection
        y = self.dropout(self.proj_down(h_state))
        return y, {"mlstm_state": mlstm_state, "conv_state": conv_state}

    def reset_parameters(self):
        # init inproj
        small_init_init_(self.proj_up.weight, dim=self.config.embedding_dim)
        if self.proj_up.bias is not None:
            nn.init.zeros_(self.proj_up.bias)
        # init outproj
        wang_init_(self.proj_down.weight, dim=self.config.embedding_dim, num_blocks=self.config._num_blocks)
        if self.proj_down.bias is not None:
            nn.init.zeros_(self.proj_down.bias)

        nn.init.ones_(self.learnable_skip)

        def _init_qkv_proj(qkv_proj: LinearHeadwiseExpand):
            # use the embedding dim instead of the inner embedding dim
            small_init_init_(qkv_proj.weight, dim=self.config.embedding_dim)
            if qkv_proj.bias is not None:
                nn.init.zeros_(qkv_proj.bias)

        _init_qkv_proj(self.q_proj)
        _init_qkv_proj(self.k_proj)
        _init_qkv_proj(self.v_proj)

        self.mlstm_cell.reset_parameters()

```

