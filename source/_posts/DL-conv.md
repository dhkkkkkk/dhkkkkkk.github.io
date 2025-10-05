---
title: CNN扫盲
date: 2025-10-05 15:17:22
tags: 深度学习
mathjax: true
---

# 卷积与互相关

对于离散信号，卷积的定义为：
$$
(f * g)(n) = \sum_{\tau = -\infty}^{\infty} f(\tau) g(n - \tau)
$$
其中g就是滤波器（卷积核），公式的理解就是将g先进行-τ的翻转，再进行+n的滑动以对其输入信号f，卷积操作的输出即为经过滤波器的输出信号

而在深度学习中，卷积核的翻转其实是不必要的，因此省去了翻转，直接对卷积核进行滑动的操作，也就是互相关操作

来源：[【CNN】很详细的讲解什么以及为什么是卷积（Convolution）！-腾讯云开发者社区-腾讯云](https://cloud.tencent.com/developer/article/2127881)

卷积核根据参数的不同可以实现提取低频、高频特征（边缘特征），平滑、锐化处理等操作

# 参数理解

## 数据形状

以二维卷积为例，在pytorch中，conv2d的定义为：

```python
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',  # TODO: refine this type
        device=None,
        dtype=None
    )
```

* in_channels决定了卷积核的**深度**，每层卷积核输出的数据最终会相加，也就是说一个卷积核输出一个单通道数据
* out_channels决定了卷积核的**个数**，决定生成多少个上条中说到的单通道数据
* kernel_size决定了每个卷积核的**长宽**（参数量）

这三个参数共同决定了该卷积层的参数量，具体图片等参考：[(1 封私信 / 80 条消息) Depthwise卷积与Pointwise卷积 - 知乎](https://zhuanlan.zhihu.com/p/80041030)

对于一维卷积，当不考虑channels的改变（且为1）时，卷积核为1*k时等价于对其进行一维卷积；例：对于形状为(b,t,f)的频谱图，直接对其进行一维卷积就是在对其frequency bin进行卷积操作

## 感受野大小与步长

* 非1*1卷积核：`3*3`和`5*5`具有较好的细粒度特征提取能力且计算开销适中，对于有特殊要求的数据也可以使用不规则形状；而使用大卷积核可以获取更大的局部感受野，对大面积的特征会更加敏感，但计算量也会变大
* 1*1卷积核：`1*1`目前我见到过的有如下几种用法：1.用于对数据进行单纯的channel维度上\下采样；2.用于Depthwise Separable Convolution中的pointwise 卷积；3.用于代替某些attention结构中qkv的linear计算；4.inception模块中的多感受野融合

对于步长，常用的就是1和2，1适合用于特征提取阶段；选择2时输出形状会缩减约一半，常用于下采样

# 卷积种类

## 常规卷积

卷积权重形状为(out_channels, inchannels, H, W)，可以理解为有out_channels个(inchannels, H, W)的卷积核。

常规卷积group=1，具体看下一小节

## Depthwise Separable Convolution

由pointwise conv 和 depthwise conv组合而成（先depth再pointwise）。其中，pointwise conv就是卷积核为1*1的常规卷积，在efficient net中其扩大了输出channels；而depthwise conv 则是通过调整nn.conv2d的groups选项实现：

卷积权重形状完整写法其实为(out_channels, inchannels/groups, H, W)，对于depthwise conv：**groups=in_channels=out_channels** ，groups必须要能被in_channels整除；

举个例子，对于一个in_channels=4的数据来说，先将其分为4个channel=1的数据，因此每个卷积核的大小只能为(1,H,W)；而对于卷积核的个数，每组数据分到out_channels/groups个（depthwise中为该结果为1），最后拼接起来即为out_channels

Depthwise Separable Convolution可以有效缩减参数量，以达到更深的网络结构：假设in/out_channels=4，卷积核3*3：

常规卷积参数：`4*4*3*3=144`

Depthwise Separable：`4*4*1*1+4*1*3*3=52`

### 转置卷积

通过对输入进行padding的方式，可以实现输入H、W维度的上采样，生成更高分辨率的数据

# 搭建EfficientNet_v2

本节以efficientnetv2_s为例

## 主模型类

```python
class EfficientNetV2(nn.Module):
    def __init__(self,
                 model_cnf: list, #传入格式为[[repeat, kernel, stride, expansion, in_c, out_c, operator, se_ratio],...],其中op参数为是否使用fuse mbconv
                 num_classes: int = 1000,
                 num_features: int = 1280,
                 dropout_rate: float = 0.2,
                 drop_connect_rate: float = 0.2):
        super(EfficientNetV2, self).__init__()

        for cnf in model_cnf:
            assert len(cnf) == 8、
            
		#functools.partial可以在每次调用时都传回一个新的函数对象，对于不改变参数的网络结构更简洁
        norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.1)
        
        #头部卷积单独实现，其卷积核数量需匹配后续模型in_channels
        stem_filter_num = model_cnf[0][4]
		
        #简单的conv2d+norm+act，不再展开
        self.stem = ConvBNAct(3,
                              stem_filter_num,
                              kernel_size=3,
                              stride=2,
                              norm_layer=norm_layer)  # 激活函数默认是SiLU
		
       	#i[0]为该block的重复次数
        total_blocks = sum([i[0] for i in model_cnf])
        block_id = 0
        blocks = []
        for cnf in model_cnf:
            repeats = cnf[0]
            op = FusedMBConv if cnf[-2] == 0 else MBConv
            for i in range(repeats):
                blocks.append(op(kernel_size=cnf[1],
                                 input_c=cnf[4] if i == 0 else cnf[5],
                                 out_c=cnf[5],
                                 expand_ratio=cnf[3],
                                 stride=cnf[2] if i == 0 else 1,
                                 se_ratio=cnf[-1],
                                 drop_rate=drop_connect_rate * block_id / total_blocks,
                                 norm_layer=norm_layer))
                block_id += 1
                
        #*操作会按添加顺序返回list，是模块化神经网络的常用操作
        self.blocks = nn.Sequential(*blocks)

        #构建分类头
        head_input_c = model_cnf[-1][-3]
        head = OrderedDict()
        #orderdict是一个有序字典，其update方法在插入新键时按顺序插入，覆盖已有键时不改变顺序
        head.update({"project_conv": ConvBNAct(head_input_c,
                                               num_features,
                                               kernel_size=1,
                                               norm_layer=norm_layer)})  # 激活函数默认是SiLU
        #可以看出，该分类头是直接将H，W池化为1*1，将channels维作为最后linear的输入维度
        head.update({"avgpool": nn.AdaptiveAvgPool2d(1)})
        head.update({"flatten": nn.Flatten()})

        if dropout_rate > 0:
            head.update({"dropout": nn.Dropout(p=dropout_rate, inplace=True)})
        head.update({"classifier": nn.Linear(num_features, num_classes)})

        #dict和list都能直接传入nn.Sequential
        self.head = nn.Sequential(head)

        # initial weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)

        return x
```

## MBConv

对于v2版本，其前半部分网络使用的fuse-MBConv将原1*1conv+depthwise替换为了一个33conv，不再额外说明

```python
class MBConv(nn.Module):
    def __init__(self,
                 kernel_size: int,
                 input_c: int,
                 out_c: int,
                 expand_ratio: int,
                 stride: int,
                 se_ratio: float,
                 drop_rate: float,
                 norm_layer: Callable[..., nn.Module]):
        super(MBConv, self).__init__()

        if stride not in [1, 2]:
            raise ValueError("illegal stride value.")

        self.has_shortcut = (stride == 1 and input_c == out_c)

        activation_layer = nn.SiLU  # alias Swish
        expanded_c = input_c * expand_ratio

        assert expand_ratio != 1
        # channel 采样
        self.expand_conv = ConvBNAct(input_c,
                                     expanded_c,
                                     kernel_size=1,
                                     norm_layer=norm_layer,
                                     activation_layer=activation_layer)

        # 标准depthwise conv
        self.dwconv = ConvBNAct(expanded_c,
                                expanded_c,
                                kernel_size=kernel_size,
                                stride=stride,
                                groups=expanded_c,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer)

        #se模块
        self.se = SqueezeExcite(input_c, expanded_c, se_ratio) if se_ratio > 0 else nn.Identity()

        # Point-wise linear projection
        self.project_conv = ConvBNAct(expanded_c,
                                      out_planes=out_c,
                                      kernel_size=1,
                                      norm_layer=norm_layer,
                                      activation_layer=nn.Identity)  # 注意这里没有激活函数，传入的是Identity

        self.out_channels = out_c

        # 只有在使用shortcut连接时才使用dropout层
        self.drop_rate = drop_rate
        if self.has_shortcut and drop_rate > 0:
            self.dropout = DropPath(drop_rate)

    def forward(self, x: Tensor) -> Tensor:
        result = self.expand_conv(x)
        result = self.dwconv(result)
        result = self.se(result)
        result = self.project_conv(result)

        if self.has_shortcut:
            if self.drop_rate > 0:
                result = self.dropout(result)
            result += x

        return result
```

## SE模块

```python
class SqueezeExcite(nn.Module):
    def __init__(self,
                 input_c: int,   # block input channel
                 expand_c: int,  # block expand channel
                 se_ratio: float = 0.25):
        super(SqueezeExcite, self).__init__()
        squeeze_c = int(input_c * se_ratio)
        #将linear替换为了1*1conv，这样做可以不用变换维度，是常见做法
        self.conv_reduce = nn.Conv2d(expand_c, squeeze_c, 1)
        self.act1 = nn.SiLU()  # alias Swish
        self.conv_expand = nn.Conv2d(squeeze_c, expand_c, 1)
        self.act2 = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        scale = x.mean((2, 3), keepdim=True)#相当于avgpool
        scale = self.conv_reduce(scale) 
        scale = self.act1(scale)
        scale = self.conv_expand(scale)
        scale = self.act2(scale)
        #scale会学习到哪些通道更重要，相当于对x的通道维度施加了一个注意力因子
        return scale * x
```

