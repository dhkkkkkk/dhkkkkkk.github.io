---
title: LLM学习
date: 2025-11-21 09:27:07
tags: 深度学习
---

本章是在有一定深度学习基础，熟悉attention和transformer结构的基础上写的

# Transformer

## attention的抽象理解

还是以”I love you“到”我爱你“的翻译任务举例：

* encoder的自注意力：在模型逐字翻译中文的过程中，encoder只计算一次。encoder主要是为了得到英文中“I love you”的语义关系，例如主谓宾结构，或者“这是一个表达情感的语句“等
* decoder的自注意力：当已完成”我“的翻译时，现在decoder的输入则是”我“，在自注意力计算中，decoder则会生成中文”我“的语义
* decoder的交叉注意力：由于”我“这个中文语义中缺失谓语，因此<u>在Q与K的计算中</u>，”love“这个表示谓语的词语会得到更高的关注度（只是举例，实际上不只有这一个原因，但具体训练过程其实我们人类是难以理解的）；将关注度进行softmax后<u>与V相乘</u>，最终会输出一个**与"love"强相关的"I love you"的深层语义表达**（可以理解为：decoder当前需要翻译“love”，并且考虑了“I”和“you”在全文中的含义）。
* decoder尾部：线性层和softmax最后会基于之前的语义表达，输出一个与中文“爱”相似度最高的embedding

## 语言模型类型

语言模型通常分为三种架构类别：

* **仅编码器模型**（如 BERT）：这些模型使用双向方法来理解来自两个方向的上下文。它们最适合需要深入理解文本的任务，如**分类、命名实体识别和问答**。
* **仅解码器模型**（如 GPT、Llama）：这些模型从左到右处理文本，特别擅长**文本生成**任务。它们可以根据提示完成句子、写文章，甚至生成代码。
* **编码器-解码器模型**（如 T5、BART）：这些模型结合了两种方法，使用编码器理解输入，使用解码器生成输出。它们在序列到序列任务中表现出色，如**翻译、摘要和问答**。

而这些模型通常有两种训练方法：

1. **掩码语言建模（MLM）**：由像 **BERT 这样的编码器模型**使用，这种方法随机掩盖输入中的一些词元，并训练模型根据周围的上下文预测原始词元。这使得模型能够学习双向上下文（同时关注被掩盖词语之前和之后的词语）。
2. **因果语言建模（CLM）**：由像 **GPT 这样的解码器模型**使用，这种方法根据序列中所有之前的词元来预测下一个词元。模型只能使用左侧（之前的词元）的上下文来预测下一个词元。

## 常规架构（encoder-decoder）

该结构常见于翻译、语音转录等需要**理解+较为严格的生成要求**的任务

* encoder用于接收源输入并提取特征（例如翻译任务中的待翻译原文）

* decoder通过encoder输出的特征和之前的输出内容，生成下一个输出

  例如，当要将"I love you"翻译为中文时，encoder接收的便是"I love you"全文，decoder则是按顺序依次生成“我”，“爱”，“你”，而当生成“爱"时，decoder会将”我”与对”I love you“的特征进行attention计算，在训练优异的情况下，”love“这个谓语会被重点关注（被认为是下一个被翻译的对象），decoder会生成一个基于”I“和”you“的”love“深层语义表达，在decoder的最后，该语义表达会与中文”爱“的匹配度最高

[Whisper](https://dhkkkkkk.github.io/2026/01/05/whisper/)就是典型的该结构（whisper实现了从头到尾的全任务适用，而不需要额外训练下游网络）

## Encoder-only

该结构强调对**输入的理解**，常用于语义理解、分类任务，本小节以BERT为例

因为强调“理解”，需要模型对文本有充分的上下文结构理解，因此该类模型通常使用**MLM训练**（见上节），该训练方法即实现了文本双向理解（不同于使用两个相反的RNN）

另外，BERT还添加了**NSP预测任务**（Next Sentence Prediction），即预测输入的两个句子是否存在上下句关系。在标注时，额外通过对所有token添加一个0or1的embedding来<u>告诉模型当前输入的是一个句子还是两个句子（不是表示它们是不是来自于同一句子！，这是模型自己要学的东西）</u>

另外，BERT添加这些不同训练任务（MLM和NSP）是为了增强模型的学习能力，因为**BERT一般作为上游模型**使用，即使用pre-train的BERT再添加下游网络进行微调从而用于特定任务

## Decoder-only

该结构常用于像GPT等的**生成任务模型**，适用于开发式生成任务。本小节以GPT为例

仅使用decoder的结构不代表GPT就不需要对输入进行“理解”，与使用encoder进行“理解”的工作不同，GPT类模型**直接使用同一个decoder进行输入的“理解”与内容的生成**

### prefill阶段

当用户输入一段话prompt时，decoder会将其中的每个token都计算其k,v向量，并存储到 **KV Cache** 中。 KV Cache 就是模型对用户输入“理解结果”的物理存储。在后续生成（Decoding）时，模型不需要重新计算 Prompt，直接通过查询这些 Cache （即将q与prompt的KV cache进行注意力计算）就能“回想起”用户到底问了什么。

另外，该阶段为<u>并行计算</u>，**其计算得到的 KV cache会直接与decoding阶段的KV cache直接拼接**（append）

为了加速 Prefill 阶段，工业界采用了多种优化手段：

- **FlashAttention:** 通过优化 GPU 上 SRAM 和 HBM 之间的数据传输，极大提升了计算 $O(N^2)$ 注意力矩阵的速度。
- **Chunked Prefill:** 将超长的 Prompt 切分成小块（Chunks）处理，防止长文本预填充时瞬间撑爆显存。
- **Parallel Prefill & Decode:** 允许模型在处理一个请求的 Prefill 时，同时处理另一个请求的 Decoding，提高吞吐量。

### decoding阶段

就是正常的decoder生成流程了，现在普遍的做法都是使用KV cache，这样就可以在每次生成新token时**仅使用上一个token**作为decoder输入（q向量），**因为之前的所有已生成token和用户输入都已经作为 KV cache被存储了**

## 自编码器模型（Autoencoder，AE）

自编码器为transformer结构在**<u>无监督学习</u>**上的核心模型。一般为encoder+decoder的**完整端到端<u>串联</u>结构**，这与传统的transformer结构不同。其核心逻辑始终围绕着：通过“压缩-还原”的过程，**学习数据最本质的特征**。

在自编码器模型中，encoder本质为“特征压缩器”，负责将高维输入映射到**低维空间**（隐向量，与数据语义高度相关）；而decoder则为“特征重建器”，将压缩后的隐向量**还原**为原始维度，在通过解码器重建时，模型必须丢弃噪声和冗余信息——这种“信息瓶颈”（Information Bottleneck）迫使它只**保留最关键的特征**（例如手写数字的轮廓、纹理等）。

### 卷积自编码器（CAE）

用卷积层替代全连接层（编码器用Conv+Pooling，解码器用ConvTranspose反卷积）。保留图像的**空间局部性**特征，在图像任务中性能远超全连接自编码器。

### 降噪自编码器（DAE）

与MAE类似，在输入主动添加噪声（高斯噪声，随机遮挡mask等），让模型<u>重建无噪声原始输入</u>。**<u>可直接用于去噪任务，也可单独取训练完的encoder用作上游模型</u>**

### 变分自编码器（VAE）

传统的AE的encoder输出的是低维隐向量z，而VAE则是**输出x的概率分布参数：均值和方差（均值代表x在隐空间中的最具代表性位置，方差代表不确定性or容错范围）**，避免了传统z离散无约束的缺点，使隐空间符合正态分布。

#### 重参数化

由于直接从encoder输出的分布中采样是一个随机过程，无法求导。因此VAE需要引入重参数化：
$$
z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$
其中ε为从标准正态分布中采样的一个**噪声变量**，这样随机性就被转移了，便可以对均值和方差进行反向传播

#### 损失函数

一般的AE一般只使用MSE（连续）或交叉熵（离散），VAE引入了KL散度**正则**项，即衡量encoder输出的分布z与先验分布（一般为标准正态分布）的差异，**强制所有编码在潜在空间中靠近原点且具有一定的重叠。这保证了潜在空间的连续性和完整性。**

## 音频Transformer

### 模型输入格式

* 文本输入：通常出现在文本到语音的任务中（TTS），与原始Transformer或任何其他NLP模型的工作方式相同：**首先对文本进行标记化**（tokenization），得到一系列文本标记。然后将此序列**通过输入嵌入层**，将标记转换为512维向量。然后将这些嵌入向量传递到Transformer编码器（一般为decoder）中。
* 波形输入：**Wav2Vec2**和**HuBERT**一类的模型直接使用音频波形作为模型的输入。我们首先将原始波形**标准化为零均值和单位方差的序列**，这有助于标准化不同音量（振幅）的音频样本。对于这类模型，encoder前一般会用一个小型CNN进行下采样和提取局部特征，减少序列长度
* 时频谱输入：老朋友了，通过时频域转化可以大幅缩减样本尺寸。该类模型通常也会使用一个小型CNN提取局部特征和修改尺寸

### 模型输出格式

* 文本输出：和语言模型相同，将decoder输出的输出嵌入向量通过一个线性头和sofrmax转换为词汇表中文本id的概率（也就是**最终输出维度=词汇表大小**）
* 直接波形输出：有些模型可以直接输出波形，但较少
* 时频谱输出：对于该类模型，由于我们最终还是需要输出一个波形，因此通常有两种做法：
  * 基于istft：如果我们输入模型的时频谱是stft得到的，也可以通过istft还原。但此时我们需要知道**幅值和相位**两部分的信息，而一般音频模型输入仅使用基于幅值信息的功率谱，因此需要一个额外的网络估计其相位信息（也有其他方法，跟模型输入有关）
  * 基于神经网络：直接再使用一个神经网络将decoder输出嵌入转换为波形（Vocoder声码器）

## 两种结构

### CTC结构

CTC结构（Connectionist Temporal Classification）是一种**仅**使用Transformer编码器（encoder）结构的**语音识别**（ASR）模型。使用该架构的模型包括Wav2Vec2、HuBERT、M-CTC-T等等。

在CTC结构模型中，所使用的词汇表通常是小词汇表（字符、音素等）。该类模型通常将若干ms的样本切片输出为一个token，由于一个字母发音可能包含多个切片，**模型便会输出多个重复字母**，因为每个token必须对应一个结果。而CTC算法就是通过一个特殊标记（blank token），压缩模型输出的重复或空白内容

对于空白标记，其是通过**特殊的损失函数**让模型学习何时该输出空白标记的，因此CTC模型使用的loss不是标准的交叉熵。**除了词汇表中添加空白标记、仅使用encoder和使用特殊的训练策略之外**，该类模型就没有什么特殊的点了。

对于只考虑单字符的CTC模型，可能会输出听起来正确但拼写不正确的单词，因此可以使用**额外的语言模型**来提高音频的转录质量。这个语言模型实际上是作为了CTC输出的**拼写检查器**。

### Seq2seq结构

比CTC结构的模型能力更强，使用标准的transformer结构，与语言模型基本一致。因此其最终的输出和语言模型一样，都是subword，对于whisper，其使用的就是GPT2的分词器。在ASR任务中，其使用交叉熵作为损失函数。

# 🤗Transformers库基础🤗

## pipeline

transformer库直接调用已确定具体功能的模型的函数

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
classifier(
    [
        "I've been waiting for a HuggingFace course my whole life.",
        "I hate this so much!",
    ]
)

>>>[{'label': 'POSITIVE', 'score': 0.9598047137260437},
>>> {'label': 'NEGATIVE', 'score': 0.9994558095932007}]
```

对于一个语言模型，其大致可以分为两个部分，即tokenizer和model

## tokenizer

### 原理

tokenizer的作用就是将输入的文本转化为模型可处理的tensor（**即编码encode**），同时生成每句话所对应的一些额外信息，如token_type_ids和attention_mask等

基本的tokenizer可以分为三类：

* word-based：直接以空格分隔单词，将每个单词（word）对应一个唯一ID。缺点即是需要大量的token库，以存储所有单词
* Character-based：直接拆为单个字符，将每个字符对应唯一ID。缺点是对于一个句子，模型需要处理大量token，并且每个字符本身没有太大意义（对于英文来说）
* subword：**几乎所有大模型都在使用的分词策略**，原则：常用词不应被分解为更小的子词，但罕见词应被分解为有意义的子词。例如，对于"tokenization"，则可以被分为"token"和"ization"，因为这两个subword出现更为频繁，并且这样分词也可以保留其意义

每个模型都有自己的具体tokenizer以实现subword策略

transformers例程：

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

tokens = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
```

token此时为一个字典，包括了编码和注意力掩码（指padding mask）

### 具体实现

编码分为两步，即分词+转化为ID

```python
#仅分词
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
sequence = "Using a Transformer network is simple"
tokens = tokenizer.tokenize(sequence)
>>>['Using', 'a', 'transform', '##er', 'network', 'is', 'simple']

#将分词转为ID
ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)
>>>[7993, 170, 11303, 1200, 2443, 1110, 3014]

#解码
decoded_string = tokenizer.decode([7993, 170, 11303, 1200, 2443, 1110, 3014])
print(decoded_string)
>>>'Using a Transformer network is simple'
```

对于不同的模型，其还要求了其他的额外输入，因此其tokenizer也具有除编码外的其他功能：

```python
# 将句子序列填充到最长句子的长度
model_inputs = tokenizer(sequences, padding="longest")

# 将句子序列填充到模型的最大长度
# (512 for BERT or DistilBERT)
model_inputs = tokenizer(sequences, padding="max_length")

# 将句子序列填充到指定的最大长度
model_inputs = tokenizer(sequences, padding="max_length", max_length=8)

# 将截断比模型最大长度长的句子序列
# (512 for BERT or DistilBERT)
model_inputs = tokenizer(sequences, truncation=True)

# 将截断长于指定最大长度的句子序列
model_inputs = tokenizer(sequences, max_length=8, truncation=True)

# 返回 PyTorch tensors
model_inputs = tokenizer(sequences, padding=True, return_tensors="pt")

# 返回 TensorFlow tensors
model_inputs = tokenizer(sequences, padding=True, return_tensors="tf")

# 返回 NumPy arrays
model_inputs = tokenizer(sequences, padding=True, return_tensors="np")
```

## model

### 创建模型

```python
from transformers import BertConfig, BertModel

config = BertConfig()
model = BertModel(config)
```

### 加载预训练权重

```python
model = BertModel.from_pretrained("bert-base-cased")

#或是直接通过AutoModel类
checkpoint = "bert-base-cased"
model = AutoModel.from_pretrained(checkpoint)
```

需要注意的是，此处的模型输入需参考各模型具体要求，一般为一个字典，例如：

```
{'input_ids': tensor([[  101,  1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 			12172,     2607,  2026,  2878,  2166,  1012,   102],
        [  101,  2061,  2031,  1045,   999,   102,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0]]), 
 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])}
```

这些可以被模型对应的tokenizer直接生成，仅需：

```python
model(**tokens) 
"""
python的字典解包机制，等同于model(
    input_ids=tensor(...),
    attention_mask=tensor(...)
"""
```

## dataset

```python
from datasets import load_dataset

raw_datasets = load_dataset("glue", "mrpc")
raw_train_dataset = raw_datasets["train"]
sentence1 = raw_train_dataset['sentence1']
sample = raw_train_dataset[0]
```

### Datadict

该函数会返回一个<u>DatasetDict</u>数据结构，这是一个类似字典的结构

{% asset_img 1.png This is an image %} 

对其进行索引可以得到单个数据集结构<u>Dataset</u>

### Dataset

对于Dataset结构，也就是`raw_train_dataset`，其存储方式是列式存储，对于本代码，其每一列为：

```
'sentence1', 'sentence2', 'label', 'idx'
```

对这些键名进行索引即可返回单个<u>Column</u>结构

也可以对其进行**行索引**，则会返回单个样本的每列信息（**一个标准字典**）：

{% asset_img 2.png This is an image %} 

其中，对于本代码来说，label则是代表sentence1和sentence2**是否**同义

### 返回token的dataset

```python
def tokenize_function(examples):
    return tokenizer(
        examples["sentence1"],
        examples["sentence2"],
        truncation=True
    )
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
```

.map方法可以将每个文本经过tokenization后的结果添加到原本的Datadict中：

```pyhton
DatasetDict({
    train: Dataset({
        features: ['attention_mask', 'idx', 'input_ids', 'label', 'sentence1', 'sentence2', 'token_type_ids'],
        num_rows: 3668
    })
    validation: Dataset({
        features: ['attention_mask', 'idx', 'input_ids', 'label', 'sentence1', 'sentence2', 'token_type_ids'],
        num_rows: 408
    })
    test: Dataset({
        features: ['attention_mask', 'idx', 'input_ids', 'label', 'sentence1', 'sentence2', 'token_type_ids'],
        num_rows: 1725
    })
})
```

当不需要其中的某些features时，可以：

```python
samples = tokenized_datasets["train"][:]
samples = {k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]}
```

或者也可以调用Datasetdict的方法：

```python
tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
```

### 动态填充

在训练 LLM时，只需要对每个 batch 进行**动态 padding**（因为模型的输入必须是规则的），而不是对整个数据集进行统一 padding，因为这样会大量浪费计算资源。

```python
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
samples = tokenized_datasets["train"][:8]
samples = {k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]}
batch = data_collator(samples)
```

其中batch则是n个样本（代码中为8）所对应的经过tokenizaiton后的samples，并且其中的id全部都被padding至8个样本中最长文本id的长度（batch.item()和samples的数据结构是相同的，都是一个标准字典，仅有id发生了padding）

# 模型微调

## 调用Trainer API

```python
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer


raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
training_args = TrainingArguments("test-trainer")
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()
```

在使用`AutoModelForSequenceClassification`实例化模型时，会收到一个警告，这是因为 BERT 没有在句子对分类方面进行过预训练，所以**预训练模型的 head 已经被丢弃**，而是**添加了一个适合句子序列分类的新头部**。这些警告表明一些权重没有使用（对应于被放弃的预训练头的权重），而有些权重被随机初始化（对应于新 head 的权重）。

### 评估

对于模型的输出：

```python
predictions = trainer.predict(tokenized_datasets["validation"])
```

其会返回一个元组：`(predictions,label_ids,metrics)`；其中predictions为一个形状为(batchsize,cls_num)的二维张量，第二维度为每个样本的logits，其中的最大值则为预测结果；label_ids为样本真实标签，metrics为自定义评估指标，默认只返回loss

因此我们可以定义一个评估函数：

```python
def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
```

其中.compute方法会返回准确率与f1分数

## 不使用Trainer

### 使用torch的Dataloader加载数据集

```python
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

#删去不需要列，将label改名为labels（模型默认输入格式）
tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")
#>>>["attention_mask", "input_ids", "labels", "token_type_ids"]

from torch.utils.data import DataLoader
#Dataloader兼容transformers的DataCollatorWithPadding
train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
)
```

此时dataloader返回的每个batch为：

```
{'attention_mask': torch.Size([8, 65]),
 'input_ids': torch.Size([8, 65]),
 'labels': torch.Size([8]),
 'token_type_ids': torch.Size([8, 65])}
```

该batch可以直接输入到AutoModelForSequenceClassification实例化的模型中：

```python
from transformers import get_scheduler
import torch

optimizer = AdamW(model.parameters(), lr=5e-5)
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(	#学习率线性衰减
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
model.to(device)

for epoch in range(num_epoch):
    model.train()
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
    
    model.eval()
	for batch in eval_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
    	with torch.no_grad():
        	outputs = model(**batch)
        logits = outputs.logits
        ...
```

