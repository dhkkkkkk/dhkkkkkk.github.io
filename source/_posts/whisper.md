---
title: 微调Whisper
date: 2026-01-05 14:36:22
tags: 深度学习
mathjax: true
---

本文的模型微调基于hugging face生态，但未使用Trainer API，以transcribe任务为例

# 论文精读

[OpenAI Whisper 精读【论文精读·45】_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1VG4y1t74x/?spm_id_from=333.788&vd_source=f274b4fe1db1741680aed4c118a32b27)

## 现存ASR问题

像wav2vec等主流ASR模型都是使用无监督学习pre训练encoder，再使用监督学习微调一个decoder，也就是说**无法避免使用监督学习与微调**，而微调容易对特定label过拟合

因此作者提出可以通过加大数据量+质量略微下降（也就是weak supervise）的方法**力大砖飞**（680k小时数据集）

## 方法

### 清洗数据

核心思路：音频质量多样性有助于提高robust，**但是文本质量没有类似好处，需要筛选**

* （主）删去机器翻译内容，避免“转录腔”。通过检测**文本是否经过标准化（**标点符号，大小写等）
* （主）删去音频、文本语言不同的情况：使用CLD2（自家软件）检测**音频所使用语言是否与文本对应**
* 通过一个initial model检测错误率一直特别高的数据，再检查是否有问题

### 模型

直接使用标准encoder+decoder的Transformer，encoder中使用2个一维卷积(大小为3)缩短mel谱时间维度，其他没有太大变化

### 多任务模式（核心）

实现一个结构的模型可以实现多种任务，whisper主要实现了：

* en to en 的**转录**
* any to any的**转录**（需对应）
* any to en 的**翻译**（注意whisper的翻译可以翻译任何内容，包括混合，而不限制源语言）
* 无语音检测

实现方法：

bert的方法是使用不同的输出层训练不同任务，而whisper是使用了类似prompt的操作，即在decoder的输入tokens前额外加入带有任务标志的token，这样做的好处就是**从头到尾都可以完完全全地使用同一个模型**

### 其他训练细节

* 使用FP16
* dynamic  loss scaling
* **epoch为2-3**
* zero shot评估，即使用完全unseen的数据集eval 

综上，whisper的主要核心就是**超大规模<u>监督训练</u>**+**prompt型多任务训练**，也就是证明了ASR领域也可以通过提高数据集数量实现力大砖飞，而不仅是调整模型（因为whisper基本没有改模型，直接用的Transformer）

# 数据集准备

本文使用hugging face上的一个阿尔巴尼亚语数据集[Kushtrim/common_voice_18_sq · Datasets at Hugging Face](https://huggingface.co/datasets/Kushtrim/common_voice_18_sq)，我是直接在网站上预先下载的parquet格式数据集，也可以直接在代码中下载

```python
from datasets import load_dataset, Audio
from transformers import WhisperProcessor,WhisperForConditionalGeneration
from transformers.models.whisper.tokenization_whisper import TO_LANGUAGE_CODE  #可查看whisper支持语言
import soundfile as sf
import io
from typing import Any, Dict, List, Union
from dataclasses import dataclass
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm import tqdm


processor = WhisperProcessor.from_pretrained(
    "openai/whisper-small", language="albanian", task="transcribe")

alb_dataset = load_dataset("parquet", data_files={'train': 'train.parquet', 'test': 'test.parquet'})
alb_dataset = alb_dataset.cast_column("audio", Audio(decode=False))

def prepare_dataset(example):
    audio_bytes = example["audio"]['bytes']
    audio, samplerate = sf.read(io.BytesIO(audio_bytes))

    example = processor(audio=audio,
                        sampling_rate=16000,
                        text=example["sentence"])
    example["input_length"] = len(audio) / samplerate
    return example
#此处alb_dataset.column_names["train"]是返回的所有column的名字，指删去原datasetdict所有同名列，不是指删去原训练集
dataset = alb_dataset.map(prepare_dataset,remove_columns=alb_dataset.column_names["train"],
                            num_proc=1,
                            batch_size=100)
# sample = dataset["train"][1] #feature形状:(1,80,3000)的list

```
## processor调用

此处调用了与模型配套的processor，这是一个与Whisper配套的类似tokenizer的工具箱对象，负责：

* 将音频重采样，转为mel频谱，归一化，统一pad至30s
* 将文本转为ID，也就是tokenizer的任务

## 禁用torchcodec，使用bytes读取

由于我直接读取单个数据时会出现torchcodec库的解码报错，由于我的torch版本较低，因此无法修复该报错。以下是我的解决方法：

* `alb_dataset.cast_column("audio", Audio(decode=False))`禁用torchcodec解码

* 对于可以正常解码的情况，声音波形的解码数据可以直接传入processor，而此处，通过读取声音的原始bytes复原声音的解码数据（对于抱抱脸的dataset统一数据格式可查看[LLM学习 | 小董的BLOG](https://dhkkkkkk.github.io/2025/11/21/LLM/)

  ```python
  audio_bytes = example["audio"]['bytes']
  audio, samplerate = sf.read(io.BytesIO(audio_bytes))
  ```

如果这里对sample进行debug，则可以观察到其数据格式：

{% asset_img 1.png This is an image %} 

为一个3key字典，其中feature为list1的原因是其被封装为了(1,80,3000)形状，80是mel滤波器数量，3000是长度

## 数据处理与dataloader加载

```python
''' 
@dataclass等同于
def __init__(self, processor):
self.processor = processor
'''
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any  #不强制约束类型

    #令对象可像函数一样直接调用
    def __call__(
            self, features: List[Dict[str, Union[List[int], torch.Tensor]]] #输入为一个list，内容为一个字典，键为str，值为list或tensor之一
    ) -> Dict[str, torch.Tensor]: #返回值为一个字典，类似C语言的函数定义

        input_features = [
            {"input_features": feature["input_features"][0]} for feature in features #feature["input_features"]形状为(1,80,3000),因此要取[0]
        ]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt") #processor已经填充过，此处仅为转tensor
        #feature_extractor.pad接口协议是List[{"input_features": (feature_dim, time)}]

        #label同理，但是label是没有提前填充的
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        #返回{"input_ids": Tensor(batch, max_len),"attention_mask": Tensor(batch, max_len)}
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100     #pytorch 交叉熵默认-100不参与loss计算
        )#用-100代替填充词元，labels_batch.attention_mask.ne(1)可以根据attention_mask生成一个bool mask，pad token为True

        # 如果在之前分词时添加了 bos 词元，那就剪切掉，因为之后还会加上的
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
# 定义 DataLoader
train_dataloader = DataLoader(
    dataset["train"],
    batch_size=2,
    shuffle=True,
    collate_fn=data_collator
)
eval_dataloader = DataLoader(
    dataset["test"],
    batch_size=2,
    collate_fn=data_collator
)
```

`DataCollatorSpeechSeq2SeqWithPadding`在这里的主要作用是

* 将特征转为tensor
* 填充label（token ID），并转为tensor
* 将padding token转化为-100

其他需要注意的就是`@dataclass`、`__call__`和列表推导式的语法技巧，都已用注释标出。最后，由于后续训练调用Seq2SeqTrainer API时总是会出现梯度计算图的重复计算报错，而hugging face封装的密不透风的对象我实在是做不到debug，干脆就索性直接调用DataLoader，自己写训练过程了。

**<u>DataLoader天然支持hugging face风格的dataset和data_collator，可以非常方便的直接使用</u>**

# 简陋微调一下

因为直接调API有bug，就自己简单实现了一下训练，以下代码只实现了基本训练功能，亲测可以跑通

```python
device = torch.device("cuda")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

for epoch in range(num_epochs):
    model.train()
    train_loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader),
                 desc=f'Train_Epoch {epoch + 1}/{num_epochs}', unit='batch')
    for idx, batch in train_loop:
        batch = {k: v.to(device) for k, v in batch.items()}


        outputs = model(**batch)
        loss = outputs.loss

        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        train_loop.set_postfix({'loss': '{0:1.5f}'.format(loss.item())})

    model.eval()
    total_loss = 0
    eval_loop = tqdm(enumerate(eval_dataloader), total=len(eval_dataloader),
                 desc=f'Eval_Epoch {epoch + 1}/{num_epochs}', unit='batch')
    for idx, batch in eval_loop:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        total_loss += outputs.loss.item()

    avg_loss = total_loss / len(eval_dataloader)
    print(f"Evaluation Loss: {avg_loss}")
```

训练相关代码好像没啥好说的了，都是一些经典操作，其中`self.processor.feature_extractor.pad`返回的batch格式如下，其是一个类似字典的结构，需要注意数据的读取方式

{% asset_img 2.png This is an image %} 

# 文本任务的评估指标

**WER** 的全称是 **Word Error Rate**（词错误率）。它是衡量语音识别（ASR）或机器翻译系统准确性最标准、最通用的指标。

```python
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

normalizer = BasicTextNormalizer()#文本标准化，重要


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # 用 pad_token_id 替换 -100
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # 我们希望在计算指标时不要组合起词元（解码结果为词元(subword)，而不是完整单词）
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    # 计算普通的 WER
    wer_ortho = 100 * metric.compute(predictions=pred_str, references=label_str)

    # 计算标准化的 WER
    pred_str_norm = [normalizer(pred) for pred in pred_str]
    label_str_norm = [normalizer(label) for label in label_str]
    # 过滤，从而在评估时只计算 reference 非空的样本
    pred_str_norm = [
        pred_str_norm[i] for i in range(len(pred_str_norm)) if len(label_str_norm[i]) > 0
    ]
    label_str_norm = [
        label_str_norm[i]
        for i in range(len(label_str_norm))
        if len(label_str_norm[i]) > 0
    ]

    wer = 100 * metric.compute(predictions=pred_str_norm, references=label_str_norm)

    return {"wer_ortho": wer_ortho, "wer": wer}
```

`BasicTextNormalizer` 是 Whisper 官方提供的一个文本“清洗”工具。

- **为什么要标准化？** 在阿尔巴尼亚语或英语中，同一个意思可能有不同写法（比如大小写、多余空格、标点符号）。
- **它做了什么？**
  - 将文本全部转为**小写**。
  - 移除所有的**标点符号**。
  - 移除**多余的空格**（连续空格合并为一个）。
  - 移除 Unicode 标记。
- **结果**：它让模型只关注“词是否选对了”，而不关注“标点符号是否点对了”，这能提供一个更客观的语言理解指标。

## WER原理

WER 的计算基于**编辑距离**的概念。想象你有一行<u>模型输出</u>的文字（Hypothesis），通过以下三种操作将其<u>修改为参考文本</u>（Reference）：

1. **替换 (Substitution, S)**：把错误的词换成正确的词。
2. **插入 (Insertion, I)**：在漏掉的地方补上缺失的词。
3. **删除 (Deletion, D)**：删掉模型多出的冗余词。

### 2. 计算公式

WER 的计算公式如下：
$$
WER = \frac{S + D + I}{N}
$$
其中：

- **S**：替换的次数
- **D**：删除的次数
- **I**：插入的次数
- **N**：参考文本（标准答案）的总词数

**注意：** WER 的**值越低越好（0 表示完美匹配）。**由于存在“插入”操作，WER 的值**有可能超过 100%**。

# 进阶微调

## 调整精度

需注意，调整精度和量化是不同的概念，调整精度的目的是**加速模型运算的速度**，而量化则是通过**改变预训练的模型权重存储格式以<u>降低其显存占用</u>**

### 直接在模型加载中指定精度

大部分模型的默认精度都是FP32，我们可以将其在加载时调整至FP16，这样可以直接使显存占用几乎减半：

```python
#对于transformers库加载模型：
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small",
                                                       ,torch_dype=torch.bfloat16)
#对于nn.Moudle对象：
model = MyModel() # 默认 FP32
model.to(torch.float16) # 整体转为 FP16
```

BFP16数值范围更广，不容易梯度爆炸

注：输入模型的数据也要同步调整至FP16：`input_features.to(torch.float16)`

### 自动混合精度AMP（工程常用）

在forward中使用FP16，但在反向传播更新梯度时使用FP32保证精度

需要使用 `torch.cuda.amp` 来实现：

```python
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to("cuda")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# 初始化梯度缩放器，防止 FP16 梯度下溢
scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()
    
    # 1. 在 autocast 上下文中运行前向传播
    with autocast(dtype=torch.float16):
        outputs = model(input_features=batch["input_features"], labels=batch["labels"])
        loss = outputs.loss

    # 2. 使用 scaler 缩放 loss 并反向传播
    scaler.scale(loss).backward()
    
    # 3. 更新参数
    scaler.step(optimizer)
    scaler.update()
```

需要注意的是，使用autocast时会自动在forward过程中将数据精度调整至fp16，由于backward时还是fp32，所以**不要手动调整输入数据精度！**

## LoRA

使用PEFT库

低秩适应是一种 PEFT 方法，它将一个大矩阵分解为两个较小的低秩矩阵，**用于注意力层**。这极大地减少了需要微调的参数数量。**<u>主要用于优化训练时间</u>**

### LoraConfig

每个 PEFT 方法都由一个 PeftConfig 类定义，该类存储了构建 PeftModel 的所有重要参数。对于LoRA微调，有：

```python
from peft import LoraConfig
config = LoraConfig(r=32, 
                    lora_alpha=64, 
                    target_modules=["q_proj", "v_proj"],
                    lora_dropout=0.05)

```

对于该类的详细使用可查阅官方文档：[LoRA - Hugging Face 文档](https://hugging-face.cn/docs/peft/package_reference/lora)

其中必填的参数有：

* r (int) — Lora 注意力维度（“秩”）
* lora_alpha (int) — Lora 的 alpha 参数，用于缩放。
* target_modules(Optional[Union[List[str], str]]) — 要应用适配器的模块名称。如果将其指定为“all-linear”，则选择所有线性/Conv1D 模块。虽然官网写的是Optional，但是对于大部分模型还是需要指定，对于具体名称则可以通过`for name, parameter in model.named_parameters():  print(name)`查看
* lora_dropout (float) — Lora 层的 dropout 概率。

接下来我们可以查看可训练的参数占比：

```python
from peft import LoraConfig, get_peft_model

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
config = LoraConfig(r=32, 
                    lora_alpha=64, 
                    target_modules=["q_proj", "v_proj"],
                    lora_dropout=0.05)

model = get_peft_model(model, config)
# 打印训练参数
model.print_trainable_parameters()

'''
结果
trainable params: 3,538,944 || all params: 245,273,856 || trainable%: 1.4429
即在微调中只有1.44%的参数参与训练
'''
```

接下来直接对新的peft模型进行训练即可

## QLoRA(4-bit 量化微调)

使用BitsAndBytesConfig方法实现量化config，这里展示4-bit，但其实8-bit也很常用

```python
from transformers import BitsAndBytesConfig
from peft import  get_peft_model, prepare_model_for_kbit_training

# 配置 4-bit 量化
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16, 
    bnb_4bit_quant_type="nf4",            # 使用正态浮点 4 位，精度更好
    bnb_4bit_use_double_quant=True        # 嵌套量化，进一步省显存
)

model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-small", 
    quantization_config=bnb_config
)

#量化后的权重是不可训练的。如果你要微调，必须配合 LoRA 使用！！
model = prepare_model_for_kbit_training(model)
...
lora_model = get_peft_model(model, lora_config)
```

注意，bnb_4bit_compute_dtype指定精度最好和前文中的精度调整（如AMP）中指定的精度相同，不然训练时会频繁调整格式。

同时，常使用的4-bit是NF4正态浮点4位，而不是INT4，因为NF4的数值分布是非线性的

## 总结

本节使用的微调技术包括了LoRA，量化（组合起来就算QLoRA）和精度调整

* LoRA将一个大矩阵分解为两个较小的低秩矩阵，**用于注意力层**，提高训练**<u>速度</u>**
* 量化是调整预训练权重存储格式，降低**<u>显存占用</u>**
* 精度调整是调整训练过程中的数据精度，提高训练**<u>速度</u>**

## 遇到的问题及解决方案

当使用QLoRA时，适当**提高学习率**，设置**学习率预热**有助于稳定高效地学习

