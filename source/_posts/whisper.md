---
title: 微调Whisper
date: 2026-01-05 14:36:22
tags: 深度学习
mathjax: true
---

本文的模型微调基于hugging face生态，但未使用Trainer API，以transcribe任务为例

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

训练相关代码好像没啥好说的了，都是一些经典操作，其中dataloader返回的batch格式如下，其是一个类似字典的结构，需要注意数据的读取方式

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
