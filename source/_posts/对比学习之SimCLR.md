---
title: 对比学习之SimCLR
date: 2024-09-14 11:00:51
tags: 深度学习
---

对比学习入门，记录一下simclr的学习过程

# 1.模型建立

基于lightly框架

```python
import torch
import torchvision
from torch import nn

from lightly.loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead
from lightly.transforms.simclr_transform import SimCLRTransform


class SimCLR(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone	#resnet最后平均池化后为512*1*1
        self.projection_head = SimCLRProjectionHead(512, 512, 128)#非线性化

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1) #nn.flatten为展平为1维
        z = self.projection_head(x)	#
        return z


resnet = torchvision.models.resnet18()	#使用resnet作为特征识别网络
backbone = nn.Sequential(*list(resnet.children())[:-1])#返回残差结构迭代器列表并拼接为resnet，此处resnet.children()不会返回resnet本来的全连接层！仅到全连接层前的平均池化
model = SimCLR(backbone)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
```

```python
#resnet.children()测试
import torchvision
from torch import nn

resnet = torchvision.models.resnet18()
backbone = nn.Sequential(*list(resnet.children())[:-1])
print(backbone)

'''
Sequential(
  (0): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (2): ReLU(inplace=True)
  (3): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  (4): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (1): BasicBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (5): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (6): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (7): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (8): AdaptiveAvgPool2d(output_size=(1, 1))
)
'''

```

# 2.transform、dataset

```python
transform = SimCLRTransform(input_size=32, gaussian_blur=0.0)
dataset = torchvision.datasets.CIFAR10(
    "datasets/cifar10", download=True, transform=transform
)



class SimCLRTransform(MultiViewTransform):	#继承MultiViewTransform父类
    def __init__(
        self,
        input_size: int = 224,
        cj_prob: float = 0.8,
        cj_strength: float = 1.0,
        cj_bright: float = 0.8,
        cj_contrast: float = 0.8,
        cj_sat: float = 0.8,
        cj_hue: float = 0.2,
        min_scale: float = 0.08,
        random_gray_scale: float = 0.2,
        gaussian_blur: float = 0.5,
        kernel_size: Optional[float] = None,
        sigmas: Tuple[float, float] = (0.1, 2),
        vf_prob: float = 0.0,
        hf_prob: float = 0.5,
        rr_prob: float = 0.0,
        rr_degrees: Optional[Union[float, Tuple[float, float]]] = None,
        normalize: Union[None, Dict[str, List[float]]] = IMAGENET_NORMALIZE,
    ):
        view_transform = SimCLRViewTransform(
            input_size=input_size,
            cj_prob=cj_prob,
            cj_strength=cj_strength,
            cj_bright=cj_bright,
            cj_contrast=cj_contrast,
            cj_sat=cj_sat,
            cj_hue=cj_hue,
            min_scale=min_scale,
            random_gray_scale=random_gray_scale,
            gaussian_blur=gaussian_blur,
            kernel_size=kernel_size,
            sigmas=sigmas,
            vf_prob=vf_prob,
            hf_prob=hf_prob,
            rr_prob=rr_prob,
            rr_degrees=rr_degrees,
            normalize=normalize,
        )
        super().__init__(transforms=[view_transform, view_transform])
			#通过父类MultiViewTransform生成一张图片的不同“视角”
         
class SimCLRViewTransform:
    def __init__(
        self,
        input_size: int = 224,
        cj_prob: float = 0.8,
        cj_strength: float = 1.0,
        cj_bright: float = 0.8,
        cj_contrast: float = 0.8,
        cj_sat: float = 0.8,
        cj_hue: float = 0.2,
        min_scale: float = 0.08,
        random_gray_scale: float = 0.2,
        gaussian_blur: float = 0.5,
        kernel_size: Optional[float] = None,
        sigmas: Tuple[float, float] = (0.1, 2),
        vf_prob: float = 0.0,
        hf_prob: float = 0.5,
        rr_prob: float = 0.0,
        rr_degrees: Optional[Union[float, Tuple[float, float]]] = None,
        normalize: Union[None, Dict[str, List[float]]] = IMAGENET_NORMALIZE,
    ):
        color_jitter = T.ColorJitter(
            brightness=cj_strength * cj_bright,
            contrast=cj_strength * cj_contrast,
            saturation=cj_strength * cj_sat,
            hue=cj_strength * cj_hue,
        )
			#随机数据增强
        transform = [
            T.RandomResizedCrop(size=input_size, scale=(min_scale, 1.0)),
            random_rotation_transform(rr_prob=rr_prob, rr_degrees=rr_degrees),
            T.RandomHorizontalFlip(p=hf_prob),
            T.RandomVerticalFlip(p=vf_prob),
            T.RandomApply([color_jitter], p=cj_prob),	#颜色抖动，对于对比学习很重要
            T.RandomGrayscale(p=random_gray_scale),	#随机灰度，同上
            GaussianBlur(kernel_size=kernel_size, sigmas=sigmas, prob=gaussian_blur),
            T.ToTensor(),
        ]
        if normalize:
            #归一化，此处列表可直接通过+=添加新元素
            transform += [T.Normalize(mean=normalize["mean"], std=normalize["std"])]
        self.transform = T.Compose(transform)

    def __call__(self, image: Union[Tensor, Image]) -> Tensor:  #该方法会在访问dataloader实例时自动调用（理解为图片在使用前才进行设置好的预处理），每次调用时transform的结果都是随机的
        transformed: Tensor = self.transform(image)
        return transformed
```

# 3.训练

```python
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=256,
    shuffle=True,
    drop_last=True,
    num_workers=8,
)
criterion = NTXentLoss()	#loss，用于计算2个输入张量的相似度
optimizer = torch.optim.SGD(model.parameters(), lr=0.06)

print("Starting Training")
for epoch in range(10):
    total_loss = 0
    for batch in dataloader:
        x0, x1 = batch[0]	#batch[0]为image，[1]为label，对比学习不需要label
        x0 = x0.to(device)
        x1 = x1.to(device)
        z0 = model(x0)
        z1 = model(x1)
        loss = criterion(z0, z1)	#基于相似度的loss
        total_loss += loss.detach()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    avg_loss = total_loss / len(dataloader)
    print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")
```

在`for batch in dataloader:`时，经过transform预处理后一张图片会返回2个随机view，也就是说x0,x1各自长度都为256

因此NTXentLoss()函数会比较一个批次中**所有图像视图**的嵌入表示（`z0` 和 `z1`），不仅仅是同一图像的不同视图（即正样本对，positive pair），还**包括不同图像的不同视图**（即负样本对，negative pair）。

# 3.基于simclr的下游分类任务

要使用经过 SimCLR 训练的模型进行图片分类任务，需要将其转换为一个**有监督的分类模型**。SimCLR 通过对比学习学习到的是**图像的特征表示，但并没有学到类别信息**。因此，接下来需要**在 SimCLR 训练好的特征提取器基础上**，加上一个新的分类头（如全连接层），并对这个分类器进行训练。

```python
class ClassificationHead(nn.Module):
    def __init__(self, backbone, num_classes=10):# cirf10有10个类别
        super(ClassificationHead, self).__init__()
        self.backbone = backbone
        self.fc = nn.Linear(512, num_classes) #添加1个全连接层，输出为10个类别，此处512需根据使用的backbone种类确定（我也不知道是不是都是512）

    def forward(self, x):
        features = self.backbone(x).flatten(start_dim=1)	#记得全连接之前要展成1维
        return self.fc(features)

model.load_state_dict(torch.load('simclr.pth'))
backbone = model.backbone  # 从 训练好的SimCLR 模型中获取 backbone
    
# 冻结 backbone 的参数
for param in backbone.parameters():
    param.requires_grad = False

# 实例化分类模型
model = ClassificationHead(backbone, num_classes=10)
model = model.to(device)

# 加载 CIFAR-10 训练集和测试集（带标签的有监督训练）
class_transform = torchvision.transforms.Compose(
        [torchvision.transforms.Resize((32, 32)),
         torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = torchvision.datasets.CIFAR10(
        root="dataset", train=True, download=True, transform=class_transform
    )
test_dataset = torchvision.datasets.CIFAR10(
        root="dataset", train=False, download=True, transform=class_transform
    )

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    # 定义损失函数和优化器
    fc_loss = nn.CrossEntropyLoss()  # 分类任务的损失函数，使用交叉熵损失函数
    optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)  # 只更新全连接层的参数

    # 训练分类器
    print('classification fc training start')
    for epoch in range(10):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 前向传播
            outputs = model(inputs)
            loss = fc_loss(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        print(f"stage2 Epoch [{epoch + 1}/5], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    # 测试集
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_accuracy = 100. * correct / total
    print(f"stage2 Test Accuracy: {test_accuracy:.2f}%")
```

## 关于model.train和.eval

* 训练模式（Training Mode）：如表格所示，在此模式下，模型会进行前向传播、反向传播以及参数更新。某些层，如Dropout层和BatchNorm层，在此模式下的行为会与评估模式下不同。例如，Dropout层会在训练过程中随机将一部分输入设置为0，以防止过拟合。

* 评估模式（Evaluation Mode）：如表格所示，在此模式下，模型只会进行前向传播，不会进行反向传播和参数更新。Dropout层会停止dropout，BatchNorm层会使用在训练阶段计算得到的全局统计数据，而不是测试集中的批统计数据。

评估模式一般用于验证、测试集

## 关于torch.max

`_, predicted = outputs.max(1)`与`torch.max(outputs, dim=1)`相同，对于其返回值：

* [0]：在全连接层所有输出中的最大值
* [1]：最大值对应的标签值，通常我们只关心标签值（索引），不关心[0]

# 4.predict

```python
import torch
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn
from PIL import Image
from examples.pytorch.simclr import SimCLR
from examples.pytorch.simclr_classification import ClassificationHead


def main():
    transform = transforms.Compose(
        [transforms.Resize((32, 32)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

   #实例化simclr模型并加载训练好的权重
    resnet = torchvision.models.resnet18()
    backbone = nn.Sequential(*list(resnet.children())[:-1])
    simclr_model = SimCLR(backbone)
    simclr_model.load_state_dict(torch.load('simclr.pth'))
   
   #将训练好的simclr的backbone取出与全连接层拼成新模型
    backbone = simclr_model.backbone
    class_model = ClassificationHead(backbone, num_classes=10)
   
   #加载已训练好的全连接层权重
    class_model.load_state_dict(torch.load('simclr_classification.pth'))

    class_model.eval()
    im = Image.open('1.jpg')
    im = transform(im)  # [C, H, W]
    im = torch.unsqueeze(im, dim=0)  # [N, C, H, W]

    with torch.no_grad():
        outputs = class_model(im)
        predict = outputs.max(1)[1].numpy()
    print(classes[int(predict)])


if __name__ == '__main__':
    main()
```

# 总结

* 先建立基于resnet18的模型，使用backbone这种形式大概是为了方便在下游任务使用？
* 对数据集进行随机裁剪、颜色抖动等操作，生成同一张图的不同view(通过dataloader访问数据集时会通过transform生成2张图)
* 使用NTXentLoss进行对比学习，该损失函数会会比较一个批次中**所有图像视图**，经过对比学习后的backbone会有更好的**特征提取能力**
* 若要用于分类任务，则将对比学习训练后的backbone接上全连接层并对全连接层**单独进行监督训练**
* 进行predict时，需先加载训练好的resnet backbone参数，再加载训练好的全连接层参数

