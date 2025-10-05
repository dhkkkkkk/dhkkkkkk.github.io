---
title: 经典CNN
date: 2024-09-04 09:50:08
tags: 深度学习
---

# Lenet

## model

卷积后的矩阵深度由卷积核组数决定，大小由卷积核尺寸通过公式计算得到

```python
# 使用torch.nn包来构建神经网络.
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module): 					# 继承于nn.Module这个父类
    def __init__(self):						# 初始化网络结构
        super(LeNet, self).__init__()    	# 多继承需用到super函数
        self.conv1 = nn.Conv2d(3, 16, 5)	
        #2维卷积：输入channel（矩阵深度），输出channel（卷积核组数），卷积核尺寸
        self.pool1 = nn.MaxPool2d(2, 2)
        #最大池化：池化核大小，布距
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32*5*5, 120)	#全连接层，输入需将矩阵转为1维
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)	#最后一层根据需求更改输出个数

    def forward(self, x):			 # 正向传播过程，pytorch 中tensor的定义：[batch，channel，height，weight]
        x = F.relu(self.conv1(x)) # input(3, 32, 32) output(16, 28, 28)
        x = self.pool1(x)            # output(16, 14, 14) 池化层只改变H，W
        x = F.relu(self.conv2(x))    # output(32, 10, 10)
        x = self.pool2(x)            # output(32, 5, 5)
        x = x.view(-1, 32*5*5)       # output(32*5*5)，tensor展平
        x = F.relu(self.fc1(x))      # output(120)
        x = F.relu(self.fc2(x))      # output(84)
        x = self.fc3(x)              # output(10)
        return x

```

## train

### 数据预处理

```python
transform = transforms.Compose(
   [transforms.ToTensor(), #将图片转化为torch的tensor格式
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))#归一化
   ])

'''
还可以添加
transforms.RandomResizedCrop(size)      # 随机裁剪，再缩放成 size*size
transforms.RandomHorizontalFlip(p=0.5) 	#概率为0.5的随机水平翻转
'''

```

### 下载数据集并通过dataloader加载

对于打包的训练集需要通过dataloader加载并设置batchsize等信息

#### 训练集

```python
train_set = torchvision.datasets.CIFAR10(root='./data', train=True,download=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=36,shuffle=True, num_workers=0)#shuffle:打乱
```

#### 测试集

```python
val_set = torchvision.datasets.CIFAR10(root='./data', train=False,download=False, transform=transform)
#测试集train为false
val_loader = torch.utils.data.DataLoader(val_set, batch_size=5000,shuffle=True, num_workers=0)
```

### 关于dataloader

如果对dataloader对象取迭代器，则其返回为值与标签

```python
#创建迭代器方便访问（dataloader返回值为数据和标签）
val_data_iter = iter(val_loader)
val_image, val_label = next(val_data_iter)
```

若对其：

```python
for step, data in enumerate(train_loader, start=0):
```

则enumerate函数会返回当前的索引步数和data，其中data组成为[value，label]

### 加载lenet模型、损失函数、优化器

```python
net = LeNet().to(device)#若使用gpu，则所有张量（模型、经过transform处理的对象）都要转到gpu上
loss_function = nn.CrossEntropyLoss()#交叉熵损失函数
optimizer = optim.Adam(net.parameters(), lr=0.001)#adam优化器
```

### 开始训练

```python
for epoch in range(7): 

  running_loss = 0.0
  for step, data in enumerate(train_loader, start=0):
      #enumerate将dataloader转化为索引值+数据的迭代器结构，其中数据data包括图像值和标签
      inputs, labels = data

      # zero the parameter gradients
      optimizer.zero_grad()
      
      # forward + backward + optimize
      outputs = net(inputs.to(device))	#正向传播
      loss = loss_function(outputs, labels.to(device))	
      loss.backward()	#根据损失函数反向传播
      optimizer.step()	#优化器更新

      # print statistics
      running_loss += loss.item()
      if step % 500 == 499:    # print every 500 mini-batches
         with torch.no_grad():	#在with块中临时关闭梯度计算，因为此处为验证而非训练
             outputs = net(val_image.to(device))  #将验证集放入训练好的模型中
             predict_y = torch.max(outputs, dim=1)[1]	#选取输出中概率最大的类别作为预测结果
             accuracy = torch.eq(predict_y, val_label.to(device)).sum().item() / 					 val_label.size(0)
             '''
             torch.eq(predict_y, val_label.to(device)) 比较预测结果 predict_y 和实际标签 				val_label 是否相等，生成一个布尔张量，值为 True 表示预测正确，False 表示预测错误。
				sum() 计算预测正确的样本数，.item() 将结果从张量转换为Python数值类型。最后，用正确预					测的数量除以验证集的总样本数 val_label.size(0)，得到准确率。
             '''
             print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %(epoch + 1, 					 step + 1, running_loss / 500, accuracy))
             running_loss = 0.0

```

## predict

需注意单张图片的处理格式，在train中，由于训练集（验证集）都经过了dataloader的batchsize的设置，最终输入到模型中的格式为[N, C, H, W]，预测中由于是单张图片，因此需要手动通过torch.unsqueeze设置为[1, C, H, W]

```python
transform = transforms.Compose(
        	[transforms.Resize((32, 32)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
im = Image.open('1.jpg')
im = transform(im)  # [C, H, W]
im = torch.unsqueeze(im, dim=0)  # [N, C, H, W]
with torch.no_grad():
   	outputs = net(im)
```

## model的一些改进

```python
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
			#在更为复杂的模型中常使用nn.Sequential集成CNN的各部分
        self.feature = nn.Sequential(
            nn.Conv2d(3, 16, 5),
            nn.ReLU(inplace=True),	#该方法可节约内存
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
      	#全连接层
        self.classifiter = nn.Sequential(
            nn.Linear(32 * 5 * 5, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.feature(x)
        x = x.view(-1, 32*5*5)	#输入全连接层时记得展平为1维
        x = self.classifiter(x)
        return x

```

# Alexnet

{% asset_img 1.jpg This is an image %} 

由图可得，Alexnet共有三次池化，且2-3之间堆叠了3次3x3的卷积

# Resnet

## 传统CNN缺点

实验证明，当网络堆叠到一定深度时，会出现两个问题：

* 梯度消失或梯度爆炸
* 退化问题(degradation problem)：在解决了梯度消失、爆炸问题后，仍然存在深层网络的效果可能比浅层网络差的现象

而Resnet通过以下方法解决了这些问题：

* 对于梯度消失或梯度爆炸问题，ResNet论文提出通过数据的预处理以及在网络中使用 **BN（Batch Normalization）层**来解决。

* 对于退化问题，ResNet论文提出了 **<u>residual结构（残差结构）</u>**来减轻退化问题，并且随着网络的不断加深，效果并没有变差，而是变的更好了。

## 残差结构

2个适用于不同深度神经网络的残差结构。

{% asset_img 2.jpg This is an image %} 

人为地让神经网络某些层跳过下一层神经元的连接，隔层相连，**弱化每层之间的强联系**。这种神经网络被称为 **残差网络** (**ResNets**)。

这里要注意最下方是**求和后**再经过激活函数，其中1x1的卷积核通过调整步数从而达到降维效果

## resnet所有网络的具体信息

{% asset_img 4.jpg This is an image %} 

**需注意的是18、32的输出深度是512，而50以上的输出深度是2048**

## resnet18结构

{% asset_img 3.jpg This is an image %} 

由于特征矩阵相加时深度必须相同，因此虚线部分需通过1x1矩阵降维

## pytorch搭建残差网络

```python
import torch.nn as nn
import torch


# ResNet18/34的残差结构，用的是2个3x3的卷积
class BasicBlock(nn.Module):
    expansion = 1  # 残差结构中，主分支的卷积核个数是否发生变化，不变则为1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):  				#downsample对应虚线残差结构
        super(BasicBlock, self).__init__()
    #主线路    
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                             kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
      #下采样，表示虚线部分的降维，不一定每个残差结构都有   
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:  # 虚线残差结构，需要下采样
            identity = self.downsample(x)  # 捷径分支 short cut

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity 	#相加后再relu
        out = self.relu(out)

        return out

# ResNet50/101/152的残差结构，用的是1x1+3x3+1x1的卷积
class Bottleneck(nn.Module):
    expansion = 4  # 残差结构中第三层卷积核个数是第一/二层卷积核个数的4倍

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(out_channel)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel * self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)  # 捷径分支 short cut

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    # block = BasicBlock or Bottleneck
    # block_num为残差结构中conv2_x~conv5_x中残差块个数，是一个列表，共4层，每层有若干卷积层
    def __init__(self, block, blocks_num, num_classes=1000, include_top=True):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])             
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2) 
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)  
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2) 
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    #重点函数
    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None

        # ResNet50/101/152的残差结构，block.expansion=4，满足条件则生成下采样降维的旁支
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, 					  stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel, channel, downsample=downsample, stride=stride))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel))

        return nn.Sequential(*layers)
   #对列表取*相对于将其元素依次顺序传入，此时nn.Sequential则会顺序拼接各个残差块的内容

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x


def resnet34(num_classes=1000, include_top=True):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=1000, include_top=True):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)


```

值得注意的是，对于nn.Sequential，其可以通过`nn.Sequential(class1,class2)`的方式顺序拼接两个神经网络块，也可以通过lenet中的方式依次集成神经网络

由于残差结构（块）的具体参数（卷积层数，是否下采样）需要根据使用的resnet的深度决定，因此使用这种拼接的方式会比较方便

