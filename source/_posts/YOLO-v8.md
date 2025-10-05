---
title: YOLO_v8
date: 2024-07-23 23:34:29
tags: 深度学习
---

大部分都是网上粘的，只是记录一下方便自己查找

# 置信度

在YOLO中，置信度是一个介于0和1之间的数值，表示模型对检测到的目标的确信程度。如果置信度接近1，那么模型相信该框中包含了目标对象。如果置信度接近0，模型认为该框中可能没有目标。所以，置信度可以看作是一个概率值，表示目标的存在概率。

在YOLO中，置信度代表了算法对于其预测结果的自信程度。简单地说，**就是算法觉得“这个框里真的有一个物体”的概率**。

# 开始

在当前目录下

```
pip install ultralytics
```

# 训练.yaml配置

```yaml
# 这里的path需要指向你项目中数据集的目录
path: C:/Users/admin/Desktop/CSDN/YOLOV8_DEF/ultralytics-detect/datasets/coco128/ 
# 这里分别指向你训练、验证、测试的文件地址，只需要指向图片的文件夹即可。但是要注意图片和labels名称要对应
train: images/train2017  # train images (relative to 'path') 128 images
val: images/test2017  # val images (relative to 'path') 128 images
test: images/test2017 # test images (optional)
 
# Classes
names:
  0: person
  1: bicycle
  2: car
  3: motorcycle
  4: airplane
  ......
  75: vase
  76: scissors
  77: teddy bear
  78: hair drier
  79: toothbrush
```

# 数据集标注信息(txt)

* 类别ID：对应.yaml中的names，代表对象的类别
* 中心X、Y坐标：对象**边界框中心**的水平、垂直方向坐标（比例）
* 边界框宽、高度：这个值是相对于整个图像的比例

若一个图片的txt标注文件中某一行为：1 0.5 0.5 0.2 0.3则代表：

* 类别：1
* 边界框**中心坐标**对应图像水平、垂直的各50%处（中心）
* 边界框**宽度**为图像宽度的20%，**高度**为图像高度的30%

# 训练对应的各参数说明

[超详细YOLOv8全解：学v8只用看一篇文章-CSDN博客](https://blog.csdn.net/weixin_45303602/article/details/139798347?spm=1001.2014.3001.5506)

最好在运行时通过命令行或.train函数添加参数实现，不在default.yaml中修改

# 预测大概步骤

* 先从头创建新模型并训练

  [YOLOv8训练参数详解（全面详细、重点突出、大白话阐述小白也能看懂）-CSDN博客](https://blog.csdn.net/qq_37553692/article/details/130898732)

  可以使用命令行格式或python代码格式

  ```pyhton
  from ultralytics import YOLO
  
  model=YOLO('[模型，.pt or .yaml]')
  result=model.trian(data='自己的.yaml',epochs=[训练轮数],lr0=[初始学习率]))
  ```

* 使用训练好的模型进行预测

  ```python
  model= YOLO('runs/detect/v8s01/weights/best.pt')
  result=model.predict(source='[要预测的图的路径]')
  ```

# 训练自带的coco128数据集

```python
from ultralytics import YOLO

model = YOLO("./yolov8n.pt")

if __name__ == '__main__':

    result = model.train(data='coco128.yaml',epochs=5, lr0=0.01)
```

查看coco128.yaml：

```yaml
path: ../datasets/coco128 # dataset root dir
train: images/train2017 # train images (relative to 'path') 128 images
val: images/train2017 # val images (relative to 'path') 128 images
test: # test images (optional)

# Classes
names:
  0: person
  1: bicycle
  2: car
  3: motorcycle
  4: airplane
  5: bus
  6: train
  7: truck
  8: boat
  9: traffic light
  10: fire hydrant
  11: stop sign
  12: parking meter
  13: bench
  14: bird
  15: cat
  16: dog
  17: horse
  18: sheep
  19: cow
  20: elephant
  21: bear
  22: zebra
  23: giraffe
  24: backpack
  25: umbrella
  26: handbag
  27: tie
  28: suitcase
  29: frisbee
  30: skis
  31: snowboard
  32: sports ball
  33: kite
  34: baseball bat
  35: baseball glove
  36: skateboard
  37: surfboard
  38: tennis racket
  39: bottle
  40: wine glass
  41: cup
  42: fork
  43: knife
  44: spoon
  45: bowl
  46: banana
  47: apple
  48: sandwich
  49: orange
  50: broccoli
  51: carrot
  52: hot dog
  53: pizza
  54: donut
  55: cake
  56: chair
  57: couch
  58: potted plant
  59: bed
  60: dining table
  61: toilet
  62: tv
  63: laptop
  64: mouse
  65: remote
  66: keyboard
  67: cell phone
  68: microwave
  69: oven
  70: toaster
  71: sink
  72: refrigerator
  73: book
  74: clock
  75: vase
  76: scissors
  77: teddy bear
  78: hair drier
  79: toothbrush
```

在数据集中随便找张图和其标注信息：

{% asset_img 1.jpg This is an image %} 

训练完毕后用训练后的模型预测这张图：

```python
result=model.predict(source='F:\\python\\ultralytics-main\\ultralytics-main\\ultralytics\\models\\yolo\\detect\\datasets\\coco128\\images\\train2017\\000000000009.jpg',save=True)
```

可得：

{% asset_img 2.jpg This is an image %}

注意：

用命令行得到的数据在文件根目录的runs文件夹下，而python代码得到的数据在ultralytics/runs下 

# 结果分析

[yolov8实战第二天——yolov8训练过程、结果分析（保姆式解读）_yolov8跑出来的指标怎么看-CSDN博客](https://blog.csdn.net/qq_34717531/article/details/135016961)

