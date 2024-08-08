# 多物体跟踪数据集概述

> 原文：[`docs.ultralytics.com/datasets/track/`](https://docs.ultralytics.com/datasets/track/)

## 数据集格式（即将推出）

多物体检测器无需独立训练，直接支持预训练的检测、分割或姿态模型。独立跟踪器训练的支持即将推出。

## 用法

示例

```py
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
results = model.track(source="https://youtu.be/LNwODJXcvt4", conf=0.3, iou=0.5, show=True) 
```

```py
yolo  track  model=yolov8n.pt  source="https://youtu.be/LNwODJXcvt4"  conf=0.3,  iou=0.5  show 
```

## FAQ

### 我如何使用 Ultralytics YOLO 进行多物体跟踪？

要使用 Ultralytics YOLO 进行多物体跟踪，您可以从提供的 Python 或 CLI 示例开始。以下是如何开始的方法：

示例

```py
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # Load the YOLOv8 model
results = model.track(source="https://youtu.be/LNwODJXcvt4", conf=0.3, iou=0.5, show=True) 
```

```py
yolo  track  model=yolov8n.pt  source="https://youtu.be/LNwODJXcvt4"  conf=0.3  iou=0.5  show 
```

这些命令加载了 YOLOv8 模型，并使用特定置信度（`conf`）和 IoU（Intersection over Union，`iou`）阈值跟踪给定视频源中的物体。有关更多详细信息，请参阅跟踪模式文档。

### Ultralytics 用于训练跟踪器的即将推出的功能有哪些？

Ultralytics 正在不断增强其 AI 模型。即将推出的功能将支持独立跟踪器的训练。在此之前，多物体检测器利用预训练的检测、分割或姿态模型进行跟踪，无需独立训练。通过关注我们的[博客](https://www.ultralytics.com/blog)或查看即将推出的功能，保持更新。

### 为什么我应该使用 Ultralytics YOLO 进行多物体跟踪？

Ultralytics YOLO 是一种以其实时性能和高准确性而闻名的最先进的目标检测模型。使用 YOLO 进行多物体跟踪具有多个优势：

+   **实时跟踪：** 实现高效率和高速度的跟踪，非常适合动态环境。

+   **使用预训练模型的灵活性：** 无需从头开始训练；直接使用预训练的检测、分割或姿态模型。

+   **易于使用：** 简单的 Python 和 CLI API 集成使得设置跟踪流水线变得简单直接。

+   **广泛的文档和社区支持：** Ultralytics 提供了全面的文档和一个活跃的社区论坛，以解决问题并增强您的跟踪模型。

有关设置和使用 YOLO 进行跟踪的更多详细信息，请访问我们的跟踪使用指南。

### 我可以使用自定义数据集来进行 Ultralytics YOLO 的多物体跟踪吗？

是的，您可以使用自定义数据集来进行 Ultralytics YOLO 的多物体跟踪。虽然独立跟踪器训练的支持即将推出，但您已经可以在自定义数据集上使用预训练模型。准备符合 YOLO 兼容的适当格式的数据集，并按照文档集成它们。

### 我如何解释 Ultralytics YOLO 跟踪模型的结果？

在使用 Ultralytics YOLO 运行跟踪作业后，结果包括跟踪对象的各种数据点，如跟踪的物体 ID、它们的边界框和置信度分数。这里是如何解释这些结果的简要概述：

+   **跟踪的 ID：** 每个物体都被分配了一个唯一的 ID，这有助于在帧间进行跟踪。

+   **边界框：** 这些指示了帧内跟踪对象的位置。

+   **置信度分数：** 这些反映了模型对检测到的跟踪对象的信心。

对于详细的解释和可视化这些结果的指导，请参考结果处理指南。
