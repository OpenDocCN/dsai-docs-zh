# 姿势估计数据集概述

> 原文：[`docs.ultralytics.com/datasets/pose/`](https://docs.ultralytics.com/datasets/pose/)

## 支持的数据集格式

### Ultralytics YOLO 格式

用于训练 YOLO 姿势模型的数据集标签格式如下：

1.  每个图像对应一个文本文件：数据集中的每个图像都有一个与图像文件同名且带有 ".txt" 扩展名的文本文件。

1.  每个对象一行：文本文件中的每行对应图像中的一个对象实例。

1.  每行包含有关对象实例的以下信息：

    +   对象类索引：表示对象类的整数（例如，人为 0，汽车为 1 等）。

    +   对象中心坐标：对象中心的 x 和 y 坐标，归一化到 0 到 1 之间。

    +   对象宽度和高度：对象的宽度和高度，归一化到 0 到 1 之间。

    +   对象关键点坐标：对象的关键点，归一化到 0 到 1 之间。

这是姿势估计任务标签格式的示例：

使用 Dim = 2 进行格式化

```py
`<class-index> <x> <y> <width> <height> <px1> <py1> <px2> <py2> ... <pxn> <pyn>` 
```

使用 Dim = 3 进行格式化

```py
`<class-index> <x> <y> <width> <height> <px1> <py1> <p1-visibility> <px2> <py2> <p2-visibility> <pxn> <pyn> <p2-visibility>` 
```

在此格式中，`<class-index>` 是对象类的索引，`<x> <y> <width> <height>` 是边界框的坐标，`<px1> <py1> <px2> <py2> ... <pxn> <pyn>` 是关键点的像素坐标。坐标之间用空格分隔。

### 数据集 YAML 格式

Ultralytics 框架使用 YAML 文件格式定义用于训练检测模型的数据集和模型配置。以下是用于定义检测数据集的 YAML 格式示例：

```py
`# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..] path:  ../datasets/coco8-pose  # dataset root dir train:  images/train  # train images (relative to 'path') 4 images val:  images/val  # val images (relative to 'path') 4 images test:  # test images (optional)  # Keypoints kpt_shape:  [17,  3]  # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible) flip_idx:  [0,  2,  1,  4,  3,  6,  5,  8,  7,  10,  9,  12,  11,  14,  13,  16,  15]  # Classes dictionary names:   0:  person` 
```

`train` 和 `val` 字段指定了包含训练和验证图像的目录路径。

`names` 是一个类名字典。名称的顺序应与 YOLO 数据集文件中对象类索引的顺序相匹配。

（可选）如果点是对称的，则需要 flip_idx，例如人体或面部的左右侧。例如，如果我们假设面部标志的五个关键点为 [左眼、右眼、鼻子、左嘴、右嘴]，原始索引为 [0, 1, 2, 3, 4]，那么 flip_idx 就是 [1, 0, 2, 4, 3]（仅交换左右索引，即 0-1 和 3-4，并保持其余像鼻子不变）。

## 用法

示例

```py
`from ultralytics import YOLO  # Load a model model = YOLO("yolov8n-pose.pt")  # load a pretrained model (recommended for training)  # Train the model results = model.train(data="coco8-pose.yaml", epochs=100, imgsz=640)` 
```

```py
`# Start training from a pretrained *.pt model yolo  pose  train  data=coco8-pose.yaml  model=yolov8n-pose.pt  epochs=100  imgsz=640` 
```

## 支持的数据集

本节概述了与 Ultralytics YOLO 格式兼容且可用于训练姿势估计模型的数据集：

### COCO-Pose

+   **描述**：COCO-Pose 是一个大规模对象检测、分割和姿势估计数据集。它是流行的 COCO 数据集的子集，专注于人体姿势估计。COCO-Pose 包括每个人体实例的多个关键点。

+   **标签格式**：与上述的 Ultralytics YOLO 格式相同，带有人体姿势的关键点。

+   **类别数量**：1（人类）。

+   **关键点**：包括鼻子、眼睛、耳朵、肩膀、肘部、手腕、臀部、膝盖和脚踝等 17 个关键点。

+   **用途**：适用于训练人体姿势估计模型。

+   **附加说明**：该数据集丰富多样，包含超过 20 万张标注图像。

+   了解更多关于 COCO-Pose 的信息

### COCO8-Pose

+   **描述**：[Ultralytics](https://ultralytics.com) COCO8-Pose 是一个小而多功能的姿态检测数据集，由 COCO 训练 2017 集的前 8 张图像组成，4 张用于训练，4 张用于验证。

+   **标签格式**：与上述描述的 Ultralytics YOLO 格式相同，包含人类姿态的关键点。

+   **类别数量**：1（人类）。

+   **关键点**：17 个关键点，包括鼻子、眼睛、耳朵、肩膀、肘部、手腕、臀部、膝盖和脚踝。

+   **用法**：适合测试和调试对象检测模型，或用于尝试新的检测方法。

+   **附加说明**：COCO8-Pose 非常适合进行合理性检查和 CI 检查。

+   了解更多关于 COCO8-Pose 的信息

### Tiger-Pose

+   **描述**：[Ultralytics](https://ultralytics.com) 这个动物姿态数据集包含 263 张来自 [YouTube 视频](https://www.youtube.com/watch?v=MIBAT6BGE6U&pp=ygUbVGlnZXIgd2Fsa2luZyByZWZlcmVuY2UubXA0) 的图像，其中 210 张用于训练，53 张用于验证。

+   **标签格式**：与上述描述的 Ultralytics YOLO 格式相同，包含 12 个动物姿态的关键点，并且没有可见的维度。

+   **类别数量**：1（老虎）。

+   **关键点**：12 个关键点。

+   **用法**：非常适合动物姿态或任何其他非人类的姿态。

+   了解更多关于 Tiger-Pose 的信息

### 添加你自己的数据集

如果你有自己的数据集并希望使用它来训练 Ultralytics YOLO 格式的姿态估计模型，请确保它遵循上述“Ultralytics YOLO 格式”中指定的格式。将你的注释转换为所需格式，并在 YAML 配置文件中指定路径、类别数量和类别名称。

### 转换工具

Ultralytics 提供了一个方便的转换工具，可以将流行的 COCO 数据集格式的标签转换为 YOLO 格式：

示例

```py
`from ultralytics.data.converter import convert_coco  convert_coco(labels_dir="path/to/coco/annotations/", use_keypoints=True)` 
```

该转换工具可用于将 COCO 数据集或任何 COCO 格式的数据集转换为 Ultralytics YOLO 格式。`use_keypoints` 参数指定是否在转换的标签中包含关键点（用于姿态估计）。

## 常见问题解答

### Ultralytics YOLO 格式的姿态估计是什么？

Ultralytics YOLO 格式的姿态估计数据集涉及为每张图像标注一个相应的文本文件。文本文件的每一行存储有关对象实例的信息：

+   对象类别索引

+   对象中心坐标（归一化的 x 和 y）

+   对象宽度和高度（归一化）

+   对象关键点坐标（归一化的 pxn 和 pyn）

对于 2D 姿态，关键点包括像素坐标。对于 3D，每个关键点还具有可见性标志。有关更多详细信息，请参见 Ultralytics YOLO 格式。

### 我如何使用 COCO-Pose 数据集与 Ultralytics YOLO？

要在 Ultralytics YOLO 中使用 COCO-Pose 数据集：1\. 下载数据集并准备 YOLO 格式的标签文件。2\. 创建一个 YAML 配置文件，指定训练和验证图像的路径，关键点形状和类名。3\. 使用配置文件进行训练：

```py
```` ```pypython from ultralytics import YOLO  model = YOLO("yolov8n-pose.pt")  # load pretrained model results = model.train(data="coco-pose.yaml", epochs=100, imgsz=640) ```  欲了解更多信息，请访问 COCO-Pose 和训练部分。 ```py` 
```

### 如何在 Ultralytics YOLO 中添加自己的姿势估计数据集？

要添加你的数据集：1\. 将你的标注转换为 Ultralytics YOLO 格式。2\. 创建一个 YAML 配置文件，指定数据集路径、类别数量和类名。3\. 使用配置文件训练你的模型：

```py
```` ```pypython from ultralytics import YOLO  model = YOLO("yolov8n-pose.pt") results = model.train(data="your-dataset.yaml", epochs=100, imgsz=640) ```  完整步骤，请查看添加自己数据集部分。 ```py` 
```

### Ultralytics YOLO 中的数据集 YAML 文件的目的是什么？

Ultralytics YOLO 中的数据集 YAML 文件定义了训练的数据集和模型配置。它指定了训练、验证和测试图像的路径，关键点形状，类名以及其他配置选项。这种结构化格式有助于简化数据集管理和模型训练。以下是一个 YAML 格式的示例：

```py
`path:  ../datasets/coco8-pose train:  images/train val:  images/val names:   0:  person` 
```

更多关于创建数据集 YAML 配置文件的信息，请阅读 Dataset YAML 格式。

### 如何将 COCO 数据集标签转换为 Ultralytics YOLO 格式，用于姿势估计？

Ultralytics 提供一个转换工具，将 COCO 数据集标签转换为 YOLO 格式，包括关键点信息：

```py
`from ultralytics.data.converter import convert_coco  convert_coco(labels_dir="path/to/coco/annotations/", use_keypoints=True)` 
```

此工具帮助无缝集成 COCO 数据集到 YOLO 项目中。详情请参考 Conversion Tool 部分。
