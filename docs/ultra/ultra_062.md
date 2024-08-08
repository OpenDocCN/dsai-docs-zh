# 图像分类数据集概述

> 原文：[`docs.ultralytics.com/datasets/classify/`](https://docs.ultralytics.com/datasets/classify/)

### YOLO 分类任务的数据集结构

对于[Ultralytics](https://ultralytics.com) YOLO 分类任务，数据集必须按照特定的分割目录结构组织在`root`目录下，以便于正确的训练、测试和可选的验证过程。该结构包括训练（`train`）和测试（`test`）阶段各自的单独目录，以及一个可选的验证（`val`）目录。

每个目录应包含该数据集中每个类别的一个子目录。子目录以相应的类别命名，并包含该类别的所有图像。确保每个图像文件具有唯一的名称，并以 JPEG 或 PNG 等通用格式存储。

**文件夹结构示例**

以 CIFAR-10 数据集为例。文件夹结构应如下所示：

```py
`cifar-10-/ | |-- train/ |   |-- airplane/ |   |   |-- 10008_airplane.png |   |   |-- 10009_airplane.png |   |   |-- ... |   | |   |-- automobile/ |   |   |-- 1000_automobile.png |   |   |-- 1001_automobile.png |   |   |-- ... |   | |   |-- bird/ |   |   |-- 10014_bird.png |   |   |-- 10015_bird.png |   |   |-- ... |   | |   |-- ... | |-- test/ |   |-- airplane/ |   |   |-- 10_airplane.png |   |   |-- 11_airplane.png |   |   |-- ... |   | |   |-- automobile/ |   |   |-- 100_automobile.png |   |   |-- 101_automobile.png |   |   |-- ... |   | |   |-- bird/ |   |   |-- 1000_bird.png |   |   |-- 1001_bird.png |   |   |-- ... |   | |   |-- ... | |-- val/ (optional) |   |-- airplane/ |   |   |-- 105_airplane.png |   |   |-- 106_airplane.png |   |   |-- ... |   | |   |-- automobile/ |   |   |-- 102_automobile.png |   |   |-- 103_automobile.png |   |   |-- ... |   | |   |-- bird/ |   |   |-- 1045_bird.png |   |   |-- 1046_bird.png |   |   |-- ... |   | |   |-- ...` 
```

这种结构化方法确保模型在训练阶段能够有效地从组织良好的课程中学习，并在测试和验证阶段准确评估性能。

## 使用方法

示例

```py
`from ultralytics import YOLO  # Load a model model = YOLO("yolov8n-cls.pt")  # load a pretrained model (recommended for training)  # Train the model results = model.train(data="path/to/dataset", epochs=100, imgsz=640)` 
```

```py
`# Start training from a pretrained *.pt model yolo  detect  train  data=path/to/data  model=yolov8n-cls.pt  epochs=100  imgsz=640` 
```

## 支持的数据集

Ultralytics 支持以下数据集的自动下载：

+   Caltech 101：一个包含 101 个物体类别图像的数据集，用于图像分类任务。

+   Caltech 256：Caltech 101 的扩展版本，包含 256 个物体类别和更具挑战性的图像。

+   CIFAR-10：一个包含 60,000 张 32x32 彩色图像的数据集，分为 10 类，每类包含 6,000 张图像。

+   CIFAR-100：CIFAR-10 的扩展版本，包含 100 个物体类别和每类 600 张图像。

+   Fashion-MNIST：一个包含 70,000 张 10 种时尚类别灰度图像的数据集，用于图像分类任务。

+   ImageNet：一个大规模的用于目标检测和图像分类的数据集，包含超过 1400 万张图片和 2 万个类别。

+   ImageNet-10：ImageNet 的一个较小子集，包含 10 个类别，用于更快的实验和测试。

+   Imagenette：ImageNet 的一个较小子集，包含 10 个易于区分的类别，用于更快的训练和测试。

+   Imagewoof：ImageNet 的一个更具挑战性的子集，包含 10 个狗品种类别，用于图像分类任务。

+   MNIST：一个包含 70,000 张手写数字灰度图像的数据集，用于图像分类任务。

### 添加自己的数据集

如果您有自己的数据集，并希望将其用于训练 Ultralytics 的分类模型，请确保其遵循上述“数据集格式”中指定的格式，然后将您的`data`参数指向数据集目录。

## 常见问题解答

### 我如何为 YOLO 分类任务结构化我的数据集？

欲为 Ultralytics YOLO 分类任务结构化您的数据集，应遵循特定的分割目录格式。将您的数据集组织成单独的`train`、`test`和可选的`val`目录。每个目录应包含以各类别命名的子目录，其中包含相应的图像。这有助于流畅的训练和评估过程。例如，考虑 CIFAR-10 数据集的格式：

```py
`cifar-10-/ |-- train/ |   |-- airplane/ |   |-- automobile/ |   |-- bird/ |   ... |-- test/ |   |-- airplane/ |   |-- automobile/ |   |-- bird/ |   ... |-- val/ (optional) |   |-- airplane/ |   |-- automobile/ |   |-- bird/ |   ...` 
```

欲了解更多详细信息，请访问 YOLO 分类任务的数据集结构。

### Ultralytics YOLO 支持哪些用于图像分类的数据集？

Ultralytics YOLO 支持自动下载多个用于图像分类的数据集，包括：

+   Caltech 101

+   Caltech 256

+   CIFAR-10

+   CIFAR-100

+   Fashion-MNIST

+   ImageNet

+   ImageNet-10

+   Imagenette

+   Imagewoof

+   MNIST

这些数据集结构化有序，易于与 YOLO 配合使用。每个数据集页面提供有关其结构和应用的详细信息。

### 如何为 YOLO 图像分类添加自己的数据集？

欲将您自己的数据集用于 Ultralytics YOLO，请确保其遵循分类任务所需的指定目录格式，其中包括单独的`train`、`test`和可选的`val`目录，以及每个类别包含相应图像的子目录。一旦您的数据集正确结构化，请在初始化训练脚本时将`data`参数指向您的数据集根目录。以下是 Python 示例：

```py
`from ultralytics import YOLO  # Load a model model = YOLO("yolov8n-cls.pt")  # load a pretrained model (recommended for training)  # Train the model results = model.train(data="path/to/your/dataset", epochs=100, imgsz=640)` 
```

欲了解更多详细信息，请参阅添加自己的数据集部分。

### 我为什么要使用 Ultralytics YOLO 进行图像分类？

Ultralytics YOLO 为图像分类提供多种好处，包括：

+   **预训练模型**：加载预训练模型如`yolov8n-cls.pt`可加快您的训练过程。

+   **易用性**：简单的 API 和 CLI 命令用于训练和评估。

+   **高性能**：领先的准确性和速度，非常适合实时应用。

+   **多数据集支持**：与诸如 CIFAR-10、ImageNet 等多种流行数据集无缝集成。

+   **社区和支持**：可访问广泛的文档和活跃的社区进行故障排除和改进。

欲获取更多洞见及实际应用，您可以探索[Ultralytics YOLO](https://www.ultralytics.com/yolo)。

### 如何使用 Ultralytics YOLO 训练模型？

使用 Ultralytics YOLO 训练模型可轻松在 Python 和 CLI 中完成。以下是一个示例：

示例

```py
`from ultralytics import YOLO  # Load a model model = YOLO("yolov8n-cls.pt")  # load a pretrained model  # Train the model results = model.train(data="path/to/dataset", epochs=100, imgsz=640)` 
```

```py
`# Start training from a pretrained *.pt model yolo  detect  train  data=path/to/data  model=yolov8n-cls.pt  epochs=100  imgsz=640` 
```

这些示例展示了使用任一方法训练 YOLO 模型的简单过程。欲了解更多信息，请访问使用部分。
