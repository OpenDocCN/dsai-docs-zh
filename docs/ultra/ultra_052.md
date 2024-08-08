# 实例分割数据集概述

> 原文：[`docs.ultralytics.com/datasets/segment/`](https://docs.ultralytics.com/datasets/segment/)

## 支持的数据集格式

### Ultralytics YOLO 格式

用于训练 YOLO 分割模型的数据集标签格式如下：

1.  每个图像一个文本文件：数据集中每个图像都有一个相应的文本文件，文件名与图像文件相同，扩展名为".txt"。

1.  每个对象一行：文本文件中的每一行对应图像中的一个对象实例。

1.  每行的对象信息：每行包含对象实例的以下信息：

    +   对象类索引：表示对象类的整数（例如，人为 0，汽车为 1 等）。

    +   对象边界坐标：围绕掩模区域的边界坐标，归一化为 0 到 1 之间。

分割数据集文件中单行的格式如下：

```py
`<class-index> <x1> <y1> <x2> <y2> ... <xn> <yn>` 
```

在此格式中，`<类索引>` 是对象的类索引，`<x1> <y1> <x2> <y2> ... <xn> <yn>` 是对象分割掩模的边界坐标。坐标之间用空格分隔。

这是 YOLO 数据集格式的单个图像示例，包含由 3 点段和 5 点段组成的两个对象。

```py
`0 0.681 0.485 0.670 0.487 0.676 0.487 1 0.504 0.000 0.501 0.004 0.498 0.004 0.493 0.010 0.492 0.0104` 
```

提示

+   每行的长度**不需要**相等。

+   每个分割标签必须至少有 3 个 xy 点：`<类索引> <x1> <y1> <x2> <y2> <x3> <y3>`

### 数据集 YAML 格式

Ultralytics 框架使用 YAML 文件格式定义用于训练检测模型的数据集和模型配置。以下是用于定义检测数据集的 YAML 格式示例：

```py
`# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..] path:  ../datasets/coco8-seg  # dataset root dir train:  images/train  # train images (relative to 'path') 4 images val:  images/val  # val images (relative to 'path') 4 images test:  # test images (optional)  # Classes (80 COCO classes) names:   0:  person   1:  bicycle   2:  car   # ...   77:  teddy bear   78:  hair drier   79:  toothbrush` 
```

`train` 和 `val` 字段指定分别包含训练和验证图像的目录路径。

`names` 是类名的字典。名称的顺序应与 YOLO 数据集文件中对象类索引的顺序相匹配。

## 用法

示例

```py
`from ultralytics import YOLO  # Load a model model = YOLO("yolov8n-seg.pt")  # load a pretrained model (recommended for training)  # Train the model results = model.train(data="coco8-seg.yaml", epochs=100, imgsz=640)` 
```

```py
`# Start training from a pretrained *.pt model yolo  segment  train  data=coco8-seg.yaml  model=yolov8n-seg.pt  epochs=100  imgsz=640` 
```

## 支持的数据集

## 支持的数据集

+   COCO：一个全面的对象检测、分割和字幕数据集，涵盖了各种类别的超过 200K 张标记图像。

+   COCO8-seg：COCO 的紧凑版，包含 8 张图像，用于快速测试分割模型训练，在`ultralytics`存储库中进行 CI 检查和工作流验证时非常理想。

+   Carparts-seg：专注于汽车部件分割的专业数据集，非常适合汽车应用。它包括多种车辆，具有详细的个别汽车组件注释。

+   Crack-seg：专为各种表面裂缝分割而设计的数据集。对于基础设施维护和质量控制至关重要，提供详细的图像用于训练模型识别结构弱点。

+   Package-seg：专注于不同类型包装材料和形状分割的数据集。它对物流和仓储自动化特别有用，有助于开发包装处理和分类系统。

### 添加您自己的数据集

如果您有自己的数据集，并希望将其用于使用 Ultralytics YOLO 格式训练分割模型，请确保其遵循上述“Ultralytics YOLO 格式”中指定的格式。将您的注释转换为所需格式，并在 YAML 配置文件中指定路径、类别数量和类名。

## 转换或转换标签格式

### 将 COCO 数据集格式转换为 YOLO 格式

您可以使用以下代码片段将流行的 COCO 数据集格式标签轻松转换为 YOLO 格式：

示例

```py
`from ultralytics.data.converter import convert_coco  convert_coco(labels_dir="path/to/coco/annotations/", use_segments=True)` 
```

此转换工具可用于将 COCO 数据集或任何 COCO 格式的数据集转换为 Ultralytics YOLO 格式。

请务必仔细检查您想使用的数据集是否与您的模型兼容，并遵循必要的格式约定。正确格式化的数据集对于成功训练对象检测模型至关重要。

## 自动标注

自动标注是一个重要的功能，允许您使用预训练检测模型生成分割数据集。它使您能够快速准确地对大量图像进行注释，无需手动标注，从而节省时间和精力。

### 使用检测模型生成分割数据集

要使用 Ultralytics 框架自动标注您的数据集，可以如下所示使用 `auto_annotate` 函数：

示例

```py
`from ultralytics.data.annotator import auto_annotate  auto_annotate(data="path/to/images", det_model="yolov8x.pt", sam_model="sam_b.pt")` 
```

| 参数 | 类型 | 描述 | 默认值 |
| --- | --- | --- | --- |
| `data` | `str` | 包含要注释图像的文件夹的路径。 | `None` |
| `det_model` | `str，可选` | 预训练的 YOLO 检测模型。默认为 `'yolov8x.pt'`。 | `'yolov8x.pt'` |
| `sam_model` | `str，可选` | 预训练的 SAM 分割模型。默认为 `'sam_b.pt'`。 | `'sam_b.pt'` |
| `device` | `str，可选` | 运行模型的设备。默认为空字符串（CPU 或 GPU，如果可用）。 | `''` |
| `output_dir` | `str 或 None，可选` | 保存注释结果的目录。默认为与 `'data'` 目录相同的 `'labels'` 文件夹。 | `None` |

`auto_annotate` 函数接受您的图像路径，以及用于指定预训练检测和 SAM 分割模型、运行模型的设备以及保存注释结果的输出目录的可选参数。

利用预训练模型的力量，自动标注可以显著减少创建高质量分割数据集所需的时间和精力。这一特性特别适用于处理大量图像集合的研究人员和开发人员，因为它允许他们集中精力进行模型开发和评估，而不是手动标注。

## 常见问题解答

### Ultralytics YOLO 支持哪些数据集格式来进行实例分割？

Ultralytics YOLO 支持多种数据集格式，例如实例分割，其中主要格式是其自身的 Ultralytics YOLO 格式。数据集中的每个图像都需要一个对应的文本文件，其中包含分割成多行的对象信息（每个对象一行），列出类索引和归一化的边界框坐标。有关 YOLO 数据集格式的详细说明，请访问 Instance Segmentation Datasets Overview。

### 我如何将 COCO 数据集注释转换为 YOLO 格式？

使用 Ultralytics 工具将 COCO 格式的注释转换为 YOLO 格式非常简单。您可以使用`ultralytics.data.converter`模块中的`convert_coco`函数：

```py
`from ultralytics.data.converter import convert_coco  convert_coco(labels_dir="path/to/coco/annotations/", use_segments=True)` 
```

这个脚本将您的 COCO 数据集注释转换为所需的 YOLO 格式，适用于训练您的 YOLO 模型。有关详细信息，请参阅 Port or Convert Label Formats。

### 我如何为训练 Ultralytics YOLO 模型准备一个 YAML 文件？

要为使用 Ultralytics 训练 YOLO 模型做准备，您需要定义数据集路径和类名。以下是一个 YAML 配置的示例：

```py
`path:  ../datasets/coco8-seg  # dataset root dir train:  images/train  # train images (relative to 'path')  val:  images/val  # val images (relative to 'path')   names:   0:  person   1:  bicycle   2:  car   # ...` 
```

确保根据您的数据集更新路径和类名。有关更多信息，请查看 Dataset YAML Format 部分。

### Ultralytics YOLO 中的自动注释功能是什么？

Ultralytics YOLO 中的自动注释允许您使用预训练的检测模型为您的数据集生成分割注释。这显著减少了手动标注的需求。您可以如下使用`auto_annotate`函数：

```py
`from ultralytics.data.annotator import auto_annotate  auto_annotate(data="path/to/images", det_model="yolov8x.pt", sam_model="sam_b.pt")` 
```

这个函数自动化了注释过程，使其更快速、高效。有关详细信息，请探索自动注释部分。
