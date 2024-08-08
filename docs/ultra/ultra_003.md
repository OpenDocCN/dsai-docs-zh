# Ultralytics YOLOv8 模式

> 原文：[`docs.ultralytics.com/modes/`](https://docs.ultralytics.com/modes/)

![Ultralytics YOLO 生态系统与集成](img/1933b0eeaf180eaa6d0c37f29931fb7d.png)

## 介绍

Ultralytics YOLOv8 不仅仅是另一个目标检测模型；它是一个多功能框架，旨在覆盖机器学习模型的整个生命周期——从数据摄入和模型训练到验证、部署和实时跟踪。每种模式都有特定的目的，并且旨在为不同任务和用例提供所需的灵活性和效率。

[`www.youtube.com/embed/j8uQc0qB91s?si=dhnGKgqvs7nPgeaM`](https://www.youtube.com/embed/j8uQc0qB91s?si=dhnGKgqvs7nPgeaM)

**观看：** Ultralytics 模式教程：训练、验证、预测、导出和基准测试。

### 模式一览

了解 Ultralytics YOLOv8 支持的不同**模式**对于充分利用您的模型至关重要：

+   **训练**模式：在自定义或预加载数据集上进行模型微调。

+   **验证**模式：用于验证模型性能的训练后检查点。

+   **预测**模式：释放您的模型在真实世界数据上的预测能力。

+   **导出**模式：使您的模型能够在各种格式下进行部署。

+   **追踪**模式：将您的目标检测模型扩展到实时跟踪应用程序中。

+   **基准测试**模式：分析您的模型在不同部署环境中的速度和准确性。

本全面指南旨在为您提供每种模式的概述和实用见解，帮助您充分利用 YOLOv8 的潜力。

## 训练

训练模式用于在自定义数据集上训练 YOLOv8 模型。在此模式下，使用指定的数据集和超参数训练模型。训练过程涉及优化模型的参数，以便它能够准确预测图像中对象的类别和位置。

训练示例

## 验证

验证模式用于在模型训练后验证 YOLOv8 模型。在此模式下，模型在验证集上评估其准确性和泛化性能。可以使用此模式调整模型的超参数以提高其性能。

验证示例

## 预测

预测模式用于在新图像或视频上使用训练后的 YOLOv8 模型进行预测。在这种模式下，模型从检查点文件加载，用户可以提供图像或视频进行推理。模型预测输入图像或视频中对象的类别和位置。

预测示例

## 导出

导出模式用于将 YOLOv8 模型导出为可用于部署的格式。在此模式下，模型被转换为其他软件应用程序或硬件设备可以使用的格式。在将模型部署到生产环境时，此模式非常有用。

导出示例

## 追踪

Track mode 用于使用 YOLOv8 模型实时跟踪对象。在此模式下，模型从检查点文件加载，用户可以提供实时视频流执行实时对象跟踪。此模式适用于监视系统或自动驾驶汽车等应用。

Track Examples

## Benchmark

Benchmark 模式用于分析 YOLOv8 各种导出格式的速度和准确性。基准测试提供了关于导出格式大小、`mAP50-95` 指标（用于目标检测、分割和姿态）或 `accuracy_top5` 指标（用于分类）以及每张图像的推理时间（以毫秒为单位），跨多种导出格式如 ONNX、OpenVINO、TensorRT 等。此信息可帮助用户根据其对速度和准确性要求选择最佳的导出格式。

Benchmark Examples

## FAQ

### 如何使用 Ultralytics YOLOv8 训练自定义目标检测模型？

使用 Ultralytics YOLOv8 训练自定义目标检测模型涉及使用训练模式。您需要一个格式为 YOLO 的数据集，包含图像和相应的标注文件。使用以下命令开始训练过程：

Example

```py
`from ultralytics import YOLO  # Train a custom model model = YOLO("yolov8n.pt") model.train(data="path/to/dataset.yaml", epochs=100, imgsz=640)` 
```

```py
`yolo  train  data=path/to/dataset.yaml  epochs=100  imgsz=640` 
```

欲获取更详细的说明，请参阅 Ultralytics Train Guide。

### Ultralytics YOLOv8 使用哪些指标来验证模型的性能？

Ultralytics YOLOv8 在验证过程中使用多种指标评估模型性能。这些指标包括：

+   **mAP (平均精度均值)**：评估目标检测的准确性。

+   **IOU (预测框与真实框的交并比)**：衡量预测框与真实框之间的重叠度。

+   **精确率和召回率**：精确率衡量真正检测到的正例与所有检测到的正例的比例，而召回率衡量真正检测到的正例与所有实际正例的比例。

您可以运行以下命令开始验证：

Example

```py
`from ultralytics import YOLO  # Validate the model model = YOLO("yolov8n.pt") model.val(data="path/to/validation.yaml")` 
```

```py
`yolo  val  data=path/to/validation.yaml` 
```

欲了解更多详细信息，请参阅 Validation Guide。

### 如何导出我的 YOLOv8 模型以进行部署？

Ultralytics YOLOv8 提供导出功能，将您训练好的模型转换为各种部署格式，如 ONNX、TensorRT、CoreML 等。使用以下示例导出您的模型：

Example

```py
`from ultralytics import YOLO  # Export the model model = YOLO("yolov8n.pt") model.export(format="onnx")` 
```

```py
`yolo  export  model=yolov8n.pt  format=onnx` 
```

每种导出格式的详细步骤可在 Export Guide 中找到。

### Ultralytics YOLOv8 的 Benchmark 模式的目的是什么？

Ultralytics YOLOv8 中的 Benchmark 模式用于分析各种导出格式（如 ONNX、TensorRT 和 OpenVINO）的速度和准确性。它提供了模型大小、目标检测的 `mAP50-95` 以及在不同硬件设置下的推理时间等指标，帮助您选择最适合部署需求的格式。

Example

```py
`from ultralytics.utils.benchmarks import benchmark  # Benchmark on GPU benchmark(model="yolov8n.pt", data="coco8.yaml", imgsz=640, half=False, device=0)` 
```

```py
`yolo  benchmark  model=yolov8n.pt  data='coco8.yaml'  imgsz=640  half=False  device=0` 
```

欲获取更多细节，请参阅 Benchmark Guide。

### 如何使用 Ultralytics YOLOv8 进行实时对象跟踪？

使用 Ultralytics YOLOv8 中的跟踪模式可以实现实时对象跟踪。该模式扩展了对象检测能力，可以跨视频帧或实时流跟踪对象。使用以下示例启用跟踪：

示例

```py
`from ultralytics import YOLO  # Track objects in a video model = YOLO("yolov8n.pt") model.track(source="path/to/video.mp4")` 
```

```py
`yolo  track  source=path/to/video.mp4` 
```

欲了解详细说明，请访问跟踪指南。
