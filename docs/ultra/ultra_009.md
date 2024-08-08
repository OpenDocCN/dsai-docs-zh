# 使用 Ultralytics YOLO 进行模型基准测试

> 原文：[`docs.ultralytics.com/modes/benchmark/`](https://docs.ultralytics.com/modes/benchmark/)

![Ultralytics YOLO 生态系统和集成](img/1933b0eeaf180eaa6d0c37f29931fb7d.png)

## 介绍

一旦您的模型经过训练和验证，下一个合乎逻辑的步骤就是在各种真实场景中评估其性能。Ultralytics YOLOv8 的基准模式通过提供一个强大的框架，为您的模型在一系列导出格式中评估速度和准确性提供了一个坚实的基础。

[`www.youtube.com/embed/j8uQc0qB91s?start=105`](https://www.youtube.com/embed/j8uQc0qB91s?start=105)

**观看：** Ultralytics 模式教程：基准测试

## 为什么基准测试至关重要？

+   **明智的决策：** 深入了解速度和准确性之间的权衡。

+   **资源分配：** 了解不同导出格式在不同硬件上的性能表现。

+   **优化：** 了解哪种导出格式对于您特定的用例提供最佳性能。

+   **成本效率：** 根据基准测试结果更有效地利用硬件资源。

### 基准模式中的关键指标

+   **mAP50-95：** 用于目标检测、分割和姿态估计。

+   **accuracy_top5：** 用于图像分类。

+   **推理时间：** 每张图像所需的时间（毫秒）。

### 支持的导出格式

+   **ONNX：** 用于最佳的 CPU 性能

+   **TensorRT：** 实现最大的 GPU 效率

+   **OpenVINO：** 适用于英特尔硬件优化

+   **CoreML、TensorFlow SavedModel 等等：** 适用于多样化的部署需求。

提示

+   导出到 ONNX 或 OpenVINO 可以实现高达 3 倍的 CPU 加速。

+   导出到 TensorRT 可以实现高达 5 倍的 GPU 加速。

## 使用示例

在所有支持的导出格式上运行 YOLOv8n 基准测试，包括 ONNX、TensorRT 等。请查看下面的参数部分，了解完整的导出参数列表。

示例

```py
from ultralytics.utils.benchmarks import benchmark

# Benchmark on GPU
benchmark(model="yolov8n.pt", data="coco8.yaml", imgsz=640, half=False, device=0) 
```

```py
yolo  benchmark  model=yolov8n.pt  data='coco8.yaml'  imgsz=640  half=False  device=0 
```

## 参数

参数如 `model`、`data`、`imgsz`、`half`、`device` 和 `verbose` 为用户提供了灵活性，可以根据其特定需求微调基准测试，并轻松比较不同导出格式的性能。

| 键 | 默认值 | 描述 |
| --- | --- | --- |
| `model` | `None` | 指定模型文件的路径。接受 `.pt` 和 `.yaml` 格式，例如，`"yolov8n.pt"` 用于预训练模型或配置文件。 |
| `data` | `None` | 定义用于基准测试的数据集的 YAML 文件路径，通常包括验证数据的路径和设置。示例："coco8.yaml"。 |
| `imgsz` | `640` | 模型的输入图像大小。可以是一个整数用于方形图像，或者是一个元组 `(width, height)` 用于非方形图像，例如 `(640, 480)`。 |
| `half` | `False` | 启用 FP16（半精度）推理，减少内存使用量，并可能在兼容硬件上增加速度。使用 `half=True` 来启用。 |
| `int8` | `False` | 激活 INT8 量化，以进一步优化支持设备上的性能，特别适用于边缘设备。设置 `int8=True` 来使用。 |
| `device` | `None` | 定义基准测试的计算设备，如 `"cpu"`、`"cuda:0"`，或像 `"cuda:0,1"` 这样的多 GPU 设置。 |
| `verbose` | `False` | 控制日志输出的详细级别。布尔值；设置 `verbose=True` 可获取详细日志，或设置浮点数以进行错误阈值设定。 |

## 导出格式

基准测试将尝试自动运行所有可能的导出格式。

| 格式 | `format` 参数 | 模型 | 元数据 | 参数 |
| --- | --- | --- | --- | --- |
| [PyTorch](https://pytorch.org/) | - | `yolov8n.pt` | ✅ | - |
| TorchScript | `torchscript` | `yolov8n.torchscript` | ✅ | `imgsz`, `optimize`, `batch` |
| ONNX | `onnx` | `yolov8n.onnx` | ✅ | `imgsz`, `half`, `dynamic`, `simplify`, `opset`, `batch` |
| OpenVINO | `openvino` | `yolov8n_openvino_model/` | ✅ | `imgsz`, `half`, `int8`, `batch`, `dynamic` |
| TensorRT | `engine` | `yolov8n.engine` | ✅ | `imgsz`, `half`, `dynamic`, `simplify`, `workspace`, `int8`, `batch` |
| CoreML | `coreml` | `yolov8n.mlpackage` | ✅ | `imgsz`, `half`, `int8`, `nms`, `batch` |
| TF SavedModel | `saved_model` | `yolov8n_saved_model/` | ✅ | `imgsz`, `keras`, `int8`, `batch` |
| TF GraphDef | `pb` | `yolov8n.pb` | ❌ | `imgsz`, `batch` |
| TF Lite | `tflite` | `yolov8n.tflite` | ✅ | `imgsz`, `half`, `int8`, `batch` |
| TF Edge TPU | `edgetpu` | `yolov8n_edgetpu.tflite` | ✅ | `imgsz` |
| TF.js | `tfjs` | `yolov8n_web_model/` | ✅ | `imgsz`, `half`, `int8`, `batch` |
| PaddlePaddle | `paddle` | `yolov8n_paddle_model/` | ✅ | `imgsz`, `batch` |
| NCNN | `ncnn` | `yolov8n_ncnn_model/` | ✅ | `imgsz`, `half`, `batch` |

查看导出页面的完整导出详情。

## 常见问题

### 如何使用 Ultralytics 对我的 YOLOv8 模型进行基准测试？

Ultralytics YOLOv8 提供了一个基准模式，可以评估模型在不同导出格式下的性能。该模式提供关键指标，如平均精度（mAP50-95）、准确性以及推断时间（毫秒）。要运行基准测试，可以使用 Python 或 CLI 命令。例如，在 GPU 上运行基准测试：

示例

```py
from ultralytics.utils.benchmarks import benchmark

# Benchmark on GPU
benchmark(model="yolov8n.pt", data="coco8.yaml", imgsz=640, half=False, device=0) 
```

```py
yolo  benchmark  model=yolov8n.pt  data='coco8.yaml'  imgsz=640  half=False  device=0 
```

有关基准参数的更多详情，请访问参数部分。

### 导出 YOLOv8 模型到不同格式有哪些好处？

将 YOLOv8 模型导出到不同格式，如 ONNX、TensorRT 和 OpenVINO，可以根据部署环境优化性能。例如：

+   **ONNX：**提供最多 3 倍的 CPU 加速。

+   **TensorRT：**提供最多 5 倍的 GPU 加速。

+   **OpenVINO：**专为 Intel 硬件优化。这些格式提升了模型的速度和准确性，使其在各种实际应用中更加高效。访问导出页面获取完整详情。

### 为什么基准测试在评估 YOLOv8 模型时至关重要？

对您的 YOLOv8 模型进行基准测试至关重要，理由如下：

+   **明智决策：**理解速度和准确性之间的权衡。

+   **资源分配：** 评估在不同硬件选项上的性能。

+   **优化：** 确定哪种导出格式针对特定用例提供最佳性能。

+   **成本效率：** 根据基准测试结果优化硬件使用。关键指标如 mAP50-95、Top-5 准确性和推理时间有助于进行这些评估。有关更多信息，请参阅关键指标部分。

### YOLOv8 支持哪些导出格式，它们各自有什么优势？

YOLOv8 支持多种导出格式，每种都针对特定的硬件和用例进行了定制：

+   **ONNX：** 最适合 CPU 性能。

+   **TensorRT：** 理想的 GPU 效率。

+   **OpenVINO：** 针对 Intel 硬件优化。

+   **CoreML & TensorFlow：** 适用于 iOS 和一般 ML 应用程序。有关支持的所有格式及其各自优势的完整列表，请查看支持的导出格式部分。

### 我可以使用哪些参数来优化我的 YOLOv8 基准测试？

运行基准测试时，可以自定义多个参数以满足特定需求：

+   **模型：** 模型文件的路径（例如，"yolov8n.pt"）。

+   **数据：** 定义数据集的 YAML 文件路径（例如，"coco8.yaml"）。

+   **imgsz：** 输入图像大小，可以是单个整数或元组。

+   **half：** 启用 FP16 推理以获得更好的性能。

+   **int8：** 为边缘设备激活 INT8 量化。

+   **设备：** 指定计算设备（例如，"cpu"，"cuda:0"）。

+   **详细模式：** 控制日志详细程度。有关所有参数的完整列表，请参阅参数部分。
