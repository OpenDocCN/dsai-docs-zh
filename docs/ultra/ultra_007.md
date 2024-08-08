# 使用 Ultralytics YOLO 导出模型

> 原文：[`docs.ultralytics.com/modes/export/`](https://docs.ultralytics.com/modes/export/)

![Ultralytics YOLO 生态系统和集成](img/1933b0eeaf180eaa6d0c37f29931fb7d.png)

## 简介

训练模型的最终目标是在实际应用中部署它。Ultralytics YOLOv8 的导出模式提供了多种选项，可将训练好的模型导出至不同格式，从而使其能够在各种平台和设备上部署。本详尽指南旨在引导您了解模型导出的细节，展示如何实现最大的兼容性和性能。

[`www.youtube.com/embed/WbomGeoOT_k?si=aGmuyooWftA0ue9X`](https://www.youtube.com/embed/WbomGeoOT_k?si=aGmuyooWftA0ue9X)

**观看：** 如何导出自定义训练的 Ultralytics YOLOv8 模型，并在网络摄像头上进行实时推理。

## 为什么选择 YOLOv8 的导出模式？

+   **多功能性：** 导出至包括 ONNX、TensorRT、CoreML 等多种格式。

+   **性能：** 使用 TensorRT 可获得最多 5 倍的 GPU 加速，使用 ONNX 或 OpenVINO 可获得最多 3 倍的 CPU 加速。

+   **兼容性：** 使您的模型能够普遍适用于多种硬件和软件环境。

+   **易用性：** 简单的命令行界面和 Python API，便于快速和直接的模型导出。

### 导出模式的关键特性

下面是一些突出的功能：

+   **一键导出：** 简单命令，可导出至不同格式。

+   **批量导出：** 导出支持批处理推理的模型。

+   **优化推理速度：** 导出模型经过优化，推理速度更快。

+   **教程视频：** 深入指南和教程，帮助您顺利进行导出操作。

提示

+   导出至 ONNX 或 OpenVINO，CPU 速度提升最多 3 倍。

+   导出至 TensorRT，GPU 速度提升最多 5 倍。

## 使用示例

将 YOLOv8n 模型导出至 ONNX 或 TensorRT 等不同格式。查看下面的参数部分，了解所有导出参数的完整列表。

示例

```py
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # load an official model
model = YOLO("path/to/best.pt")  # load a custom trained model

# Export the model
model.export(format="onnx") 
```

```py
yolo  export  model=yolov8n.pt  format=onnx  # export official model
yolo  export  model=path/to/best.pt  format=onnx  # export custom trained model 
```

## 参数

此表详细描述了将 YOLO 模型导出至不同格式的配置和选项。这些设置对优化导出模型的性能、大小和在各种平台和环境中的兼容性至关重要。适当的配置确保模型能够在预期应用中以最佳效率部署。

| 参数 | 类型 | 默认值 | 描述 |
| --- | --- | --- | --- |
| `format` | `str` | `'torchscript'` | 导出模型的目标格式，如 `'onnx'`、`'torchscript'`、`'tensorflow'` 等，定义与各种部署环境的兼容性。 |
| `imgsz` | `int` 或 `tuple` | `640` | 模型输入的期望图像尺寸。可以是整数表示正方形图像，也可以是元组 `(height, width)` 表示具体尺寸。 |
| `keras` | `bool` | `False` | 启用导出至 TensorFlow SavedModel 的 Keras 格式，提供与 TensorFlow Serving 和 API 的兼容性。 |
| `optimize` | `bool` | `False` | 在导出 TorchScript 到移动设备时应用优化，可能减小模型大小并提高性能。 |
| `half` | `bool` | `False` | 启用 FP16（半精度）量化，减小模型大小并在支持的硬件上加快推断速度。 |
| `int8` | `bool` | `False` | 激活 INT8 量化，进一步压缩模型并在几乎不损失精度的情况下加快推断速度，主要用于边缘设备。 |
| `dynamic` | `bool` | `False` | 允许 ONNX、TensorRT 和 OpenVINO 导出使用动态输入尺寸，增强处理不同图像尺寸的灵活性。 |
| `simplify` | `bool` | `False` | 使用 `onnxslim` 简化 ONNX 导出的模型图，可能提高性能和兼容性。 |
| `opset` | `int` | `None` | 指定 ONNX opset 版本，以便与不同的 ONNX 解析器和运行时兼容。如果未设置，将使用支持的最新版本。 |
| `workspace` | `float` | `4.0` | 设置 TensorRT 优化的最大工作空间大小（单位：GiB），平衡内存使用和性能。 |
| `nms` | `bool` | `False` | 在 CoreML 导出中添加非最大抑制（NMS），用于精确和高效的检测后处理。 |
| `batch` | `int` | `1` | 指定导出模型的批量推断大小，或者导出模型在 `predict` 模式下并发处理的最大图像数量。 |

调整这些参数允许定制导出过程，以适应特定的需求，如部署环境、硬件约束和性能目标。选择合适的格式和设置对于实现模型大小、速度和精度的最佳平衡至关重要。

## 导出格式

下面的表格列出了可用的 YOLOv8 导出格式。您可以使用 `format` 参数导出到任何格式，例如 `format='onnx'` 或 `format='engine'`。导出完成后，您可以直接预测或验证导出的模型，例如 `yolo predict model=yolov8n.onnx`。下面展示了导出后您模型的使用示例。

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

## 常见问题解答

### 如何将 YOLOv8 模型导出为 ONNX 格式？

使用 Ultralytics 导出 YOLOv8 模型到 ONNX 格式非常简单，提供了 Python 和 CLI 方法来导出模型。

示例

```py
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # load an official model
model = YOLO("path/to/best.pt")  # load a custom trained model

# Export the model
model.export(format="onnx") 
```

```py
yolo  export  model=yolov8n.pt  format=onnx  # export official model
yolo  export  model=path/to/best.pt  format=onnx  # export custom trained model 
```

关于包括处理不同输入尺寸在内的高级选项，更多详细流程请参考 ONNX 部分。

### 使用 TensorRT 进行模型导出的好处是什么？

使用 TensorRT 进行模型导出能显著提升性能。导出到 TensorRT 的 YOLOv8 模型可以实现多达 5 倍的 GPU 加速，非常适合实时推理应用。

+   **通用性：** 为特定硬件设置优化模型。

+   **速度：** 通过先进优化实现更快推理速度。

+   **兼容性：** 与 NVIDIA 硬件无缝集成。

要了解更多有关集成 TensorRT 的信息，请参阅 TensorRT 集成指南。

### 如何在导出 YOLOv8 模型时启用 INT8 量化？

INT8 量化是压缩模型并加速推理的优秀方式，尤其适用于边缘设备。以下是如何启用 INT8 量化的方法：

示例

```py
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # Load a model
model.export(format="onnx", int8=True) 
```

```py
yolo  export  model=yolov8n.pt  format=onnx  int8=True  # export model with INT8 quantization 
```

INT8 量化可以应用于多种格式，如 TensorRT 和 CoreML。更多详细信息请参考导出部分。

### 在导出模型时，为什么动态输入尺寸很重要？

动态输入尺寸允许导出的模型处理不同的图像尺寸，为不同用例提供灵活性并优化处理效率。当导出到 ONNX 或 TensorRT 等格式时，启用动态输入尺寸可以确保模型能够无缝适应不同的输入形状。

要启用此功能，在导出时使用`dynamic=True`标志：

示例

```py
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.export(format="onnx", dynamic=True) 
```

```py
yolo  export  model=yolov8n.pt  format=onnx  dynamic=True 
```

关于动态输入尺寸配置的更多上下文，请参考。

### 如何优化模型性能的关键导出参数是什么？

理解和配置导出参数对优化模型性能至关重要：

+   **`format:`** 导出模型的目标格式（例如`onnx`、`torchscript`、`tensorflow`）。

+   **`imgsz:`** 模型输入的期望图像尺寸（例如`640`或`(height, width)`）。

+   **`half:`** 启用 FP16 量化，减小模型大小并可能加快推理速度。

+   **`optimize:`** 为移动或受限环境应用特定优化。

+   **`int8:`** 启用 INT8 量化，对边缘部署极为有益。

想了解所有导出参数的详细列表和解释，请访问导出参数部分。
