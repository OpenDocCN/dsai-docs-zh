# YOLOv5

> 原文：[`docs.ultralytics.com/models/yolov5/`](https://docs.ultralytics.com/models/yolov5/)

## 概述

YOLOv5u 代表了物体检测方法论的进步。源自 Ultralytics 开发的 YOLOv5 模型的基础架构，YOLOv5u 集成了无锚点、无对象性的分割头部，这一特性此前已在 YOLOv8 模型中引入。这种适应性调整优化了模型的架构，在物体检测任务中实现了更好的准确度和速度权衡。根据实证结果及其派生特性，YOLOv5u 为那些在研究和实际应用中寻求强大解决方案的人提供了高效的替代选择。

![Ultralytics YOLOv5](img/043a7987b73c701bfe07aa6ab67c7f4c.png)

## 主要特点

+   **无锚点分割 Ultralytics 头：** 传统的物体检测模型依赖预定义的锚框来预测物体位置。然而，YOLOv5u 现代化了这种方法。通过采用无锚点分割 Ultralytics 头，它确保了更灵活和适应性更强的检测机制，从而在多种场景中提高了性能。

+   **优化的准确度-速度权衡：** 速度和准确度常常相互制约。但 YOLOv5u 挑战了这种权衡。它提供了一个校准的平衡，确保实时检测而不会牺牲准确性。这一特性在需要快速响应的应用中尤为宝贵，如自动驾驶车辆、机器人技术和实时视频分析。

+   **各种预训练模型：** 了解到不同任务需要不同的工具集，YOLOv5u 提供了大量预训练模型。无论您是专注于推理、验证还是训练，都有一个专门为您等待的量身定制的模型。这种多样性确保您不仅使用一种“一刀切”的解决方案，而是一种专门为您独特挑战进行了优化调整的模型。

## 支持的任务和模式

YOLOv5u 模型以各种预训练权重在物体检测任务中表现卓越。它们支持广泛的模式，适用于从开发到部署的各种应用。

| 模型类型 | 预训练权重 | 任务 | 推理 | 验证 | 训练 | 导出 |
| --- | --- | --- | --- | --- | --- | --- |
| YOLOv5u | `yolov5nu`, `yolov5su`, `yolov5mu`, `yolov5lu`, `yolov5xu`, `yolov5n6u`, `yolov5s6u`, `yolov5m6u`, `yolov5l6u`, `yolov5x6u` | 物体检测 | ✅ | ✅ | ✅ | ✅ |

此表详细介绍了 YOLOv5u 模型变体的概述，突出了它们在物体检测任务中的适用性以及对推理、验证、训练和导出等各种操作模式的支持。这种全面的支持确保用户能充分利用 YOLOv5u 模型在各种物体检测场景中的能力。

## 性能指标

性能

请查看检测文档，了解在 COCO 数据集上训练的这些模型的使用示例，其中包括 80 个预训练类别。

| Model | YAML | size ^((pixels)) | mAP^(val 50-95) | Speed ^(CPU ONNX

(ms)) | Speed ^(A100 TensorRT

(ms)) | params ^((M)) | FLOPs ^((B)) |

| --- | --- | --- | --- | --- | --- | --- | --- |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [yolov5nu.pt](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov5nu.pt) | [yolov5n.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5.yaml) | 640 | 34.3 | 73.6 | 1.06 | 2.6 | 7.7 |
| [yolov5su.pt](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov5su.pt) | [yolov5s.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5.yaml) | 640 | 43.0 | 120.7 | 1.27 | 9.1 | 24.0 |
| [yolov5mu.pt](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov5mu.pt) | [yolov5m.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5.yaml) | 640 | 49.0 | 233.9 | 1.86 | 25.1 | 64.2 |
| [yolov5lu.pt](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov5lu.pt) | [yolov5l.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5.yaml) | 640 | 52.2 | 408.4 | 2.50 | 53.2 | 135.0 |
| [yolov5xu.pt](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov5xu.pt) | [yolov5x.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5.yaml) | 640 | 53.2 | 763.2 | 3.81 | 97.2 | 246.4 |
|  |  |  |  |  |  |  |  |
| [yolov5n6u.pt](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov5n6u.pt) | [yolov5n6.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5-p6.yaml) | 1280 | 42.1 | 211.0 | 1.83 | 4.3 | 7.8 |
| [yolov5s6u.pt](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov5s6u.pt) | [yolov5s6.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5-p6.yaml) | 1280 | 48.6 | 422.6 | 2.34 | 15.3 | 24.6 |
| [yolov5m6u.pt](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov5m6u.pt) | [yolov5m6.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5-p6.yaml) | 1280 | 53.6 | 810.9 | 4.36 | 41.2 | 65.7 |
| [yolov5l6u.pt](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov5l6u.pt) | [yolov5l6.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5-p6.yaml) | 1280 | 55.7 | 1470.9 | 5.47 | 86.1 | 137.4 |
| [yolov5x6u.pt](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov5x6u.pt) | [yolov5x6.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5-p6.yaml) | 1280 | 56.8 | 2436.5 | 8.98 | 155.4 | 250.7 |

## Usage Examples

This example provides simple YOLOv5 training and inference examples. For full documentation on these and other modes see the Predict, Train, Val and Export docs pages.

Example

PyTorch 预训练的`*.pt`模型以及配置`*.yaml`文件可以传递给`YOLO()`类，以在 Python 中创建模型实例：

```py
`from ultralytics import YOLO  # Load a COCO-pretrained YOLOv5n model model = YOLO("yolov5n.pt")  # Display model information (optional) model.info()  # Train the model on the COCO8 example dataset for 100 epochs results = model.train(data="coco8.yaml", epochs=100, imgsz=640)  # Run inference with the YOLOv5n model on the 'bus.jpg' image results = model("path/to/bus.jpg")` 
```

CLI 命令可直接运行模型：

```py
`# Load a COCO-pretrained YOLOv5n model and train it on the COCO8 example dataset for 100 epochs yolo  train  model=yolov5n.pt  data=coco8.yaml  epochs=100  imgsz=640  # Load a COCO-pretrained YOLOv5n model and run inference on the 'bus.jpg' image yolo  predict  model=yolov5n.pt  source=path/to/bus.jpg` 
```

## 引文和致谢

如果您在研究中使用 YOLOv5 或 YOLOv5u，请引用 Ultralytics YOLOv5 库的存储库如下：

```py
`@software{yolov5,   title  =  {Ultralytics YOLOv5},   author  =  {Glenn Jocher},   year  =  {2020},   version  =  {7.0},   license  =  {AGPL-3.0},   url  =  {https://github.com/ultralytics/yolov5},   doi  =  {10.5281/zenodo.3908559},   orcid  =  {0000-0001-5950-6979} }` 
```

请注意，YOLOv5 模型根据[AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)和[企业](https://ultralytics.com/license)许可提供。

## 常见问题解答

### Ultralytics YOLOv5u 是什么，它与 YOLOv5 有什么不同？

Ultralytics YOLOv5u 是 YOLOv5 的高级版本，集成了无锚点、无对象性分裂头部，增强了实时目标检测任务的精度和速度折衷。与传统的 YOLOv5 不同，YOLOv5u 采用无锚点检测机制，使其在不同场景中更加灵活和适应性强。关于其特性的更多详细信息，请参考 YOLOv5 概述。

### 无锚点的 Ultralytics 头部如何提高 YOLOv5u 中的目标检测性能？

YOLOv5u 中的无锚点 Ultralytics 头部通过消除对预定义锚点框的依赖来提高目标检测性能。这导致了更灵活、适应性更强的检测机制，可以更高效地处理各种大小和形状的物体。这种增强直接促成了精度和速度之间的平衡折衷，使 YOLOv5u 适用于实时应用。在关键特性部分了解其架构的更多信息。

### 我可以将预训练的 YOLOv5u 模型用于不同的任务和模式吗？

是的，您可以将预训练的 YOLOv5u 模型用于物体检测等多种任务。这些模型支持包括推断、验证、训练和导出在内的多种模式。这种灵活性使用户能够在不同的操作需求下利用 YOLOv5u 模型的能力。详细概述，请查看支持的任务和模式部分。

### YOLOv5u 模型在不同平台上的性能指标如何比较？

YOLOv5u 模型的性能指标因平台和硬件的不同而有所不同。例如，YOLOv5nu 模型在 COCO 数据集上的 mAP 达到 34.3，在 CPU（ONNX）上的速度为 73.6 毫秒，在 A100 TensorRT 上为 1.06 毫秒。详细的不同 YOLOv5u 模型的性能指标可以在性能指标部分找到，该部分提供了跨各种设备的全面比较。

### 如何使用 Ultralytics Python API 训练 YOLOv5u 模型？

您可以通过加载预训练模型并使用您的数据集运行训练命令来训练 YOLOv5u 模型。以下是一个快速示例：

示例

```py
`from ultralytics import YOLO  # Load a COCO-pretrained YOLOv5n model model = YOLO("yolov5n.pt")  # Display model information (optional) model.info()  # Train the model on the COCO8 example dataset for 100 epochs results = model.train(data="coco8.yaml", epochs=100, imgsz=640)` 
```

```py
`# Load a COCO-pretrained YOLOv5n model and train it on the COCO8 example dataset for 100 epochs yolo  train  model=yolov5n.pt  data=coco8.yaml  epochs=100  imgsz=640` 
```

欲了解更详细的说明，请访问使用示例部分。
