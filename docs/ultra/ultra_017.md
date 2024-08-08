# YOLOv3、YOLOv3-Ultralytics 和 YOLOv3u

> 原文：[`docs.ultralytics.com/models/yolov3/`](https://docs.ultralytics.com/models/yolov3/)

## 概述

本文介绍了三个密切相关的目标检测模型的概述，分别是[YOLOv3](https://pjreddie.com/darknet/yolo/)、[YOLOv3-Ultralytics](https://github.com/ultralytics/yolov3)和[YOLOv3u](https://github.com/ultralytics/ultralytics)。

1.  **YOLOv3:** 这是 You Only Look Once (YOLO)目标检测算法的第三个版本。由 Joseph Redmon 最初开发，YOLOv3 通过引入多尺度预测和三种不同尺寸的检测核心来改进其前身。

1.  **YOLOv3-Ultralytics:** 这是 Ultralytics 对 YOLOv3 模型的实现。它复制了原始的 YOLOv3 架构，并提供了额外的功能，例如支持更多预训练模型和更简单的定制选项。

1.  **YOLOv3u:** 这是 YOLOv3-Ultralytics 的更新版本，采用了 YOLOv8 模型中使用的无锚点、无物体性分离头。YOLOv3u 保留了 YOLOv3 相同的主干和颈部架构，但使用了 YOLOv8 的更新检测头。

![Ultralytics YOLOv3](img/f50df2eb05ecc42b4900c27d1abb4812.png)

## 主要特点

+   **YOLOv3:** 引入了三种不同尺度的检测方式，利用了三种不同大小的检测核心：13x13、26x26 和 52x52。这显著提高了对不同尺寸物体的检测精度。此外，YOLOv3 还增加了诸如每个边界框的多标签预测和更好的特征提取网络等功能。

+   **YOLOv3-Ultralytics:** Ultralytics 对 YOLOv3 的实现提供了与原始模型相同的性能，但增加了对更多预训练模型、额外的训练方法和更简单的定制选项的支持。这使得它在实际应用中更加多功能和用户友好。

+   **YOLOv3u:** 这个更新的模型采用了 YOLOv8 模型中使用的无锚点、无物体性分离头。通过消除预定义的锚框和物体性评分的需求，这种检测头设计可以提高模型对各种大小和形状物体的检测能力。这使得 YOLOv3u 在目标检测任务中更加稳健和准确。

## 支持的任务和模式

YOLOv3 系列，包括 YOLOv3、YOLOv3-Ultralytics 和 YOLOv3u，专为目标检测任务而设计。这些模型在各种实际场景中以其在精度和速度之间的平衡而闻名。每个变种都提供独特的功能和优化，适用于一系列应用场景。

所有三个模型支持全面的模式集合，确保在模型部署和开发的各个阶段具有多样性。这些模式包括推断、验证、训练和导出，为用户提供了完整的工具包，用于有效的目标检测。

| 模型类型 | 支持的任务 | 推断 | 验证 | 训练 | 导出 |
| --- | --- | --- | --- | --- | --- |
| YOLOv3 | 目标检测 | ✅ | ✅ | ✅ | ✅ |
| YOLOv3-Ultralytics | 目标检测 | ✅ | ✅ | ✅ | ✅ |
| YOLOv3u | 目标检测 | ✅ | ✅ | ✅ | ✅ |

此表提供了每个 YOLOv3 变体的功能一览，突显了它们在各种任务和操作模式中在目标检测工作流中的多用途性和适用性。

## 使用示例

此示例提供了简单的 YOLOv3 训练和推断示例。有关这些及其他模式的完整文档，请参阅预测、训练、验证和导出文档页面。

示例

可以将 PyTorch 预训练的 `*.pt` 模型及配置 `*.yaml` 文件传递给 `YOLO()` 类，在 Python 中创建模型实例：

```py
from ultralytics import YOLO

# Load a COCO-pretrained YOLOv3n model
model = YOLO("yolov3n.pt")

# Display model information (optional)
model.info()

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

# Run inference with the YOLOv3n model on the 'bus.jpg' image
results = model("path/to/bus.jpg") 
```

可通过 CLI 命令直接运行模型：

```py
# Load a COCO-pretrained YOLOv3n model and train it on the COCO8 example dataset for 100 epochs
yolo  train  model=yolov3n.pt  data=coco8.yaml  epochs=100  imgsz=640

# Load a COCO-pretrained YOLOv3n model and run inference on the 'bus.jpg' image
yolo  predict  model=yolov3n.pt  source=path/to/bus.jpg 
```

## 引用和致谢

如果您在研究中使用 YOLOv3，请引用原始 YOLO 论文和 Ultralytics YOLOv3 仓库：

```py
@article{redmon2018yolov3,
  title={YOLOv3: An Incremental Improvement},
  author={Redmon, Joseph and Farhadi, Ali},
  journal={arXiv preprint arXiv:1804.02767},
  year={2018}
} 
```

感谢 Joseph Redmon 和 Ali Farhadi 开发原始 YOLOv3。

## 常见问题解答

### YOLOv3、YOLOv3-Ultralytics 和 YOLOv3u 之间有何区别？

YOLOv3 是由 Joseph Redmon 开发的 YOLO（You Only Look Once）目标检测算法的第三个版本，以其在准确性和速度上的平衡而闻名，利用三种不同的尺度（13x13、26x26 和 52x52）进行检测。YOLOv3-Ultralytics 是 Ultralytics 对 YOLOv3 的适配版本，增加了对更多预训练模型的支持，并简化了模型定制过程。YOLOv3u 是 YOLOv3-Ultralytics 的升级变体，集成了来自 YOLOv8 的无锚点、无对象性分割头部，提升了对各种目标尺寸的检测鲁棒性和准确性。关于这些变体的更多细节，请参阅 [YOLOv3 系列](https://github.com/ultralytics/yolov3)。

### 如何使用 Ultralytics 训练 YOLOv3 模型？

使用 Ultralytics 训练 YOLOv3 模型非常简单。您可以使用 Python 或 CLI 来训练模型：

示例

```py
from ultralytics import YOLO

# Load a COCO-pretrained YOLOv3n model
model = YOLO("yolov3n.pt")

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data="coco8.yaml", epochs=100, imgsz=640) 
```

```py
# Load a COCO-pretrained YOLOv3n model and train it on the COCO8 example dataset for 100 epochs
yolo  train  model=yolov3n.pt  data=coco8.yaml  epochs=100  imgsz=640 
```

若要了解更全面的训练选项和指南，请访问我们的训练模式文档。

### YOLOv3u 在目标检测任务中如何提升准确性？

YOLOv3u 改进了 YOLOv3 和 YOLOv3-Ultralytics，引入了 YOLOv8 模型中使用的无锚点、无对象性分割头部。此升级消除了预定义锚点框和对象性分数的需求，增强了检测不同大小和形状对象的精确性。这使得 YOLOv3u 在复杂和多样化的目标检测任务中更为优选。有关更多信息，请参阅 Why YOLOv3u 部分。

### 如何使用 YOLOv3 模型进行推断？

您可以通过 Python 脚本或 CLI 命令执行 YOLOv3 模型推断：

示例

```py
from ultralytics import YOLO

# Load a COCO-pretrained YOLOv3n model
model = YOLO("yolov3n.pt")

# Run inference with the YOLOv3n model on the 'bus.jpg' image
results = model("path/to/bus.jpg") 
```

```py
# Load a COCO-pretrained YOLOv3n model and run inference on the 'bus.jpg' image
yolo  predict  model=yolov3n.pt  source=path/to/bus.jpg 
```

若要了解有关运行 YOLO 模型的详细信息，请参阅推断模式文档。

### YOLOv3 及其变体支持哪些任务？

YOLOv3、YOLOv3-Ultralytics 和 YOLOv3u 主要支持目标检测任务。这些模型可用于模型部署和开发的各个阶段，例如推断、验证、训练和导出。有关支持的全面任务集合和更深入的详细信息，请访问我们的目标检测任务文档。

### 我在哪里可以找到引用 YOLOv3 在我的研究中所需的资源？

如果您在研究中使用了 YOLOv3，请引用原始的 YOLO 论文和 Ultralytics YOLOv3 代码库。示例 BibTeX 引用：

```py
@article{redmon2018yolov3,
  title={YOLOv3: An Incremental Improvement},
  author={Redmon, Joseph and Farhadi, Ali},
  journal={arXiv preprint arXiv:1804.02767},
  year={2018}
} 
```

有关更多引用详细信息，请参阅引文和致谢部分。
