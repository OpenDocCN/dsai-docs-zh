# Ultralytics 支持的模型

> 原文：[`docs.ultralytics.com/models/`](https://docs.ultralytics.com/models/)

欢迎访问 Ultralytics 的模型文档！我们支持多种模型，每个模型都专为特定任务如对象检测、实例分割、图像分类、姿态估计和多对象跟踪而设计。如果您有兴趣将您的模型架构贡献给 Ultralytics，请查阅我们的贡献指南。

## 特色模型

这里列出了一些主要支持的模型：

1.  **YOLOv3**: YOLO 模型系列的第三个版本，最初由 Joseph Redmon 开发，以其高效的实时对象检测能力而闻名。

1.  **YOLOv4**: 由 Alexey Bochkovskiy 在 2020 年发布的 darknet 原生更新版 YOLOv3。

1.  **YOLOv5**: Ultralytics 改进的 YOLO 架构版本，提供比之前版本更好的性能和速度权衡。

1.  **YOLOv6**: 2022 年由[美团](https://about.meituan.com/)发布，并在该公司许多自主送餐机器人中使用。

1.  **YOLOv7**: 2022 年发布的更新版 YOLO 模型，由 YOLOv4 的作者发布。

1.  **YOLOv8 NEW 🚀**: YOLO 系列的最新版本，具有增强的能力，如实例分割、姿态/关键点估计和分类。

1.  **YOLOv9**: 在 Ultralytics YOLOv5 代码库上训练的实验性模型，实现可编程梯度信息（PGI）。

1.  **YOLOv10**: 清华大学发布，采用无 NMS 训练和效率-精度驱动架构，提供最先进的性能和延迟。

1.  **Segment Anything Model (SAM)**: Meta 原始的 Segment Anything 模型（SAM）。

1.  **Segment Anything Model 2 (SAM2)**: Meta 的下一代视频和图像 Segment Anything 模型（SAM）。

1.  **Mobile Segment Anything Model (MobileSAM)**: MobileSAM 是由庆熙大学推出的面向移动应用的模型。

1.  **Fast Segment Anything Model (FastSAM)**: 中国科学院自动化研究所的 Image & Video Analysis Group 推出的 FastSAM。

1.  **YOLO-NAS**: YOLO 神经架构搜索（NAS）模型。

1.  **Realtime Detection Transformers (RT-DETR)**: 百度的 PaddlePaddle 实时检测变换器（RT-DETR）模型。

1.  **YOLO-World**: 腾讯 AI 实验室发布的实时开放词汇对象检测模型。

[`www.youtube.com/embed/MWq1UxqTClU?si=nHAW-lYDzrz68jR0`](https://www.youtube.com/embed/MWq1UxqTClU?si=nHAW-lYDzrz68jR0)

**Watch:** 仅需几行代码即可运行 Ultralytics 的 YOLO 模型。

## 入门：使用示例

该示例提供了简单的 YOLO 训练和推断示例。有关这些和其他模式的完整文档，请参阅 Predict、Train、Val 和 Export 文档页面。

注意下面的例子是关于 YOLOv8 Detect 模型进行对象检测。有关其他支持的任务，请参阅 Segment、Classify 和 Pose 文档。

例子

可以将预训练的 PyTorch `*.pt`模型以及配置`*.yaml`文件传递给`YOLO()`、`SAM()`、`NAS()`和`RTDETR()`类，在 Python 中创建一个模型实例：

```py
`from ultralytics import YOLO  # Load a COCO-pretrained YOLOv8n model model = YOLO("yolov8n.pt")  # Display model information (optional) model.info()  # Train the model on the COCO8 example dataset for 100 epochs results = model.train(data="coco8.yaml", epochs=100, imgsz=640)  # Run inference with the YOLOv8n model on the 'bus.jpg' image results = model("path/to/bus.jpg")` 
```

可以使用 CLI 命令直接运行模型：

```py
`# Load a COCO-pretrained YOLOv8n model and train it on the COCO8 example dataset for 100 epochs yolo  train  model=yolov8n.pt  data=coco8.yaml  epochs=100  imgsz=640  # Load a COCO-pretrained YOLOv8n model and run inference on the 'bus.jpg' image yolo  predict  model=yolov8n.pt  source=path/to/bus.jpg` 
```

## 贡献新模型

感兴趣将您的模型贡献给 Ultralytics 吗？太棒了！我们始终欢迎扩展我们的模型组合。

1.  **分叉存储库**：首先分叉[Ultralytics GitHub 存储库](https://github.com/ultralytics/ultralytics)。

1.  **克隆您的分支**：将您的分支克隆到本地机器，并创建一个新分支进行操作。

1.  **实现您的模型**：按照我们提供的贡献指南中的编码标准和准则添加您的模型。

1.  **彻底测试**：务必对您的模型进行严格测试，无论是独立进行还是作为管道的一部分。

1.  **创建拉取请求**：一旦您满意您的模型，请创建一个拉取请求到主存储库进行审查。

1.  **代码审查和合并**：经过审查，如果您的模型符合我们的标准，将合并到主存储库中。

详细步骤，请参阅我们的贡献指南。

## 常见问题

### 使用 Ultralytics YOLOv8 进行目标检测的关键优势是什么？

Ultralytics YOLOv8 提供了增强功能，如实时目标检测、实例分割、姿态估计和分类。其优化的架构确保高速性能，不会牺牲准确性，使其非常适合各种应用。YOLOv8 还包括与流行数据集和模型的内置兼容性，详细信息请参阅 YOLOv8 文档页面。

### 如何在自定义数据上训练 YOLOv8 模型？

使用 Ultralytics 库可以轻松地在自定义数据上训练 YOLOv8 模型。以下是一个快速示例：

示例

```py
`from ultralytics import YOLO  # Load a YOLOv8n model model = YOLO("yolov8n.pt")  # Train the model on custom dataset results = model.train(data="custom_data.yaml", epochs=100, imgsz=640)` 
```

```py
`yolo  train  model=yolov8n.pt  data='custom_data.yaml'  epochs=100  imgsz=640` 
```

获取更详细的指导，请访问 Train 文档页面。

### Ultralytics 支持哪些 YOLO 版本？

Ultralytics 支持从 YOLOv3 到 YOLOv10 等全面的 YOLO（You Only Look Once）版本，以及 NAS、SAM 和 RT-DETR 等模型。每个版本都针对检测、分割和分类等各种任务进行了优化。有关每个模型的详细信息，请参阅 Ultralytics 支持的模型文档。

### 我为什么应该使用 Ultralytics HUB 进行机器学习项目？

Ultralytics HUB 为训练、部署和管理 YOLO 模型提供了一个无代码、端到端的平台。它简化了复杂的工作流程，使用户能够专注于模型性能和应用。HUB 还提供云训练能力、全面的数据集管理和用户友好的界面。更多信息请访问 Ultralytics HUB 文档页面。

### YOLOv8 可以执行哪些类型的任务，以及与其他 YOLO 版本相比有何优势？

YOLOv8 是一个多功能模型，能够执行包括目标检测、实例分割、分类和姿态估计等任务。与 YOLOv3 和 YOLOv4 等早期版本相比，YOLOv8 在速度和准确性方面都有显著提升，这归功于其优化的架构。有关更详细的比较，请参考 YOLOv8 文档和任务页面，了解特定任务的更多细节。
