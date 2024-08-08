# 首页

> 原文：[`docs.ultralytics.com/`](https://docs.ultralytics.com/)

![Ultralytics YOLO 横幅](https://github.com/ultralytics/assets/releases/tag/v8.2.0) [中文](https://docs.ultralytics.com/zh/) | [한국어](https://docs.ultralytics.com/ko/) | [日本語](https://docs.ultralytics.com/ja/) | [Русский](https://docs.ultralytics.com/ru/) | [Deutsch](https://docs.ultralytics.com/de/) | [Français](https://docs.ultralytics.com/fr/) | [Español](https://docs.ultralytics.com/es/) | [Português](https://docs.ultralytics.com/pt/) | [Türkçe](https://docs.ultralytics.com/tr/) | [Tiếng Việt](https://docs.ultralytics.com/vi/) | [हिन्दी](https://docs.ultralytics.com/hi/) | [العربية](https://docs.ultralytics.com/ar/)

![Ultralytics CI](https://github.com/ultralytics/ultralytics/actions/workflows/ci.yaml) ![YOLOv8 引用](https://zenodo.org/badge/latestdoi/264818686) ![Docker 拉取](https://hub.docker.com/r/ultralytics/ultralytics) ![Discord](https://ultralytics.com/discord) ![Ultralytics 论坛](https://community.ultralytics.com)

![在 Gradient 上运行](https://console.paperspace.com/github/ultralytics/ultralytics) ![在 Colab 中打开](https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/tutorial.ipynb) ![在 Kaggle 中打开](https://www.kaggle.com/ultralytics/yolov8)

介绍 [Ultralytics](https://ultralytics.com) [YOLOv8](https://github.com/ultralytics/ultralytics)，这是备受赞誉的实时目标检测和图像分割模型的最新版本。YOLOv8 基于深度学习和计算机视觉的前沿进展，提供无与伦比的速度和准确性。其简化的设计使其适用于各种应用，并且可以轻松适应不同的硬件平台，从边缘设备到云 API。

探索 YOLOv8 文档，这是一个全面的资源，旨在帮助您理解和利用其功能和能力。无论您是经验丰富的机器学习从业者还是新手，本中心旨在最大化 YOLOv8 在您项目中的潜力。

![Ultralytics GitHub](https://github.com/ultralytics) ![space](img/bea28c9c7f1a0c4c2108b8795e6e2889.png) ![Ultralytics LinkedIn](https://www.linkedin.com/company/ultralytics/) ![space](img/bea28c9c7f1a0c4c2108b8795e6e2889.png) ![Ultralytics Twitter](https://twitter.com/ultralytics) ![space](img/bea28c9c7f1a0c4c2108b8795e6e2889.png) ![Ultralytics YouTube](https://youtube.com/ultralytics?sub_confirmation=1) ![space](img/bea28c9c7f1a0c4c2108b8795e6e2889.png) ![Ultralytics TikTok](https://www.tiktok.com/@ultralytics) ![space](img/bea28c9c7f1a0c4c2108b8795e6e2889.png) ![Ultralytics BiliBili](https://ultralytics.com/bilibili) ![space](img/bea28c9c7f1a0c4c2108b8795e6e2889.png) ![Ultralytics Discord](https://ultralytics.com/discord)

## 从哪里开始

+   使用 pip **安装** `ultralytics`，几分钟内即可开始使用   开始使用

+   使用 YOLOv8 **预测** 新的图像和视频   在图像上预测

+   **Train** 在您自己的定制数据集上训练新的 YOLOv8 模型   训练一个模型

+   **Tasks** YOLOv8 任务如分段、分类、姿势和跟踪   探索任务

+   **NEW 🚀 探索** 带有高级语义和 SQL 搜索功能的数据集   探索数据集

[`www.youtube.com/embed/LNwODJXcvt4?si=7n1UvGRLSd9p5wKs`](https://www.youtube.com/embed/LNwODJXcvt4?si=7n1UvGRLSd9p5wKs)

**Watch:** 如何在 [Google Colab](https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/tutorial.ipynb) 上训练 YOLOv8 模型的视频。

## YOLO：简史

[YOLO](https://arxiv.org/abs/1506.02640)（You Only Look Once）是一种流行的目标检测和图像分割模型，由华盛顿大学的 Joseph Redmon 和 Ali Farhadi 开发。YOLO 由于其高速和高准确性，在 2015 年发布后迅速受到欢迎。

+   [YOLOv2](https://arxiv.org/abs/1612.08242)，发布于 2016 年，通过引入批量归一化、锚框和维度聚类，改进了原始模型。

+   [YOLOv3](https://pjreddie.com/media/files/papers/YOLOv3.pdf)，于 2018 年发布，通过更高效的骨干网络、多个锚点和空间金字塔池化进一步提升了模型的性能。

+   [YOLOv4](https://arxiv.org/abs/2004.10934) 于 2020 年发布，引入了 Mosaic 数据增强、新的无锚检测头部和新的损失函数等创新。

+   [YOLOv5](https://github.com/ultralytics/yolov5) 进一步提升了模型的性能，并增加了超参数优化、集成实验追踪和自动导出到流行的导出格式等新功能。

+   [YOLOv6](https://github.com/meituan/YOLOv6) 由 [美团](https://about.meituan.com/) 在 2022 年开源，并在该公司的许多自动配送机器人中使用。

+   [YOLOv7](https://github.com/WongKinYiu/yolov7) 在 COCO 关键点数据集上增加了姿势估计等附加任务。

+   [YOLOv8](https://github.com/ultralytics/ultralytics) 是由 Ultralytics 推出的最新版本 YOLO。作为先进的模型，YOLOv8 在之前版本的成功基础上引入了新功能和改进，提升了性能、灵活性和效率。YOLOv8 支持包括检测、分割、姿势估计、跟踪和分类在内的全方位视觉 AI 任务。这种多功能性使用户可以在各种应用和领域中充分利用 YOLOv8 的能力。

+   YOLOv9 引入了像可编程梯度信息（PGI）和广义高效层聚合网络（GELAN）等创新方法。

+   YOLOv10 是由 [清华大学](https://www.tsinghua.edu.cn/en/) 的研究人员使用 [Ultralytics](https://ultralytics.com/) 的 [Python package](https://pypi.org/project/ultralytics/) 创建的。这个版本通过引入端到端头部消除了非最大抑制（NMS）要求，提供了实时目标检测的进展。

## YOLO 许可证：Ultralytics YOLO 如何许可？

Ultralytics 提供两种许可选项以适应不同的使用场景：

+   **AGPL-3.0 许可证**：这个[OSI 批准的](https://opensource.org/licenses/)开源许可证非常适合学生和爱好者，促进开放协作和知识共享。详见[LICENSE](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)文件获取更多详情。

+   **企业许可证**：设计用于商业使用，此许可证允许无缝集成 Ultralytics 软件和 AI 模型到商业产品和服务中，绕过 AGPL-3.0 许可证的开源要求。如果您的情况涉及将我们的解决方案嵌入到商业产品中，请通过[Ultralytics Licensing](https://ultralytics.com/license)联系。

我们的许可策略旨在确保对我们开源项目的任何改进都能回馈给社区。我们深知开源原则的重要性 ❤️，我们的使命是确保我们的贡献可以以有益于所有人的方式被利用和扩展。

## 常见问题解答

### 什么是 Ultralytics YOLO 以及它如何改善目标检测？

Ultralytics YOLO 是备受赞誉的 YOLO（You Only Look Once）系列的最新进展，用于实时目标检测和图像分割。它通过引入新功能和改进来建立在之前版本的基础上，提升了性能、灵活性和效率。YOLOv8 支持多种视觉 AI 任务，如检测、分割、姿态估计、跟踪和分类。其先进的架构确保了超高的速度和精度，适用于各种应用场景，包括边缘设备和云 API。

### 如何开始使用 YOLO 进行安装和设置？

快速开始 YOLO 非常简单直接。您可以使用 pip 安装 Ultralytics 包，并在几分钟内运行起来。以下是一个基本的安装命令：

```py
pip  install  ultralytics 
```

对于全面的逐步指南，请访问我们的快速入门指南。这个资源将帮助您完成安装指导、初始设置和运行您的第一个模型。

### 如何在我的数据集上训练自定义 YOLO 模型？

在您的数据集上训练自定义 YOLO 模型涉及几个详细步骤：

1.  准备您的标注数据集。

1.  在一个 YAML 文件中配置训练参数。

1.  使用`yolo train`命令开始训练。

这里是一个示例命令：

```py
yolo  train  model=yolov8n.pt  data=coco128.yaml  epochs=100  imgsz=640 
```

欲了解详细步骤，请查看我们的模型训练指南，其中包括示例和优化训练过程的技巧。

### Ultralytics YOLO 有哪些许可选项？

Ultralytics 为 YOLO 提供了两种许可选项：

+   **AGPL-3.0 许可证**：这个开源许可证非常适合教育和非商业用途，促进开放协作。

+   **企业许可证**：这个许可证专为商业应用设计，允许无缝集成 Ultralytics 软件到商业产品中，无需遵守 AGPL-3.0 许可证的限制。

欲了解更多详情，请访问我们的[许可](https://ultralytics.com/license)页面。

### Ultralytics YOLO 如何用于实时目标跟踪？

Ultralytics YOLO 支持高效且可定制的多目标跟踪。要利用跟踪功能，可以使用`yolo track`命令，如下所示：

```py
yolo  track  model=yolov8n.pt  source=video.mp4 
```

有关设置和运行目标跟踪的详细指南，请查看我们的跟踪模式文档，其中解释了配置和在实时场景中的实际应用。
