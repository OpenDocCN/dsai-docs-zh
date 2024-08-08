# Ultralytics YOLOv5 全面指南

> 原文：[`docs.ultralytics.com/yolov5/`](https://docs.ultralytics.com/yolov5/)

![Ultralytics YOLOv5 v7.0 横幅](https://ultralytics.com/yolov5)

![YOLOv5 CI](https://github.com/ultralytics/yolov5/actions/workflows/ci-testing.yml) ![YOLOv5 引用](https://zenodo.org/badge/latestdoi/264818686) ![Docker 拉取](https://hub.docker.com/r/ultralytics/yolov5)

![在 Gradient 上运行](https://bit.ly/yolov5-paperspace-notebook) ![在 Colab 中打开](https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb) ![在 Kaggle 中打开](https://www.kaggle.com/ultralytics/yolov5)

欢迎来到 Ultralytics 的 [YOLOv5](https://github.com/ultralytics/yolov5)🚀 文档！YOLOv5，这一革命性的"You Only Look Once"目标检测模型的第五代，旨在实时提供高速、高精度的结果。

基于 PyTorch 构建，这个强大的深度学习框架因其多功能性、易用性和高性能而广受欢迎。我们的文档将指导您完成安装过程，解释模型的架构细节，展示各种使用案例，并提供一系列详细的教程。这些资源将帮助您充分利用 YOLOv5 在计算机视觉项目中的潜力。让我们开始吧！

## 探索与学习

这里是一系列全面的教程，将指导您了解 YOLOv5 的不同方面。

+   训练自定义数据 🚀 推荐：学习如何在您的自定义数据集上训练 YOLOv5 模型。

+   最佳训练结果的技巧 ☘️：揭示优化模型训练过程的实用技巧。

+   多 GPU 训练：学习如何利用多个 GPU 加快训练速度。

+   PyTorch Hub 🌟 新功能：学习如何通过 PyTorch Hub 加载预训练模型。

+   TFLite、ONNX、CoreML、TensorRT 导出 🚀：了解如何将您的模型导出到不同的格式。

+   测试时间增强（TTA）：探索如何使用 TTA 提高模型预测的准确性。

+   模型集成：学习将多个模型组合以提升性能的策略。

+   模型修剪/稀疏性：了解修剪和稀疏性概念，以及如何创建更高效的模型。

+   超参数演进：探索自动化超参数调整过程，以提升模型性能。

+   冻结层的迁移学习：学习如何在 YOLOv5 中通过冻结层实现迁移学习。

+   架构摘要 🌟 深入了解 YOLOv5 模型的结构细节。

+   Roboflow 用于数据集：了解如何利用 Roboflow 进行数据集管理、标注和主动学习。

+   ClearML 日志记录 🌟 学习如何集成 ClearML，在模型训练过程中实现高效的日志记录。

+   YOLOv5 与 Neural Magic：了解如何使用 Neural Magic 的 Deepsparse 对 YOLOv5 模型进行剪枝和量化。

+   Comet 日志记录 🌟 新功能：探索如何利用 Comet 实现改进的模型训练日志记录。

## 支持的环境

Ultralytics 提供一系列预装有必要依赖项如 [CUDA](https://developer.nvidia.com/cuda)，[CUDNN](https://developer.nvidia.com/cudnn)，[Python](https://www.python.org/) 和 [PyTorch](https://pytorch.org/) 的即用环境，以快速启动你的项目。

+   **免费 GPU 笔记本**: ![在 Gradient 上运行](https://bit.ly/yolov5-paperspace-notebook) ![在 Colab 上打开](https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb) ![在 Kaggle 上打开](https://www.kaggle.com/ultralytics/yolov5)

+   **Google Cloud**: GCP 快速入门指南

+   **亚马逊**: AWS 快速入门指南

+   **Azure**: AzureML 快速入门指南

+   **Docker**: Docker 快速入门指南 ![Docker 拉取](https://hub.docker.com/r/ultralytics/yolov5)

## 项目状态

![YOLOv5 CI](https://github.com/ultralytics/yolov5/actions/workflows/ci-testing.yml)

这个徽章表示所有 [YOLOv5 GitHub Actions](https://github.com/ultralytics/yolov5/actions) 持续集成（CI）测试都成功通过。这些 CI 测试严格检查 YOLOv5 在训练，验证，推理，导出和基准测试等各个关键方面的功能和性能。它们确保在 macOS，Windows 和 Ubuntu 上的一致和可靠的运行，每 24 小时和每次新提交都进行测试。

![Ultralytics GitHub](https://github.com/ultralytics) ![space](img/bea28c9c7f1a0c4c2108b8795e6e2889.png) ![Ultralytics LinkedIn](https://www.linkedin.com/company/ultralytics/) ![space](img/bea28c9c7f1a0c4c2108b8795e6e2889.png) ![Ultralytics Twitter](https://twitter.com/ultralytics) ![space](img/bea28c9c7f1a0c4c2108b8795e6e2889.png) ![Ultralytics YouTube](https://youtube.com/ultralytics?sub_confirmation=1) ![space](img/bea28c9c7f1a0c4c2108b8795e6e2889.png) ![Ultralytics TikTok](https://www.tiktok.com/@ultralytics) ![space](img/bea28c9c7f1a0c4c2108b8795e6e2889.png) ![Ultralytics BiliBili](https://ultralytics.com/bilibili) ![space](img/bea28c9c7f1a0c4c2108b8795e6e2889.png) ![Ultralytics Discord](https://ultralytics.com/discord)

## 连接和贡献

你的 YOLOv5 之旅不必是孤独的。加入我们在 [GitHub](https://github.com/ultralytics/yolov5) 上充满活力的社区，通过 [LinkedIn](https://www.linkedin.com/company/ultralytics/) 连接专业人士，分享你的成果在 [Twitter](https://twitter.com/ultralytics)，并在 [YouTube](https://youtube.com/ultralytics?sub_confirmation=1) 上找到教育资源。在 [TikTok](https://www.tiktok.com/@ultralytics) 和 [BiliBili](https://ultralytics.com/bilibili) 关注我们获取更多互动内容。

想要做出贡献吗？我们欢迎各种形式的贡献，从代码改进和 bug 报告到文档更新。查看我们的贡献指南获取更多信息。

我们很期待看到您将如何创新地使用 YOLOv5。快来吧，进行探索，彻底改变您的计算机视觉项目！🚀

## 常见问题

### Ultralytics YOLOv5 的关键特性是什么？

Ultralytics YOLOv5 因其高速和高准确性的目标检测能力而闻名。基于 PyTorch 构建，它灵活易用，适用于各种计算机视觉项目。关键特性包括实时推断、支持诸如测试时增强（TTA）和模型集成等多种训练技巧，以及与 TFLite、ONNX、CoreML 和 TensorRT 等导出格式的兼容性。要深入了解如何通过 Ultralytics YOLOv5 提升项目，请浏览我们的 TFLite、ONNX、CoreML、TensorRT 导出指南。

### 如何在我的数据集上训练自定义 YOLOv5 模型？

在您的数据集上训练自定义 YOLOv5 模型涉及几个关键步骤。首先，按照要求的格式准备数据集，并标注标签。然后，配置 YOLOv5 训练参数，并使用 `train.py` 脚本开始训练过程。要深入了解此过程，请查阅我们的训练自定义数据指南。它提供了逐步指导，确保针对您特定用例的最佳结果。

### 为什么应该选择 Ultralytics YOLOv5 而不是像 RCNN 这样的其他目标检测模型？

与基于区域的 RCNN 的多次传递相比，Ultralytics YOLOv5 在实时目标检测中因其卓越的速度和准确性而被优先选择。YOLOv5 一次性处理整个图像，因此比 RCNN 更快。此外，YOLOv5 与各种导出格式的无缝集成以及广泛的文档使其成为初学者和专业人士的优秀选择。在我们的架构摘要中了解更多关于架构优势的信息。

### 在训练期间如何优化 YOLOv5 模型性能？

优化 YOLOv5 模型性能涉及调整各种超参数和应用数据增强、迁移学习等技术。Ultralytics 提供了关于超参数进化和修剪/稀疏化的全面资源，以提高模型效率。在我们的最佳训练结果技巧指南中，您可以发现实用的提示，为训练期间实现最佳性能提供行动建议。

### YOLOv5 应用程序支持哪些运行环境？

Ultralytics YOLOv5 支持各种环境，包括 Gradient 上的免费 GPU 笔记本、Google Colab、Kaggle，以及 Google Cloud、Amazon AWS 和 Azure 等主要云平台。还提供 Docker 映像，方便设置。有关设置这些环境的详细指南，请查看我们的支持环境部分，其中包含每个平台的逐步说明。
