# Ultralytics Explorer

> 原文：[`docs.ultralytics.com/datasets/explorer/`](https://docs.ultralytics.com/datasets/explorer/)

![Ultralytics Explorer 截图 1](img/16813c5c76de99fa62271e29dc570958.png)

![在 Colab 中打开](https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/docs/en/datasets/explorer/explorer.ipynb) Ultralytics Explorer 是一个用于通过语义搜索、SQL 查询、向量相似性搜索甚至使用自然语言探索 CV 数据集的工具。它还是一个 Python API，用于访问相同的功能。

[`www.youtube.com/embed/3VryynorQeo`](https://www.youtube.com/embed/3VryynorQeo)

**Watch:** Ultralytics Explorer API | 语义搜索、SQL 查询和 Ask AI 功能

### 可选依赖项的安装

Explorer 依赖于外部库来实现其某些功能。这些功能在使用时会自动安装。要手动安装这些依赖项，请使用以下命令：

```py
pip  install  ultralytics[explorer] 
```

提示

Explorer 支持嵌入式/语义搜索和 SQL 查询，并由[LanceDB](https://lancedb.com/)无服务器向量数据库驱动。与传统的内存数据库不同，它在不牺牲性能的情况下持久化存储在磁盘上，因此可以在本地扩展到像 COCO 这样的大型数据集而不会耗尽内存。

### Explorer API

这是一个用于探索数据集的 Python API。它还驱动 GUI Explorer。您可以使用它创建自己的探索性笔记本或脚本，以深入了解您的数据集。

在这里了解更多关于 Explorer API 的信息。

## GUI Explorer 使用

GUI 演示在您的浏览器中运行，允许您为数据集创建嵌入并搜索相似图像，运行 SQL 查询和执行语义搜索。可以使用以下命令运行它：

```py
yolo  explorer 
```

注意

Ask AI 功能使用 OpenAI，因此当您首次运行 GUI 时，将提示您设置 OpenAI 的 API 密钥。您可以像这样设置它 - `yolo settings openai_api_key="..."`

![Ultralytics Explorer OpenAI 集成](img/9fb0ad10c094a36f84bf5fe39589baf7.png)

## 常见问题解答

### 什么是 Ultralytics Explorer 及其如何帮助 CV 数据集？

Ultralytics Explorer 是一个功能强大的工具，旨在通过语义搜索、SQL 查询、向量相似性搜索甚至自然语言，探索计算机视觉（CV）数据集。这个多功能工具提供了 GUI 和 Python API，允许用户与其数据集无缝交互。通过利用像 LanceDB 这样的技术，Ultralytics Explorer 确保高效、可扩展地访问大型数据集，而不会过度使用内存。无论是进行详细的数据集分析还是探索数据模式，Ultralytics Explorer 都简化了整个流程。

了解更多关于 Explorer API 的信息。

### 如何安装 Ultralytics Explorer 的依赖项？

要手动安装 Ultralytics Explorer 所需的可选依赖项，可以使用以下`pip`命令：

```py
pip  install  ultralytics[explorer] 
```

这些依赖项对语义搜索和 SQL 查询的完整功能至关重要。通过包含由 [LanceDB](https://lancedb.com/) 提供支持的库，安装确保数据库操作保持高效且可扩展，即使是像 COCO 这样的大型数据集。

### 如何使用 Ultralytics Explorer 的 GUI 版本？

使用 Ultralytics Explorer 的 GUI 版本非常简单。在安装必要的依赖项后，您可以使用以下命令启动 GUI：

```py
yolo  explorer 
```

GUI 提供了一个用户友好的界面，用于创建数据集嵌入、搜索相似图像、运行 SQL 查询以及进行语义搜索。此外，与 OpenAI 的 Ask AI 功能集成，允许您使用自然语言查询数据集，增强了灵活性和易用性。

如需存储和可扩展性信息，请查看我们的安装指南。

### Ultralytics Explorer 中的 Ask AI 功能是什么？

Ultralytics Explorer 中的 Ask AI 功能允许用户使用自然语言查询与其数据集进行交互。由 OpenAI 提供支持，此功能使您能够提出复杂问题并获得深刻的答案，无需编写 SQL 查询或类似命令。要使用此功能，您需要在首次运行 GUI 时设置您的 OpenAI API 密钥：

```py
yolo  settings  openai_api_key="YOUR_API_KEY" 
```

关于此功能及其集成方式的更多信息，请参阅我们的 GUI Explorer 使用部分。

### 我可以在 Google Colab 上运行 Ultralytics Explorer 吗？

是的，您可以在 Google Colab 中运行 Ultralytics Explorer，为数据集探索提供便捷且强大的环境。您可以通过打开提供的 Colab 笔记本来开始，该笔记本已预先配置了所有必要的设置：

![在 Colab 中打开](https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/docs/en/datasets/explorer/explorer.ipynb)

此设置使您能够充分探索您的数据集，利用 Google 的云资源。在我们的 Google Colab 指南中了解更多。
