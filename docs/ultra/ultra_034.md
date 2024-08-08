# Ultralytics Explorer API

> 原文：[`docs.ultralytics.com/datasets/explorer/api/`](https://docs.ultralytics.com/datasets/explorer/api/)

## 简介

![在 Colab 中打开](https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/docs/en/datasets/explorer/explorer.ipynb) Explorer API 是一个用于探索数据集的 Python API。它支持使用 SQL 查询、向量相似性搜索和语义搜索对数据集进行过滤和搜索。

[`www.youtube.com/embed/3VryynorQeo?start=279`](https://www.youtube.com/embed/3VryynorQeo?start=279)

**观看：** Ultralytics Explorer API 概述

## 安装

Explorer 依赖于一些外部库来完成其功能。在使用时会自动安装这些依赖项。要手动安装这些依赖项，请使用以下命令：

```py
`pip  install  ultralytics[explorer]` 
```

## 使用方法

```py
`from ultralytics import Explorer  # Create an Explorer object explorer = Explorer(data="coco128.yaml", model="yolov8n.pt")  # Create embeddings for your dataset explorer.create_embeddings_table()  # Search for similar images to a given image/images dataframe = explorer.get_similar(img="path/to/image.jpg")  # Or search for similar images to a given index/indices dataframe = explorer.get_similar(idx=0)` 
```

注意

对于给定的数据集和模型对，嵌入表仅创建一次并重复使用。这些在幕后使用 [LanceDB](https://lancedb.github.io/lancedb/)，它在磁盘上扩展，因此您可以在不耗尽内存的情况下为诸如 COCO 等大型数据集创建和重用嵌入。

如果您想强制更新嵌入表，可以将 `force=True` 传递给 `create_embeddings_table` 方法。

您可以直接访问 LanceDB 表对象进行高级分析。在“使用嵌入表”部分了解更多信息

## 1\. 相似性搜索

相似性搜索是一种寻找与给定图像相似的图像的技术。它基于相似图像将具有相似嵌入的想法。一旦建立了嵌入表，您可以以以下任何方式之一运行语义搜索：

+   对于数据集中的给定索引或索引列表：`exp.get_similar(idx=[1,10], limit=10)`

+   对于数据集中不存在的任何图像或图像列表：`exp.get_similar(img=["path/to/img1", "path/to/img2"], limit=10)`

如果有多个输入，则使用它们的嵌入的聚合。

您将获得一个带有 `limit` 数量最相似数据点的 pandas dataframe，以及它们在嵌入空间中的距离。您可以使用此数据集进行进一步的筛选

语义搜索

```py
`from ultralytics import Explorer  # create an Explorer object exp = Explorer(data="coco128.yaml", model="yolov8n.pt") exp.create_embeddings_table()  similar = exp.get_similar(img="https://ultralytics.com/images/bus.jpg", limit=10) print(similar.head())  # Search using multiple indices similar = exp.get_similar(     img=["https://ultralytics.com/images/bus.jpg", "https://ultralytics.com/images/bus.jpg"],     limit=10, ) print(similar.head())` 
```

```py
`from ultralytics import Explorer  # create an Explorer object exp = Explorer(data="coco128.yaml", model="yolov8n.pt") exp.create_embeddings_table()  similar = exp.get_similar(idx=1, limit=10) print(similar.head())  # Search using multiple indices similar = exp.get_similar(idx=[1, 10], limit=10) print(similar.head())` 
```

### 绘制相似图像

您还可以使用 `plot_similar` 方法绘制相似图像。此方法接受与 `get_similar` 相同的参数，并在网格中绘制相似图像。

绘制相似图像

```py
`from ultralytics import Explorer  # create an Explorer object exp = Explorer(data="coco128.yaml", model="yolov8n.pt") exp.create_embeddings_table()  plt = exp.plot_similar(img="https://ultralytics.com/images/bus.jpg", limit=10) plt.show()` 
```

```py
`from ultralytics import Explorer  # create an Explorer object exp = Explorer(data="coco128.yaml", model="yolov8n.pt") exp.create_embeddings_table()  plt = exp.plot_similar(idx=1, limit=10) plt.show()` 
```

## 2\. 询问 AI（自然语言查询）

这使您可以使用自然语言编写您想要过滤数据集的方式。您不必精通编写 SQL 查询。我们的 AI 动力查询生成器将在幕后自动执行此操作。例如，您可以说：“显示 100 张只有一个人和 2 只狗的图像。也可能有其他物体”，它会在内部生成查询并显示结果。注意：这使用 LLMs 在幕后工作，因此结果是概率性的，有时可能出错。

询问 AI

```py
`from ultralytics import Explorer from ultralytics.data.explorer import plot_query_result  # create an Explorer object exp = Explorer(data="coco128.yaml", model="yolov8n.pt") exp.create_embeddings_table()  df = exp.ask_ai("show me 100 images with exactly one person and 2 dogs. There can be other objects too") print(df.head())  # plot the results plt = plot_query_result(df) plt.show()` 
```

## 3\. SQL 查询

您可以使用`sql_query`方法在数据集上运行 SQL 查询。此方法接受 SQL 查询作为输入，并返回带有结果的 pandas 数据框。

SQL 查询

```py
`from ultralytics import Explorer  # create an Explorer object exp = Explorer(data="coco128.yaml", model="yolov8n.pt") exp.create_embeddings_table()  df = exp.sql_query("WHERE labels LIKE '%person%' AND labels LIKE '%dog%'") print(df.head())` 
```

### 绘制 SQL 查询结果

您还可以使用`plot_sql_query`方法绘制 SQL 查询结果。此方法接受与`sql_query`相同的参数，并在网格中绘制结果。

绘制 SQL 查询结果

```py
`from ultralytics import Explorer  # create an Explorer object exp = Explorer(data="coco128.yaml", model="yolov8n.pt") exp.create_embeddings_table()  # plot the SQL Query exp.plot_sql_query("WHERE labels LIKE '%person%' AND labels LIKE '%dog%' LIMIT 10")` 
```

## 4\. 使用嵌入表

您还可以直接使用嵌入表。一旦创建了嵌入表，您可以使用`Explorer.table`访问它。

Explorer 在内部使用[LanceDB](https://lancedb.github.io/lancedb/)表。您可以直接使用`Explorer.table`对象访问此表，并运行原始查询、推送预处理和后处理过滤等。

```py
`from ultralytics import Explorer  exp = Explorer() exp.create_embeddings_table() table = exp.table` 
```

以下是您可以使用表格执行的一些示例：

### 获取原始嵌入

示例

```py
`from ultralytics import Explorer  exp = Explorer() exp.create_embeddings_table() table = exp.table  embeddings = table.to_pandas()["vector"] print(embeddings)` 
```

### 使用预处理和后处理过滤进行高级查询

示例

```py
`from ultralytics import Explorer  exp = Explorer(model="yolov8n.pt") exp.create_embeddings_table() table = exp.table  # Dummy embedding embedding = [i for i in range(256)] rs = table.search(embedding).metric("cosine").where("").limit(10)` 
```

### 创建向量索引

在使用大型数据集时，您还可以创建一个专用的向量索引以加快查询速度。这可以通过在 LanceDB 表上使用`create_index`方法来完成。

```py
`table.create_index(num_partitions=..., num_sub_vectors=...)` 
```

查找更多有关可用类型向量索引和参数的详细信息[这里](https://lancedb.github.io/lancedb/ann_indexes/#types-of-index)。将来，我们将支持直接从 Explorer API 创建向量索引。

## 5\. 嵌入应用

您可以使用嵌入表执行各种探索性分析。以下是一些示例：

### 相似性指数

Explorer 提供了`similarity_index`操作：

+   它试图估计每个数据点与数据集中其余数据点的相似程度。

+   它通过计算与生成的嵌入空间中当前图像距离小于`max_dist`的图像数量来实现，每次考虑`top_k`相似图像。

返回一个带有以下列的 pandas 数据框：

+   `idx`: 数据集中图像的索引

+   `im_file`: 图像文件的路径

+   `count`: 数据集中与当前图像距离小于`max_dist`的图像数量

+   `sim_im_files`: 列表，包含`count`个相似图像的路径

小贴士

对于给定的数据集、模型、`max_dist`和`top_k`，生成一次相似性指数后将重复使用。如果您的数据集已更改或者您只需重新生成相似性指数，可以传递`force=True`。

相似性指数

```py
`from ultralytics import Explorer  exp = Explorer() exp.create_embeddings_table()  sim_idx = exp.similarity_index()` 
```

您可以使用相似性指数构建自定义条件来过滤数据集。例如，您可以使用以下代码过滤掉与数据集中任何其他图像不相似的图像：

```py
`import numpy as np  sim_count = np.array(sim_idx["count"]) sim_idx["im_file"][sim_count > 30]` 
```

### 可视化嵌入空间

您还可以使用您选择的绘图工具来可视化嵌入空间。例如，这是使用 matplotlib 的一个简单示例：

```py
`import matplotlib.pyplot as plt from sklearn.decomposition import PCA  # Reduce dimensions using PCA to 3 components for visualization in 3D pca = PCA(n_components=3) reduced_data = pca.fit_transform(embeddings)  # Create a 3D scatter plot using Matplotlib Axes3D fig = plt.figure(figsize=(8, 6)) ax = fig.add_subplot(111, projection="3d")  # Scatter plot ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2], alpha=0.5) ax.set_title("3D Scatter Plot of Reduced 256-Dimensional Data (PCA)") ax.set_xlabel("Component 1") ax.set_ylabel("Component 2") ax.set_zlabel("Component 3")  plt.show()` 
```

使用 Explorer API 开始创建您自己的 CV 数据集探索报告。为了灵感，请查看

## 使用 Ultralytics Explorer 构建的应用程序

使用基于 Explorer API 的 GUI 演示

## 即将推出

+   [ ] 合并数据集中特定的标签。例如 - 从 COCO 导入所有`person`标签和从 Cityscapes 导入所有`car`标签

+   [ ] 删除具有比给定阈值更高相似性索引的图像

+   [ ] 在合并/移除条目后自动持久化新数据集

+   [ ] 高级数据集可视化

## 常见问题解答

### Ultralytics Explorer API 用于什么目的？

Ultralytics Explorer API 旨在进行全面的数据集探索。它允许用户使用 SQL 查询、向量相似性搜索和语义搜索来过滤和搜索数据集。这个功能强大的 Python API 可以处理大型数据集，非常适合使用 Ultralytics 模型进行各种计算机视觉任务。

### 如何安装 Ultralytics Explorer API？

要安装 Ultralytics Explorer API 及其依赖项，请使用以下命令：

```py
`pip  install  ultralytics[explorer]` 
```

这将自动安装探索器 API 功能所需的所有外部库。有关其他设置详细信息，请参阅我们文档的安装部分。

### 如何使用 Ultralytics Explorer API 进行相似性搜索？

您可以使用 Ultralytics 探索器 API 创建嵌入表，并查询类似图像进行相似性搜索。以下是一个基本示例：

```py
`from ultralytics import Explorer  # Create an Explorer object explorer = Explorer(data="coco128.yaml", model="yolov8n.pt") explorer.create_embeddings_table()  # Search for similar images to a given image similar_images_df = explorer.get_similar(img="path/to/image.jpg") print(similar_images_df.head())` 
```

欲了解更多详情，请访问相似性搜索部分。

### 使用 LanceDB 与 Ultralytics Explorer 有什么好处？

Ultralytics Explorer 内部使用的 LanceDB 提供可扩展的磁盘嵌入表。这确保您可以为诸如 COCO 这样的大型数据集创建和重用嵌入，而无需担心内存不足问题。这些表仅创建一次，并可重复使用，提升数据处理效率。

### Ultralytics Explorer API 中的 Ask AI 功能是如何工作的？

Ask AI 功能允许用户使用自然语言查询来过滤数据集。此功能利用 LLMs 在后台将这些查询转换为 SQL 查询。以下是一个示例：

```py
`from ultralytics import Explorer  # Create an Explorer object explorer = Explorer(data="coco128.yaml", model="yolov8n.pt") explorer.create_embeddings_table()  # Query with natural language query_result = explorer.ask_ai("show me 100 images with exactly one person and 2 dogs. There can be other objects too") print(query_result.head())` 
```

欲了解更多示例，请查看 Ask AI 部分。
