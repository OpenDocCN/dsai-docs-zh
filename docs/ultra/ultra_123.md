# Ultralytics Explorer API

> 原文：[`docs.ultralytics.com/datasets/explorer/api/`](https://docs.ultralytics.com/datasets/explorer/api/)

## 简介

![在 Colab 中打开](https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/docs/en/datasets/explorer/explorer.ipynb) Explorer API 是用于探索数据集的 Python API。支持使用 SQL 查询、向量相似性搜索和语义搜索进行过滤和搜索您的数据集。

[`www.youtube.com/embed/3VryynorQeo?start=279`](https://www.youtube.com/embed/3VryynorQeo?start=279)

**Watch:** Ultralytics Explorer API 概述

## 安装

Explorer 依赖于某些功能的外部库。这些会在使用时自动安装。要手动安装这些依赖项，请使用以下命令：

```py
pip  install  ultralytics[explorer] 
```

## 用法

```py
from ultralytics import Explorer

# Create an Explorer object
explorer = Explorer(data="coco128.yaml", model="yolov8n.pt")

# Create embeddings for your dataset
explorer.create_embeddings_table()

# Search for similar images to a given image/images
dataframe = explorer.get_similar(img="path/to/image.jpg")

# Or search for similar images to a given index/indices
dataframe = explorer.get_similar(idx=0) 
```

注意

为给定的数据集和模型对创建的嵌入表只会创建一次并重复使用。这些在幕后使用 [LanceDB](https://lancedb.github.io/lancedb/)，它在磁盘上扩展，因此您可以为像 COCO 这样的大型数据集创建和重复使用嵌入，而无需耗尽内存。

如果您想要强制更新嵌入表，可以将 `force=True` 传递给 `create_embeddings_table` 方法。

您可以直接访问 LanceDB 表对象以执行高级分析。在“使用嵌入表”部分了解更多信息

## 1\. 相似性搜索

相似性搜索是一种查找给定图像相似图像的技术。它基于相似图像将具有相似嵌入的想法。一旦建立了嵌入表，您可以以以下任何一种方式运行语义搜索：

+   在数据集中给定索引或索引列表上：`exp.get_similar(idx=[1,10], limit=10)`

+   对于数据集中不存在的任何图像或图像列表：`exp.get_similar(img=["path/to/img1", "path/to/img2"], limit=10)`

对于多个输入情况，使用它们的嵌入的聚合。

您可以获得一个带有`limit`个最相似数据点的 pandas 数据框，以及它们在嵌入空间中的距离。您可以使用此数据集进行进一步的过滤。

语义搜索

```py
from ultralytics import Explorer

# create an Explorer object
exp = Explorer(data="coco128.yaml", model="yolov8n.pt")
exp.create_embeddings_table()

similar = exp.get_similar(img="https://ultralytics.com/images/bus.jpg", limit=10)
print(similar.head())

# Search using multiple indices
similar = exp.get_similar(
    img=["https://ultralytics.com/images/bus.jpg", "https://ultralytics.com/images/bus.jpg"],
    limit=10,
)
print(similar.head()) 
```

```py
from ultralytics import Explorer

# create an Explorer object
exp = Explorer(data="coco128.yaml", model="yolov8n.pt")
exp.create_embeddings_table()

similar = exp.get_similar(idx=1, limit=10)
print(similar.head())

# Search using multiple indices
similar = exp.get_similar(idx=[1, 10], limit=10)
print(similar.head()) 
```

### 绘制相似图像

您还可以使用 `plot_similar` 方法绘制相似图像。此方法接受与 `get_similar` 相同的参数，并在网格中绘制相似图像。

绘制相似图像

```py
from ultralytics import Explorer

# create an Explorer object
exp = Explorer(data="coco128.yaml", model="yolov8n.pt")
exp.create_embeddings_table()

plt = exp.plot_similar(img="https://ultralytics.com/images/bus.jpg", limit=10)
plt.show() 
```

```py
from ultralytics import Explorer

# create an Explorer object
exp = Explorer(data="coco128.yaml", model="yolov8n.pt")
exp.create_embeddings_table()

plt = exp.plot_similar(idx=1, limit=10)
plt.show() 
```

## 2\. 问答 AI（自然语言查询）

这使您可以以自然语言编写想要如何过滤数据集的方式。您无需精通编写 SQL 查询。我们的 AI 动力查询生成器会在幕后自动生成查询。例如 - 您可以说 - “显示我有正好一个人和两只狗的 100 张图像。也可以有其他对象”，它会在内部生成查询并显示这些结果。注意：这是在幕后使用 LLMs 运行，因此结果是概率性的，有时可能会出错。

问答 AI

```py
from ultralytics import Explorer
from ultralytics.data.explorer import plot_query_result

# create an Explorer object
exp = Explorer(data="coco128.yaml", model="yolov8n.pt")
exp.create_embeddings_table()

df = exp.ask_ai("show me 100 images with exactly one person and 2 dogs. There can be other objects too")
print(df.head())

# plot the results
plt = plot_query_result(df)
plt.show() 
```

## 3\. SQL 查询

您可以使用 `sql_query` 方法在数据集上运行 SQL 查询。此方法接受 SQL 查询作为输入，并返回包含结果的 pandas 数据帧。

SQL 查询

```py
from ultralytics import Explorer

# create an Explorer object
exp = Explorer(data="coco128.yaml", model="yolov8n.pt")
exp.create_embeddings_table()

df = exp.sql_query("WHERE labels LIKE '%person%' AND labels LIKE '%dog%'")
print(df.head()) 
```

### 绘制 SQL 查询结果

您还可以使用 `plot_sql_query` 方法绘制 SQL 查询的结果。此方法接受与 `sql_query` 相同的参数，并在网格中绘制结果。

绘制 SQL 查询结果

```py
from ultralytics import Explorer

# create an Explorer object
exp = Explorer(data="coco128.yaml", model="yolov8n.pt")
exp.create_embeddings_table()

# plot the SQL Query
exp.plot_sql_query("WHERE labels LIKE '%person%' AND labels LIKE '%dog%' LIMIT 10") 
```

## 4\. 使用嵌入表

您也可以直接使用嵌入表。一旦创建了嵌入表，可以使用 `Explorer.table` 访问它。

Explorer 内部使用 [LanceDB](https://lancedb.github.io/lancedb/) 表。您可以直接访问此表，使用 `Explorer.table` 对象运行原始查询，推送预过滤器和后过滤器等。

```py
from ultralytics import Explorer

exp = Explorer()
exp.create_embeddings_table()
table = exp.table 
```

以下是您可以使用该表执行的一些示例操作：

### 获取原始嵌入

示例

```py
from ultralytics import Explorer

exp = Explorer()
exp.create_embeddings_table()
table = exp.table

embeddings = table.to_pandas()["vector"]
print(embeddings) 
```

### 使用预过滤器和后过滤器进行高级查询

示例

```py
from ultralytics import Explorer

exp = Explorer(model="yolov8n.pt")
exp.create_embeddings_table()
table = exp.table

# Dummy embedding
embedding = [i for i in range(256)]
rs = table.search(embedding).metric("cosine").where("").limit(10) 
```

### 创建向量索引

当使用大型数据集时，您还可以为更快的查询创建专用的向量索引。这可以通过在 LanceDB 表上使用 `create_index` 方法完成。

```py
table.create_index(num_partitions=..., num_sub_vectors=...) 
```

在此处查找有关可用类型向量索引和参数的更多详细信息 [here](https://lancedb.github.io/lancedb/ann_indexes/#types-of-index)。未来，我们将支持直接从 Explorer API 创建向量索引。

## 5\. 嵌入应用

您可以使用嵌入表执行各种探索性分析。以下是一些示例：

### 相似性索引

Explorer 提供了一个 `similarity_index` 操作：

+   它试图估计每个数据点与数据集中其余数据点的相似度。

+   它通过计算在生成的嵌入空间中比当前图像更接近的图像嵌入的数量，考虑一次 `top_k` 个相似图像来实现。

它返回一个包含以下列的 pandas 数据帧：

+   `idx`：数据集中图像的索引

+   `im_file`：图像文件的路径

+   `count`：比当前图像更接近的数据集中图像数量

+   `sim_im_files`：路径列表，包含 `count` 个相似图像

提示

对于给定的数据集、模型、`max_dist` 和 `top_k`，一旦生成相似性索引，将会重复使用。如果您的数据集发生变化，或者仅需重新生成相似性索引，可以传递 `force=True`。

相似性索引

```py
from ultralytics import Explorer

exp = Explorer()
exp.create_embeddings_table()

sim_idx = exp.similarity_index() 
```

您可以使用相似性索引来构建自定义条件，以过滤数据集。例如，您可以使用以下代码过滤掉与数据集中任何其他图像不相似的图像：

```py
import numpy as np

sim_count = np.array(sim_idx["count"])
sim_idx["im_file"][sim_count > 30] 
```

### 可视化嵌入空间

您还可以使用所选的绘图工具可视化嵌入空间。例如，这里是使用 matplotlib 的简单示例：

```py
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Reduce dimensions using PCA to 3 components for visualization in 3D
pca = PCA(n_components=3)
reduced_data = pca.fit_transform(embeddings)

# Create a 3D scatter plot using Matplotlib Axes3D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

# Scatter plot
ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2], alpha=0.5)
ax.set_title("3D Scatter Plot of Reduced 256-Dimensional Data (PCA)")
ax.set_xlabel("Component 1")
ax.set_ylabel("Component 2")
ax.set_zlabel("Component 3")

plt.show() 
```

开始使用 Explorer API 创建自己的 CV 数据集探索报告。作为灵感，查看

## 使用 Ultralytics Explorer 构建的应用程序

尝试我们基于 Explorer API 的 GUI 演示

## 即将推出

+   [ ] 从数据集中合并特定标签。示例 - 从 COCO 导入所有 `person` 标签和从 Cityscapes 导入 `car` 标签

+   [ ] 移除相似性指数高于给定阈值的图像

+   [ ] 在合并/移除条目后自动持久化新数据集

+   [ ] 高级数据集可视化

## 常见问题解答

### Ultralytics Explorer API 的用途是什么？

Ultralytics Explorer API 旨在进行全面的数据集探索。它允许用户使用 SQL 查询、向量相似性搜索和语义搜索来过滤和搜索数据集。这个强大的 Python API 可以处理大型数据集，非常适合使用 Ultralytics 模型的各种计算机视觉任务。

### 我该如何安装 Ultralytics Explorer API？

要安装 Ultralytics Explorer API 及其依赖项，请使用以下命令：

```py
pip  install  ultralytics[explorer] 
```

这将自动安装 Explorer API 功能所需的所有外部库。有关其他设置细节，请参阅我们文档的安装部分。

### 我该如何使用 Ultralytics Explorer API 进行相似性搜索？

你可以使用 Ultralytics Explorer API 通过创建嵌入表并查询相似图像来执行相似性搜索。以下是一个基本示例：

```py
from ultralytics import Explorer

# Create an Explorer object
explorer = Explorer(data="coco128.yaml", model="yolov8n.pt")
explorer.create_embeddings_table()

# Search for similar images to a given image
similar_images_df = explorer.get_similar(img="path/to/image.jpg")
print(similar_images_df.head()) 
```

欲了解更多详情，请访问相似性搜索部分。

### 使用 LanceDB 与 Ultralytics Explorer 的好处是什么？

LanceDB 在 Ultralytics Explorer 的底层使用，提供可扩展的磁盘嵌入表。这确保你可以为像 COCO 这样的大型数据集创建和重用嵌入，而不会耗尽内存。这些表只创建一次，可以重复使用，从而提高数据处理的效率。

### 问 AI 功能在 Ultralytics Explorer API 中是如何工作的？

问 AI 功能允许用户使用自然语言查询来过滤数据集。此功能利用 LLMs 将这些查询转换为后台的 SQL 查询。以下是一个示例：

```py
from ultralytics import Explorer

# Create an Explorer object
explorer = Explorer(data="coco128.yaml", model="yolov8n.pt")
explorer.create_embeddings_table()

# Query with natural language
query_result = explorer.ask_ai("show me 100 images with exactly one person and 2 dogs. There can be other objects too")
print(query_result.head()) 
```

欲了解更多示例，请查看问 AI 部分。
