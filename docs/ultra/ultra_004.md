# 使用 Ultralytics YOLO 进行模型训练

> 原文：[`docs.ultralytics.com/modes/train/`](https://docs.ultralytics.com/modes/train/)

![Ultralytics YOLO 生态系统和集成](img/1933b0eeaf180eaa6d0c37f29931fb7d.png)

## 介绍

训练深度学习模型涉及提供数据并调整其参数，以便它能进行准确预测。Ultralytics YOLOv8 的训练模式专为有效和高效地训练目标检测模型而设计，充分利用现代硬件能力。本指南旨在涵盖使用 YOLOv8 丰富功能集训练自己的模型所需的所有详细信息。

[`www.youtube.com/embed/LNwODJXcvt4?si=7n1UvGRLSd9p5wKs`](https://www.youtube.com/embed/LNwODJXcvt4?si=7n1UvGRLSd9p5wKs)

**观看：** 如何在 Google Colab 上训练自定义数据集的 YOLOv8 模型。

## 为什么选择 Ultralytics YOLO 进行训练？

以下是选择 YOLOv8 训练模式的一些引人注目的理由：

+   **效率：** 充分利用您的硬件资源，无论您使用单 GPU 设置还是跨多 GPU 扩展。

+   **多功能性：** 除了像 COCO、VOC 和 ImageNet 这样的现成数据集外，还可以训练自定义数据集。

+   **用户友好：** 简单而强大的 CLI 和 Python 接口，提供直观的训练体验。

+   **超参数灵活性：** 广泛的可定制超参数范围，以微调模型性能。

### 训练模式的关键特点

以下是 YOLOv8 训练模式的一些显著特点：

+   **自动数据集下载：** 类似 COCO、VOC 和 ImageNet 的标准数据集在首次使用时会自动下载。

+   **多 GPU 支持：** 跨多个 GPU 无缝扩展训练效果，加快进程。

+   **超参数配置：** 可通过 YAML 配置文件或 CLI 参数修改超参数的选项。

+   **可视化与监控：** 实时跟踪训练指标，并可视化学习过程，以获取更好的见解。

提示

+   YOLOv8 数据集如 COCO、VOC、ImageNet 等，在第一次使用时会自动下载，例如 `yolo train data=coco.yaml`

## 使用示例

在图像大小为 640 的 COCO8 数据集上对 YOLOv8n 进行 100 个 epoch 的训练。可以使用`device`参数指定训练设备。如果未传递参数，则如果可用，则将使用 GPU `device=0`，否则将使用`device='cpu'`。有关所有训练参数的完整列表，请参见下面的参数部分。

单 GPU 和 CPU 训练示例

设备会自动确定。如果 GPU 可用，则会使用 GPU，否则将在 CPU 上开始训练。

```py
`from ultralytics import YOLO  # Load a model model = YOLO("yolov8n.yaml")  # build a new model from YAML model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training) model = YOLO("yolov8n.yaml").load("yolov8n.pt")  # build from YAML and transfer weights  # Train the model results = model.train(data="coco8.yaml", epochs=100, imgsz=640)` 
```

```py
`# Build a new model from YAML and start training from scratch yolo  detect  train  data=coco8.yaml  model=yolov8n.yaml  epochs=100  imgsz=640  # Start training from a pretrained *.pt model yolo  detect  train  data=coco8.yaml  model=yolov8n.pt  epochs=100  imgsz=640  # Build a new model from YAML, transfer pretrained weights to it and start training yolo  detect  train  data=coco8.yaml  model=yolov8n.yaml  pretrained=yolov8n.pt  epochs=100  imgsz=640` 
```

### 多 GPU 训练

多 GPU 训练通过在多个 GPU 上分发训练负载，有效利用可用硬件资源。此功能可通过 Python API 和命令行界面使用。要启用多 GPU 训练，请指定要使用的 GPU 设备 ID。

多 GPU 训练示例

要使用 2 个 GPU 进行训练，CUDA 设备 0 和 1，请使用以下命令。根据需要扩展到更多 GPU。

```py
`from ultralytics import YOLO  # Load a model model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)  # Train the model with 2 GPUs results = model.train(data="coco8.yaml", epochs=100, imgsz=640, device=[0, 1])` 
```

```py
`# Start training from a pretrained *.pt model using GPUs 0 and 1 yolo  detect  train  data=coco8.yaml  model=yolov8n.pt  epochs=100  imgsz=640  device=0,1` 
```

### Apple M1 和 M2 MPS 训练

支持 Apple M1 和 M2 芯片集成在 Ultralytics YOLO 模型中，现在可以在使用强大的 Metal Performance Shaders（MPS）框架的设备上训练模型。MPS 提供了在 Apple 定制硅上执行计算和图像处理任务的高性能方式。

要在 Apple M1 和 M2 芯片上进行训练，你应该在启动训练过程时将'device'指定为'mps'。以下是在 Python 中和通过命令行如何实现的示例：

MPS 训练示例

```py
`from ultralytics import YOLO  # Load a model model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)  # Train the model with 2 GPUs results = model.train(data="coco8.yaml", epochs=100, imgsz=640, device="mps")` 
```

```py
`# Start training from a pretrained *.pt model using GPUs 0 and 1 yolo  detect  train  data=coco8.yaml  model=yolov8n.pt  epochs=100  imgsz=640  device=mps` 
```

在利用 M1/M2 芯片的计算能力的同时，这使得训练任务的处理更加高效。有关更详细的指导和高级配置选项，请参阅[PyTorch MPS 文档](https://pytorch.org/docs/stable/notes/mps.html)。

### 恢复中断的训练

当使用深度学习模型时，从先前保存的状态恢复训练是一个关键特性。这在各种情况下都很有用，例如当训练过程意外中断或希望使用新数据或更多 epochs 继续训练模型时。

当恢复训练时，Ultralytics YOLO 会从最后保存的模型加载权重，并恢复优化器状态、学习率调度器和轮数。这样可以无缝地从中断的地方继续训练过程。

在 Ultralytics YOLO 中通过将`resume`参数设置为`True`并指定包含部分训练模型权重的`.pt`文件的路径，可以轻松恢复训练。

以下是如何使用 Python 和命令行恢复中断训练的示例：

恢复训练示例

```py
`from ultralytics import YOLO  # Load a model model = YOLO("path/to/last.pt")  # load a partially trained model  # Resume training results = model.train(resume=True)` 
```

```py
`# Resume an interrupted training yolo  train  resume  model=path/to/last.pt` 
```

通过设置`resume=True`，`train`函数将从存储在'path/to/last.pt'文件中的状态继续训练。如果省略`resume`参数或将其设置为`False`，`train`函数将启动新的训练会话。

请记住，默认情况下，每个 epoch 结束时或使用`save_period`参数以固定间隔保存检查点，所以您必须完成至少 1 个 epoch 才能恢复训练运行。

## 训练设置

YOLO 模型的训练设置涵盖了训练过程中使用的各种超参数和配置。这些设置影响模型的性能、速度和准确性。关键的训练设置包括批量大小、学习率、动量和权重衰减。此外，优化器的选择、损失函数和训练数据集的组成也会影响训练过程。精心调整和对这些设置进行实验对于优化性能至关重要。

| 参数 | 默认值 | 描述 |
| --- | --- | --- |
| `model` | `None` | 指定用于训练的模型文件。接受`.pt`预训练模型或`.yaml`配置文件的路径。定义模型结构或初始化权重至关重要。 |
| `data` | `None` | 数据集配置文件的路径（例如`coco8.yaml`）。该文件包含数据集特定的参数，包括训练和验证数据的路径、类名和类的数量。 |
| `epochs` | `100` | 总训练轮数。每个 epoch 表示对整个数据集的完整遍历。调整此值会影响训练持续时间和模型性能。 |
| `time` | `None` | 最大训练时间（小时）。如果设置了此参数，则会覆盖`epochs`参数，允许在指定的持续时间后自动停止训练。适用于时间受限的训练场景。 |
| `patience` | `100` | 在验证指标没有改善的情况下等待的轮数（epochs），用于提前停止训练以防止过拟合。 |
| `batch` | `16` | 批量大小，有三种模式：设置为整数（例如`batch=16`）、自动模式以利用 60%的 GPU 内存（`batch=-1`）或带有指定利用率分数的自动模式（`batch=0.70`）。 |
| `imgsz` | `640` | 训练的目标图像尺寸。所有图像在输入模型之前都会被调整到这个尺寸。影响模型的准确性和计算复杂度。 |
| `save` | `True` | 启用训练检查点和最终模型权重的保存。有助于恢复训练或模型部署。 |
| `save_period` | `-1` | 模型检查点保存频率，以 epochs 为单位。设置为-1 时禁用此功能。在长时间训练会话中保存中间模型非常有用。 |
| `cache` | `False` | 启用数据集图像的缓存，可选`True`/`ram`（内存中）、`disk`（磁盘中）或禁用`False`。减少磁盘 I/O 以提高训练速度，但会增加内存使用。 |
| `device` | `None` | 指定用于训练的计算设备：单个 GPU（`device=0`）、多个 GPU（`device=0,1`）、CPU（`device=cpu`）或 Apple Silicon 的 MPS（`device=mps`）。 |
| `workers` | `8` | 数据加载的工作线程数（每个`RANK`如果是多 GPU 训练）。影响数据预处理和输入模型的速度，特别适用于多 GPU 设置。 |
| `project` | `None` | 训练输出保存的项目目录名称。允许组织不同实验的存储。 |
| `name` | `None` | 训练运行的名称。用于在项目文件夹内创建子目录，存储训练日志和输出。 |
| `exist_ok` | `False` | 如果为 True，允许覆盖现有项目/名称目录。有助于在不需手动清除先前输出的情况下进行迭代实验。 |
| `pretrained` | `True` | 确定是否从预训练模型开始训练。可以是布尔值或指定模型路径的字符串，以加载权重。提升训练效率和模型性能。 |
| `optimizer` | `'auto'` | 训练的优化器选择。选项包括 `SGD`、`Adam`、`AdamW`、`NAdam`、`RAdam`、`RMSProp` 等，或者 `auto` 根据模型配置自动选择。影响收敛速度和稳定性。 |
| `verbose` | `False` | 在训练过程中启用详细输出，提供详细日志和进度更新。用于调试和紧密监控训练过程。 |
| `seed` | `0` | 设置训练的随机种子，确保在相同配置下运行时结果的可重现性。 |
| `deterministic` | `True` | 强制使用确定性算法，确保结果的可重现性，但可能会影响性能和速度，因为对非确定性算法施加了限制。 |
| `single_cls` | `False` | 在训练期间将多类数据集中的所有类别视为单一类别。适用于二元分类任务或关注对象存在而非分类。 |
| `rect` | `False` | 启用矩形训练，优化批次构成以减少填充。可以提高效率和速度，但可能影响模型精度。 |
| `cos_lr` | `False` | 使用余弦学习率调度器，在 epochs 上按余弦曲线调整学习率。有助于管理学习率以实现更好的收敛。 |
| `close_mosaic` | `10` | 在完成训练之前的最后 N 个 epochs 禁用马赛克数据增强，以稳定训练。设置为 0 会禁用此功能。 |
| `resume` | `False` | 从最后保存的检查点恢复训练。自动加载模型权重、优化器状态和 epoch 计数，无缝继续训练。 |
| `amp` | `True` | 启用自动混合精度（AMP）训练，减少内存使用并可能加快训练速度，对精度影响较小。 |
| `fraction` | `1.0` | 指定用于训练的数据集分数。允许在完整数据集的子集上进行训练，适用于实验或资源有限的情况。 |
| `profile` | `False` | 在训练期间启用 ONNX 和 TensorRT 速度的分析，有助于优化模型部署。 |
| `freeze` | `None` | 冻结模型的前 N 层或指定索引的层，减少可训练参数的数量。用于微调或迁移学习。 |
| `lr0` | `0.01` | 初始学习率（例如 `SGD=1E-2`，`Adam=1E-3`）。调整此值对优化过程至关重要，影响模型权重更新的速度。 |
| `lrf` | `0.01` | 最终学习率作为初始学习率的一部分 = (`lr0 * lrf`)，与调度器配合使用以随时间调整学习率。 |
| `momentum` | `0.937` | SGD 的动量因子或 Adam 优化器的 beta1，影响当前更新中过去梯度的纳入。 |
| `weight_decay` | `0.0005` | L2 正则化项，惩罚大权重以防止过拟合。 |
| `warmup_epochs` | `3.0` | 学习率预热的轮数，逐渐将学习率从低值增加到初始学习率，以稳定早期训练。 |
| `warmup_momentum` | `0.8` | 预热阶段的初始动量，逐渐调整到预热期间设定的动量。 |
| `warmup_bias_lr` | `0.1` | 预热阶段偏置参数的学习率，帮助稳定模型在初始轮次的训练。 |
| `box` | `7.5` | 损失函数中框损失组件的权重，影响对准确预测边界框坐标的重视程度。 |
| `cls` | `0.5` | 分类损失在总损失函数中的权重，影响正确类别预测相对于其他组件的重要性。 |
| `dfl` | `1.5` | 分布焦点损失的权重，在某些 YOLO 版本中用于细粒度分类。 |
| `pose` | `12.0` | 在为姿态估计训练的模型中，姿态损失的权重，影响对准确预测姿态关键点的重视程度。 |
| `kobj` | `2.0` | 关键点物体性损失在姿态估计模型中的权重，平衡检测置信度与姿态准确性。 |
| `label_smoothing` | `0.0` | 应用标签平滑，将硬标签软化为目标标签和标签均匀分布的混合，有助于提高泛化能力。 |
| `nbs` | `64` | 用于损失归一化的名义批量大小。 |
| `overlap_mask` | `True` | 确定在训练期间分割掩码是否应重叠，适用于实例分割任务。 |
| `mask_ratio` | `4` | 分割掩码的下采样比例，影响训练期间使用的掩码分辨率。 |
| `dropout` | `0.0` | 分类任务中的正则化丢弃率，通过在训练期间随机省略单元来防止过拟合。 |
| `val` | `True` | 在训练期间启用验证，允许定期评估模型在单独数据集上的性能。 |
| `plots` | `False` | 生成并保存训练和验证指标的图表，以及预测示例，提供对模型性能和学习进展的可视化洞察。 |

批量大小设置说明 |

`batch`参数可以通过三种方式进行配置： |

+   **固定批量大小**：设置一个整数值（例如，`batch=16`），直接指定每个批次的图像数量。 |

+   **自动模式（60% GPU 内存）**：使用`batch=-1`自动调整批量大小，以实现大约 60%的 CUDA 内存利用率。 |

+   **自动模式与利用率分数**：设置一个分数值（例如，`batch=0.70`），根据指定的 GPU 内存使用分数调整批量大小。 |

## 增强设置和超参数 |

数据增强技术对改善 YOLO 模型的鲁棒性和性能至关重要，通过向训练数据引入变异性，帮助模型更好地泛化到未见数据。以下表格详细描述了每种增强参数的目的和效果：

| 参数 | 类型 | 默认 | 范围 | 描述 |
| --- | --- | --- | --- | --- |
| `hsv_h` | `float` | `0.015` | `0.0 - 1.0` | 通过颜色轮的一部分调整图像的色调，引入颜色变化。有助于模型在不同光照条件下泛化。 |
| `hsv_s` | `float` | `0.7` | `0.0 - 1.0` | 改变图像的饱和度，影响颜色的强度。模拟不同环境条件很有用。 |
| `hsv_v` | `float` | `0.4` | `0.0 - 1.0` | 通过一部分修改图像的值（亮度），帮助模型在各种光照条件下表现良好。 |
| `degrees` | `float` | `0.0` | `-180 - +180` | 在指定角度范围内随机旋转图像，提高模型识别各种方向物体的能力。 |
| `translate` | `float` | `0.1` | `0.0 - 1.0` | 通过图像尺寸的一部分水平和垂直平移图像，有助于学习检测部分可见对象。 |
| `scale` | `float` | `0.5` | `>=0.0` | 通过增益因子缩放图像，模拟相机距离不同的物体。 |
| `shear` | `float` | `0.0` | `-180 - +180` | 按指定角度剪切图像，模拟从不同角度观察物体的效果。 |
| `perspective` | `float` | `0.0` | `0.0 - 0.001` | 对图像应用随机透视变换，增强模型理解三维空间中的物体能力。 |
| `flipud` | `float` | `0.0` | `0.0 - 1.0` | 将图像上下翻转，指定的概率下，增加数据的变异性，而不影响物体的特征。 |
| `fliplr` | `float` | `0.5` | `0.0 - 1.0` | 将图像左右翻转，指定的概率下，有助于学习对称物体并增加数据集的多样性。 |
| `bgr` | `float` | `0.0` | `0.0 - 1.0` | 将图像通道从 RGB 翻转为 BGR，指定的概率为，有助于增强对不正确通道排序的鲁棒性。 |
| `mosaic` | `float` | `1.0` | `0.0 - 1.0` | 将四个训练图像合并成一个，模拟不同的场景组合和物体交互。对于复杂场景理解非常有效。 |
| `mixup` | `float` | `0.0` | `0.0 - 1.0` | 混合两幅图像及其标签，创建混合图像。通过引入标签噪声和视觉变化，增强模型的泛化能力。 |
| `copy_paste` | `float` | `0.0` | `0.0 - 1.0` | 从一幅图像复制对象并粘贴到另一幅图像中，有助于增加对象实例并学习对象遮挡。 |
| `auto_augment` | `str` | `randaugment` | - | 自动应用预定义的增强策略（`randaugment`、`autoaugment`、`augmix`），通过增加视觉特征的多样性来优化分类任务。 |
| `erasing` | `float` | `0.4` | `0.0 - 0.9` | 在分类训练期间随机擦除图像的一部分，鼓励模型专注于识别不太明显的特征。 |
| `crop_fraction` | `float` | `1.0` | `0.1 - 1.0` | 将分类图像裁剪为其大小的一部分，以突出中央特征并适应对象的比例，减少背景干扰。 |

可以调整这些设置以满足数据集和当前任务的具体要求。尝试不同的值可以帮助找到最佳的增强策略，从而获得最佳的模型性能。

信息

获取有关训练增强操作的更多信息，请参阅参考部分。

## 记录

在训练 YOLOv8 模型时，您可能会发现跟踪模型随时间性能变化很有价值。这就是记录的作用。Ultralytics 的 YOLO 支持三种类型的记录器 - Comet、ClearML 和 TensorBoard。

要使用记录器，请从上面的代码片段的下拉菜单中选择并运行它。所选记录器将被安装和初始化。

### Comet

Comet 是一个平台，允许数据科学家和开发人员跟踪、比较、解释和优化实验和模型。它提供实时指标、代码差异和超参数跟踪等功能。

要使用 Comet：

示例

```py
`# pip install comet_ml import comet_ml  comet_ml.init()` 
```

记得在 Comet 的网站上登录您的账户并获取您的 API 密钥。您需要将其添加到环境变量或脚本中以记录您的实验。

### ClearML

[ClearML](https://www.clear.ml/)是一个开源平台，自动化实验跟踪，并帮助高效共享资源。它旨在帮助团队更有效地管理、执行和复现他们的机器学习工作。

使用 ClearML：

示例

```py
`# pip install clearml import clearml  clearml.browser_login()` 
```

运行此脚本后，您需要在浏览器上登录您的[CearML](https://www.clear.ml/)账户并验证您的会话。

### TensorBoard

[TensorBoard](https://www.tensorflow.org/tensorboard)是 TensorFlow 的可视化工具包。它允许您可视化 TensorFlow 图，绘制关于图执行的定量指标，并显示通过图像传递的其他数据。

要在[Google Colab](https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/tutorial.ipynb)中使用 TensorBoard：

示例

```py
`load_ext  tensorboard tensorboard  --logdir  ultralytics/runs  # replace with 'runs' directory` 
```

要在本地使用 TensorBoard，请运行下面的命令，并在 http://localhost:6006/ 查看结果。

示例

```py
`tensorboard  --logdir  ultralytics/runs  # replace with 'runs' directory` 
```

这将加载 TensorBoard 并将其定向到保存训练日志的目录。

设置好记录器后，您可以开始模型训练。所有训练指标将自动记录在您选择的平台上，您可以访问这些日志以随时监控模型的性能，比较不同模型，并确定改进的方向。

## 常见问题

### 如何使用 Ultralytics YOLOv8 训练目标检测模型？

要使用 Ultralytics YOLOv8 训练目标检测模型，您可以使用 Python API 或 CLI。以下是两者的示例：

单 GPU 和 CPU 训练示例

```py
`from ultralytics import YOLO  # Load a model model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)  # Train the model results = model.train(data="coco8.yaml", epochs=100, imgsz=640)` 
```

```py
`yolo  detect  train  data=coco8.yaml  model=yolov8n.pt  epochs=100  imgsz=640` 
```

欲了解更多详情，请参阅训练设置部分。

### Ultralytics YOLOv8 的训练模式的关键特性是什么？

Ultralytics YOLOv8 的训练模式的关键特性包括：

+   **自动数据集下载：** 自动下载标准数据集，如 COCO、VOC 和 ImageNet。

+   **多 GPU 支持：** 可以跨多个 GPU 进行训练，加速处理速度。

+   **超参数配置：** 通过 YAML 文件或 CLI 参数自定义超参数。

+   **可视化和监控：** 实时跟踪训练指标，以获得更好的洞察力。

这些功能使得训练高效且可根据您的需求定制。有关详细信息，请参阅训练模式的关键特性部分。

### 如何从中断的会话中恢复 Ultralytics YOLOv8 的训练？

要从中断的会话恢复训练，请将`resume`参数设置为`True`并指定最后保存的检查点路径。

恢复训练示例

```py
`from ultralytics import YOLO  # Load the partially trained model model = YOLO("path/to/last.pt")  # Resume training results = model.train(resume=True)` 
```

```py
`yolo  train  resume  model=path/to/last.pt` 
```

查看中断训练部分以获取更多信息。

### 我能在 Apple M1 和 M2 芯片上训练 YLOv8 模型吗？

是的，Ultralytics YOLOv8 支持在 Apple M1 和 M2 芯片上使用 Metal Performance Shaders（MPS）框架进行训练。请将训练设备设置为 'mps'。

MPS 训练示例

```py
`from ultralytics import YOLO  # Load a pretrained model model = YOLO("yolov8n.pt")  # Train the model on M1/M2 chip results = model.train(data="coco8.yaml", epochs=100, imgsz=640, device="mps")` 
```

```py
`yolo  detect  train  data=coco8.yaml  model=yolov8n.pt  epochs=100  imgsz=640  device=mps` 
```

欲了解更多详情，请参阅 Apple M1 和 M2 MPS 训练部分。

### 常见的训练设置是什么，如何配置它们？

Ultralytics YOLOv8 允许通过参数配置各种训练设置，如批量大小、学习率、时代等。以下是简要概述：

| Argument | Default | Description |
| --- | --- | --- |
| `model` | `None` | 用于训练的模型文件路径。 |
| `data` | `None` | 数据集配置文件的路径（例如 `coco8.yaml`）。 |
| `epochs` | `100` | 总训练时期数。 |
| `batch` | `16` | 批量大小，可以设置为整数或自动模式。 |
| `imgsz` | `640` | 训练的目标图像尺寸。 |
| `device` | `None` | 用于训练的计算设备，例如 `cpu`、`0`、`0,1`或`mps`。 |
| `save` | `True` | 启用保存训练检查点和最终模型权重。 |

欲了解训练设置的详细指南，请参阅训练设置部分。
