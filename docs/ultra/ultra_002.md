# 快速入门

> 原文：[`docs.ultralytics.com/quickstart/`](https://docs.ultralytics.com/quickstart/)

## 安装 Ultralytics

Ultralytics 提供各种安装方法，包括 pip、conda 和 Docker。通过`ultralytics` pip 包安装最新稳定版本的 YOLOv8，或通过克隆[Ultralytics GitHub 仓库](https://github.com/ultralytics/ultralytics)获取最新版本。Docker 可用于在隔离的容器中执行该软件包，避免本地安装。

[`www.youtube.com/embed/_a7cVL9hqnk`](https://www.youtube.com/embed/_a7cVL9hqnk)

**观看：** Ultralytics YOLO 快速入门指南

安装

![PyPI - Python Version](img/c1c96b5d0d3404f036367589f6993a54.png)

使用 pip 安装`ultralytics`包，或通过运行`pip install -U ultralytics`更新现有安装。有关`ultralytics`包的更多详情，请访问 Python 包索引（PyPI）：[`pypi.org/project/ultralytics/`](https://pypi.org/project/ultralytics/)。

![PyPI - 版本](https://pypi.org/project/ultralytics/) ![下载量](https://pepy.tech/project/ultralytics)

```py
# Install the ultralytics package from PyPI
pip  install  ultralytics 
```

您还可以直接从 GitHub [仓库](https://github.com/ultralytics/ultralytics)安装`ultralytics`包。如果您需要最新的开发版本，这可能很有用。请确保在系统上安装了 Git 命令行工具。`@main`命令安装`main`分支，并可以修改为其他分支，例如`@my-branch`，或完全删除以默认使用`main`分支。

```py
# Install the ultralytics package from GitHub
pip  install  git+https://github.com/ultralytics/ultralytics.git@main 
```

Conda 是 pip 的另一种替代包管理器，也可用于安装。有关更多细节，请访问 Anaconda：[`anaconda.org/conda-forge/ultralytics`](https://anaconda.org/conda-forge/ultralytics)。更新 conda 包的 Ultralytics feedstock 仓库位于[`github.com/conda-forge/ultralytics-feedstock/`](https://github.com/conda-forge/ultralytics-feedstock/)。

![Conda 版本](https://anaconda.org/conda-forge/ultralytics) ![Conda 下载量](https://anaconda.org/conda-forge/ultralytics) ![Conda 配方](https://anaconda.org/conda-forge/ultralytics) ![Conda 平台](https://anaconda.org/conda-forge/ultralytics)

```py
# Install the ultralytics package using conda
conda  install  -c  conda-forge  ultralytics 
```

注意

如果您在 CUDA 环境中安装，最佳做法是在同一命令中安装`ultralytics`，`pytorch`和`pytorch-cuda`，以允许 conda 包管理器解决任何冲突，或者安装`pytorch-cuda`时最后进行以允许其覆盖 CPU 特定的`pytorch`包。

```py
# Install all packages together using conda
conda  install  -c  pytorch  -c  nvidia  -c  conda-forge  pytorch  torchvision  pytorch-cuda=11.8  ultralytics 
```

### Conda Docker 镜像

Ultralytics Conda Docker 镜像也可从[DockerHub](https://hub.docker.com/r/ultralytics/ultralytics)获取。这些镜像基于[Miniconda3](https://docs.conda.io/projects/miniconda/en/latest/)，是在 Conda 环境中开始使用`ultralytics`的简单方法。

```py
# Set image name as a variable
t=ultralytics/ultralytics:latest-conda

# Pull the latest ultralytics image from Docker Hub
sudo  docker  pull  $t

# Run the ultralytics image in a container with GPU support
sudo  docker  run  -it  --ipc=host  --gpus  all  $t  # all GPUs
sudo  docker  run  -it  --ipc=host  --gpus  '"device=2,3"'  $t  # specify GPUs 
```

如果您有兴趣贡献代码或希望使用最新的源代码进行实验，请克隆`ultralytics`仓库。克隆后，进入目录并使用 pip 以可编辑模式`-e`安装包。

![GitHub 最后提交](https://github.com/ultralytics/ultralytics) ![GitHub 提交活动](https://github.com/ultralytics/ultralytics)

```py
# Clone the ultralytics repository
git  clone  https://github.com/ultralytics/ultralytics

# Navigate to the cloned directory
cd  ultralytics

# Install the package in editable mode for development
pip  install  -e  . 
```

利用 Docker 在隔离的容器中轻松执行`ultralytics`包，确保在各种环境中保持一致且流畅的性能。通过选择[Docker Hub](https://hub.docker.com/r/ultralytics/ultralytics) 中的官方`ultralytics`镜像之一，您不仅避免了本地安装的复杂性，还能从验证过的工作环境中获益。Ultralytics 提供了 5 种主要支持的 Docker 镜像，每种都设计用于不同平台和用例的高兼容性和效率：

![Docker 镜像版本](https://hub.docker.com/r/ultralytics/ultralytics) ![Docker 拉取次数](https://hub.docker.com/r/ultralytics/ultralytics)

+   **Dockerfile：** 推荐用于训练的 GPU 镜像。

+   **Dockerfile-arm64：** 针对 ARM64 架构优化，允许在像树莓派和其他基于 ARM64 的平台上部署。

+   **Dockerfile-cpu：** 基于 Ubuntu 的仅 CPU 版本，适用于推理和没有 GPU 的环境。

+   **Dockerfile-jetson：** 专为 NVIDIA Jetson 设备定制，集成了针对这些平台优化的 GPU 支持。

+   **Dockerfile-python：** 只包含 Python 和必要依赖项的最小镜像，非常适合轻量级应用和开发。

+   **Dockerfile-conda：** 基于 Miniconda3，通过 conda 安装`ultralytics`包。

下面是获取最新镜像并执行的命令：

```py
# Set image name as a variable
t=ultralytics/ultralytics:latest

# Pull the latest ultralytics image from Docker Hub
sudo  docker  pull  $t

# Run the ultralytics image in a container with GPU support
sudo  docker  run  -it  --ipc=host  --gpus  all  $t  # all GPUs
sudo  docker  run  -it  --ipc=host  --gpus  '"device=2,3"'  $t  # specify GPUs 
```

上述命令初始化了一个带有最新`ultralytics`镜像的 Docker 容器。`-it`标志分配了一个伪 TTY 并保持标准输入打开，使您能够与容器进行交互。`--ipc=host`标志设置 IPC（进程间通信）命名空间为主机，这对于在进程之间共享内存至关重要。`--gpus all`标志启用了对容器内所有可用 GPU 的访问，这对需要 GPU 计算的任务至关重要。

注意：要在容器内与本地机器上的文件一起工作，请使用 Docker 卷将本地目录挂载到容器中：

```py
# Mount local directory to a directory inside the container
sudo  docker  run  -it  --ipc=host  --gpus  all  -v  /path/on/host:/path/in/container  $t 
```

使用本地机器上的目录路径替换`/path/on/host`，并在 Docker 容器中使用所需路径替换`/path/in/container`以便访问。

对于高级的 Docker 使用，请随时查阅 Ultralytics Docker 指南。

查看`ultralytics` [pyproject.toml](https://github.com/ultralytics/ultralytics/blob/main/pyproject.toml) 文件以获取依赖项列表。请注意，上述所有示例均安装了所有所需的依赖项。

提示

PyTorch 的要求因操作系统和 CUDA 要求而异，因此建议首先根据[`pytorch.org/get-started/locally`](https://pytorch.org/get-started/locally)的说明安装 PyTorch。

![PyTorch 安装说明](https://pytorch.org/get-started/locally/)

## 使用 CLI 与 Ultralytics

Ultralytics 命令行界面（CLI）允许简单的单行命令，无需 Python 环境。CLI 无需定制或 Python 代码。您可以通过`yolo`命令直接从终端运行所有任务。查看 CLI 指南，了解更多关于如何从命令行使用 YOLOv8 的信息。

示例

Ultralytics `yolo`命令使用以下语法：

```py
yolo  TASK  MODE  ARGS 
```

+   `TASK`（可选）是（detect, segment, classify, pose, obb）之一

+   `MODE`（必需）是（train, val, predict, export, track, benchmark）之一

+   `ARGS`（可选）是`arg=value`对，例如`imgsz=640`，用于覆盖默认值。

请查看完整的配置指南或使用`yolo cfg` CLI 命令中的所有`ARGS`。

使用初始学习率为 0.01 训练 10 个 epoch 的检测模型

```py
yolo  train  data=coco8.yaml  model=yolov8n.pt  epochs=10  lr0=0.01 
```

预测使用预训练分割模型在图像大小为 320 的 YouTube 视频：

```py
yolo  predict  model=yolov8n-seg.pt  source='https://youtu.be/LNwODJXcvt4'  imgsz=320 
```

在批量大小为 1 且图像大小为 640 时评估预训练检测模型：

```py
yolo  val  model=yolov8n.pt  data=coco8.yaml  batch=1  imgsz=640 
```

将 YOLOv8n 分类模型导出到 ONNX 格式，图像大小为 224 乘以 128（无需任务要求）

```py
yolo  export  model=yolov8n-cls.pt  format=onnx  imgsz=224,128 
```

运行特殊命令以查看版本、查看设置、运行检查等：

```py
yolo  help
yolo  checks
yolo  version
yolo  settings
yolo  copy-cfg
yolo  cfg 
```

警告

参数必须以`arg=val`对形式传递，由等号`=`分隔，并且在对中使用空格分隔。不要使用`--`前缀或逗号`,`分隔参数。

+   `yolo predict model=yolov8n.pt imgsz=640 conf=0.25` ✅

+   `yolo predict model yolov8n.pt imgsz 640 conf 0.25` ❌（缺少`=`）

+   `yolo predict model=yolov8n.pt, imgsz=640, conf=0.25` ❌（不要使用`,`）

+   `yolo predict --model yolov8n.pt --imgsz 640 --conf 0.25` ❌（不要使用`--`）

CLI 指南

## 使用 Python 与 Ultralytics

YOLOv8 的 Python 接口允许将其无缝集成到您的 Python 项目中，使您能够轻松加载、运行和处理模型的输出。设计时考虑了简易性和易用性，Python 接口使用户能够快速实现对象检测、分割和分类。这使得 YOLOv8 的 Python 接口成为任何希望将这些功能集成到其 Python 项目中的人的宝贵工具。

例如，用户可以加载模型，训练模型，在验证集上评估其性能，甚至仅使用几行代码将其导出为 ONNX 格式。查看 Python 指南，了解如何在 Python 项目中使用 YOLOv8 更多信息。

示例

```py
from ultralytics import YOLO

# Create a new YOLO model from scratch
model = YOLO("yolov8n.yaml")

# Load a pretrained YOLO model (recommended for training)
model = YOLO("yolov8n.pt")

# Train the model using the 'coco8.yaml' dataset for 3 epochs
results = model.train(data="coco8.yaml", epochs=3)

# Evaluate the model's performance on the validation set
results = model.val()

# Perform object detection on an image using the model
results = model("https://ultralytics.com/images/bus.jpg")

# Export the model to ONNX format
success = model.export(format="onnx") 
```

Python 指南

## Ultralytics 设置

Ultralytics 库提供了一个强大的设置管理系统，可以精细控制您的实验。通过使用`ultralytics.utils`模块中的`SettingsManager`，用户可以轻松访问和修改他们的设置。这些设置存储在一个 YAML 文件中，可以直接在 Python 环境中或通过命令行界面（CLI）中查看或修改。

### 检查设置

要了解当前设置的配置，您可以直接查看它们：

查看设置

您可以使用 Python 查看您的设置。首先从`ultralytics`模块导入`settings`对象。使用以下命令打印和返回设置：

```py
from ultralytics import settings

# View all settings
print(settings)

# Return a specific setting
value = settings["runs_dir"] 
```

或者，命令行界面允许您使用简单的命令检查您的设置：

```py
yolo  settings 
```

### 修改设置

Ultralytics 允许用户轻松修改其设置。可以通过以下方式进行更改：

更新设置

在 Python 环境中，调用`settings`对象的`update`方法来更改您的设置：

```py
from ultralytics import settings

# Update a setting
settings.update({"runs_dir": "/path/to/runs"})

# Update multiple settings
settings.update({"runs_dir": "/path/to/runs", "tensorboard": False})

# Reset settings to default values
settings.reset() 
```

如果您更喜欢使用命令行界面，以下命令将允许您修改您的设置：

```py
# Update a setting
yolo  settings  runs_dir='/path/to/runs'

# Update multiple settings
yolo  settings  runs_dir='/path/to/runs'  tensorboard=False

# Reset settings to default values
yolo  settings  reset 
```

### 理解设置

下表提供了在 Ultralytics 中可调整的设置概述。每个设置都包括示例值、数据类型和简要描述。

| 名称 | 示例值 | 数据类型 | 描述 |
| --- | --- | --- | --- |
| `settings_version` | `'0.0.4'` | `str` | Ultralytics *settings*版本（与 Ultralytics [pip](https://pypi.org/project/ultralytics/)版本不同） |
| `datasets_dir` | `'/path/to/datasets'` | `str` | 存储数据集的目录 |
| `weights_dir` | `'/path/to/weights'` | `str` | 存储模型权重的目录 |
| `runs_dir` | `'/path/to/runs'` | `str` | 存储实验运行的目录 |
| `uuid` | `'a1b2c3d4'` | `str` | 当前设置的唯一标识符 |
| `sync` | `True` | `bool` | 是否将分析和崩溃同步到 HUB |
| `api_key` | `''` | `str` | Ultralytics HUB [API Key](https://hub.ultralytics.com/settings?tab=api+keys) |
| `clearml` | `True` | `bool` | 是否使用 ClearML 进行日志记录 |
| `comet` | `True` | `bool` | 是否使用[Comet ML](https://bit.ly/yolov8-readme-comet)进行实验跟踪和可视化 |
| `dvc` | `True` | `bool` | 是否使用[DVC 进行实验跟踪](https://dvc.org/doc/dvclive/ml-frameworks/yolo)和版本控制 |
| `hub` | `True` | `bool` | 是否使用[Ultralytics HUB](https://hub.ultralytics.com)集成 |
| `mlflow` | `True` | `bool` | 是否使用 MLFlow 进行实验跟踪 |
| `neptune` | `True` | `bool` | 是否使用 Neptune 进行实验跟踪 |
| `raytune` | `True` | `bool` | 是否使用 Ray Tune 进行超参数调优 |
| `tensorboard` | `True` | `bool` | 是否使用 TensorBoard 进行可视化 |
| `wandb` | `True` | `bool` | 是否使用 Weights & Biases 日志记录 |

当您浏览您的项目或实验时，请确保定期检查这些设置，以确保它们针对您的需求进行了最佳配置。

## 常见问题解答

### 如何使用 pip 安装 Ultralytics YOLOv8？

要使用 pip 安装 Ultralytics YOLOv8，请执行以下命令：

```py
pip  install  ultralytics 
```

对于最新的稳定版本发布，这将直接从 Python 包索引（PyPI）安装 `ultralytics` 包。欲了解更多详细信息，请访问 [PyPI 上的 ultralytics 包](https://pypi.org/project/ultralytics/)。

或者，您可以直接从 GitHub 安装最新的开发版本：

```py
pip  install  git+https://github.com/ultralytics/ultralytics.git 
```

确保在你的系统上安装了 Git 命令行工具。

### 我能使用 conda 安装 Ultralytics YOLOv8 吗？

是的，你可以通过以下 conda 命令安装 Ultralytics YOLOv8：

```py
conda  install  -c  conda-forge  ultralytics 
```

这种方法是 pip 的一个很好的替代方案，并确保与您环境中的其他包兼容。对于 CUDA 环境，最好同时安装 `ultralytics`、`pytorch` 和 `pytorch-cuda` 以解决任何冲突：

```py
conda  install  -c  pytorch  -c  nvidia  -c  conda-forge  pytorch  torchvision  pytorch-cuda=11.8  ultralytics 
```

欲获取更多指导，请查看 Conda 快速入门指南。

### 使用 Docker 运行 Ultralytics YOLOv8 的优势是什么？

使用 Docker 运行 Ultralytics YOLOv8 可提供隔离和一致的环境，确保在不同系统上的平稳运行。它还消除了本地安装的复杂性。Ultralytics 官方提供了适用于 GPU、CPU、ARM64、NVIDIA Jetson 和 Conda 环境的不同变体的 Docker 镜像，可以在 [Docker Hub](https://hub.docker.com/r/ultralytics/ultralytics) 上获取最新镜像并运行以下命令：

```py
# Pull the latest ultralytics image from Docker Hub
sudo  docker  pull  ultralytics/ultralytics:latest

# Run the ultralytics image in a container with GPU support
sudo  docker  run  -it  --ipc=host  --gpus  all  ultralytics/ultralytics:latest 
```

欲了解更详细的 Docker 指南，请查看 Docker 快速入门指南。

### 如何克隆 Ultralytics 代码库以进行开发？

要克隆 Ultralytics 代码库并设置开发环境，请按以下步骤操作：

```py
# Clone the ultralytics repository
git  clone  https://github.com/ultralytics/ultralytics

# Navigate to the cloned directory
cd  ultralytics

# Install the package in editable mode for development
pip  install  -e  . 
```

这种方法允许您贡献到项目或使用最新的源代码进行实验。欲了解更多详细信息，请访问 [Ultralytics GitHub 代码库](https://github.com/ultralytics/ultralytics)。

### 为什么要使用 Ultralytics YOLOv8 CLI？

Ultralytics YOLOv8 命令行界面（CLI）简化了运行对象检测任务的流程，无需编写 Python 代码。您可以直接从终端执行单行命令，例如训练、验证和预测任务。`yolo` 命令的基本语法如下：

```py
yolo  TASK  MODE  ARGS 
```

例如，使用指定参数训练检测模型：

```py
yolo  train  data=coco8.yaml  model=yolov8n.pt  epochs=10  lr0=0.01 
```

查看完整的 CLI 指南以探索更多命令和用法示例。
