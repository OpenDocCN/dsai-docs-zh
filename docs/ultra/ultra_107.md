# Triton 推理服务器与 Ultralytics YOLOv8。

> 原文：[`docs.ultralytics.com/guides/triton-inference-server/`](https://docs.ultralytics.com/guides/triton-inference-server/)

[Triton 推理服务器](https://developer.nvidia.com/nvidia-triton-inference-server)（以前称为 TensorRT 推理服务器）是 NVIDIA 开发的开源软件解决方案。它提供了一个针对 NVIDIA GPU 优化的云推理解决方案。Triton 简化了生产环境中大规模部署 AI 模型的过程。将 Ultralytics YOLOv8 与 Triton 推理服务器集成，可以部署可扩展、高性能的深度学习推理工作负载。本指南提供了设置和测试集成的步骤。

[`www.youtube.com/embed/NQDtfSi5QF4`](https://www.youtube.com/embed/NQDtfSi5QF4)

**观看：**开始使用 NVIDIA Triton 推理服务器。

## 什么是 Triton 推理服务器？

Triton 推理服务器旨在生产部署各种 AI 模型，支持 TensorFlow、PyTorch、ONNX Runtime 等广泛的深度学习和机器学习框架。其主要用例包括：

+   从单个服务器实例中服务多个模型。

+   动态模型加载和卸载，无需服务器重启。

+   集成推理，允许多个模型一起使用以实现结果。

+   为 A/B 测试和滚动更新进行模型版本控制。

## 先决条件

在继续之前，请确保具备以下先决条件：

+   您的机器上安装了 Docker。

+   安装`tritonclient`：

    ```py
    `pip  install  tritonclient[all]` 
    ```

## 将 YOLOv8 导出为 ONNX 格式

在将模型部署到 Triton 之前，必须将其导出为 ONNX 格式。ONNX（开放神经网络交换）是一种允许在不同深度学习框架之间转移模型的格式。使用`YOLO`类的`export`功能：

```py
`from ultralytics import YOLO  # Load a model model = YOLO("yolov8n.pt")  # load an official model  # Export the model onnx_file = model.export(format="onnx", dynamic=True)` 
```

## 设置 Triton 模型仓库

Triton 模型仓库是 Triton 可以访问和加载模型的存储位置。

1.  创建必要的目录结构：

    ```py
    `from pathlib import Path  # Define paths model_name = "yolo" triton_repo_path = Path("tmp") / "triton_repo" triton_model_path = triton_repo_path / model_name  # Create directories (triton_model_path / "1").mkdir(parents=True, exist_ok=True)` 
    ```

1.  将导出的 ONNX 模型移至 Triton 仓库：

    ```py
    `from pathlib import Path  # Move ONNX model to Triton Model path Path(onnx_file).rename(triton_model_path / "1" / "model.onnx")  # Create config file (triton_model_path / "config.pbtxt").touch()` 
    ```

## 运行 Triton 推理服务器

使用 Docker 运行 Triton 推理服务器：

```py
`import contextlib import subprocess import time  from tritonclient.http import InferenceServerClient  # Define image https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver tag = "nvcr.io/nvidia/tritonserver:23.09-py3"  # 6.4 GB  # Pull the image subprocess.call(f"docker pull {tag}", shell=True)  # Run the Triton server and capture the container ID container_id = (     subprocess.check_output(         f"docker run -d --rm -v {triton_repo_path}:/models -p 8000:8000 {tag} tritonserver --model-repository=/models",         shell=True,     )     .decode("utf-8")     .strip() )  # Wait for the Triton server to start triton_client = InferenceServerClient(url="localhost:8000", verbose=False, ssl=False)  # Wait until model is ready for _ in range(10):     with contextlib.suppress(Exception):         assert triton_client.is_model_ready(model_name)         break     time.sleep(1)` 
```

然后使用 Triton 服务器模型进行推理：

```py
`from ultralytics import YOLO  # Load the Triton Server model model = YOLO("http://localhost:8000/yolo", task="detect")  # Run inference on the server results = model("path/to/image.jpg")` 
```

清理容器：

```py
`# Kill and remove the container at the end of the test subprocess.call(f"docker kill {container_id}", shell=True)` 
```

* * *

遵循以上步骤，您可以在 Triton 推理服务器上高效部署和运行 Ultralytics YOLOv8 模型，为深度学习推理任务提供可扩展和高性能的解决方案。如果遇到任何问题或有进一步的疑问，请参阅[官方 Triton 文档](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html)或联系 Ultralytics 社区获取支持。

## 常见问题

### 如何设置 Ultralytics YOLOv8 与 NVIDIA Triton 推理服务器？

使用[NVIDIA Triton 推理服务器](https://developer.nvidia.com/nvidia-triton-inference-server)设置[Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8)涉及几个关键步骤：

1.  **将 YOLOv8 导出为 ONNX 格式**：

    ```py
    `from ultralytics import YOLO  # Load a model model = YOLO("yolov8n.pt")  # load an official model  # Export the model to ONNX format onnx_file = model.export(format="onnx", dynamic=True)` 
    ```

1.  **设置 Triton 模型仓库**：

    ```py
    `from pathlib import Path  # Define paths model_name = "yolo" triton_repo_path = Path("tmp") / "triton_repo" triton_model_path = triton_repo_path / model_name  # Create directories (triton_model_path / "1").mkdir(parents=True, exist_ok=True) Path(onnx_file).rename(triton_model_path / "1" / "model.onnx") (triton_model_path / "config.pbtxt").touch()` 
    ```

1.  **运行 Triton 服务器**：

    ```py
    `import contextlib import subprocess import time  from tritonclient.http import InferenceServerClient  # Define image https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver tag = "nvcr.io/nvidia/tritonserver:23.09-py3"  subprocess.call(f"docker pull {tag}", shell=True)  container_id = (     subprocess.check_output(         f"docker run -d --rm -v {triton_repo_path}/models -p 8000:8000 {tag} tritonserver --model-repository=/models",         shell=True,     )     .decode("utf-8")     .strip() )  triton_client = InferenceServerClient(url="localhost:8000", verbose=False, ssl=False)  for _ in range(10):     with contextlib.suppress(Exception):         assert triton_client.is_model_ready(model_name)         break     time.sleep(1)` 
    ```

此设置可帮助您高效地在 Triton 推断服务器上部署 YOLOv8 模型，用于高性能 AI 模型推断。

### 使用 Ultralytics YOLOv8 与 NVIDIA Triton 推断服务器有什么好处？

将[Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8)与[NVIDIA Triton Inference Server](https://developer.nvidia.com/nvidia-triton-inference-server)集成，具有多个优势：

+   **可扩展的 AI 推断**：Triton 允许从单个服务器实例中服务多个模型，支持动态模型的加载和卸载，因此对各种 AI 工作负载具有高度可扩展性。

+   **高性能**：针对 NVIDIA GPU 进行优化，Triton 推断服务器确保高速推断操作，非常适合实时目标检测等实时应用。

+   **集成和模型版本控制**：Triton 的集成模式允许组合多个模型以提高结果，其模型版本控制支持 A/B 测试和滚动更新。

有关设置和运行 YOLOv8 与 Triton 的详细说明，请参考设置指南。

### 为什么在使用 Triton 推断服务器之前需要将 YOLOv8 模型导出为 ONNX 格式？

在部署在[NVIDIA Triton Inference Server](https://developer.nvidia.com/nvidia-triton-inference-server)上之前，为您的 Ultralytics YOLOv8 模型使用 ONNX（开放神经网络交换格式）提供了几个关键的好处：

+   **互操作性**：ONNX 格式支持不同深度学习框架（如 PyTorch、TensorFlow）之间的转换，确保更广泛的兼容性。

+   **优化**：包括 Triton 在内的许多部署环境都为 ONNX 进行了优化，实现更快的推断和更好的性能。

+   **部署简便性**：ONNX 在各种操作系统和硬件配置中广泛支持，简化了部署过程。

要导出您的模型，请使用：

```py
`from ultralytics import YOLO  model = YOLO("yolov8n.pt") onnx_file = model.export(format="onnx", dynamic=True)` 
```

您可以按照导出指南中的步骤完成该过程。

### 我可以在 Triton 推断服务器上使用 Ultralytics YOLOv8 模型进行推断吗？

是的，您可以在[NVIDIA Triton Inference Server](https://developer.nvidia.com/nvidia-triton-inference-server)上运行 Ultralytics YOLOv8 模型进行推断。一旦您的模型设置在 Triton 模型存储库中并且服务器正在运行，您可以加载并运行推断模型如下：

```py
`from ultralytics import YOLO  # Load the Triton Server model model = YOLO("http://localhost:8000/yolo", task="detect")  # Run inference on the server results = model("path/to/image.jpg")` 
```

有关设置和运行 Triton 服务器与 YOLOv8 的深入指南，请参考运行 Triton 推断服务器部分。

### Ultralytics YOLOv8 在部署时与 TensorFlow 和 PyTorch 模型有何区别？

[Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8)相比于 TensorFlow 和 PyTorch 模型，在部署时提供了几个独特的优势：

+   **实时性能**：优化用于实时目标检测任务，YOLOv8 提供了最先进的精度和速度，非常适合需要实时视频分析的应用。

+   **易用性**：YOLOv8 与 Triton 推理服务器无缝集成，并支持多种导出格式（ONNX、TensorRT、CoreML），使其在各种部署场景下具备灵活性。

+   **高级功能**：YOLOv8 包括动态模型加载、模型版本管理和集成推理等功能，对于可扩展和可靠的 AI 部署至关重要。

有关更多详细信息，请比较模型部署指南中的部署选项。
