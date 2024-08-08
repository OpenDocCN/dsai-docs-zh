# 姿势估计

> 原文：[`docs.ultralytics.com/tasks/pose/`](https://docs.ultralytics.com/tasks/pose/)

![姿势估计示例](img/48a7c9a6d42399ce7eddc7553488109a.png)

姿势估计是一项任务，涉及在图像中识别特定点的位置，通常称为关键点。关键点可以代表对象的各种部位，如关节、地标或其他显著特征。关键点的位置通常表示为一组 2D `[x, y]` 或 3D `[x, y, visible]`坐标。

姿势估计模型的输出是一组点，表示图像中对象的关键点，通常还包括每个点的置信度分数。当您需要识别场景中对象特定部分及其相互位置时，姿势估计是一个不错的选择。

|

[`www.youtube.com/embed/Y28xXQmju64?si=pCY4ZwejZFu6Z4kZ`](https://www.youtube.com/embed/Y28xXQmju64?si=pCY4ZwejZFu6Z4kZ)

**观看：** 与 Ultralytics YOLOv8 一起的姿势估计。

[`www.youtube.com/embed/aeAX6vWpfR0`](https://www.youtube.com/embed/aeAX6vWpfR0)

**观看：** 与 Ultralytics HUB 一起的姿势估计。

提示

YOLOv8 *pose*模型使用`-pose`后缀，例如`yolov8n-pose.pt`。这些模型是在[COCO 关键点](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco-pose.yaml)数据集上训练的，适用于各种姿势估计任务。

在默认的 YOLOv8 姿势模型中，有 17 个关键点，每个代表人体不同部位。以下是每个索引与其对应身体关节的映射：

0: 鼻子 1: 左眼 2: 右眼 3: 左耳 4: 右耳 5: 左肩 6: 右肩 7: 左肘 8: 右肘 9: 左腕 10: 右腕 11: 左髋 12: 右髋 13: 左膝 14: 右膝 15: 左踝 16: 右踝

## [模型](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/v8)

YOLOv8 预训练的姿势模型显示在这里。检测、分割和姿势模型在[COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml)数据集上进行了预训练，而分类模型则在[ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml)数据集上进行了预训练。

[模型](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models)将在首次使用时自动从最新的 Ultralytics [发布](https://github.com/ultralytics/assets/releases)中下载。

| 模型 | 大小 ^((像素)) | mAP^(姿势 50-95) | mAP^(姿势 50) | 速度 ^(CPU ONNX

(ms)) | 速度 ^(A100 TensorRT

(ms)) | 参数 ^((M)) | FLOPs ^((B)) |

| --- | --- | --- | --- | --- | --- | --- | --- |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [YOLOv8n-pose](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-pose.pt) | 640 | 50.4 | 80.1 | 131.8 | 1.18 | 3.3 | 9.2 |
| [YOLOv8s-pose](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s-pose.pt) | 640 | 60.0 | 86.2 | 233.2 | 1.42 | 11.6 | 30.2 |
| [YOLOv8m-pose](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m-pose.pt) | 640 | 65.0 | 88.8 | 456.3 | 2.00 | 26.4 | 81.0 |
| [YOLOv8l-pose](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8l-pose.pt) | 640 | 67.6 | 90.0 | 784.5 | 2.59 | 44.4 | 168.6 |
| [YOLOv8x-pose](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x-pose.pt) | 640 | 69.2 | 90.2 | 1607.1 | 3.73 | 69.4 | 263.2 |
| [YOLOv8x-pose-p6](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x-pose-p6.pt) | 1280 | 71.6 | 91.2 | 4088.7 | 10.04 | 99.1 | 1066.4 |

+   **mAP^(val)**值适用于[COCO 关键点 val2017](https://cocodataset.org)数据集上的单一模型单一尺度。

    通过`yolo val pose data=coco-pose.yaml device=0`重现

+   **速度**是使用[Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/)实例对 COCO 验证图像进行平均化。

    通过`yolo val pose data=coco8-pose.yaml batch=1 device=0|cpu`重现

## 训练

在 COCO128-pose 数据集上训练 YOLOv8-pose 模型。

示例

```py
`from ultralytics import YOLO  # Load a model model = YOLO("yolov8n-pose.yaml")  # build a new model from YAML model = YOLO("yolov8n-pose.pt")  # load a pretrained model (recommended for training) model = YOLO("yolov8n-pose.yaml").load("yolov8n-pose.pt")  # build from YAML and transfer weights  # Train the model results = model.train(data="coco8-pose.yaml", epochs=100, imgsz=640)` 
```

```py
`# Build a new model from YAML and start training from scratch yolo  pose  train  data=coco8-pose.yaml  model=yolov8n-pose.yaml  epochs=100  imgsz=640  # Start training from a pretrained *.pt model yolo  pose  train  data=coco8-pose.yaml  model=yolov8n-pose.pt  epochs=100  imgsz=640  # Build a new model from YAML, transfer pretrained weights to it and start training yolo  pose  train  data=coco8-pose.yaml  model=yolov8n-pose.yaml  pretrained=yolov8n-pose.pt  epochs=100  imgsz=640` 
```

### 数据集格式

YOLO 姿势数据集格式详见数据集指南。要将现有数据集（如 COCO 等）转换为 YOLO 格式，请使用[Ultralytics 的 JSON2YOLO](https://github.com/ultralytics/JSON2YOLO)工具。

## 验证

在 COCO128-pose 数据集上验证训练的 YOLOv8n-pose 模型准确性。作为模型属性，不需要传递任何参数。

示例

```py
`from ultralytics import YOLO  # Load a model model = YOLO("yolov8n-pose.pt")  # load an official model model = YOLO("path/to/best.pt")  # load a custom model  # Validate the model metrics = model.val()  # no arguments needed, dataset and settings remembered metrics.box.map  # map50-95 metrics.box.map50  # map50 metrics.box.map75  # map75 metrics.box.maps  # a list contains map50-95 of each category` 
```

```py
`yolo  pose  val  model=yolov8n-pose.pt  # val official model yolo  pose  val  model=path/to/best.pt  # val custom model` 
```

## 预测

使用训练的 YOLOv8n-pose 模型对图像进行预测。

示例

```py
`from ultralytics import YOLO  # Load a model model = YOLO("yolov8n-pose.pt")  # load an official model model = YOLO("path/to/best.pt")  # load a custom model  # Predict with the model results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image` 
```

```py
`yolo  pose  predict  model=yolov8n-pose.pt  source='https://ultralytics.com/images/bus.jpg'  # predict with official model yolo  pose  predict  model=path/to/best.pt  source='https://ultralytics.com/images/bus.jpg'  # predict with custom model` 
```

在预测页面查看完整的`predict`模式详细信息。

## 导出

将 YOLOv8n Pose 模型导出到 ONNX、CoreML 等不同格式。

示例

```py
`from ultralytics import YOLO  # Load a model model = YOLO("yolov8n-pose.pt")  # load an official model model = YOLO("path/to/best.pt")  # load a custom trained model  # Export the model model.export(format="onnx")` 
```

```py
`yolo  export  model=yolov8n-pose.pt  format=onnx  # export official model yolo  export  model=path/to/best.pt  format=onnx  # export custom trained model` 
```

可用的 YOLOv8-pose 导出格式如下表所示。您可以使用`format`参数导出到任何格式，例如`format='onnx'`或`format='engine'`。导出完成后，您可以直接在导出的模型上进行预测或验证，例如`yolo predict model=yolov8n-pose.onnx`。使用示例在导出模型后显示您的模型。

| 格式 | `format`参数 | 模型 | 元数据 | 参数 |
| --- | --- | --- | --- | --- |
| [PyTorch](https://pytorch.org/) | - | `yolov8n-pose.pt` | ✅ | - |
| TorchScript | `torchscript` | `yolov8n-pose.torchscript` | ✅ | `imgsz`, `optimize`, `batch` |
| ONNX | `onnx` | `yolov8n-pose.onnx` | ✅ | `imgsz`, `half`, `dynamic`, `simplify`, `opset`, `batch` |
| OpenVINO | `openvino` | `yolov8n-pose_openvino_model/` | ✅ | `imgsz`, `half`, `int8`, `batch`, `dynamic` |
| TensorRT | `engine` | `yolov8n-pose.engine` | ✅ | `imgsz`, `half`, `dynamic`, `simplify`, `workspace`, `int8`, `batch` |
| CoreML | `coreml` | `yolov8n-pose.mlpackage` | ✅ | `imgsz`, `half`, `int8`, `nms`, `batch` |
| TF SavedModel | `saved_model` | `yolov8n-pose_saved_model/` | ✅ | `imgsz`, `keras`, `int8`, `batch` |
| TF GraphDef | `pb` | `yolov8n-pose.pb` | ❌ | `imgsz`, `batch` |
| TF Lite | `tflite` | `yolov8n-pose.tflite` | ✅ | `imgsz`, `half`, `int8`, `batch` |
| TF Edge TPU | `edgetpu` | `yolov8n-pose_edgetpu.tflite` | ✅ | `imgsz` |
| TF.js | `tfjs` | `yolov8n-pose_web_model/` | ✅ | `imgsz`, `half`, `int8`, `batch` |
| PaddlePaddle | `paddle` | `yolov8n-pose_paddle_model/` | ✅ | `imgsz`, `batch` |
| NCNN | `ncnn` | `yolov8n-pose_ncnn_model/` | ✅ | `imgsz`, `half`, `batch` |

请参阅导出页面以查看完整的导出细节。

## 常见问题解答

### 什么是使用 Ultralytics YOLOv8 进行姿势估计，它是如何工作的？

使用 Ultralytics YOLOv8 进行姿势估计涉及在图像中识别特定点，称为关键点。这些关键点通常代表对象的关节或其他重要特征。输出包括每个点的`[x, y]`坐标和置信度分数。YOLOv8-pose 模型专门设计用于此任务，并使用`-pose`后缀，如`yolov8n-pose.pt`。这些模型预先在数据集（如[COCO 关键点](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco-pose.yaml)）上进行了训练，并可用于各种姿势估计任务。欲了解更多信息，请访问姿势估计页面。

### 如何在自定义数据集上训练 YOLOv8-pose 模型？

在自定义数据集上训练 YOLOv8-pose 模型涉及加载模型，可以是由 YAML 文件定义的新模型或预训练模型。然后，您可以使用指定的数据集和参数开始训练过程。

```py
`from ultralytics import YOLO  # Load a model model = YOLO("yolov8n-pose.yaml")  # build a new model from YAML model = YOLO("yolov8n-pose.pt")  # load a pretrained model (recommended for training)  # Train the model results = model.train(data="your-dataset.yaml", epochs=100, imgsz=640)` 
```

欲了解有关训练的详细信息，请参阅训练部分。

### 如何验证已训练的 YOLOv8-pose 模型？

验证 YOLOv8-pose 模型涉及使用在训练期间保留的相同数据集参数来评估其准确性。以下是一个示例：

```py
`from ultralytics import YOLO  # Load a model model = YOLO("yolov8n-pose.pt")  # load an official model model = YOLO("path/to/best.pt")  # load a custom model  # Validate the model metrics = model.val()  # no arguments needed, dataset and settings remembered` 
```

欲了解更多信息，请访问验证部分。

### 我可以将 YOLOv8-pose 模型导出为其他格式吗？如何操作？

是的，您可以将 YOLOv8-pose 模型导出为 ONNX、CoreML、TensorRT 等各种格式。可以使用 Python 或命令行界面（CLI）来完成此操作。

```py
`from ultralytics import YOLO  # Load a model model = YOLO("yolov8n-pose.pt")  # load an official model model = YOLO("path/to/best.pt")  # load a custom trained model  # Export the model model.export(format="onnx")` 
```

欲了解更多详细信息，请参阅导出部分。

### 可用的 Ultralytics YOLOv8-pose 模型及其性能指标是什么？

Ultralytics YOLOv8 提供各种预训练姿势模型，如 YOLOv8n-pose、YOLOv8s-pose、YOLOv8m-pose 等。这些模型在尺寸、准确性（mAP）和速度上有所不同。例如，YOLOv8n-pose 模型实现了 mAP^(pose)50-95 为 50.4 和 mAP^(pose)50 为 80.1。有关完整列表和性能详细信息，请访问模型部分。
