# 使用 Ultralytics YOLOv8 进行对象模糊 🚀

> 原文：[`docs.ultralytics.com/guides/object-blurring/`](https://docs.ultralytics.com/guides/object-blurring/)

## 什么是对象模糊？

使用[Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics/)进行对象模糊处理涉及对图像或视频中特定检测到的对象应用模糊效果。利用 YOLOv8 模型的能力来识别和操作给定场景中的对象，从而实现此目的。

[`www.youtube.com/embed/ydGdibB5Mds`](https://www.youtube.com/embed/ydGdibB5Mds)

**观看：** 使用 Ultralytics YOLOv8 进行对象模糊

## 对象模糊的优势？

+   **隐私保护**：对象模糊是通过在图像或视频中隐藏敏感或个人可识别信息来有效保护隐私的工具。

+   **选择性焦点**：YOLOv8 允许选择性模糊，使用户能够针对特定对象进行模糊处理，从而在隐私保护和保留相关视觉信息之间取得平衡。

+   **实时处理**：YOLOv8 的高效性使其能够在实时中进行对象模糊处理，适用于需要在动态环境中进行即时隐私增强的应用。

使用 YOLOv8 示例进行对象模糊处理

```py
`import cv2  from ultralytics import YOLO from ultralytics.utils.plotting import Annotator, colors  model = YOLO("yolov8n.pt") names = model.names  cap = cv2.VideoCapture("path/to/video/file.mp4") assert cap.isOpened(), "Error reading video file" w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))  # Blur ratio blur_ratio = 50  # Video writer video_writer = cv2.VideoWriter("object_blurring_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))  while cap.isOpened():     success, im0 = cap.read()     if not success:         print("Video frame is empty or video processing has been successfully completed.")         break      results = model.predict(im0, show=False)     boxes = results[0].boxes.xyxy.cpu().tolist()     clss = results[0].boxes.cls.cpu().tolist()     annotator = Annotator(im0, line_width=2, example=names)      if boxes is not None:         for box, cls in zip(boxes, clss):             annotator.box_label(box, color=colors(int(cls), True), label=names[int(cls)])              obj = im0[int(box[1]) : int(box[3]), int(box[0]) : int(box[2])]             blur_obj = cv2.blur(obj, (blur_ratio, blur_ratio))              im0[int(box[1]) : int(box[3]), int(box[0]) : int(box[2])] = blur_obj      cv2.imshow("ultralytics", im0)     video_writer.write(im0)     if cv2.waitKey(1) & 0xFF == ord("q"):         break  cap.release() video_writer.release() cv2.destroyAllWindows()` 
```

### 参数 `model.predict`

| 名称 | 类型 | 默认值 | 描述 |
| --- | --- | --- | --- |
| `source` | `str` | `'ultralytics/assets'` | 图像或视频的源目录 |
| `conf` | `float` | `0.25` | 检测的对象置信度阈值 |
| `iou` | `float` | `0.7` | NMS 的交并比（IoU）阈值 |
| `imgsz` | `int or tuple` | `640` | 图像大小，可以是标量或（h, w）列表，例如（640, 480） |
| `half` | `bool` | `False` | 使用半精度（FP16） |
| `device` | `None or str` | `None` | 运行设备，例如 cuda device=0/1/2/3 或 device=cpu |
| `max_det` | `int` | `300` | 每张图像的最大检测数 |
| `vid_stride` | `bool` | `False` | 视频帧率步进 |
| `stream_buffer` | `bool` | `False` | 缓冲所有流帧（True），或返回最近的帧（False） |
| `visualize` | `bool` | `False` | 可视化模型特征 |
| `augment` | `bool` | `False` | 对预测来源应用图像增强 |
| `agnostic_nms` | `bool` | `False` | 无类别 NMS |
| `classes` | `list[int]` | `None` | 按类别过滤结果，例如 classes=0 或 classes=[0,2,3] |
| `retina_masks` | `bool` | `False` | 使用高分辨率分割掩模 |
| `embed` | `list[int]` | `None` | 返回指定层的特征向量/嵌入 |

## 常见问题解答

### 什么是使用 Ultralytics YOLOv8 的对象模糊？

使用[Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics/)进行对象模糊涉及自动检测并对图像或视频中的特定对象应用模糊效果。此技术通过隐藏敏感信息同时保留相关视觉数据，增强隐私。YOLOv8 的实时处理能力使其适用于需要动态环境中即时隐私增强和选择性聚焦调整的应用。

### 如何使用 YOLOv8 实现实时物体模糊？

要使用 YOLOv8 实现实时物体模糊，请参考提供的 Python 示例。这涉及使用 YOLOv8 进行物体检测和 OpenCV 应用模糊效果。以下是简化版本：

```py
`import cv2  from ultralytics import YOLO  model = YOLO("yolov8n.pt") cap = cv2.VideoCapture("path/to/video/file.mp4")  while cap.isOpened():     success, im0 = cap.read()     if not success:         break      results = model.predict(im0, show=False)     for box in results[0].boxes.xyxy.cpu().tolist():         obj = im0[int(box[1]) : int(box[3]), int(box[0]) : int(box[2])]         im0[int(box[1]) : int(box[3]), int(box[0]) : int(box[2])] = cv2.blur(obj, (50, 50))      cv2.imshow("YOLOv8 Blurring", im0)     if cv2.waitKey(1) & 0xFF == ord("q"):         break  cap.release() cv2.destroyAllWindows()` 
```

### 使用 Ultralytics 的 YOLOv8 进行物体模糊的好处是什么？

Ultralytics 的 YOLOv8 在物体模糊方面具有多个优势：

+   **隐私保护**：有效模糊敏感或可识别信息。

+   **选择性焦点**：针对特定物体进行模糊，保持基本的视觉内容。

+   **实时处理**：在动态环境中高效执行物体模糊，适合即时隐私增强。

欲了解更详细的应用，请查看物体模糊部分的优势。

### 我可以使用 Ultralytics 的 YOLOv8 在视频中模糊面部以保护隐私吗？

是的，Ultralytics 的 YOLOv8 可以配置为检测和模糊视频中的面部以保护隐私。通过训练或使用预训练模型来专门识别面部，检测结果可以通过 OpenCV 处理以应用模糊效果。请参考我们关于[使用 YOLOv8 进行物体检测](https://docs.ultralytics.com/models/yolov8)的指南，并修改代码以针对面部检测。

### YOLOv8 与其他物体检测模型（如 Faster R-CNN）在物体模糊方面有何区别？

Ultralytics 的 YOLOv8 通常在速度方面优于 Faster R-CNN 等模型，使其更适合实时应用。虽然两种模型都提供准确的检测，但 YOLOv8 的架构针对快速推断进行了优化，这对于实时物体模糊等任务至关重要。详细了解技术差异和性能指标，请参阅我们的[YOLOv8 文档](https://docs.ultralytics.com/models/yolov8)。
