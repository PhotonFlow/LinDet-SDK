# Model Preparation Guide

This directory is where you place your TensorRT `.engine` files.

## Object Detection (YOLOv8n)

YOLOv8n is pre-trained on COCO-80, which includes:
- **person** (class 0)
- **car** (class 2), **motorcycle** (class 3), **bus** (class 5), **truck** (class 7)

### Export to TensorRT

```bash
# Install ultralytics (on the Jetson or a compatible machine)
pip install ultralytics

# Export YOLOv8n to TensorRT FP16 engine
yolo export model=yolov8n.pt format=engine half=True imgsz=640

# Move the engine file here
mv yolov8n.engine /lindet_ws/models/yolov8n.engine
```

> **Important:** The `.engine` file must be built on the **same GPU architecture**
> as the deployment target. Build on the Jetson Nano directly, or use
> `docker buildx` with matching CUDA/TRT versions.

## Pose Estimation (YOLOv8n-pose)

```bash
yolo export model=yolov8n-pose.pt format=engine half=True imgsz=640
mv yolov8n-pose.engine /lindet_ws/models/yolov8n-pose.engine
```

## Custom Models

The engine wrapper accepts any TensorRT `.engine` file with:
- **Detection**: Input `[1, 3, H, W]`, output `[1, 4+num_classes, N]` (YOLOv8 format)
- **Pose**: Input `[1, 3, H, W]`, output keypoints `(17, 3)` per person

Update `config/detection.yaml` and `config/pose.yaml` with the model path.
