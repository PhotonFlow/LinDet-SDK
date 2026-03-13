# LinDet-SDK

**Real-Time Vision Pipeline for NVIDIA Jetson Nano**

A production-grade ROS 2 Humble workspace providing real-time object detection, multi-object tracking, pose estimation, and downstream analytics — all running with zero-copy intra-process communication and TensorRT-accelerated inference.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ComposableNodeContainer (zero-copy)                      │
│                                                                             │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                │
│  │  GStreamer    │────▶│  TensorRT    │────▶│  ByteTrack   │                │
│  │  Camera Node │     │  Detection   │     │  Tracker     │                │
│  │  (C++)       │     │  (C++)       │     │  (Python)    │                │
│  └──────┬───────┘     └──────┬───────┘     └──────┬───────┘                │
│         │                    │                     │                        │
│         │              ┌─────▼──────┐        ┌─────▼──────┐                │
│         │              │   Pose     │        │  Analytics  │                │
│         │              │ Estimation │        │  (Zone/Line │                │
│         │              │  (Python)  │        │   /Speed)   │                │
│         │              └─────┬──────┘        └─────┬───────┘               │
│         │                    │                     │                        │
│         └─────────┬──────────┴─────────────────────┘                       │
│                   ▼                                                         │
│          ┌──────────────┐                                                  │
│          │ Visualization│                                                  │
│          │   (Python)   │                                                  │
│          └──────────────┘                                                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Features

| Feature | Description |
|---|---|
| **Object Detection** | TensorRT-accelerated YOLOv8 inference (person, car, motorcycle, bus, truck) |
| **Multi-Object Tracking** | ByteTrack with Kalman filter + two-stage IoU association |
| **Pose Estimation** | Top-down keypoint detection on person crops (17 COCO keypoints) |
| **Zone Intrusion** | Polygon-based geofencing with entry/exit detection |
| **Line Counting** | Virtual counting line with directional in/out counts |
| **Speed Estimation** | Track velocity with optional homography calibration |
| **Visualization** | Real-time overlay with bboxes, skeletons, track IDs, and analytics |

## Packages

| Package | Type | Description |
|---|---|---|
| `lindet_msgs` | C++ (msgs) | Custom message definitions |
| `lindet_camera` | C++ | GStreamer camera driver (CSI/USB/RTSP/test) |
| `lindet_detection` | C++ | TensorRT object detection |
| `lindet_tracker` | Python | ByteTrack multi-object tracker |
| `lindet_pose` | Python | TensorRT pose estimation |
| `lindet_analytics` | Python | Zone intrusion, line counting, speed estimation |
| `lindet_visualization` | Python | Overlay rendering |
| `lindet_bringup` | C++ | Launch files |

## Quick Start

### 1. Prepare Models

```bash
# On the Jetson Nano (or matching architecture):
pip install ultralytics
yolo export model=yolov8n.pt format=engine half=True imgsz=640
yolo export model=yolov8n-pose.pt format=engine half=True imgsz=640
cp yolov8n.engine yolov8n-pose.engine models/
```

### 2. Build & Run with Docker

```bash
cd docker
docker compose up --build
```

### 3. Run with Custom Camera

```bash
# USB camera
docker compose run lindet ros2 launch lindet_bringup lindet_system.launch.py \
    source_type:=usb device:=/dev/video0 \
    model_path:=/lindet_ws/models/yolov8n.engine

# CSI camera (Jetson)
docker compose run lindet ros2 launch lindet_bringup lindet_system.launch.py \
    source_type:=csi \
    model_path:=/lindet_ws/models/yolov8n.engine

# RTSP stream
docker compose run lindet ros2 launch lindet_bringup lindet_system.launch.py \
    source_type:=rtsp device:=rtsp://192.168.1.100:554/stream \
    model_path:=/lindet_ws/models/yolov8n.engine
```

### 4. View Topics

```bash
# List all topics
ros2 topic list

# Expected:
#   /camera/image_raw
#   /detection/detections
#   /tracker/tracked_objects
#   /pose/poses
#   /analytics/events
#   /visualization/image

# Monitor detection rate
ros2 topic hz /detection/detections
```

## Configuration

All parameters are configurable via YAML files in `config/`:

| File | Description |
|---|---|
| `camera.yaml` | Source type, resolution, FPS |
| `detection.yaml` | Model path, confidence/NMS thresholds |
| `tracker.yaml` | ByteTrack parameters |
| `pose.yaml` | Pose model path, keypoint threshold |
| `analytics.yaml` | Zone polygons, counting lines, speed threshold |

## Requirements

- NVIDIA Jetson Nano (JetPack 4.6 / L4T R32.7.1)
- Docker with NVIDIA runtime
- ROS 2 Humble
- TensorRT 8.x, CUDA 10.2

## License

Apache-2.0
