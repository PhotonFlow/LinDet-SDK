# Copyright 2024 LinDet Team. Apache-2.0 license.
"""ROS 2 pose estimation node using TensorRT.

This node subscribes to raw images and person detections, crops each
person bounding box, runs a TensorRT pose model, and publishes
PoseArray2D messages.

The TensorRT engine is loaded from a .engine file specified via the
'model_path' parameter.  The engine wrapper is fully isolated from ROS
so models can be swapped without touching node logic.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image
from lindet_msgs.msg import (
    Detection2DArray,
    Keypoint2D,
    Pose2D,
    PoseArray2D,
)

import message_filters


# ─── COCO keypoint skeleton ──────────────────────────────────────────────────

COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # head
    (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),  # arms
    (5, 11), (6, 12), (11, 12),  # torso
    (11, 13), (12, 14), (13, 15), (14, 16),  # legs
]

PERSON_CLASS_ID = 0


class PoseEngineStub:
    """Placeholder TensorRT pose engine.

    Replace this with actual TensorRT inference when deploying on Jetson.
    The interface is intentionally simple:
        infer(bgr_crop: np.ndarray) -> np.ndarray  shape (17, 3)  [x, y, conf]
    """

    def __init__(self, model_path: str = ""):
        self.loaded = False
        if model_path:
            self._load(model_path)

    def _load(self, path: str):
        """Load TensorRT engine from file.

        TODO: Implement with pycuda + tensorrt or use the C++ TRT wrapper
        exposed via pybind11.
        """
        try:
            # Placeholder: in production, deserialize the .engine file here
            import os
            if os.path.isfile(path):
                self.loaded = True
        except Exception:
            self.loaded = False

    def infer(self, bgr_crop: np.ndarray) -> np.ndarray:
        """Run pose estimation on a BGR crop.

        Args:
            bgr_crop: (H, W, 3) uint8 BGR image crop of a person.

        Returns:
            (17, 3) array of [x, y, confidence] for each COCO keypoint,
            coordinates normalized to [0, 1] relative to the crop.
        """
        if not self.loaded:
            # Return dummy keypoints at center for testing
            kpts = np.zeros((17, 3), dtype=np.float32)
            kpts[:, 0] = 0.5  # x center
            kpts[:, 1] = np.linspace(0.1, 0.9, 17)  # spread vertically
            kpts[:, 2] = 0.0  # zero confidence = not real
            return kpts

        # TODO: actual TRT inference
        # 1. Preprocess: resize to model input size, normalize
        # 2. Run inference
        # 3. Post-process: extract keypoints from heatmaps
        return np.zeros((17, 3), dtype=np.float32)


class PoseNode(Node):
    """Pose estimation node — subscribes to images + detections,
    publishes PoseArray2D."""

    def __init__(self):
        super().__init__("pose_node")

        # ── Parameters ─────────────────────────────────────────────────────
        self.declare_parameter("model_path", "")
        self.declare_parameter("keypoint_threshold", 0.3)
        self.declare_parameter("person_class_id", PERSON_CLASS_ID)

        model_path = self.get_parameter("model_path").value
        self.kpt_thresh = self.get_parameter("keypoint_threshold").value
        self.person_cls = self.get_parameter("person_class_id").value

        # ── Engine ─────────────────────────────────────────────────────────
        self.engine = PoseEngineStub(model_path)
        if self.engine.loaded:
            self.get_logger().info(f"Pose engine loaded: {model_path}")
        else:
            self.get_logger().warn(
                "No pose model loaded — running in stub mode. "
                "Set 'model_path' parameter to enable inference."
            )

        # ── Synchronized subscribers ───────────────────────────────────────
        self.sub_image = message_filters.Subscriber(
            self, Image, "/camera/image_raw",
            qos_profile=qos_profile_sensor_data,
        )
        self.sub_dets = message_filters.Subscriber(
            self, Detection2DArray, "/detection/detections",
            qos_profile=qos_profile_sensor_data,
        )

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.sub_image, self.sub_dets],
            queue_size=10,
            slop=0.05,
        )
        self.sync.registerCallback(self.synced_callback)

        # ── Publisher ──────────────────────────────────────────────────────
        self.pub = self.create_publisher(
            PoseArray2D, "~/poses", qos_profile_sensor_data
        )

        self.get_logger().info("Pose node started")

    def synced_callback(self, img_msg: Image, det_msg: Detection2DArray):
        """Process synchronized image + detections."""
        # Convert image to numpy
        if img_msg.encoding == "bgr8":
            img = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(
                img_msg.height, img_msg.width, 3
            )
        elif img_msg.encoding == "rgb8":
            raw = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(
                img_msg.height, img_msg.width, 3
            )
            img = raw[:, :, ::-1]  # RGB → BGR
        else:
            self.get_logger().warn_throttle(
                self.get_clock(), 2.0,
                f"Unsupported encoding: {img_msg.encoding}"
            )
            return

        h, w = img.shape[:2]

        # Filter person detections
        person_dets = [
            d for d in det_msg.detections
            if d.class_id == self.person_cls
        ]

        out = PoseArray2D()
        out.header = img_msg.header

        for det in person_dets:
            # Convert normalized coords to pixel coords
            cx = det.x_center * w
            cy = det.y_center * h
            bw = det.width * w
            bh = det.height * h

            x1 = max(0, int(cx - bw / 2))
            y1 = max(0, int(cy - bh / 2))
            x2 = min(w, int(cx + bw / 2))
            y2 = min(h, int(cy + bh / 2))

            if x2 <= x1 or y2 <= y1:
                continue

            crop = img[y1:y2, x1:x2]
            kpts = self.engine.infer(crop)  # (17, 3)

            pose = Pose2D()
            pose.header = img_msg.header
            pose.track_id = -1  # will be linked by visualization
            pose.bbox_x = det.x_center
            pose.bbox_y = det.y_center
            pose.bbox_w = det.width
            pose.bbox_h = det.height

            for i, (kx, ky, kc) in enumerate(kpts):
                kp = Keypoint2D()
                # Convert crop-relative coords to image-relative normalized
                kp.x = (x1 + kx * (x2 - x1)) / w
                kp.y = (y1 + ky * (y2 - y1)) / h
                kp.confidence = float(kc)
                kp.id = i
                pose.keypoints.append(kp)

            out.poses.append(pose)

        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = PoseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
