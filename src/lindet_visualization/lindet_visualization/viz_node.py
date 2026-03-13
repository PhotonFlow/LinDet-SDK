# Copyright 2024 LinDet Team. Apache-2.0 license.
"""ROS 2 visualization node — draws detection, tracking, pose, and analytics
overlays on the camera image."""

import json
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image
from lindet_msgs.msg import (
    AnalyticsEvent,
    PoseArray2D,
    TrackedObject2DArray,
)

import message_filters

from lindet_visualization.drawing import (
    draw_bbox,
    draw_skeleton,
    get_class_color,
    get_track_color,
)


class VizNode(Node):
    """Visualization overlay node.

    Subscribes to:
      - /camera/image_raw (sensor_msgs/Image)
      - /tracker/tracked_objects (TrackedObject2DArray)
      - /pose/poses (PoseArray2D)

    Publishes:
      - ~/image (sensor_msgs/Image) with all overlays drawn
    """

    def __init__(self):
        super().__init__("viz_node")

        # ── Parameters ─────────────────────────────────────────────────────
        self.declare_parameter("draw_detections", True)
        self.declare_parameter("draw_tracks", True)
        self.declare_parameter("draw_poses", True)
        self.declare_parameter("keypoint_threshold", 0.3)

        self.draw_dets = self.get_parameter("draw_detections").value
        self.draw_tracks = self.get_parameter("draw_tracks").value
        self.draw_poses = self.get_parameter("draw_poses").value
        self.kpt_thresh = self.get_parameter("keypoint_threshold").value

        # ── Synchronized subscribers ───────────────────────────────────────
        self.sub_image = message_filters.Subscriber(
            self, Image, "/camera/image_raw",
            qos_profile=qos_profile_sensor_data,
        )
        self.sub_tracks = message_filters.Subscriber(
            self, TrackedObject2DArray, "/tracker/tracked_objects",
            qos_profile=qos_profile_sensor_data,
        )
        self.sub_poses = message_filters.Subscriber(
            self, PoseArray2D, "/pose/poses",
            qos_profile=qos_profile_sensor_data,
        )

        # Approximate time sync — tolerates slight misalignment
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.sub_image, self.sub_tracks, self.sub_poses],
            queue_size=10,
            slop=0.1,
        )
        self.sync.registerCallback(self.synced_callback)

        # Analytics events (not time-synced, just cached)
        self.latest_events: list = []
        self.sub_events = self.create_subscription(
            AnalyticsEvent,
            "/analytics/events",
            self.event_callback,
            qos_profile_sensor_data,
        )

        # ── Publisher ──────────────────────────────────────────────────────
        self.pub = self.create_publisher(
            Image, "~/image", qos_profile_sensor_data
        )

        self.get_logger().info("Visualization node started")

    def event_callback(self, msg: AnalyticsEvent):
        """Cache latest analytics events for overlay."""
        self.latest_events.append(msg)
        # Keep only last 20 events
        if len(self.latest_events) > 20:
            self.latest_events = self.latest_events[-20:]

    def synced_callback(
        self,
        img_msg: Image,
        tracks_msg: TrackedObject2DArray,
        poses_msg: PoseArray2D,
    ):
        """Draw all overlays on the synchronized image."""
        # Convert to numpy
        if img_msg.encoding == "bgr8":
            img = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(
                img_msg.height, img_msg.width, 3
            ).copy()
        elif img_msg.encoding == "rgb8":
            raw = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(
                img_msg.height, img_msg.width, 3
            )
            img = raw[:, :, ::-1].copy()
        else:
            return

        # ── Draw tracked objects with bboxes ───────────────────────────────
        if self.draw_tracks:
            for tobj in tracks_msg.tracked_objects:
                det = tobj.detection
                color = get_track_color(tobj.track_id)
                cls_name = det.class_name if det.class_name else f"cls{det.class_id}"
                label = f"ID:{tobj.track_id} {cls_name} {det.confidence:.2f}"
                draw_bbox(img, det.x_center, det.y_center,
                          det.width, det.height, color, label)

        # ── Draw pose skeletons ────────────────────────────────────────────
        if self.draw_poses:
            for pose in poses_msg.poses:
                color = get_track_color(pose.track_id) if pose.track_id >= 0 else (0, 255, 0)
                draw_skeleton(img, pose.keypoints, color, self.kpt_thresh)

        # ── Draw analytics event indicators ────────────────────────────────
        try:
            import cv2
            y_offset = 30
            for ev in self.latest_events[-5:]:
                text = f"[{ev.event_type}] T{ev.track_id}"
                cv2.putText(img, text, (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (0, 255, 255), 1, cv2.LINE_AA)
                y_offset += 18
        except ImportError:
            pass

        # ── Publish ────────────────────────────────────────────────────────
        out = Image()
        out.header = img_msg.header
        out.height = img.shape[0]
        out.width = img.shape[1]
        out.encoding = "bgr8"
        out.is_bigendian = False
        out.step = img.shape[1] * 3
        out.data = img.tobytes()

        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = VizNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
