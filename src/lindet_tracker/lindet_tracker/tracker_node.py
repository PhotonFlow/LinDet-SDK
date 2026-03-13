# Copyright 2024 LinDet Team. Apache-2.0 license.
"""ROS 2 tracker node — wraps ByteTrack for multi-object tracking."""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from lindet_msgs.msg import (
    Detection2DArray,
    TrackedObject2D,
    TrackedObject2DArray,
)
from lindet_tracker.byte_track import ByteTrack


class TrackerNode(Node):
    """Multi-object tracker node using ByteTrack."""

    def __init__(self):
        super().__init__("tracker_node")

        # ── Parameters ─────────────────────────────────────────────────────
        self.declare_parameter("high_thresh", 0.5)
        self.declare_parameter("low_thresh", 0.1)
        self.declare_parameter("match_thresh", 0.3)
        self.declare_parameter("max_age", 30)
        self.declare_parameter("min_hits", 3)

        # ── Initialize tracker ─────────────────────────────────────────────
        self.tracker = ByteTrack(
            high_thresh=self.get_parameter("high_thresh").value,
            low_thresh=self.get_parameter("low_thresh").value,
            match_thresh=self.get_parameter("match_thresh").value,
            max_age=self.get_parameter("max_age").value,
            min_hits=self.get_parameter("min_hits").value,
        )

        # ── Pub / Sub ──────────────────────────────────────────────────────
        self.sub_ = self.create_subscription(
            Detection2DArray,
            "/detection/detections",
            self.detection_callback,
            qos_profile_sensor_data,
        )

        self.pub_ = self.create_publisher(
            TrackedObject2DArray,
            "~/tracked_objects",
            qos_profile_sensor_data,
        )

        self.get_logger().info("Tracker node started (ByteTrack)")

    def detection_callback(self, msg: Detection2DArray):
        """Process detections and publish tracked objects."""
        if len(msg.detections) == 0:
            # Still publish empty message to maintain header chain
            out = TrackedObject2DArray()
            out.header = msg.header
            self.pub_.publish(out)
            return

        # Convert to numpy arrays
        bboxes = np.array(
            [[d.x_center, d.y_center, d.width, d.height] for d in msg.detections],
            dtype=np.float32,
        )
        class_ids = np.array([d.class_id for d in msg.detections], dtype=np.int32)
        confidences = np.array(
            [d.confidence for d in msg.detections], dtype=np.float32
        )

        # Run tracker
        confirmed_tracks = self.tracker.update(bboxes, class_ids, confidences)

        # Build output message
        out = TrackedObject2DArray()
        out.header = msg.header

        for track in confirmed_tracks:
            tobj = TrackedObject2D()
            tobj.detection.header = msg.header
            tobj.detection.x_center = float(track.bbox[0])
            tobj.detection.y_center = float(track.bbox[1])
            tobj.detection.width = float(track.bbox[2])
            tobj.detection.height = float(track.bbox[3])
            tobj.detection.confidence = track.confidence
            tobj.detection.class_id = track.class_id
            tobj.track_id = track.track_id
            tobj.age = track.age
            vel = track.velocity
            tobj.vx = float(vel[0])
            tobj.vy = float(vel[1])
            out.tracked_objects.append(tobj)

        self.pub_.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = TrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
