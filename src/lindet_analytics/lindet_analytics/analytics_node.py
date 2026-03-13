# Copyright 2024 LinDet Team. Apache-2.0 license.
"""ROS 2 analytics node — runs pluggable analytics modules on tracked objects."""

import json
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from lindet_msgs.msg import AnalyticsEvent, TrackedObject2DArray

from lindet_analytics.zone_intrusion import ZoneIntrusion
from lindet_analytics.line_counter import LineCounter
from lindet_analytics.speed_estimator import SpeedEstimator


class AnalyticsNode(Node):
    """Analytics node — subscribes to tracked objects and publishes events."""

    def __init__(self):
        super().__init__("analytics_node")

        # ── Parameters ─────────────────────────────────────────────────────
        # Zone polygons: JSON string {"zone_name": [[x1,y1],[x2,y2],...]}
        self.declare_parameter("zones_json", "{}")
        # Counting line: JSON string {"p1": [x,y], "p2": [x,y], "name": "line_0"}
        self.declare_parameter("lines_json", "[]")
        # Speed alert threshold
        self.declare_parameter("speed_threshold", 100.0)
        self.declare_parameter("fps", 30.0)

        # ── Initialize modules ─────────────────────────────────────────────
        zones_raw = json.loads(self.get_parameter("zones_json").value)
        zones = {
            name: [(p[0], p[1]) for p in pts]
            for name, pts in zones_raw.items()
        }
        self.zone_module = ZoneIntrusion(zones)

        lines_raw = json.loads(self.get_parameter("lines_json").value)
        self.line_modules = []
        for line in lines_raw:
            self.line_modules.append(LineCounter(
                tuple(line["p1"]),
                tuple(line["p2"]),
                line.get("name", "line_0"),
            ))

        self.speed_module = SpeedEstimator(
            fps=self.get_parameter("fps").value,
            speed_alert_threshold=self.get_parameter("speed_threshold").value,
        )

        # ── Pub / Sub ──────────────────────────────────────────────────────
        self.sub = self.create_subscription(
            TrackedObject2DArray,
            "/tracker/tracked_objects",
            self.tracked_callback,
            qos_profile_sensor_data,
        )

        self.pub = self.create_publisher(
            AnalyticsEvent, "~/events", qos_profile_sensor_data
        )

        self.get_logger().info(
            f"Analytics node started: {len(zones)} zones, "
            f"{len(self.line_modules)} lines"
        )

    def tracked_callback(self, msg: TrackedObject2DArray):
        """Process tracked objects through all analytics modules."""
        for tobj in msg.tracked_objects:
            tid = tobj.track_id
            cx = tobj.detection.x_center
            cy = tobj.detection.y_center
            vx = tobj.vx
            vy = tobj.vy

            all_events = []

            # Zone intrusion
            all_events.extend(self.zone_module.check(tid, cx, cy))

            # Line crossing
            for lm in self.line_modules:
                all_events.extend(lm.check(tid, cx, cy))

            # Speed estimation
            all_events.extend(self.speed_module.check(tid, vx, vy, cx, cy))

            # Publish events
            for ev in all_events:
                event_msg = AnalyticsEvent()
                event_msg.header = msg.header
                event_msg.event_type = ev["type"]
                event_msg.event_data = json.dumps(ev)
                event_msg.track_id = tid
                self.pub.publish(event_msg)

                self.get_logger().info(
                    f"[{ev['type']}] track={tid} {json.dumps(ev)}"
                )


def main(args=None):
    rclpy.init(args=args)
    node = AnalyticsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
