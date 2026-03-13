# Copyright 2024 LinDet Team. Apache-2.0 license.
"""Full LinDet-SDK system launch file.

Launches all nodes:
  - GStreamer Camera (C++ composable component)
  - TensorRT Detection (C++ composable component)
  - ByteTrack Tracker (Python)
  - Pose Estimation (Python)
  - Analytics (Python)
  - Visualization (Python)

C++ components are loaded into a ComposableNodeContainer for zero-copy
intra-process communication.  Python nodes run alongside in the same
launch context.
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    # ── Launch arguments ──────────────────────────────────────────────────
    source_type_arg = DeclareLaunchArgument(
        "source_type", default_value="test",
        description="Camera source: csi, usb, rtsp, test"
    )
    device_arg = DeclareLaunchArgument(
        "device", default_value="/dev/video0",
        description="Camera device path or RTSP URI"
    )
    model_path_arg = DeclareLaunchArgument(
        "model_path", default_value="",
        description="Path to TensorRT detection .engine file"
    )
    pose_model_path_arg = DeclareLaunchArgument(
        "pose_model_path", default_value="",
        description="Path to TensorRT pose .engine file"
    )

    # ── C++ Composable Node Container (zero-copy intra-process) ───────────
    composable_container = ComposableNodeContainer(
        name="lindet_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container",
        composable_node_descriptions=[
            # Camera node
            ComposableNode(
                package="lindet_camera",
                plugin="lindet_camera::GstCameraNode",
                name="camera",
                namespace="",
                parameters=[{
                    "source_type": LaunchConfiguration("source_type"),
                    "device": LaunchConfiguration("device"),
                    "width": 1280,
                    "height": 720,
                    "fps": 30,
                    "frame_id": "camera_optical_frame",
                }],
                remappings=[
                    ("~/image_raw", "/camera/image_raw"),
                ],
                extra_arguments=[{
                    "use_intra_process_comms": True,
                }],
            ),
            # Detection node
            ComposableNode(
                package="lindet_detection",
                plugin="lindet_detection::DetectionNode",
                name="detection",
                namespace="",
                parameters=[{
                    "model_path": LaunchConfiguration("model_path"),
                    "confidence_threshold": 0.25,
                    "nms_threshold": 0.45,
                    "num_classes": 80,
                    "class_names": [
                        "person", "bicycle", "car", "motorcycle", "airplane",
                        "bus", "train", "truck", "boat", "traffic light",
                        "fire hydrant", "stop sign", "parking meter", "bench",
                        "bird", "cat", "dog", "horse", "sheep", "cow",
                        "elephant", "bear", "zebra", "giraffe", "backpack",
                        "umbrella", "handbag", "tie", "suitcase", "frisbee",
                        "skis", "snowboard", "sports ball", "kite",
                        "baseball bat", "baseball glove", "skateboard",
                        "surfboard", "tennis racket", "bottle", "wine glass",
                        "cup", "fork", "knife", "spoon", "bowl", "banana",
                        "apple", "sandwich", "orange", "broccoli", "carrot",
                        "hot dog", "pizza", "donut", "cake", "chair", "couch",
                        "potted plant", "bed", "dining table", "toilet", "tv",
                        "laptop", "mouse", "remote", "keyboard", "cell phone",
                        "microwave", "oven", "toaster", "sink", "refrigerator",
                        "book", "clock", "vase", "scissors", "teddy bear",
                        "hair drier", "toothbrush",
                    ],
                }],
                remappings=[
                    ("~/detections", "/detection/detections"),
                ],
                extra_arguments=[{
                    "use_intra_process_comms": True,
                }],
            ),
        ],
        output="screen",
    )

    # ── Python nodes ──────────────────────────────────────────────────────
    tracker_node = Node(
        package="lindet_tracker",
        executable="tracker_node",
        name="tracker",
        output="screen",
        parameters=[{
            "high_thresh": 0.5,
            "low_thresh": 0.1,
            "match_thresh": 0.3,
            "max_age": 30,
            "min_hits": 3,
        }],
        remappings=[
            ("~/tracked_objects", "/tracker/tracked_objects"),
        ],
    )

    pose_node = Node(
        package="lindet_pose",
        executable="pose_node",
        name="pose",
        output="screen",
        parameters=[{
            "model_path": LaunchConfiguration("pose_model_path"),
            "keypoint_threshold": 0.3,
            "person_class_id": 0,
        }],
        remappings=[
            ("~/poses", "/pose/poses"),
        ],
    )

    analytics_node = Node(
        package="lindet_analytics",
        executable="analytics_node",
        name="analytics",
        output="screen",
        parameters=[{
            "zones_json": "{}",
            "lines_json": "[]",
            "speed_threshold": 100.0,
            "fps": 30.0,
        }],
        remappings=[
            ("~/events", "/analytics/events"),
        ],
    )

    viz_node = Node(
        package="lindet_visualization",
        executable="viz_node",
        name="visualization",
        output="screen",
        parameters=[{
            "draw_detections": True,
            "draw_tracks": True,
            "draw_poses": True,
            "keypoint_threshold": 0.3,
        }],
        remappings=[
            ("~/image", "/visualization/image"),
        ],
    )

    return LaunchDescription([
        source_type_arg,
        device_arg,
        model_path_arg,
        pose_model_path_arg,
        composable_container,
        tracker_node,
        pose_node,
        analytics_node,
        viz_node,
    ])
