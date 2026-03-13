# Copyright 2024 LinDet Team. Apache-2.0 license.
"""Camera-only launch — for bring-up and debugging."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    source_type_arg = DeclareLaunchArgument(
        "source_type", default_value="test",
    )
    device_arg = DeclareLaunchArgument(
        "device", default_value="/dev/video0",
    )

    container = ComposableNodeContainer(
        name="lindet_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container",
        composable_node_descriptions=[
            ComposableNode(
                package="lindet_camera",
                plugin="lindet_camera::GstCameraNode",
                name="camera",
                parameters=[{
                    "source_type": LaunchConfiguration("source_type"),
                    "device": LaunchConfiguration("device"),
                    "width": 1280,
                    "height": 720,
                    "fps": 30,
                }],
                remappings=[("~/image_raw", "/camera/image_raw")],
                extra_arguments=[{"use_intra_process_comms": True}],
            ),
        ],
        output="screen",
    )

    return LaunchDescription([source_type_arg, device_arg, container])
