from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    fake_camera = Node(
        package="stereo_skeleton",
        executable="fake_camera_node",
        name="fake_camera_driver",
        output="screen",
    )

    processor = Node(
        package="stereo_skeleton",
        executable="dummy_processor_node",
        name="ai_processor",
        output="screen",
    )

    return LaunchDescription([fake_camera, processor])

