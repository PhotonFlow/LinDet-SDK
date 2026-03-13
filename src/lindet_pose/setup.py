from setuptools import setup

package_name = "lindet_pose"

setup(
    name=package_name,
    version="1.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="LinDet Team",
    maintainer_email="dev@lindet.local",
    description="TensorRT pose estimation node for LinDet-SDK.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "pose_node = lindet_pose.pose_node:main",
        ],
    },
)
