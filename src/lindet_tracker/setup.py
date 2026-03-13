from setuptools import setup

package_name = "lindet_tracker"

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
    description="ByteTrack multi-object tracker for LinDet-SDK.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "tracker_node = lindet_tracker.tracker_node:main",
        ],
    },
)
