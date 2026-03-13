cmake_minimum_required(VERSION 3.8)
project(lindet_msgs)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Detection2D.msg"
  "msg/Detection2DArray.msg"
  "msg/TrackedObject2D.msg"
  "msg/TrackedObject2DArray.msg"
  "msg/Keypoint2D.msg"
  "msg/Pose2D.msg"
  "msg/PoseArray2D.msg"
  "msg/AnalyticsEvent.msg"
  DEPENDENCIES std_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
