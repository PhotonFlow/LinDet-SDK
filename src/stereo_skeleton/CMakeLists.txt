cmake_minimum_required(VERSION 3.5)
project(stereo_skeleton)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)

add_executable(fake_camera_node src/fake_camera_node.cpp)
ament_target_dependencies(fake_camera_node rclcpp sensor_msgs)

add_executable(dummy_processor_node src/dummy_processor_node.cpp)
ament_target_dependencies(dummy_processor_node rclcpp sensor_msgs)

install(TARGETS
  fake_camera_node
  dummy_processor_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()

