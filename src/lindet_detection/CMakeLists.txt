cmake_minimum_required(VERSION 3.8)
project(lindet_detection)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)

# ── Dependencies ─────────────────────────────────────────────────────────────
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(lindet_msgs REQUIRED)

# CUDA & TensorRT
find_package(CUDA REQUIRED)
find_library(NVINFER_LIB nvinfer)
find_library(NVPARSERS_LIB nvparsers)

# ── Library (composable component) ───────────────────────────────────────────
add_library(detection_component SHARED
  src/detection_node.cpp
  src/trt_engine.cpp
)

target_include_directories(detection_component PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(detection_component
  ${NVINFER_LIB}
  ${CUDA_LIBRARIES}
)

ament_target_dependencies(detection_component
  rclcpp rclcpp_components sensor_msgs lindet_msgs
)

rclcpp_components_register_node(detection_component
  PLUGIN "lindet_detection::DetectionNode"
  EXECUTABLE detection_node
)

# ── Install ──────────────────────────────────────────────────────────────────
install(TARGETS detection_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/
  DESTINATION include
)

ament_package()
