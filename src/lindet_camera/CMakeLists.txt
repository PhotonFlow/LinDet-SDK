cmake_minimum_required(VERSION 3.8)
project(lindet_camera)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)

# ── Dependencies ─────────────────────────────────────────────────────────────
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)

# GStreamer
find_package(PkgConfig REQUIRED)
pkg_check_modules(GST REQUIRED gstreamer-1.0 gstreamer-app-1.0)

# ── Library (composable component) ───────────────────────────────────────────
add_library(gst_camera_component SHARED
  src/gst_camera_node.cpp
  src/gst_pipeline_builder.cpp
)

target_include_directories(gst_camera_component PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${GST_INCLUDE_DIRS}
)

target_link_libraries(gst_camera_component ${GST_LIBRARIES})

ament_target_dependencies(gst_camera_component
  rclcpp rclcpp_components sensor_msgs std_msgs
)

# Register as composable node for zero-copy intra-process
rclcpp_components_register_node(gst_camera_component
  PLUGIN "lindet_camera::GstCameraNode"
  EXECUTABLE gst_camera_node
)

# ── Install ──────────────────────────────────────────────────────────────────
install(TARGETS gst_camera_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/
  DESTINATION include
)

ament_package()
