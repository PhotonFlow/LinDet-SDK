// Copyright 2024 LinDet Team. Apache-2.0 license.

#include "lindet_camera/gst_camera_node.hpp"
#include "lindet_camera/gst_pipeline_builder.hpp"

#include <gst/gst.h>
#include <gst/app/gstappsink.h>

#include <rclcpp_components/register_node_macro.hpp>

namespace lindet_camera {

GstCameraNode::GstCameraNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("gst_camera", options)
{
  // ── Declare parameters ──────────────────────────────────────────────────
  source_type_ = declare_parameter<std::string>("source_type", "test");
  device_      = declare_parameter<std::string>("device", "/dev/video0");
  width_       = declare_parameter<int>("width", 1280);
  height_      = declare_parameter<int>("height", 720);
  fps_         = declare_parameter<int>("fps", 30);
  frame_id_    = declare_parameter<std::string>("frame_id", "camera_optical_frame");

  // ── Publisher with SensorDataQoS for real-time performance ──────────────
  pub_ = create_publisher<sensor_msgs::msg::Image>(
    "~/image_raw", rclcpp::SensorDataQoS());

  // ── Build & start GStreamer pipeline ─────────────────────────────────────
  if (!gst_is_initialized()) {
    gst_init(nullptr, nullptr);
  }

  const auto pipeline_str = GstPipelineBuilder::build(
    source_type_, device_, width_, height_, fps_);
  RCLCPP_INFO(get_logger(), "GStreamer pipeline: %s", pipeline_str.c_str());

  GError * error = nullptr;
  pipeline_ = gst_parse_launch(pipeline_str.c_str(), &error);
  if (error) {
    RCLCPP_FATAL(get_logger(), "Pipeline parse error: %s", error->message);
    g_error_free(error);
    throw std::runtime_error("Failed to create GStreamer pipeline");
  }

  appsink_ = gst_bin_get_by_name(GST_BIN(pipeline_), "appsink0");
  if (!appsink_) {
    throw std::runtime_error("appsink not found in pipeline");
  }

  // Configure appsink: drop old buffers, keep only latest
  gst_app_sink_set_max_buffers(GST_APP_SINK(appsink_), 1);
  gst_app_sink_set_drop(GST_APP_SINK(appsink_), TRUE);

  gst_element_set_state(pipeline_, GST_STATE_PLAYING);

  // ── Timer to grab frames at the pipeline framerate ──────────────────────
  const auto period = std::chrono::milliseconds(1000 / fps_);
  timer_ = create_wall_timer(period, std::bind(&GstCameraNode::grab_frame, this));

  RCLCPP_INFO(get_logger(), "Camera started: %s %dx%d@%dfps",
    source_type_.c_str(), width_, height_, fps_);
}

GstCameraNode::~GstCameraNode()
{
  if (pipeline_) {
    gst_element_set_state(pipeline_, GST_STATE_NULL);
    gst_object_unref(pipeline_);
  }
}

void GstCameraNode::grab_frame()
{
  GstSample * sample = gst_app_sink_try_pull_sample(
    GST_APP_SINK(appsink_), 0);  // non-blocking

  if (!sample) {
    return;  // no frame ready yet
  }

  auto msg = sample_to_image(sample);
  gst_sample_unref(sample);

  if (msg) {
    pub_->publish(std::move(msg));
  }
}

sensor_msgs::msg::Image::UniquePtr
GstCameraNode::sample_to_image(GstSample * sample) const
{
  GstBuffer * buffer = gst_sample_get_buffer(sample);
  if (!buffer) {
    return nullptr;
  }

  GstMapInfo map;
  if (!gst_buffer_map(buffer, &map, GST_MAP_READ)) {
    return nullptr;
  }

  auto msg = std::make_unique<sensor_msgs::msg::Image>();
  msg->header.stamp    = now();
  msg->header.frame_id = frame_id_;
  msg->height          = static_cast<uint32_t>(height_);
  msg->width           = static_cast<uint32_t>(width_);
  msg->encoding        = "bgr8";
  msg->is_bigendian    = false;
  msg->step            = static_cast<uint32_t>(width_ * 3);  // 3 channels BGR

  const size_t expected = static_cast<size_t>(msg->step * msg->height);
  if (map.size >= expected) {
    msg->data.assign(map.data, map.data + expected);
  } else {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
      "Buffer size mismatch: got %zu, expected %zu", map.size, expected);
    msg->data.assign(map.data, map.data + map.size);
  }

  gst_buffer_unmap(buffer, &map);
  return msg;
}

}  // namespace lindet_camera

RCLCPP_COMPONENTS_REGISTER_NODE(lindet_camera::GstCameraNode)
