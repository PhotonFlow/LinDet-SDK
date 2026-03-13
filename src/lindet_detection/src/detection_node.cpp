// Copyright 2024 LinDet Team. Apache-2.0 license.

#include "lindet_detection/detection_node.hpp"
#include "lindet_detection/trt_engine.hpp"

#include <lindet_msgs/msg/detection2_d.hpp>
#include <lindet_msgs/msg/detection2_d_array.hpp>

#include <rclcpp_components/register_node_macro.hpp>

namespace lindet_detection {

DetectionNode::DetectionNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("detection_node", options)
{
  // ── Parameters ──────────────────────────────────────────────────────────
  model_path_   = declare_parameter<std::string>("model_path", "");
  conf_thresh_  = declare_parameter<double>("confidence_threshold", 0.25);
  nms_thresh_   = declare_parameter<double>("nms_threshold", 0.45);
  num_classes_  = declare_parameter<int>("num_classes", 80);

  class_names_  = declare_parameter<std::vector<std::string>>(
    "class_names", std::vector<std::string>{});

  // ── Load TensorRT engine ────────────────────────────────────────────────
  engine_ = std::make_unique<TRTEngine>();
  if (!model_path_.empty()) {
    if (engine_->load(model_path_, num_classes_)) {
      RCLCPP_INFO(get_logger(), "Loaded TRT engine: %s (%d classes)",
        model_path_.c_str(), num_classes_);
    } else {
      RCLCPP_ERROR(get_logger(), "Failed to load TRT engine: %s",
        model_path_.c_str());
    }
  } else {
    RCLCPP_WARN(get_logger(),
      "No model_path set — detection node running in passthrough mode. "
      "Set 'model_path' parameter to enable inference.");
  }

  // ── Pub / Sub ───────────────────────────────────────────────────────────
  pub_ = create_publisher<lindet_msgs::msg::Detection2DArray>(
    "~/detections", rclcpp::SensorDataQoS());

  sub_ = create_subscription<sensor_msgs::msg::Image>(
    "/camera/image_raw",
    rclcpp::SensorDataQoS(),
    std::bind(&DetectionNode::image_callback, this, std::placeholders::_1));
}

DetectionNode::~DetectionNode() = default;

void DetectionNode::image_callback(
  const sensor_msgs::msg::Image::ConstSharedPtr & msg)
{
  auto out = std::make_unique<lindet_msgs::msg::Detection2DArray>();
  out->header       = msg->header;
  out->image_width  = msg->width;
  out->image_height = msg->height;

  if (engine_ && engine_->is_loaded()) {
    auto results = engine_->infer(
      msg->data.data(),
      static_cast<int>(msg->width),
      static_cast<int>(msg->height),
      conf_thresh_, nms_thresh_);

    for (const auto & r : results) {
      lindet_msgs::msg::Detection2D det;
      det.header     = msg->header;
      det.x_center   = r.x_center;
      det.y_center   = r.y_center;
      det.width      = r.width;
      det.height     = r.height;
      det.confidence = r.confidence;
      det.class_id   = r.class_id;

      if (r.class_id >= 0 &&
          r.class_id < static_cast<int>(class_names_.size())) {
        det.class_name = class_names_[r.class_id];
      }

      out->detections.push_back(std::move(det));
    }
  }

  pub_->publish(std::move(out));
}

}  // namespace lindet_detection

RCLCPP_COMPONENTS_REGISTER_NODE(lindet_detection::DetectionNode)
