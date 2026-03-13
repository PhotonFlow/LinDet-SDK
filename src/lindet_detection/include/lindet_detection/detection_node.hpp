#pragma once

#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

// Forward declaration — avoids pulling heavy lindet_msgs headers into hpp
namespace lindet_msgs { namespace msg {
class Detection2DArray;
}}  // namespace lindet_msgs::msg

namespace lindet_detection {

class TRTEngine;  // forward-declare

/// @brief ROS 2 composable node for TensorRT-based object detection.
///
/// Subscribes to raw images, runs the TRTEngine, and publishes
/// Detection2DArray messages.  All inference is isolated inside TRTEngine
/// so the model can be swapped without touching this node.
class DetectionNode : public rclcpp::Node {
public:
  explicit DetectionNode(const rclcpp::NodeOptions & options);
  ~DetectionNode() override;

private:
  void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr & msg);

  // ── Members ──────────────────────────────────────────────────────────
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;
  rclcpp::Publisher<lindet_msgs::msg::Detection2DArray>::SharedPtr pub_;

  std::unique_ptr<TRTEngine> engine_;

  // Parameters
  std::string model_path_;
  float conf_thresh_;
  float nms_thresh_;
  int   num_classes_;
  std::vector<std::string> class_names_;
};

}  // namespace lindet_detection
