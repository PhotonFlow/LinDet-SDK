#pragma once

#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

// Forward-declare GStreamer types to keep header clean
typedef struct _GstElement GstElement;
typedef struct _GstSample GstSample;

namespace lindet_camera {

/// @brief ROS 2 composable node that captures frames via a GStreamer pipeline
///        and publishes them as sensor_msgs/Image.
///
/// Designed as an rclcpp_components plugin so it can be loaded into a
/// ComposableNodeContainer for zero-copy intra-process transfer.
class GstCameraNode : public rclcpp::Node {
public:
  explicit GstCameraNode(const rclcpp::NodeOptions & options);
  ~GstCameraNode() override;

private:
  /// Timer callback that pulls a frame from the appsink and publishes it.
  void grab_frame();

  /// Convert a GstSample to a sensor_msgs::msg::Image.
  sensor_msgs::msg::Image::UniquePtr
  sample_to_image(GstSample * sample) const;

  // ── Members ──────────────────────────────────────────────────────────────
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_;
  rclcpp::TimerBase::SharedPtr timer_;

  GstElement * pipeline_ = nullptr;
  GstElement * appsink_  = nullptr;

  // Parameters
  std::string source_type_;   // csi | usb | rtsp | test
  std::string device_;        // /dev/video0, rtsp://..., etc.
  int width_  = 1280;
  int height_ = 720;
  int fps_    = 30;
  std::string frame_id_;
};

}  // namespace lindet_camera
