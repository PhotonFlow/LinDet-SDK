#pragma once

#include <string>

namespace lindet_camera {

/// @brief Builds GStreamer pipeline strings for various source types.
///
/// CSI  → nvarguscamerasrc  (NVIDIA Jetson, zero-copy NvBufSurface)
/// USB  → v4l2src           (standard V4L2 device)
/// RTSP → rtspsrc           (network camera)
/// Test → videotestsrc      (synthetic pattern for CI / offline testing)
struct GstPipelineBuilder {
  /// Build a full pipeline string ending with appsink.
  /// @param source_type  One of: csi, usb, rtsp, test
  /// @param device       Device path or URI (ignored for test)
  /// @param width        Output width in pixels
  /// @param height       Output height in pixels
  /// @param fps          Framerate
  /// @return GStreamer pipeline description string
  static std::string build(
    const std::string & source_type,
    const std::string & device,
    int width, int height, int fps);
};

}  // namespace lindet_camera
