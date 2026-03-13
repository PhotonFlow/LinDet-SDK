// Copyright 2024 LinDet Team. Apache-2.0 license.

#include "lindet_camera/gst_pipeline_builder.hpp"

#include <sstream>
#include <stdexcept>

namespace lindet_camera {

std::string GstPipelineBuilder::build(
  const std::string & source_type,
  const std::string & device,
  int width, int height, int fps)
{
  std::ostringstream ss;

  if (source_type == "csi") {
    // ── NVIDIA Jetson CSI (nvarguscamerasrc) ───────────────────────────────
    ss << "nvarguscamerasrc sensor-id=0 "
       << "! video/x-raw(memory:NVMM), "
       << "width=" << width << ", height=" << height
       << ", framerate=" << fps << "/1, format=NV12 "
       << "! nvvidconv "
       << "! video/x-raw, format=BGRx "
       << "! videoconvert "
       << "! video/x-raw, format=BGR "
       << "! appsink name=appsink0 emit-signals=false sync=false";

  } else if (source_type == "usb") {
    // ── USB camera (v4l2src) ──────────────────────────────────────────────
    ss << "v4l2src device=" << device << " "
       << "! video/x-raw, width=" << width << ", height=" << height
       << ", framerate=" << fps << "/1 "
       << "! videoconvert "
       << "! video/x-raw, format=BGR "
       << "! appsink name=appsink0 emit-signals=false sync=false";

  } else if (source_type == "rtsp") {
    // ── RTSP network camera ───────────────────────────────────────────────
    ss << "rtspsrc location=" << device << " latency=100 "
       << "! rtph264depay "
       << "! h264parse "
       << "! nvv4l2decoder "
       << "! nvvidconv "
       << "! video/x-raw, format=BGRx, width=" << width
       << ", height=" << height << " "
       << "! videoconvert "
       << "! video/x-raw, format=BGR "
       << "! appsink name=appsink0 emit-signals=false sync=false";

  } else if (source_type == "test") {
    // ── Test pattern (for CI / development) ───────────────────────────────
    ss << "videotestsrc pattern=ball is-live=true "
       << "! video/x-raw, format=BGR, width=" << width
       << ", height=" << height
       << ", framerate=" << fps << "/1 "
       << "! appsink name=appsink0 emit-signals=false sync=false";

  } else {
    throw std::invalid_argument(
      "Unknown source_type: '" + source_type +
      "'. Supported: csi, usb, rtsp, test");
  }

  return ss.str();
}

}  // namespace lindet_camera
