// Copyright 2024 LinDet Team. Apache-2.0 license.
//
// TRTEngine — Isolated TensorRT inference wrapper for YOLO-style detectors.
//
// NOTE: This file requires TensorRT headers and CUDA.  It will only compile
// on a system with the TensorRT SDK installed (e.g. inside the Jetson Docker
// container).  The PIMPL pattern keeps TensorRT headers out of the public
// header so downstream packages do not need TRT.

#include "lindet_detection/trt_engine.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <vector>

// ── TensorRT / CUDA headers (only in this .cpp) ──────────────────────────
#include <NvInfer.h>
#include <cuda_runtime_api.h>

// ── TensorRT logger ──────────────────────────────────────────────────────
class TRTLogger : public nvinfer1::ILogger {
public:
  void log(Severity severity, const char * msg) noexcept override {
    if (severity <= Severity::kWARNING) {
      fprintf(stderr, "[TRT] %s\n", msg);
    }
  }
};

static TRTLogger g_logger;

// ── CUDA check macro ─────────────────────────────────────────────────────
#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t err = (call);                                                \
    if (err != cudaSuccess) {                                                \
      throw std::runtime_error(                                              \
        std::string("CUDA error: ") + cudaGetErrorString(err));              \
    }                                                                        \
  } while (0)

namespace lindet_detection {

// ── PIMPL implementation ─────────────────────────────────────────────────
struct TRTEngine::Impl {
  nvinfer1::IRuntime *          runtime    = nullptr;
  nvinfer1::ICudaEngine *       engine     = nullptr;
  nvinfer1::IExecutionContext * context    = nullptr;

  void * gpu_buffers[2]  = {nullptr, nullptr};  // input, output
  int    input_binding   = -1;
  int    output_binding  = -1;

  size_t input_size      = 0;   // bytes
  size_t output_size     = 0;

  int    output_dim1     = 0;   // e.g., num_detections
  int    output_dim2     = 0;   // e.g., 4+1+num_classes

  // Host-side buffers
  std::vector<float> input_host;
  std::vector<float> output_host;

  ~Impl() {
    for (auto & buf : gpu_buffers) {
      if (buf) cudaFree(buf);
    }
    if (context) context->destroy();
    if (engine)  engine->destroy();
    if (runtime) runtime->destroy();
  }
};

TRTEngine::~TRTEngine() = default;

bool TRTEngine::load(const std::string & engine_path, int num_classes)
{
  // Read serialized engine
  std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    return false;
  }
  const size_t size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<char> data(size);
  file.read(data.data(), size);
  file.close();

  impl_ = std::make_unique<Impl>();
  num_classes_ = num_classes;

  // Deserialize
  impl_->runtime = nvinfer1::createInferRuntime(g_logger);
  impl_->engine  = impl_->runtime->deserializeCudaEngine(data.data(), size);
  if (!impl_->engine) {
    return false;
  }
  impl_->context = impl_->engine->createExecutionContext();

  // Determine bindings
  impl_->input_binding  = impl_->engine->getBindingIndex("images");
  impl_->output_binding = impl_->engine->getBindingIndex("output0");

  // Fallback: use index 0/1 if named bindings not found
  if (impl_->input_binding < 0) impl_->input_binding = 0;
  if (impl_->output_binding < 0) impl_->output_binding = 1;

  // Input dimensions: [1, 3, H, W]
  auto in_dims = impl_->engine->getBindingDimensions(impl_->input_binding);
  input_h_ = in_dims.d[2];
  input_w_ = in_dims.d[3];

  impl_->input_size = 1;
  for (int i = 0; i < in_dims.nbDims; ++i)
    impl_->input_size *= in_dims.d[i];
  impl_->input_size *= sizeof(float);

  impl_->input_host.resize(impl_->input_size / sizeof(float));

  // Output dimensions: [1, D1, D2]  or [1, D2, D1] depending on model export
  auto out_dims = impl_->engine->getBindingDimensions(impl_->output_binding);
  impl_->output_dim1 = out_dims.d[1];
  impl_->output_dim2 = out_dims.d[2];

  impl_->output_size = 1;
  for (int i = 0; i < out_dims.nbDims; ++i)
    impl_->output_size *= out_dims.d[i];
  impl_->output_size *= sizeof(float);

  impl_->output_host.resize(impl_->output_size / sizeof(float));

  // Allocate GPU memory
  CUDA_CHECK(cudaMalloc(&impl_->gpu_buffers[impl_->input_binding],
                         impl_->input_size));
  CUDA_CHECK(cudaMalloc(&impl_->gpu_buffers[impl_->output_binding],
                         impl_->output_size));

  loaded_ = true;
  return true;
}

// ── Inference ────────────────────────────────────────────────────────────
std::vector<DetectionResult> TRTEngine::infer(
  const uint8_t * bgr_data,
  int img_width, int img_height,
  float conf_thresh, float nms_thresh)
{
  if (!loaded_) return {};

  preprocess(bgr_data, img_width, img_height);

  // H2D
  CUDA_CHECK(cudaMemcpy(
    impl_->gpu_buffers[impl_->input_binding],
    impl_->input_host.data(),
    impl_->input_size,
    cudaMemcpyHostToDevice));

  // Execute
  impl_->context->executeV2(impl_->gpu_buffers);

  // D2H
  CUDA_CHECK(cudaMemcpy(
    impl_->output_host.data(),
    impl_->gpu_buffers[impl_->output_binding],
    impl_->output_size,
    cudaMemcpyDeviceToHost));

  return postprocess(img_width, img_height, conf_thresh, nms_thresh);
}

// ── Letterbox preprocessing ──────────────────────────────────────────────
void TRTEngine::preprocess(const uint8_t * bgr, int src_w, int src_h)
{
  // Simple resize + channel split + normalize [0, 1]
  // For production, consider CUDA-accelerated preprocessing.
  const float scale = std::min(
    static_cast<float>(input_w_) / src_w,
    static_cast<float>(input_h_) / src_h);
  const int new_w = static_cast<int>(src_w * scale);
  const int new_h = static_cast<int>(src_h * scale);
  const int pad_x = (input_w_ - new_w) / 2;
  const int pad_y = (input_h_ - new_h) / 2;

  // Fill with 114/255 (YOLO letterbox grey)
  std::fill(impl_->input_host.begin(), impl_->input_host.end(),
            114.0f / 255.0f);

  // CHW layout
  const int ch_stride = input_h_ * input_w_;
  for (int dy = 0; dy < new_h; ++dy) {
    const int sy = static_cast<int>(dy / scale);
    if (sy >= src_h) break;
    for (int dx = 0; dx < new_w; ++dx) {
      const int sx = static_cast<int>(dx / scale);
      if (sx >= src_w) break;

      const int src_idx = (sy * src_w + sx) * 3;
      const int dst_y = dy + pad_y;
      const int dst_x = dx + pad_x;

      // BGR → RGB and normalize
      impl_->input_host[0 * ch_stride + dst_y * input_w_ + dst_x] =
        bgr[src_idx + 2] / 255.0f;  // R
      impl_->input_host[1 * ch_stride + dst_y * input_w_ + dst_x] =
        bgr[src_idx + 1] / 255.0f;  // G
      impl_->input_host[2 * ch_stride + dst_y * input_w_ + dst_x] =
        bgr[src_idx + 0] / 255.0f;  // B
    }
  }
}

// ── Post-processing (YOLOv8 output format) ───────────────────────────────
std::vector<DetectionResult> TRTEngine::postprocess(
  int src_w, int src_h, float conf_thresh, float nms_thresh)
{
  // YOLOv8 output: [1, 4+num_classes, num_detections] (transposed)
  // Need to handle both [1, D, N] and [1, N, D] layouts
  const int d1 = impl_->output_dim1;
  const int d2 = impl_->output_dim2;

  const bool transposed = (d1 == 4 + num_classes_);
  const int  num_dets   = transposed ? d2 : d1;
  const int  det_dim    = transposed ? d1 : d2;
  const float * data    = impl_->output_host.data();

  // Letterbox scale info
  const float scale = std::min(
    static_cast<float>(input_w_) / src_w,
    static_cast<float>(input_h_) / src_h);
  const int pad_x = (input_w_ - static_cast<int>(src_w * scale)) / 2;
  const int pad_y = (input_h_ - static_cast<int>(src_h * scale)) / 2;

  std::vector<DetectionResult> results;
  results.reserve(128);

  for (int i = 0; i < num_dets; ++i) {
    // Get values depending on layout
    auto val = [&](int attr) -> float {
      if (transposed) {
        return data[attr * num_dets + i];
      } else {
        return data[i * det_dim + attr];
      }
    };

    // cx, cy, w, h (in letterbox pixel coords)
    float cx = val(0);
    float cy = val(1);
    float bw = val(2);
    float bh = val(3);

    // Find best class
    float max_score = 0.0f;
    int   best_cls  = 0;
    for (int c = 0; c < num_classes_; ++c) {
      float s = val(4 + c);
      if (s > max_score) {
        max_score = s;
        best_cls  = c;
      }
    }

    if (max_score < conf_thresh) continue;

    // Undo letterbox → original image coords (normalized)
    cx = (cx - pad_x) / scale / src_w;
    cy = (cy - pad_y) / scale / src_h;
    bw = bw / scale / src_w;
    bh = bh / scale / src_h;

    results.push_back({cx, cy, bw, bh, max_score, best_cls});
  }

  return nms(results, nms_thresh);
}

// ── NMS ──────────────────────────────────────────────────────────────────
std::vector<DetectionResult> TRTEngine::nms(
  std::vector<DetectionResult> & dets, float iou_thresh)
{
  // Sort by confidence descending
  std::sort(dets.begin(), dets.end(),
    [](const DetectionResult & a, const DetectionResult & b) {
      return a.confidence > b.confidence;
    });

  auto iou = [](const DetectionResult & a, const DetectionResult & b) -> float {
    float ax1 = a.x_center - a.width  / 2, ay1 = a.y_center - a.height / 2;
    float ax2 = a.x_center + a.width  / 2, ay2 = a.y_center + a.height / 2;
    float bx1 = b.x_center - b.width  / 2, by1 = b.y_center - b.height / 2;
    float bx2 = b.x_center + b.width  / 2, by2 = b.y_center + b.height / 2;

    float ix1 = std::max(ax1, bx1), iy1 = std::max(ay1, by1);
    float ix2 = std::min(ax2, bx2), iy2 = std::min(ay2, by2);
    float inter = std::max(0.0f, ix2 - ix1) * std::max(0.0f, iy2 - iy1);
    float area_a = a.width * a.height;
    float area_b = b.width * b.height;
    return inter / (area_a + area_b - inter + 1e-6f);
  };

  std::vector<bool> suppressed(dets.size(), false);
  std::vector<DetectionResult> result;

  for (size_t i = 0; i < dets.size(); ++i) {
    if (suppressed[i]) continue;
    result.push_back(dets[i]);
    for (size_t j = i + 1; j < dets.size(); ++j) {
      if (!suppressed[j] && dets[i].class_id == dets[j].class_id &&
          iou(dets[i], dets[j]) > iou_thresh) {
        suppressed[j] = true;
      }
    }
  }

  return result;
}

}  // namespace lindet_detection
