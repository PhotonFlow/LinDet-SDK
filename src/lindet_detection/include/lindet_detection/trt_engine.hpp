#pragma once

#include <memory>
#include <string>
#include <vector>

namespace lindet_detection {

/// @brief A single detection result from the TensorRT engine.
struct DetectionResult {
  float x_center;     // normalized [0, 1]
  float y_center;
  float width;
  float height;
  float confidence;
  int   class_id;
};

/// @brief Isolated TensorRT engine wrapper for object detection.
///
/// This class is completely decoupled from ROS — it can be unit-tested
/// standalone.  It loads a serialized TensorRT `.engine` file and runs
/// inference on a raw BGR image buffer.
///
/// The expected model output layout is YOLO-style:
///   [batch, num_detections, 4 + 1 + num_classes]
///   where 4 = (cx, cy, w, h), 1 = objectness, rest = class scores
///
/// Users can drop in any compatible `.engine` file without modifying
/// node logic.
class TRTEngine {
public:
  /// @brief Construct but do NOT load the engine yet.
  TRTEngine() = default;
  ~TRTEngine();

  TRTEngine(const TRTEngine &) = delete;
  TRTEngine & operator=(const TRTEngine &) = delete;

  /// @brief Load a serialized TensorRT engine from disk.
  /// @param engine_path  Path to the `.engine` file.
  /// @param num_classes  Number of object classes the model outputs.
  /// @return true on success.
  bool load(const std::string & engine_path, int num_classes);

  /// @brief Run inference on a single BGR image.
  /// @param bgr_data  Pointer to HWC BGR uint8 image data.
  /// @param img_width  Width of the input image.
  /// @param img_height Height of the input image.
  /// @param conf_thresh  Confidence threshold.
  /// @param nms_thresh   NMS IoU threshold.
  /// @return Vector of detections that survive NMS.
  std::vector<DetectionResult> infer(
    const uint8_t * bgr_data,
    int img_width, int img_height,
    float conf_thresh = 0.25f,
    float nms_thresh  = 0.45f);

  /// @brief Check whether an engine is loaded and ready.
  bool is_loaded() const { return loaded_; }

  /// @brief Get the expected input dimensions.
  void get_input_dims(int & w, int & h) const { w = input_w_; h = input_h_; }

private:
  /// Apply letterbox pre-processing (resize + pad).
  void preprocess(const uint8_t * bgr, int src_w, int src_h);

  /// Post-process raw network output → DetectionResult vector.
  std::vector<DetectionResult> postprocess(
    int src_w, int src_h, float conf_thresh, float nms_thresh);

  /// Greedy NMS on sorted detections.
  static std::vector<DetectionResult> nms(
    std::vector<DetectionResult> & dets, float iou_thresh);

  // ── Engine internals (opaque, implemented in .cpp) ───────────────────
  struct Impl;
  std::unique_ptr<Impl> impl_;

  bool loaded_     = false;
  int  num_classes_ = 0;
  int  input_w_    = 640;
  int  input_h_    = 640;
};

}  // namespace lindet_detection
