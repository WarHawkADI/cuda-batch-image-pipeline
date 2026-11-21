// Copyright 2026 CUDA at Scale Independent Project.
//
// BatchProcessor streams a directory of images through the GPU. It keeps a
// pool of "slots", each owning a CUDA stream, pinned host buffers and device
// buffers. Images are assigned to slots round-robin, so while one image is
// being transferred / processed on the GPU the host is already reading the
// next image from disk and other streams are running their kernels.

#ifndef SRC_BATCH_PROCESSOR_H_
#define SRC_BATCH_PROCESSOR_H_

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "image_io.h"
#include "operation.h"

namespace gpu_batch {

struct ProcessorConfig {
  Operation operation = Operation::kPipeline;
  float sigma = 1.5f;
  int threshold = 0;
  int num_streams = 4;
  bool save_intermediate = false;
  bool write_outputs = true;  // False in benchmark mode.
  std::string output_dir;
  bool verbose = true;
};

// Supplies image `index` of a batch. Returns nullptr and fills `error` if the
// image cannot be loaded. The pointer only needs to stay valid until the
// next call.
using ImageSource =
    std::function<const Image*(int index, std::string* error)>;

struct ImageRecord {
  std::string name;
  int width = 0;
  int height = 0;
  int channels = 0;
  float gpu_ms = 0.0f;  // H2D copy + kernels + D2H copy, timed with events.
};

struct BatchSummary {
  std::vector<ImageRecord> images;
  int failures = 0;
  double wall_ms = 0.0;
  double megapixels = 0.0;
};

class BatchProcessor {
 public:
  explicit BatchProcessor(const ProcessorConfig& config);
  ~BatchProcessor();

  BatchProcessor(const BatchProcessor&) = delete;
  BatchProcessor& operator=(const BatchProcessor&) = delete;

  // Creates streams and events and uploads the Gaussian weights.
  void Initialize();

  // Processes `names.size()` images obtained from `source` and (unless
  // disabled) writes the results to the output directory as
  // <stem>_<operation>.pgm. Loading image i+1 on the host overlaps with the
  // GPU work of the images already in flight on other streams.
  BatchSummary ProcessBatch(const std::vector<std::string>& names,
                            const ImageSource& source);

  // Runs the configured operation on one in-memory image and returns the
  // final result synchronously. Used for CPU-vs-GPU verification.
  std::vector<uint8_t> ProcessImage(const Image& image, float* gpu_ms);

 private:
  // Indices into Slot::host_output.
  enum OutputPlane { kEqualizedPlane = 0, kBlurredPlane = 1, kFinalPlane = 2 };

  struct Slot {
    cudaStream_t stream = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    size_t capacity = 0;  // Pixels per plane the buffers can hold.

    uint8_t* host_input = nullptr;  // Pinned, capacity * 3 bytes.
    uint8_t* host_output[3] = {nullptr, nullptr, nullptr};  // Pinned.

    uint8_t* device_input = nullptr;  // capacity * 3 bytes.
    uint8_t* device_gray = nullptr;
    uint8_t* device_equalized = nullptr;
    uint8_t* device_blurred = nullptr;
    uint8_t* device_output = nullptr;
    float* device_scratch = nullptr;
    unsigned int* device_histogram = nullptr;
    uint8_t* device_lut = nullptr;

    // The image currently in flight on this slot, if any.
    bool busy = false;
    int index = 0;
    std::string name;
    int width = 0;
    int height = 0;
    int channels = 0;
  };

  void EnsureCapacity(Slot* slot, size_t num_pixels);
  void FreeBuffers(Slot* slot);
  void Enqueue(Slot* slot, const Image& image);
  // Waits for the slot's image, writes its outputs and records its timing.
  void Finish(Slot* slot, int total, BatchSummary* summary);

  ProcessorConfig config_;
  int gaussian_radius_ = 0;
  std::vector<Slot> slots_;
};

}  // namespace gpu_batch

#endif  // SRC_BATCH_PROCESSOR_H_
