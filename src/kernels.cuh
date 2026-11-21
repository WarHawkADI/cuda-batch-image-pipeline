// Copyright 2026 CUDA at Scale Independent Project.
//
// Host-side launchers for the custom CUDA image-processing kernels. Every
// launcher is asynchronous with respect to the host and enqueues work on the
// given stream, so several images can be in flight on different streams.
//
// All kernels use clamp-to-edge border handling and integer / round-to-nearest
// arithmetic that matches the CPU reference implementation bit for bit.

#ifndef SRC_KERNELS_CUH_
#define SRC_KERNELS_CUH_

#include <cuda_runtime.h>

#include <cstdint>

namespace gpu_batch {

constexpr int kHistogramBins = 256;
constexpr int kMaxGaussianRadius = 15;

// Copies `count` = 2 * radius + 1 normalized Gaussian weights into constant
// memory. Must be called before LaunchGaussianBlur.
void UploadGaussianWeights(const float* weights, int count);

// Converts interleaved RGB to 8-bit luma: (77 R + 150 G + 29 B + 128) >> 8.
void LaunchRgbToGray(const uint8_t* rgb, uint8_t* gray, int num_pixels,
                     cudaStream_t stream);

// Histogram equalization: clears `histogram`, builds it with shared-memory
// atomics, scans it into a CDF and a lookup table, and remaps the image.
// `histogram` holds kHistogramBins counters, `lut` kHistogramBins bytes.
void LaunchEqualize(const uint8_t* input, uint8_t* output, int num_pixels,
                    unsigned int* histogram, uint8_t* lut,
                    cudaStream_t stream);

// Separable Gaussian blur: a horizontal pass into the float `scratch` buffer
// (width * height floats) and a vertical pass into `output`. Both passes
// stage their tile plus apron in shared memory.
void LaunchGaussianBlur(const uint8_t* input, float* scratch, uint8_t* output,
                        int width, int height, int radius,
                        cudaStream_t stream);

// Sobel gradient magnitude, rounded and clamped to [0, 255]. If `threshold`
// is positive, the result is binarized (255 where magnitude >= threshold).
void LaunchSobel(const uint8_t* input, uint8_t* output, int width, int height,
                 int threshold, cudaStream_t stream);

}  // namespace gpu_batch

#endif  // SRC_KERNELS_CUH_
