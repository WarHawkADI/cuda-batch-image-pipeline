// Copyright 2026 CUDA at Scale Independent Project.
//
// Single-threaded CPU implementations of the GPU operations. They are used
// only to verify GPU correctness (--verify) and to measure the speedup; all
// production processing runs on the GPU.

#ifndef SRC_REFERENCE_FILTERS_H_
#define SRC_REFERENCE_FILTERS_H_

#include <cstdint>
#include <vector>

#include "image_io.h"
#include "operation.h"

namespace gpu_batch {

// Returns 2 * radius + 1 normalized Gaussian weights, where
// radius = ceil(3 * sigma). Shared by the CPU and GPU paths.
std::vector<float> GaussianWeights(float sigma);

void CpuRgbToGray(const uint8_t* rgb, int num_pixels, uint8_t* gray);
void CpuEqualize(const uint8_t* input, int num_pixels, uint8_t* output);
void CpuGaussianBlur(const uint8_t* input, int width, int height,
                     const std::vector<float>& weights, uint8_t* output);
void CpuSobel(const uint8_t* input, int width, int height, int threshold,
              uint8_t* output);

// Runs `operation` on `image` exactly as the GPU pipeline does and returns
// the final 8-bit grayscale result.
std::vector<uint8_t> CpuRunOperation(const Image& image, Operation operation,
                                     const std::vector<float>& weights,
                                     int threshold);

}  // namespace gpu_batch

#endif  // SRC_REFERENCE_FILTERS_H_
