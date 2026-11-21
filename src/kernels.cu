// Copyright 2026 CUDA at Scale Independent Project.

#include "kernels.cuh"

#include "cuda_check.h"

namespace gpu_batch {
namespace {

constexpr int kTileSize = 16;           // 16 x 16 threads per 2-D block.
constexpr int kThreadsPerBlock1d = 256;
constexpr int kMaxGridBlocks1d = 1024;  // Grid-stride loops cover the rest.

__constant__ float c_gaussian_weights[2 * kMaxGaussianRadius + 1];

__device__ __forceinline__ int Clamp(int value, int low, int high) {
  return min(max(value, low), high);
}

int GridSize1d(int num_items) {
  const int blocks = (num_items + kThreadsPerBlock1d - 1) / kThreadsPerBlock1d;
  return blocks < kMaxGridBlocks1d ? blocks : kMaxGridBlocks1d;
}

dim3 GridSize2d(int width, int height) {
  return dim3((width + kTileSize - 1) / kTileSize,
              (height + kTileSize - 1) / kTileSize);
}

// ---------------------------------------------------------------------------
// Color conversion.
// ---------------------------------------------------------------------------

__global__ void RgbToGrayKernel(const uint8_t* rgb, uint8_t* gray,
                                int num_pixels) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_pixels;
       i += blockDim.x * gridDim.x) {
    const int r = rgb[3 * i];
    const int g = rgb[3 * i + 1];
    const int b = rgb[3 * i + 2];
    gray[i] = static_cast<uint8_t>((77 * r + 150 * g + 29 * b + 128) >> 8);
  }
}

// ---------------------------------------------------------------------------
// Histogram equalization.
// ---------------------------------------------------------------------------

// Each block accumulates a private histogram in shared memory (cheap
// shared-memory atomics) and merges it into the global histogram once.
__global__ void HistogramKernel(const uint8_t* input, int num_pixels,
                                unsigned int* histogram) {
  __shared__ unsigned int local[kHistogramBins];
  for (int bin = threadIdx.x; bin < kHistogramBins; bin += blockDim.x) {
    local[bin] = 0;
  }
  __syncthreads();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_pixels;
       i += blockDim.x * gridDim.x) {
    atomicAdd(&local[input[i]], 1u);
  }
  __syncthreads();
  for (int bin = threadIdx.x; bin < kHistogramBins; bin += blockDim.x) {
    if (local[bin] != 0) atomicAdd(&histogram[bin], local[bin]);
  }
}

// Single block of kHistogramBins threads: inclusive Hillis-Steele scan of the
// histogram into a CDF, then the classic equalization mapping
//   lut[v] = round((cdf[v] - cdf_min) * 255 / (N - cdf_min)),
// evaluated in 64-bit integer arithmetic so it is exact.
__global__ void BuildEqualizationLutKernel(const unsigned int* histogram,
                                           int num_pixels, uint8_t* lut) {
  __shared__ unsigned int cdf[kHistogramBins];
  __shared__ unsigned int cdf_min;
  const int bin = threadIdx.x;
  cdf[bin] = histogram[bin];
  __syncthreads();
  for (int offset = 1; offset < kHistogramBins; offset <<= 1) {
    const unsigned int addend = (bin >= offset) ? cdf[bin - offset] : 0u;
    __syncthreads();
    cdf[bin] += addend;
    __syncthreads();
  }
  // The first non-empty bin is where the CDF first becomes non-zero.
  if (cdf[bin] != 0 && (bin == 0 || cdf[bin - 1] == 0)) cdf_min = cdf[bin];
  __syncthreads();

  const unsigned long long denominator =
      static_cast<unsigned long long>(num_pixels) - cdf_min;
  if (denominator == 0) {  // Constant image: leave it unchanged.
    lut[bin] = static_cast<uint8_t>(bin);
    return;
  }
  const unsigned long long numerator =
      static_cast<unsigned long long>(cdf[bin] - cdf_min) * 255ull;
  lut[bin] = static_cast<uint8_t>(
      (cdf[bin] < cdf_min) ? 0 : (numerator + denominator / 2) / denominator);
}

__global__ void ApplyLutKernel(const uint8_t* input, const uint8_t* lut,
                               uint8_t* output, int num_pixels) {
  __shared__ uint8_t local_lut[kHistogramBins];
  for (int bin = threadIdx.x; bin < kHistogramBins; bin += blockDim.x) {
    local_lut[bin] = lut[bin];
  }
  __syncthreads();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_pixels;
       i += blockDim.x * gridDim.x) {
    output[i] = local_lut[input[i]];
  }
}

// ---------------------------------------------------------------------------
// Separable Gaussian blur.
//
// __fmul_rn / __fadd_rn prevent the compiler from fusing the multiply-add
// into an FMA, so the GPU result is bit-identical to the CPU reference.
// ---------------------------------------------------------------------------

__global__ void GaussianRowKernel(const uint8_t* input, float* output,
                                  int width, int height, int radius) {
  __shared__ float tile[kTileSize][kTileSize + 2 * kMaxGaussianRadius];
  const int x = blockIdx.x * kTileSize + threadIdx.x;
  const int y = blockIdx.y * kTileSize + threadIdx.y;
  const int row = min(y, height - 1);
  const int tile_width = kTileSize + 2 * radius;
  const int first_x = blockIdx.x * kTileSize - radius;

  for (int i = threadIdx.x; i < tile_width; i += kTileSize) {
    const int source_x = Clamp(first_x + i, 0, width - 1);
    tile[threadIdx.y][i] = input[row * width + source_x];
  }
  __syncthreads();
  if (x >= width || y >= height) return;

  float sum = 0.0f;
  for (int k = -radius; k <= radius; ++k) {
    sum = __fadd_rn(sum, __fmul_rn(c_gaussian_weights[k + radius],
                                   tile[threadIdx.y][threadIdx.x + radius + k]));
  }
  output[y * width + x] = sum;
}

__global__ void GaussianColumnKernel(const float* input, uint8_t* output,
                                     int width, int height, int radius) {
  __shared__ float tile[kTileSize + 2 * kMaxGaussianRadius][kTileSize];
  const int x = blockIdx.x * kTileSize + threadIdx.x;
  const int y = blockIdx.y * kTileSize + threadIdx.y;
  const int column = min(x, width - 1);
  const int tile_height = kTileSize + 2 * radius;
  const int first_y = blockIdx.y * kTileSize - radius;

  for (int i = threadIdx.y; i < tile_height; i += kTileSize) {
    const int source_y = Clamp(first_y + i, 0, height - 1);
    tile[i][threadIdx.x] = input[source_y * width + column];
  }
  __syncthreads();
  if (x >= width || y >= height) return;

  float sum = 0.0f;
  for (int k = -radius; k <= radius; ++k) {
    sum = __fadd_rn(sum, __fmul_rn(c_gaussian_weights[k + radius],
                                   tile[threadIdx.y + radius + k][threadIdx.x]));
  }
  output[y * width + x] =
      static_cast<uint8_t>(min(255, static_cast<int>(sum + 0.5f)));
}

// ---------------------------------------------------------------------------
// Sobel edge detection.
// ---------------------------------------------------------------------------

__global__ void SobelKernel(const uint8_t* input, uint8_t* output, int width,
                            int height, int threshold) {
  constexpr int kApronTile = kTileSize + 2;
  __shared__ int tile[kApronTile][kApronTile];
  const int first_x = blockIdx.x * kTileSize - 1;
  const int first_y = blockIdx.y * kTileSize - 1;
  const int thread_id = threadIdx.y * kTileSize + threadIdx.x;

  // 256 threads cooperatively load the 18 x 18 tile including the apron.
  for (int i = thread_id; i < kApronTile * kApronTile;
       i += kTileSize * kTileSize) {
    const int source_x = Clamp(first_x + i % kApronTile, 0, width - 1);
    const int source_y = Clamp(first_y + i / kApronTile, 0, height - 1);
    tile[i / kApronTile][i % kApronTile] = input[source_y * width + source_x];
  }
  __syncthreads();

  const int x = blockIdx.x * kTileSize + threadIdx.x;
  const int y = blockIdx.y * kTileSize + threadIdx.y;
  if (x >= width || y >= height) return;

  const int tx = threadIdx.x + 1;
  const int ty = threadIdx.y + 1;
  const int gx = (tile[ty - 1][tx + 1] + 2 * tile[ty][tx + 1] +
                  tile[ty + 1][tx + 1]) -
                 (tile[ty - 1][tx - 1] + 2 * tile[ty][tx - 1] +
                  tile[ty + 1][tx - 1]);
  const int gy = (tile[ty + 1][tx - 1] + 2 * tile[ty + 1][tx] +
                  tile[ty + 1][tx + 1]) -
                 (tile[ty - 1][tx - 1] + 2 * tile[ty - 1][tx] +
                  tile[ty - 1][tx + 1]);
  const float magnitude = sqrtf(static_cast<float>(gx * gx + gy * gy));
  const int value = min(255, static_cast<int>(magnitude + 0.5f));
  if (threshold > 0) {
    output[y * width + x] = (value >= threshold) ? 255 : 0;
  } else {
    output[y * width + x] = static_cast<uint8_t>(value);
  }
}

}  // namespace

void UploadGaussianWeights(const float* weights, int count) {
  CUDA_CHECK(cudaMemcpyToSymbol(c_gaussian_weights, weights,
                                sizeof(float) * count));
}

void LaunchRgbToGray(const uint8_t* rgb, uint8_t* gray, int num_pixels,
                     cudaStream_t stream) {
  RgbToGrayKernel<<<GridSize1d(num_pixels), kThreadsPerBlock1d, 0, stream>>>(
      rgb, gray, num_pixels);
  CUDA_CHECK(cudaGetLastError());
}

void LaunchEqualize(const uint8_t* input, uint8_t* output, int num_pixels,
                    unsigned int* histogram, uint8_t* lut,
                    cudaStream_t stream) {
  const int blocks = GridSize1d(num_pixels);
  CUDA_CHECK(cudaMemsetAsync(histogram, 0,
                             kHistogramBins * sizeof(unsigned int), stream));
  HistogramKernel<<<blocks, kThreadsPerBlock1d, 0, stream>>>(
      input, num_pixels, histogram);
  CUDA_CHECK(cudaGetLastError());
  BuildEqualizationLutKernel<<<1, kHistogramBins, 0, stream>>>(
      histogram, num_pixels, lut);
  CUDA_CHECK(cudaGetLastError());
  ApplyLutKernel<<<blocks, kThreadsPerBlock1d, 0, stream>>>(input, lut,
                                                            output, num_pixels);
  CUDA_CHECK(cudaGetLastError());
}

void LaunchGaussianBlur(const uint8_t* input, float* scratch, uint8_t* output,
                        int width, int height, int radius,
                        cudaStream_t stream) {
  const dim3 block(kTileSize, kTileSize);
  const dim3 grid = GridSize2d(width, height);
  GaussianRowKernel<<<grid, block, 0, stream>>>(input, scratch, width, height,
                                                radius);
  CUDA_CHECK(cudaGetLastError());
  GaussianColumnKernel<<<grid, block, 0, stream>>>(scratch, output, width,
                                                   height, radius);
  CUDA_CHECK(cudaGetLastError());
}

void LaunchSobel(const uint8_t* input, uint8_t* output, int width, int height,
                 int threshold, cudaStream_t stream) {
  const dim3 block(kTileSize, kTileSize);
  SobelKernel<<<GridSize2d(width, height), block, 0, stream>>>(
      input, output, width, height, threshold);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace gpu_batch
