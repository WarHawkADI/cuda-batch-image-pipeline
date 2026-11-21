// Copyright 2026 CUDA at Scale Independent Project.

#include "batch_processor.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>

#include "cuda_check.h"
#include "kernels.cuh"
#include "reference_filters.h"

namespace gpu_batch {

BatchProcessor::BatchProcessor(const ProcessorConfig& config)
    : config_(config) {}

BatchProcessor::~BatchProcessor() {
  for (Slot& slot : slots_) {
    if (slot.stream != nullptr) cudaStreamSynchronize(slot.stream);
    FreeBuffers(&slot);
    if (slot.start != nullptr) cudaEventDestroy(slot.start);
    if (slot.stop != nullptr) cudaEventDestroy(slot.stop);
    if (slot.stream != nullptr) cudaStreamDestroy(slot.stream);
  }
}

void BatchProcessor::Initialize() {
  const std::vector<float> weights = GaussianWeights(config_.sigma);
  gaussian_radius_ = static_cast<int>(weights.size() / 2);
  UploadGaussianWeights(weights.data(), static_cast<int>(weights.size()));

  slots_.resize(config_.num_streams);
  for (Slot& slot : slots_) {
    CUDA_CHECK(cudaStreamCreateWithFlags(&slot.stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreate(&slot.start));
    CUDA_CHECK(cudaEventCreate(&slot.stop));
  }
}

void BatchProcessor::FreeBuffers(Slot* slot) {
  cudaFreeHost(slot->host_input);
  for (uint8_t*& plane : slot->host_output) {
    cudaFreeHost(plane);
    plane = nullptr;
  }
  cudaFree(slot->device_input);
  cudaFree(slot->device_gray);
  cudaFree(slot->device_equalized);
  cudaFree(slot->device_blurred);
  cudaFree(slot->device_output);
  cudaFree(slot->device_scratch);
  cudaFree(slot->device_histogram);
  cudaFree(slot->device_lut);
  slot->host_input = nullptr;
  slot->device_input = nullptr;
  slot->device_gray = nullptr;
  slot->device_equalized = nullptr;
  slot->device_blurred = nullptr;
  slot->device_output = nullptr;
  slot->device_scratch = nullptr;
  slot->device_histogram = nullptr;
  slot->device_lut = nullptr;
  slot->capacity = 0;
}

void BatchProcessor::EnsureCapacity(Slot* slot, size_t num_pixels) {
  if (num_pixels <= slot->capacity) return;
  // Buffers only grow, so after the largest image no more allocations occur.
  CUDA_CHECK(cudaStreamSynchronize(slot->stream));
  FreeBuffers(slot);
  CUDA_CHECK(cudaMallocHost(&slot->host_input, num_pixels * 3));
  for (uint8_t*& plane : slot->host_output) {
    CUDA_CHECK(cudaMallocHost(&plane, num_pixels));
  }
  CUDA_CHECK(cudaMalloc(&slot->device_input, num_pixels * 3));
  CUDA_CHECK(cudaMalloc(&slot->device_gray, num_pixels));
  CUDA_CHECK(cudaMalloc(&slot->device_equalized, num_pixels));
  CUDA_CHECK(cudaMalloc(&slot->device_blurred, num_pixels));
  CUDA_CHECK(cudaMalloc(&slot->device_output, num_pixels));
  CUDA_CHECK(cudaMalloc(&slot->device_scratch, num_pixels * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&slot->device_histogram,
                        kHistogramBins * sizeof(unsigned int)));
  CUDA_CHECK(cudaMalloc(&slot->device_lut, kHistogramBins));
  slot->capacity = num_pixels;
}

void BatchProcessor::Enqueue(Slot* slot, const Image& image) {
  const int width = image.width;
  const int height = image.height;
  const int num_pixels = static_cast<int>(image.NumPixels());
  cudaStream_t stream = slot->stream;

  std::memcpy(slot->host_input, image.pixels.data(), image.SizeBytes());
  CUDA_CHECK(cudaEventRecord(slot->start, stream));
  CUDA_CHECK(cudaMemcpyAsync(slot->device_input, slot->host_input,
                             image.SizeBytes(), cudaMemcpyHostToDevice,
                             stream));

  const uint8_t* gray = slot->device_input;
  if (image.channels == 3) {
    LaunchRgbToGray(slot->device_input, slot->device_gray, num_pixels, stream);
    gray = slot->device_gray;
  }

  switch (config_.operation) {
    case Operation::kEqualize:
      LaunchEqualize(gray, slot->device_output, num_pixels,
                     slot->device_histogram, slot->device_lut, stream);
      break;
    case Operation::kBlur:
      LaunchGaussianBlur(gray, slot->device_scratch, slot->device_output,
                         width, height, gaussian_radius_, stream);
      break;
    case Operation::kSobel:
      LaunchSobel(gray, slot->device_output, width, height, config_.threshold,
                  stream);
      break;
    case Operation::kPipeline:
      LaunchEqualize(gray, slot->device_equalized, num_pixels,
                     slot->device_histogram, slot->device_lut, stream);
      LaunchGaussianBlur(slot->device_equalized, slot->device_scratch,
                         slot->device_blurred, width, height,
                         gaussian_radius_, stream);
      LaunchSobel(slot->device_blurred, slot->device_output, width, height,
                  config_.threshold, stream);
      if (config_.save_intermediate) {
        CUDA_CHECK(cudaMemcpyAsync(slot->host_output[kEqualizedPlane],
                                   slot->device_equalized, num_pixels,
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(slot->host_output[kBlurredPlane],
                                   slot->device_blurred, num_pixels,
                                   cudaMemcpyDeviceToHost, stream));
      }
      break;
  }

  CUDA_CHECK(cudaMemcpyAsync(slot->host_output[kFinalPlane],
                             slot->device_output, num_pixels,
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaEventRecord(slot->stop, stream));

  slot->busy = true;
  slot->width = width;
  slot->height = height;
  slot->channels = image.channels;
}

void BatchProcessor::Finish(Slot* slot, int total, BatchSummary* summary) {
  CUDA_CHECK(cudaEventSynchronize(slot->stop));
  slot->busy = false;

  ImageRecord record;
  record.name = slot->name;
  record.width = slot->width;
  record.height = slot->height;
  record.channels = slot->channels;
  CUDA_CHECK(cudaEventElapsedTime(&record.gpu_ms, slot->start, slot->stop));

  const std::filesystem::path out_dir(config_.output_dir);
  const std::string stem = FileStem(slot->name);
  std::string error;
  bool ok = !config_.write_outputs || WritePgm(
      (out_dir / (stem + "_" + OperationName(config_.operation) + ".pgm"))
          .string(),
      slot->width, slot->height, slot->host_output[kFinalPlane], &error);
  if (ok && config_.write_outputs && config_.save_intermediate &&
      config_.operation == Operation::kPipeline) {
    ok = WritePgm((out_dir / (stem + "_1_equalized.pgm")).string(),
                  slot->width, slot->height,
                  slot->host_output[kEqualizedPlane], &error) &&
         WritePgm((out_dir / (stem + "_2_blurred.pgm")).string(), slot->width,
                  slot->height, slot->host_output[kBlurredPlane], &error);
  }
  if (!ok) {
    std::fprintf(stderr, "  ERROR: %s\n", error.c_str());
    ++summary->failures;
    return;
  }

  summary->megapixels +=
      static_cast<double>(slot->width) * slot->height / 1.0e6;
  summary->images.push_back(record);
  if (config_.verbose) {
    std::printf("[%4d/%d] %-28s %5dx%-5d ch=%d  stream=%ld  gpu=%7.3f ms\n",
                slot->index + 1, total, stem.c_str(), slot->width,
                slot->height, slot->channels,
                static_cast<long>(slot - slots_.data()), record.gpu_ms);
  }
}

BatchSummary BatchProcessor::ProcessBatch(
    const std::vector<std::string>& names, const ImageSource& source) {
  BatchSummary summary;
  const int total = static_cast<int>(names.size());
  const auto wall_start = std::chrono::steady_clock::now();

  for (int i = 0; i < total; ++i) {
    Slot& slot = slots_[i % slots_.size()];
    if (slot.busy) Finish(&slot, total, &summary);

    // Loading this image overlaps with GPU work on the other streams.
    std::string error;
    const Image* image = source(i, &error);
    if (image == nullptr) {
      std::fprintf(stderr, "  ERROR: %s\n", error.c_str());
      ++summary.failures;
      continue;
    }
    EnsureCapacity(&slot, image->NumPixels());
    slot.index = i;
    slot.name = names[i];
    Enqueue(&slot, *image);
  }
  // Drain in submission order so the log stays ordered.
  for (int i = 0; i < total; ++i) {
    Slot& slot = slots_[i % slots_.size()];
    if (slot.busy && slot.index == i) Finish(&slot, total, &summary);
  }

  const auto wall_stop = std::chrono::steady_clock::now();
  summary.wall_ms =
      std::chrono::duration<double, std::milli>(wall_stop - wall_start)
          .count();
  return summary;
}

std::vector<uint8_t> BatchProcessor::ProcessImage(const Image& image,
                                                  float* gpu_ms) {
  Slot& slot = slots_.front();
  EnsureCapacity(&slot, image.NumPixels());
  Enqueue(&slot, image);
  CUDA_CHECK(cudaEventSynchronize(slot.stop));
  slot.busy = false;
  CUDA_CHECK(cudaEventElapsedTime(gpu_ms, slot.start, slot.stop));
  const uint8_t* result = slot.host_output[kFinalPlane];
  return std::vector<uint8_t>(result, result + image.NumPixels());
}

}  // namespace gpu_batch
