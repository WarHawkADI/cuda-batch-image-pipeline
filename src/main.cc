// Copyright 2026 CUDA at Scale Independent Project.
//
// Entry point: parses flags, runs the GPU batch over the input directory,
// prints a timing summary and optionally verifies results against the CPU.

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "batch_processor.h"
#include "command_line.h"
#include "cuda_check.h"
#include "image_io.h"
#include "operation.h"
#include "reference_filters.h"

namespace gpu_batch {
namespace {

void PrintDeviceInfo() {
  int device = 0;
  cudaDeviceProp properties;
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
  int runtime_version = 0;
  CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
  std::printf("GPU: %s (compute %d.%d, %d SMs, %.0f MiB), CUDA runtime %d.%d\n",
              properties.name, properties.major, properties.minor,
              properties.multiProcessorCount,
              properties.totalGlobalMem / (1024.0 * 1024.0),
              runtime_version / 1000, (runtime_version % 1000) / 10);
}

bool WriteCsv(const std::string& path, const BatchSummary& summary) {
  std::ofstream out(path);
  if (!out) return false;
  out << "image,width,height,channels,gpu_ms\n";
  for (const ImageRecord& record : summary.images) {
    out << FileStem(record.name) << "," << record.width << ","
        << record.height << "," << record.channels << "," << record.gpu_ms
        << "\n";
  }
  return static_cast<bool>(out);
}

// Re-runs the first `count` images on the GPU and on the CPU and compares
// the outputs pixel by pixel. Returns true if every image matches exactly.
bool Verify(const std::vector<std::string>& files, int count,
            const CommandLineOptions& options, BatchProcessor* processor) {
  const std::vector<float> weights = GaussianWeights(options.sigma);
  double cpu_ms_total = 0.0;
  double gpu_ms_total = 0.0;
  int mismatched_images = 0;
  std::printf("\nVerifying %d image(s) against the CPU reference...\n", count);
  for (int i = 0; i < count; ++i) {
    Image image;
    std::string error;
    if (!ReadNetpbm(files[i], &image, &error)) {
      std::fprintf(stderr, "  ERROR: %s\n", error.c_str());
      ++mismatched_images;
      continue;
    }
    float gpu_ms = 0.0f;
    const std::vector<uint8_t> gpu = processor->ProcessImage(image, &gpu_ms);
    const auto cpu_start = std::chrono::steady_clock::now();
    const std::vector<uint8_t> cpu = CpuRunOperation(
        image, options.operation, weights, options.threshold);
    const double cpu_ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - cpu_start)
                              .count();
    size_t differing = 0;
    int max_difference = 0;
    for (size_t p = 0; p < cpu.size(); ++p) {
      const int difference = std::abs(static_cast<int>(gpu[p]) - cpu[p]);
      if (difference != 0) ++differing;
      if (difference > max_difference) max_difference = difference;
    }
    if (differing != 0) ++mismatched_images;
    cpu_ms_total += cpu_ms;
    gpu_ms_total += gpu_ms;
    std::printf("  %-28s cpu=%9.2f ms  gpu=%7.3f ms  speedup=%7.1fx  "
                "differing_pixels=%zu  max_diff=%d  %s\n",
                FileStem(files[i]).c_str(), cpu_ms, gpu_ms, cpu_ms / gpu_ms,
                differing, max_difference, differing == 0 ? "MATCH" : "DIFF");
  }
  std::printf("Verification: %d/%d images bit-exact, overall speedup %.1fx "
              "(CPU %.1f ms vs GPU %.1f ms)\n",
              count - mismatched_images, count, cpu_ms_total / gpu_ms_total,
              cpu_ms_total, gpu_ms_total);
  return mismatched_images == 0;
}

int Run(int argc, char** argv) {
  CommandLineOptions options;
  std::string error;
  if (!ParseCommandLine(argc, argv, &options, &error)) {
    std::fprintf(stderr, "Error: %s\n\n", error.c_str());
    PrintUsage(argv[0]);
    return EXIT_FAILURE;
  }
  if (options.show_help) {
    PrintUsage(argv[0]);
    return EXIT_SUCCESS;
  }

  const std::vector<std::string> files =
      ListImageFiles(options.input_dir, options.max_images);
  if (files.empty()) {
    std::fprintf(stderr, "Error: no .pgm/.ppm images found in '%s'\n",
                 options.input_dir.c_str());
    return EXIT_FAILURE;
  }
  std::error_code mkdir_error;
  std::filesystem::create_directories(options.output_dir, mkdir_error);
  if (mkdir_error) {
    std::fprintf(stderr, "Error: cannot create '%s': %s\n",
                 options.output_dir.c_str(), mkdir_error.message().c_str());
    return EXIT_FAILURE;
  }

  PrintDeviceInfo();
  std::printf("Operation: %s | sigma=%.2f | threshold=%d | streams=%d | "
              "images=%zu | mode=%s\nInput: %s\nOutput: %s\n\n",
              OperationName(options.operation), options.sigma,
              options.threshold, options.num_streams, files.size(),
              options.benchmark ? "benchmark (preloaded, no writes)"
                                : "streaming from disk",
              options.input_dir.c_str(),
              options.benchmark ? "(none)" : options.output_dir.c_str());

  ProcessorConfig config;
  config.operation = options.operation;
  config.sigma = options.sigma;
  config.threshold = options.threshold;
  config.num_streams = options.num_streams;
  config.save_intermediate = options.save_intermediate;
  config.write_outputs = !options.benchmark;
  config.output_dir = options.output_dir;

  BatchProcessor processor(config);
  processor.Initialize();

  std::vector<Image> preloaded;
  Image current;
  ImageSource source = [&files, &current](int index, std::string* error) {
    return ReadNetpbm(files[index], &current, error) ? &current : nullptr;
  };
  if (options.benchmark) {
    preloaded.resize(files.size());
    for (size_t i = 0; i < files.size(); ++i) {
      if (!ReadNetpbm(files[i], &preloaded[i], &error)) {
        std::fprintf(stderr, "Error: %s\n", error.c_str());
        return EXIT_FAILURE;
      }
    }
    // Warm-up run so CUDA context / module setup is not counted.
    float warmup_ms = 0.0f;
    processor.ProcessImage(preloaded.front(), &warmup_ms);
    source = [&preloaded](int index, std::string*) {
      return &preloaded[index];
    };
  }
  const BatchSummary summary = processor.ProcessBatch(files, source);

  double gpu_ms_sum = 0.0;
  for (const ImageRecord& record : summary.images) gpu_ms_sum += record.gpu_ms;
  std::printf(
      "\nSummary\n"
      "  Images processed : %zu (failed: %d)\n"
      "  Total pixels     : %.2f megapixels\n"
      "  Wall-clock time  : %.1f ms (batch loop, incl. host I/O)\n"
      "  Sum of GPU times : %.1f ms (per-stream H2D + kernels + D2H)\n"
      "  Throughput       : %.1f megapixels/s end-to-end\n",
      summary.images.size(), summary.failures, summary.megapixels,
      summary.wall_ms, gpu_ms_sum,
      summary.megapixels / (summary.wall_ms / 1000.0));

  if (!options.csv_path.empty()) {
    if (WriteCsv(options.csv_path, summary)) {
      std::printf("  CSV report       : %s\n", options.csv_path.c_str());
    } else {
      std::fprintf(stderr, "Error: cannot write %s\n",
                   options.csv_path.c_str());
    }
  }

  bool verified = true;
  if (options.verify_count > 0) {
    const int count = std::min<int>(options.verify_count,
                                    static_cast<int>(files.size()));
    verified = Verify(files, count, options, &processor);
  }
  return (summary.failures == 0 && verified) ? EXIT_SUCCESS : EXIT_FAILURE;
}

}  // namespace
}  // namespace gpu_batch

int main(int argc, char** argv) { return gpu_batch::Run(argc, argv); }
