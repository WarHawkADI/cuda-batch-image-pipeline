// Copyright 2026 CUDA at Scale Independent Project.

#include "command_line.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>

namespace gpu_batch {
namespace {

constexpr float kMaxSigma = 5.0f;  // ceil(3 * 5) = 15 = kMaxGaussianRadius.
constexpr int kMaxStreams = 32;

bool ParseInt(const std::string& text, int* value) {
  char* end = nullptr;
  errno = 0;
  const long parsed = std::strtol(text.c_str(), &end, 10);
  if (text.empty() || *end != '\0' || errno != 0) return false;
  *value = static_cast<int>(parsed);
  return true;
}

bool ParseFloat(const std::string& text, float* value) {
  char* end = nullptr;
  errno = 0;
  const float parsed = std::strtof(text.c_str(), &end);
  if (text.empty() || *end != '\0' || errno != 0) return false;
  *value = parsed;
  return true;
}

bool ParseBool(const std::string& text, bool* value) {
  if (text == "true" || text == "1") {
    *value = true;
  } else if (text == "false" || text == "0") {
    *value = false;
  } else {
    return false;
  }
  return true;
}

bool ApplyFlag(const std::string& name, const std::string& value,
               CommandLineOptions* options, std::string* error) {
  bool ok = true;
  if (name == "input_dir") {
    options->input_dir = value;
  } else if (name == "output_dir") {
    options->output_dir = value;
  } else if (name == "operation") {
    ok = ParseOperation(value, &options->operation);
  } else if (name == "sigma") {
    ok = ParseFloat(value, &options->sigma) && options->sigma > 0.0f &&
         options->sigma <= kMaxSigma;
  } else if (name == "threshold") {
    ok = ParseInt(value, &options->threshold) && options->threshold >= 0 &&
         options->threshold <= 255;
  } else if (name == "streams") {
    ok = ParseInt(value, &options->num_streams) && options->num_streams >= 1 &&
         options->num_streams <= kMaxStreams;
  } else if (name == "max_images") {
    ok = ParseInt(value, &options->max_images) && options->max_images >= 0;
  } else if (name == "verify") {
    ok = ParseInt(value, &options->verify_count) && options->verify_count >= 0;
  } else if (name == "save_intermediate") {
    ok = ParseBool(value, &options->save_intermediate);
  } else if (name == "benchmark") {
    ok = ParseBool(value, &options->benchmark);
  } else if (name == "csv") {
    options->csv_path = value;
  } else {
    *error = "unknown flag --" + name;
    return false;
  }
  if (!ok) *error = "invalid value '" + value + "' for --" + name;
  return ok;
}

}  // namespace

bool ParseCommandLine(int argc, char** argv, CommandLineOptions* options,
                      std::string* error) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "-h" || arg == "--help") {
      options->show_help = true;
      continue;
    }
    if (arg.rfind("--", 0) != 0) {
      *error = "unexpected argument '" + arg + "'";
      return false;
    }
    std::string name;
    std::string value;
    const size_t equals = arg.find('=');
    if (equals != std::string::npos) {
      name = arg.substr(2, equals - 2);
      value = arg.substr(equals + 1);
    } else {
      name = arg.substr(2);
      // Boolean flags may be given without a value.
      if (name == "save_intermediate") {
        options->save_intermediate = true;
        continue;
      }
      if (name == "benchmark") {
        options->benchmark = true;
        continue;
      }
      if (i + 1 >= argc) {
        *error = "missing value for --" + name;
        return false;
      }
      value = argv[++i];
    }
    if (!ApplyFlag(name, value, options, error)) return false;
  }
  return true;
}

void PrintUsage(const char* program_name) {
  std::printf(
      "Usage: %s [flags]\n"
      "\n"
      "Batch-processes every .pgm/.ppm image in a directory on the GPU.\n"
      "\n"
      "Flags:\n"
      "  --input_dir=DIR        Input images (default: data/input)\n"
      "  --output_dir=DIR       Where results are written "
      "(default: data/output)\n"
      "  --operation=OP         equalize | blur | sobel | pipeline "
      "(default: pipeline)\n"
      "                         pipeline = equalize -> blur -> sobel\n"
      "  --sigma=S              Gaussian sigma, 0 < S <= 5 (default: 1.5)\n"
      "  --threshold=T          Binarize Sobel output at T, 0 = off "
      "(default: 0)\n"
      "  --streams=N            Concurrent CUDA streams, 1..32 (default: 4)\n"
      "  --max_images=N         Process at most N images, 0 = all "
      "(default: 0)\n"
      "  --verify=N             Compare the first N results with the CPU "
      "reference\n"
      "  --save_intermediate    Also write equalized and blurred stages\n"
      "  --benchmark            Preload all images into RAM, skip writing\n"
      "                         outputs: measures GPU throughput only\n"
      "  --csv=FILE             Write per-image timings to a CSV file\n"
      "  -h, --help             Show this message\n",
      program_name);
}

}  // namespace gpu_batch
