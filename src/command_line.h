// Copyright 2026 CUDA at Scale Independent Project.
//
// Command-line parsing. Flags use the --name=value or --name value forms.

#ifndef SRC_COMMAND_LINE_H_
#define SRC_COMMAND_LINE_H_

#include <string>

#include "operation.h"

namespace gpu_batch {

struct CommandLineOptions {
  std::string input_dir = "data/input";
  std::string output_dir = "data/output";
  Operation operation = Operation::kPipeline;
  float sigma = 1.5f;
  int threshold = 0;
  int num_streams = 4;
  int max_images = 0;    // 0 = all images in the input directory.
  int verify_count = 0;  // Number of images to check against the CPU.
  bool save_intermediate = false;
  bool benchmark = false;  // Preload images into RAM and skip writing.
  std::string csv_path;    // Empty = no CSV report.
  bool show_help = false;
};

// Parses argv into `options`. Returns false and fills `error` on bad input.
bool ParseCommandLine(int argc, char** argv, CommandLineOptions* options,
                      std::string* error);

// Prints the usage message to stdout.
void PrintUsage(const char* program_name);

}  // namespace gpu_batch

#endif  // SRC_COMMAND_LINE_H_
