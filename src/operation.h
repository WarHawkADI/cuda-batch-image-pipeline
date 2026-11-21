// Copyright 2026 CUDA at Scale Independent Project.
//
// The image operations the batch processor can run.

#ifndef SRC_OPERATION_H_
#define SRC_OPERATION_H_

#include <string>

namespace gpu_batch {

enum class Operation {
  kEqualize,  // Histogram equalization (contrast enhancement).
  kBlur,      // Separable Gaussian blur.
  kSobel,     // Sobel gradient magnitude (edge detection).
  kPipeline,  // Equalize -> Gaussian blur -> Sobel, all on the GPU.
};

// Returns the command-line / file-suffix name of `operation`.
inline const char* OperationName(Operation operation) {
  switch (operation) {
    case Operation::kEqualize:
      return "equalize";
    case Operation::kBlur:
      return "blur";
    case Operation::kSobel:
      return "sobel";
    case Operation::kPipeline:
      return "pipeline";
  }
  return "unknown";
}

// Parses `name` into `operation`. Returns false if the name is not known.
inline bool ParseOperation(const std::string& name, Operation* operation) {
  for (Operation candidate : {Operation::kEqualize, Operation::kBlur,
                              Operation::kSobel, Operation::kPipeline}) {
    if (name == OperationName(candidate)) {
      *operation = candidate;
      return true;
    }
  }
  return false;
}

}  // namespace gpu_batch

#endif  // SRC_OPERATION_H_
