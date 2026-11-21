// Copyright 2026 CUDA at Scale Independent Project.
//
// Error-checking helper for CUDA runtime calls. The Google C++ style guide
// discourages exceptions, so a failed CUDA call is treated as a fatal error:
// the location and message are printed and the process exits.

#ifndef SRC_CUDA_CHECK_H_
#define SRC_CUDA_CHECK_H_

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                   \
  do {                                                                     \
    const cudaError_t cuda_check_status = (call);                          \
    if (cuda_check_status != cudaSuccess) {                                \
      std::fprintf(stderr, "CUDA error at %s:%d: %s (%s)\n", __FILE__,     \
                   __LINE__, cudaGetErrorString(cuda_check_status), #call); \
      std::exit(EXIT_FAILURE);                                             \
    }                                                                      \
  } while (0)

#endif  // SRC_CUDA_CHECK_H_
