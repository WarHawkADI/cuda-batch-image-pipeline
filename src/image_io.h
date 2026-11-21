// Copyright 2026 CUDA at Scale Independent Project.
//
// Minimal, dependency-free image I/O for binary Netpbm files (PGM / PPM).
// The dataset preparation script converts the SIPI TIFF images to these
// formats so the CUDA program does not need FreeImage, OpenCV, etc.

#ifndef SRC_IMAGE_IO_H_
#define SRC_IMAGE_IO_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace gpu_batch {

// An 8-bit image with 1 (grayscale) or 3 (interleaved RGB) channels.
struct Image {
  int width = 0;
  int height = 0;
  int channels = 0;
  std::vector<uint8_t> pixels;

  size_t NumPixels() const { return static_cast<size_t>(width) * height; }
  size_t SizeBytes() const { return NumPixels() * channels; }
};

// Reads a binary PGM (P5) or PPM (P6) file with a max value of 255.
// Returns false and fills `error` on failure.
bool ReadNetpbm(const std::string& path, Image* image, std::string* error);

// Writes an 8-bit grayscale image as a binary PGM (P5) file.
bool WritePgm(const std::string& path, int width, int height,
              const uint8_t* pixels, std::string* error);

// Returns the sorted list of *.pgm / *.ppm files in `directory`. If
// `max_files` is positive, at most that many files are returned.
std::vector<std::string> ListImageFiles(const std::string& directory,
                                        int max_files);

// Returns the file name of `path` without its directory and extension.
std::string FileStem(const std::string& path);

}  // namespace gpu_batch

#endif  // SRC_IMAGE_IO_H_
