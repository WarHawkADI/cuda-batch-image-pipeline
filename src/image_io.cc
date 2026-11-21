// Copyright 2026 CUDA at Scale Independent Project.

#include "image_io.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <istream>

namespace gpu_batch {
namespace {

// Reads the next integer token from a Netpbm header, skipping whitespace and
// '#' comment lines.
bool ReadHeaderInt(std::istream& in, int* value) {
  int next = in.peek();
  while (next != EOF) {
    if (std::isspace(next)) {
      in.get();
    } else if (next == '#') {
      std::string comment;
      std::getline(in, comment);
    } else {
      break;
    }
    next = in.peek();
  }
  return static_cast<bool>(in >> *value);
}

std::string ToLower(std::string text) {
  std::transform(text.begin(), text.end(), text.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return text;
}

}  // namespace

bool ReadNetpbm(const std::string& path, Image* image, std::string* error) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    *error = "cannot open " + path;
    return false;
  }
  char magic[2] = {0, 0};
  if (!in.read(magic, 2) || magic[0] != 'P' ||
      (magic[1] != '5' && magic[1] != '6')) {
    *error = path + ": only binary PGM (P5) and PPM (P6) are supported";
    return false;
  }
  int width = 0;
  int height = 0;
  int max_value = 0;
  if (!ReadHeaderInt(in, &width) || !ReadHeaderInt(in, &height) ||
      !ReadHeaderInt(in, &max_value)) {
    *error = path + ": malformed header";
    return false;
  }
  if (width <= 0 || height <= 0 || max_value <= 0 || max_value > 255) {
    *error = path + ": unsupported dimensions or bit depth";
    return false;
  }
  in.get();  // Exactly one whitespace byte separates header and raster.

  image->width = width;
  image->height = height;
  image->channels = (magic[1] == '6') ? 3 : 1;
  image->pixels.resize(image->SizeBytes());
  if (!in.read(reinterpret_cast<char*>(image->pixels.data()),
               static_cast<std::streamsize>(image->pixels.size()))) {
    *error = path + ": truncated pixel data";
    return false;
  }
  return true;
}

bool WritePgm(const std::string& path, int width, int height,
              const uint8_t* pixels, std::string* error) {
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    *error = "cannot create " + path;
    return false;
  }
  out << "P5\n" << width << " " << height << "\n255\n";
  out.write(reinterpret_cast<const char*>(pixels),
            static_cast<std::streamsize>(width) * height);
  if (!out) {
    *error = "failed writing " + path;
    return false;
  }
  return true;
}

std::vector<std::string> ListImageFiles(const std::string& directory,
                                        int max_files) {
  std::vector<std::string> files;
  std::error_code error;
  for (const auto& entry :
       std::filesystem::directory_iterator(directory, error)) {
    if (!entry.is_regular_file()) continue;
    const std::string extension = ToLower(entry.path().extension().string());
    if (extension == ".pgm" || extension == ".ppm") {
      files.push_back(entry.path().string());
    }
  }
  std::sort(files.begin(), files.end());
  if (max_files > 0 && static_cast<int>(files.size()) > max_files) {
    files.resize(max_files);
  }
  return files;
}

std::string FileStem(const std::string& path) {
  return std::filesystem::path(path).stem().string();
}

}  // namespace gpu_batch
