// Copyright 2026 CUDA at Scale Independent Project.

#include "reference_filters.h"

#include <algorithm>
#include <cmath>

namespace gpu_batch {
namespace {

int ClampIndex(int value, int high) { return std::min(std::max(value, 0), high); }

}  // namespace

std::vector<float> GaussianWeights(float sigma) {
  const int radius = static_cast<int>(std::ceil(3.0f * sigma));
  std::vector<float> weights(2 * radius + 1);
  double total = 0.0;
  for (int k = -radius; k <= radius; ++k) {
    const double value = std::exp(-(k * k) / (2.0 * sigma * sigma));
    weights[k + radius] = static_cast<float>(value);
    total += value;
  }
  for (float& weight : weights) {
    weight = static_cast<float>(weight / total);
  }
  return weights;
}

void CpuRgbToGray(const uint8_t* rgb, int num_pixels, uint8_t* gray) {
  for (int i = 0; i < num_pixels; ++i) {
    gray[i] = static_cast<uint8_t>(
        (77 * rgb[3 * i] + 150 * rgb[3 * i + 1] + 29 * rgb[3 * i + 2] + 128) >>
        8);
  }
}

void CpuEqualize(const uint8_t* input, int num_pixels, uint8_t* output) {
  unsigned int cdf[256] = {0};
  for (int i = 0; i < num_pixels; ++i) ++cdf[input[i]];
  for (int bin = 1; bin < 256; ++bin) cdf[bin] += cdf[bin - 1];
  unsigned int cdf_min = 0;
  for (int bin = 0; bin < 256; ++bin) {
    if (cdf[bin] != 0) {
      cdf_min = cdf[bin];
      break;
    }
  }
  const uint64_t denominator = static_cast<uint64_t>(num_pixels) - cdf_min;
  uint8_t lut[256];
  for (int bin = 0; bin < 256; ++bin) {
    if (denominator == 0) {
      lut[bin] = static_cast<uint8_t>(bin);
    } else if (cdf[bin] < cdf_min) {
      lut[bin] = 0;
    } else {
      const uint64_t numerator =
          static_cast<uint64_t>(cdf[bin] - cdf_min) * 255u;
      lut[bin] =
          static_cast<uint8_t>((numerator + denominator / 2) / denominator);
    }
  }
  for (int i = 0; i < num_pixels; ++i) output[i] = lut[input[i]];
}

void CpuGaussianBlur(const uint8_t* input, int width, int height,
                     const std::vector<float>& weights, uint8_t* output) {
  const int radius = static_cast<int>(weights.size() / 2);
  std::vector<float> rows(static_cast<size_t>(width) * height);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      float sum = 0.0f;
      for (int k = -radius; k <= radius; ++k) {
        const float product =
            weights[k + radius] *
            static_cast<float>(input[y * width + ClampIndex(x + k, width - 1)]);
        sum += product;
      }
      rows[y * width + x] = sum;
    }
  }
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      float sum = 0.0f;
      for (int k = -radius; k <= radius; ++k) {
        const float product =
            weights[k + radius] *
            rows[ClampIndex(y + k, height - 1) * width + x];
        sum += product;
      }
      output[y * width + x] =
          static_cast<uint8_t>(std::min(255, static_cast<int>(sum + 0.5f)));
    }
  }
}

void CpuSobel(const uint8_t* input, int width, int height, int threshold,
              uint8_t* output) {
  auto at = [&](int x, int y) {
    return static_cast<int>(input[ClampIndex(y, height - 1) * width +
                                  ClampIndex(x, width - 1)]);
  };
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const int gx = (at(x + 1, y - 1) + 2 * at(x + 1, y) + at(x + 1, y + 1)) -
                     (at(x - 1, y - 1) + 2 * at(x - 1, y) + at(x - 1, y + 1));
      const int gy = (at(x - 1, y + 1) + 2 * at(x, y + 1) + at(x + 1, y + 1)) -
                     (at(x - 1, y - 1) + 2 * at(x, y - 1) + at(x + 1, y - 1));
      const float magnitude = std::sqrt(static_cast<float>(gx * gx + gy * gy));
      const int value = std::min(255, static_cast<int>(magnitude + 0.5f));
      if (threshold > 0) {
        output[y * width + x] = (value >= threshold) ? 255 : 0;
      } else {
        output[y * width + x] = static_cast<uint8_t>(value);
      }
    }
  }
}

std::vector<uint8_t> CpuRunOperation(const Image& image, Operation operation,
                                     const std::vector<float>& weights,
                                     int threshold) {
  const int num_pixels = static_cast<int>(image.NumPixels());
  std::vector<uint8_t> gray(image.pixels);
  if (image.channels == 3) {
    gray.resize(num_pixels);
    CpuRgbToGray(image.pixels.data(), num_pixels, gray.data());
  }
  std::vector<uint8_t> result(num_pixels);
  switch (operation) {
    case Operation::kEqualize:
      CpuEqualize(gray.data(), num_pixels, result.data());
      break;
    case Operation::kBlur:
      CpuGaussianBlur(gray.data(), image.width, image.height, weights,
                      result.data());
      break;
    case Operation::kSobel:
      CpuSobel(gray.data(), image.width, image.height, threshold,
               result.data());
      break;
    case Operation::kPipeline: {
      std::vector<uint8_t> equalized(num_pixels);
      std::vector<uint8_t> blurred(num_pixels);
      CpuEqualize(gray.data(), num_pixels, equalized.data());
      CpuGaussianBlur(equalized.data(), image.width, image.height, weights,
                      blurred.data());
      CpuSobel(blurred.data(), image.width, image.height, threshold,
               result.data());
      break;
    }
  }
  return result;
}

}  // namespace gpu_batch
