#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ImageIO {

struct GrayImage {
    int width = 0;
    int height = 0;
    std::vector<float> pixels;  // row-major, size = width * height
};

struct RgbImage {
    int width = 0;
    int height = 0;
    std::vector<uint8_t> rgb;  // row-major RGB, size = width * height * 3
};

// Load grayscale (8/16-bit or converted). Pixel values keep raw numeric meaning
// (e.g. uint16 DN stays 0..65535 as float).
bool LoadGray(const std::string& path, GrayImage& out, std::string& error);

// Load as 8-bit RGB (grayscale images are expanded to R=G=B).
bool LoadRgb(const std::string& path, RgbImage& out, std::string& error);

}  // namespace ImageIO
