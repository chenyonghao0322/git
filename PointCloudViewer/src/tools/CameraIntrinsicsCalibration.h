#pragma once

#include <string>
#include <vector>

namespace CameraIntrinsicsCalibration {

struct PatternConfig {
    int innerCols = 9;      // 棋盘内角点列数
    int innerRows = 6;      // 棋盘内角点行数
    double squareSizeMm = 10.0;  // 方格边长（mm），用于世界坐标尺度
};

struct CornerDetectResult {
    std::vector<float> cornersX;  // 与 cornersY 等长，像素坐标
    std::vector<float> cornersY;
    bool ok = false;
};

struct ImageObservation {
    const std::vector<uint8_t>* rgb = nullptr;
    int width = 0;
    int height = 0;
    const std::vector<float>* cornersX = nullptr;
    const std::vector<float>* cornersY = nullptr;
    bool hasCorners = false;
};

struct IntrinsicsResult {
    double fx = 0.0;
    double fy = 0.0;
    double cx = 0.0;
    double cy = 0.0;
    double k1 = 0.0;
    double k2 = 0.0;
    double p1 = 0.0;
    double p2 = 0.0;
    double k3 = 0.0;
    int imageWidth = 0;
    int imageHeight = 0;
    int imageCount = 0;
    int cornerCountPerImage = 0;
    double reprojMean = 0.0;
    double reprojMax = 0.0;
    std::vector<double> perImageReprojError;
    bool valid = false;
};

// 在整幅 RGB 图像中检测棋盘角点
bool DetectChessboardCorners(const std::vector<uint8_t>& rgb, int width, int height,
                             const PatternConfig& pattern, CornerDetectResult& result,
                             std::string& error);

// 由多张已检测角点的图像求解相机内参
bool CalibrateIntrinsics(const std::vector<ImageObservation>& images,
                         const PatternConfig& pattern, IntrinsicsResult& result,
                         std::string& error);

}  // namespace CameraIntrinsicsCalibration
