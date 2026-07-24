#pragma once

#include <string>
#include <vector>

namespace Camera2DCalibration {

struct PointPair {
    float imageU = 0.f;
    float imageV = 0.f;
    float robotX = 0.f;
    float robotY = 0.f;
    bool hasImagePoint = false;
    bool hasRobotCoord = false;
};

struct AffineResult {
    double a = 0.0;
    double b = 0.0;
    double c = 0.0;
    double d = 0.0;
    double e = 0.0;
    double f = 0.0;
    double rms = 0.0;
    int pointCount = 0;
    bool valid = false;
};

// 由图像像素 (u,v) 映射到机器人平面坐标 (X,Y)：仿射模型
// X = a*u + b*v + c,  Y = d*u + e*v + f
bool ComputeAffine(const std::vector<PointPair>& pairs, AffineResult& result, std::string& error);

void ImageToRobot(float u, float v, const AffineResult& result, float& outX, float& outY);

struct ErrorStats {
    std::vector<double> perPointError;
    double errMean = 0.0;
    double errMax = 0.0;
    int pointCount = 0;
    bool valid = false;
};

// Halcon 同款：err = sqrt((WorldX-Qx)^2 + (WorldY-Qy)^2)，Qx/Qy 为像素经仿射变换后的坐标
bool ComputeErrorStats(const std::vector<PointPair>& pairs, const AffineResult& result,
                       ErrorStats& stats, std::string& error);

struct DotDetectResult {
    float centerX = 0.f;
    float centerY = 0.f;
    float radius = 0.f;
    bool ok = false;
};

// 在 RGB 图像 ROI 内检测标定圆点（黑白圆点板），输出圆心像素坐标
bool DetectCalibrationDotCenterRgb(const std::vector<uint8_t>& rgb, int width, int height,
                                   float roiX0, float roiY0, float roiX1, float roiY1,
                                   DotDetectResult& result, std::string& error);

}  // namespace Camera2DCalibration
