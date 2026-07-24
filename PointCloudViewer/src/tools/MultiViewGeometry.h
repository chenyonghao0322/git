#pragma once

#include "core/MathTypes.h"

#include <string>
#include <vector>

namespace MultiViewGeometry {

struct Correspondence {
    float u1 = 0.f;
    float v1 = 0.f;
    float u2 = 0.f;
    float v2 = 0.f;
};

struct CameraIntrinsics {
    double fx = 1000.0;
    double fy = 1000.0;
    double cx = 0.0;
    double cy = 0.0;
};

struct FundamentalResult {
    double f[9] = {};
    std::vector<uint8_t> inlierMask;
    double meanEpipolarError = 0.0;
    double maxEpipolarError = 0.0;
    int inlierCount = 0;
    bool valid = false;
};

struct TriangulationResult {
    std::vector<Vec3> points;
    std::vector<double> reprojectionErrors;
    double meanReprojError = 0.0;
    double maxReprojError = 0.0;
    bool valid = false;
};

// 由对应点估计基础矩阵 F（RANSAC）
bool ComputeFundamental(const std::vector<Correspondence>& pairs, FundamentalResult& result,
                        std::string& error, double ransacThresh = 1.0);

// 已知内参 K 时，由 F 与对应点恢复位姿并三角化
bool TriangulateWithIntrinsics(const std::vector<Correspondence>& pairs,
                               const CameraIntrinsics& intrinsics,
                               const FundamentalResult& fundamental,
                               TriangulationResult& result, std::string& error);

// 计算点 (u,v) 在图像 1 上对应于图像 2 的极线 ax+by+c=0
bool EpipolarLineInImage2(const FundamentalResult& fundamental, float u1, float v1, double& a,
                          double& b, double& c);

}  // namespace MultiViewGeometry
