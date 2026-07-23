#pragma once

#include "core/PointCloud.h"

#include <string>

namespace PointCloudGenerator {

struct SphereParams {
    float radius = 10.f;
    int pointCount = 20000;
    float noise = 0.f;  // 沿法向噪声（绝对单位），0 = 光滑球面
    float centerX = 0.f;
    float centerY = 0.f;
    float centerZ = 0.f;
};

struct CylinderParams {
    float radius = 8.f;
    float height = 30.f;
    int pointCount = 30000;
    float noise = 0.05f;  // 径向噪声
    float centerX = 0.f;
    float centerY = 0.f;
    float centerZ = 0.f;  // 圆柱中心
};

struct DiskParams {
    float radius = 10.f;
    int pointCount = 20000;
    float noise = 0.f;  // 法向（Z）噪声，0 = 完全平整
    float centerX = 0.f;
    float centerY = 0.f;
    float centerZ = 0.f;
};

struct PlaneParams {
    float extentX = 20.f;  // X 方向半宽
    float extentY = 20.f;  // Y 方向半宽
    int pointCount = 20000;
    float noise = 0.f;  // 法向（Z）噪声
    float centerX = 0.f;
    float centerY = 0.f;
    float centerZ = 0.f;
};

bool GenerateSphere(const SphereParams& params, PointCloud& out, std::string& error,
                    bool centerToOrigin = true);
bool GenerateCylinder(const CylinderParams& params, PointCloud& out, std::string& error);
bool GenerateDisk(const DiskParams& params, PointCloud& out, std::string& error);
bool GeneratePlane(const PlaneParams& params, PointCloud& out, std::string& error);

}  // namespace PointCloudGenerator
