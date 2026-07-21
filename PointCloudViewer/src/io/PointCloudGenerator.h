#pragma once

#include "core/PointCloud.h"

#include <string>

namespace PointCloudGenerator {

struct SphereParams {
    float radius = 10.f;
    int pointCount = 20000;
    float noise = 0.05f;  // 径向噪声幅度（绝对单位）
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

bool GenerateSphere(const SphereParams& params, PointCloud& out, std::string& error);
bool GenerateCylinder(const CylinderParams& params, PointCloud& out, std::string& error);

}  // namespace PointCloudGenerator
