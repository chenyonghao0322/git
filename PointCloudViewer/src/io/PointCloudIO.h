#pragma once

// PointCloudIO — 点云与深度图读写
//
// 支持 PLY/PCD/XYZ/OBJ 加载与导出；深度图转点云（DepthMapParams）。

#include "core/PointCloud.h"

#include <string>

namespace PointCloudIO {

struct DepthMapParams {
    float pixelSizeX = 0.05f;  // mm / pixel
    float pixelSizeY = 0.05f;
    float depthScale = 1.f;    // z = raw * depthScale (mm)
    float zOffset = 0.f;
    float invalidValue = 0.f;  // skip when |raw - invalidValue| <= invalidEps
    float invalidEps = 1e-6f;
    bool skipNonFinite = true;
    bool flipY = false;        // true: image row0 at +Y top
    int step = 1;              // subsample >= 1
};

bool Load(const std::string& path, PointCloud& out, std::string& error);  // PLY/PCD/XYZ/OBJ

// 保存为世界坐标；visibleOnly 时跳过 mask==0 的点
bool Save(const std::string& path, const PointCloud& cloud, std::string& error,
          bool visibleOnly = true);

// 深度图（+可选亮度图）转点云
bool LoadDepthMaps(const std::string& depthPath, const std::string& brightnessPath,
                   const DepthMapParams& params, PointCloud& out, std::string& error);

}  // namespace PointCloudIO
