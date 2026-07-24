#pragma once

// PointCloudIO — 点云与深度图读写
//
// 支持 PLY/PCD/XYZ/OBJ 加载与导出；深度图转点云（DepthMapParams）。

#include "core/PointCloud.h"

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace PointCloudIO {

struct DepthMapParams {
    // XY：线扫图像素间隔（mm/px），X 通常为激光线方向，Y 为扫描/编码器方向
    float pixelSizeX = 0.05f;
    float pixelSizeY = 0.05f;

    // Z：线扫 16bit 量化深度  Z(mm) = raw × (zFullRangeMm / zQuantDivisor) + zOffset
    bool useQuantizedZ = true;
    float zFullRangeMm = 80.f;     // Z 向全量程 (mm)，按相机规格填写
    float zQuantDivisor = 65536.f; // 量化除数，常见 65536 (2^16)

    // 非量化深度（float TIFF 等已是 mm）：Z = raw × depthScale + zOffset
    float depthScale = 1.f;
    float zOffset = 0.f;

    float invalidValue = 0.f;  // skip when |raw - invalidValue| <= invalidEps
    float invalidEps = 1e-6f;
    bool skipNonFinite = true;
    bool skipZero = true;         // 跳过 raw≈0（线扫常见无效）
    bool skipUint16Max = true;    // 跳过 raw≈65535（16bit 无效标记）
    bool flipY = false;           // true: image row0 at +Y top
    bool swapXY = false;          // true: 列→Y、行→X（部分相机坐标系）
    bool centerToOrigin = true;   // 转换后平移到包围盒中心
    int step = 1;                 // subsample >= 1
};

// 当前参数下的 Z 换算系数 (mm/DN)
inline float EffectiveDepthScale(const DepthMapParams& params) {
    if (params.useQuantizedZ) {
        if (!(params.zQuantDivisor > 0.f) || !std::isfinite(params.zQuantDivisor)) return 0.f;
        if (!std::isfinite(params.zFullRangeMm)) return 0.f;
        return params.zFullRangeMm / params.zQuantDivisor;
    }
    return params.depthScale;
}

struct DepthGrayStats {
    std::size_t totalPixels = 0;
    std::size_t validCount = 0;
    std::size_t zeroCount = 0;
    std::size_t maxUint16Count = 0;
    float rawMin = 0.f;
    float rawMax = 0.f;
    float validMin = 0.f;
    float validMax = 0.f;
};

// 统计深度图有效像素范围（用于 UI 提示与推荐参数）
DepthGrayStats AnalyzeDepthGray(const std::vector<float>& depth, int width, int height);

// 根据深度图数值分布给出推荐转换参数（不覆盖像素尺寸）
void SuggestDepthMapParams(const std::vector<float>& depth, int width, int height,
                           DepthMapParams& inOut);

bool Load(const std::string& path, PointCloud& out, std::string& error);  // PLY/PCD/XYZ/OBJ

// 保存为世界坐标；visibleOnly 时跳过 mask==0 的点
bool Save(const std::string& path, const PointCloud& cloud, std::string& error,
          bool visibleOnly = true);

// 深度图（+可选亮度图）转点云（从文件）
bool LoadDepthMaps(const std::string& depthPath, const std::string& brightnessPath,
                   const DepthMapParams& params, PointCloud& out, std::string& error);

// 深度图（+可选亮度 RGB）转点云（内存数据，已打开的深度/亮度图）
bool BuildFromDepthGray(const std::vector<float>& depth, int width, int height,
                        const std::vector<uint8_t>* brightnessRgb,
                        const DepthMapParams& params, PointCloud& out, std::string& error,
                        const std::string& sourcePath = {});

}  // namespace PointCloudIO
