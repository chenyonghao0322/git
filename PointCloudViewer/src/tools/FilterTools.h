#pragma once

// FilterTools — 自研点云滤波（体素降采样、半径离群、统计离群）
//
// 输出 keepMask，由 Application 做预览对比后再写入 cloud_.mask。

#include "core/PointCloud.h"

#include <cstdint>
#include <string>
#include <vector>

namespace FilterTools {

// 体素网格降采样：每个体素保留一个代表点
bool VoxelDownsample(const PointCloud& cloud, float leafSize, std::vector<uint8_t>& keepMask,
                     std::string& error, int* outKept = nullptr);

// 半径离群点剔除：邻域点数不足则标记为离群
bool RadiusOutlier(const PointCloud& cloud, float radius, int minNeighbors,
                   std::vector<uint8_t>& keepMask, std::string& error, int* outKept = nullptr);

// 统计离群点剔除：基于邻域距离均值与标准差
bool StatisticalOutlier(const PointCloud& cloud, int meanK, float stdMul,
                        std::vector<uint8_t>& keepMask, std::string& error, int* outKept = nullptr);

}  // namespace FilterTools
