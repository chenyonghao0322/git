#pragma once

#include "core/PointCloud.h"

#include <cstdint>
#include <string>
#include <vector>

namespace FilterTools {

// keepMask[i]==1 保留。仅对当前可见点（mask）处理；不可见点保持原 mask。
bool VoxelDownsample(const PointCloud& cloud, float leafSize, std::vector<uint8_t>& keepMask,
                     std::string& error, int* outKept = nullptr);

bool RadiusOutlier(const PointCloud& cloud, float radius, int minNeighbors,
                   std::vector<uint8_t>& keepMask, std::string& error, int* outKept = nullptr);

bool StatisticalOutlier(const PointCloud& cloud, int meanK, float stdMul,
                        std::vector<uint8_t>& keepMask, std::string& error, int* outKept = nullptr);

}  // namespace FilterTools
