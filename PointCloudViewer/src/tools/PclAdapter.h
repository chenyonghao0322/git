#pragma once

#include "core/PointCloud.h"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <cstddef>
#include <vector>

namespace PclAdapter {

// 将可见点（及可选 indices）导出为 PCL 点云，map[i] 为原始点索引。
pcl::PointCloud<pcl::PointXYZ>::Ptr ToPclXYZ(const PointCloud& cloud,
                                             const std::vector<std::size_t>& indices,
                                             std::vector<std::size_t>& map);

// 根据 PCL 保留的子集索引（指向 map 的下标）写回 keepMask。
void ApplyKeptIndices(const PointCloud& cloud, const std::vector<std::size_t>& map,
                      const std::vector<int>& keptPclIndices, std::vector<uint8_t>& keepMask);

}  // namespace PclAdapter
