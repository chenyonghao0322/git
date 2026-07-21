#include "tools/PclAdapter.h"

#include <cmath>

namespace PclAdapter {
namespace {

inline bool Visible(const PointCloud& cloud, std::size_t i) {
    return cloud.mask.empty() || cloud.mask[i];
}

}  // namespace

pcl::PointCloud<pcl::PointXYZ>::Ptr ToPclXYZ(const PointCloud& cloud,
                                             const std::vector<std::size_t>& indices,
                                             std::vector<std::size_t>& map) {
    auto out = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
    map.clear();

    if (!indices.empty()) {
        map.reserve(indices.size());
        out->reserve(indices.size());
        for (std::size_t idx : indices) {
            if (idx >= cloud.points.size() || !Visible(cloud, idx)) continue;
            const Vec3& p = cloud.points[idx];
            out->push_back(pcl::PointXYZ(p.x, p.y, p.z));
            map.push_back(idx);
        }
        return out;
    }

    map.reserve(cloud.points.size());
    out->reserve(cloud.points.size());
    for (std::size_t i = 0; i < cloud.points.size(); ++i) {
        if (!Visible(cloud, i)) continue;
        const Vec3& p = cloud.points[i];
        out->push_back(pcl::PointXYZ(p.x, p.y, p.z));
        map.push_back(i);
    }
    return out;
}

void ApplyKeptIndices(const PointCloud& cloud, const std::vector<std::size_t>& map,
                      const std::vector<int>& keptPclIndices, std::vector<uint8_t>& keepMask) {
    keepMask.assign(cloud.points.size(), 0);
    for (std::size_t i = 0; i < cloud.points.size(); ++i) {
        if (!Visible(cloud, i)) {
            keepMask[i] = 0;
            continue;
        }
    }
    for (int pclIdx : keptPclIndices) {
        if (pclIdx < 0 || static_cast<std::size_t>(pclIdx) >= map.size()) continue;
        keepMask[map[static_cast<std::size_t>(pclIdx)]] = 1;
    }
}

}  // namespace PclAdapter
