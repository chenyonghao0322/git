#include "tools/FilterTools.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <vector>

namespace {

inline bool Visible(const PointCloud& cloud, std::size_t i) {
    return cloud.mask.empty() || cloud.mask[i];
}

inline std::int64_t PackKey(int ix, int iy, int iz) {
    // 21-bit per axis packed into 63 bits
    constexpr std::int64_t bias = 1 << 20;
    return ((static_cast<std::int64_t>(ix) + bias) << 42) |
           ((static_cast<std::int64_t>(iy) + bias) << 21) |
           (static_cast<std::int64_t>(iz) + bias);
}

struct GridHash {
    float cell = 1.f;
    float inv = 1.f;
    std::unordered_map<std::int64_t, std::vector<std::size_t>> buckets;

    void Build(const PointCloud& cloud, float cellSize) {
        cell = std::max(cellSize, 1e-6f);
        inv = 1.f / cell;
        buckets.clear();
        buckets.reserve(cloud.points.size() / 4 + 1);
        for (std::size_t i = 0; i < cloud.points.size(); ++i) {
            if (!Visible(cloud, i)) continue;
            const Vec3& p = cloud.points[i];
            const int ix = static_cast<int>(std::floor(p.x * inv));
            const int iy = static_cast<int>(std::floor(p.y * inv));
            const int iz = static_cast<int>(std::floor(p.z * inv));
            buckets[PackKey(ix, iy, iz)].push_back(i);
        }
    }

    void Neighbors(const Vec3& p, float radius, std::vector<std::size_t>& out) const {
        out.clear();
        const float r2 = radius * radius;
        const int r = std::max(1, static_cast<int>(std::ceil(radius * inv)));
        const int cx = static_cast<int>(std::floor(p.x * inv));
        const int cy = static_cast<int>(std::floor(p.y * inv));
        const int cz = static_cast<int>(std::floor(p.z * inv));
        for (int ix = cx - r; ix <= cx + r; ++ix) {
            for (int iy = cy - r; iy <= cy + r; ++iy) {
                for (int iz = cz - r; iz <= cz + r; ++iz) {
                    const auto it = buckets.find(PackKey(ix, iy, iz));
                    if (it == buckets.end()) continue;
                    for (std::size_t j : it->second) {
                        const Vec3 d = cloudPts_->at(j) - p;
                        if (d.LengthSq() <= r2) out.push_back(j);
                    }
                }
            }
        }
    }

    const std::vector<Vec3>* cloudPts_ = nullptr;
};

void InitKeepFromVisible(const PointCloud& cloud, std::vector<uint8_t>& keepMask) {
    keepMask.resize(cloud.points.size());
    for (std::size_t i = 0; i < cloud.points.size(); ++i) {
        keepMask[i] = Visible(cloud, i) ? 1 : 0;
    }
}

}  // namespace

namespace FilterTools {

bool VoxelDownsample(const PointCloud& cloud, float leafSize, std::vector<uint8_t>& keepMask,
                     std::string& error, int* outKept) {
    if (cloud.points.empty()) {
        error = u8"点云为空";
        return false;
    }
    if (leafSize <= 0.f) {
        error = u8"体素边长必须 > 0";
        return false;
    }

    InitKeepFromVisible(cloud, keepMask);
    const float inv = 1.f / leafSize;
    std::unordered_map<std::int64_t, std::size_t> chosen;
    chosen.reserve(cloud.points.size() / 8 + 1);

    for (std::size_t i = 0; i < cloud.points.size(); ++i) {
        if (!Visible(cloud, i)) {
            keepMask[i] = 0;
            continue;
        }
        const Vec3& p = cloud.points[i];
        const int ix = static_cast<int>(std::floor(p.x * inv));
        const int iy = static_cast<int>(std::floor(p.y * inv));
        const int iz = static_cast<int>(std::floor(p.z * inv));
        const std::int64_t key = PackKey(ix, iy, iz);
        auto it = chosen.find(key);
        if (it == chosen.end()) {
            chosen.emplace(key, i);
            keepMask[i] = 1;
        } else {
            keepMask[i] = 0;
        }
    }

    int kept = 0;
    for (uint8_t m : keepMask)
        if (m) ++kept;
    if (outKept) *outKept = kept;
    return true;
}

bool RadiusOutlier(const PointCloud& cloud, float radius, int minNeighbors,
                   std::vector<uint8_t>& keepMask, std::string& error, int* outKept) {
    if (cloud.points.empty()) {
        error = u8"点云为空";
        return false;
    }
    if (radius <= 0.f || minNeighbors < 1) {
        error = u8"半径与最少邻居数无效";
        return false;
    }

    InitKeepFromVisible(cloud, keepMask);
    GridHash grid;
    grid.cloudPts_ = &cloud.points;
    grid.Build(cloud, radius);

    std::vector<std::size_t> neigh;
    neigh.reserve(64);
    for (std::size_t i = 0; i < cloud.points.size(); ++i) {
        if (!Visible(cloud, i)) {
            keepMask[i] = 0;
            continue;
        }
        grid.Neighbors(cloud.points[i], radius, neigh);
        // neigh includes self
        keepMask[i] = (static_cast<int>(neigh.size()) >= minNeighbors + 1) ? 1 : 0;
    }

    int kept = 0;
    for (uint8_t m : keepMask)
        if (m) ++kept;
    if (outKept) *outKept = kept;
    return true;
}

bool StatisticalOutlier(const PointCloud& cloud, int meanK, float stdMul,
                        std::vector<uint8_t>& keepMask, std::string& error, int* outKept) {
    if (cloud.points.empty()) {
        error = u8"点云为空";
        return false;
    }
    if (meanK < 2 || stdMul <= 0.f) {
        error = u8"统计滤波参数无效";
        return false;
    }

    InitKeepFromVisible(cloud, keepMask);

    // Estimate search radius from bounds diagonal / point density heuristic
    float diag = cloud.bounds.Valid() ? cloud.bounds.Diagonal() : 1.f;
    const std::size_t vis = cloud.VisibleCount();
    const float densityLen =
        (vis > 0) ? diag / std::cbrt(static_cast<float>(vis)) : diag * 0.01f;
    const float searchR = std::max(densityLen * static_cast<float>(meanK) * 0.8f, densityLen * 2.f);

    GridHash grid;
    grid.cloudPts_ = &cloud.points;
    grid.Build(cloud, searchR * 0.5f);

    std::vector<float> meanDist(cloud.points.size(), 0.f);
    std::vector<std::size_t> neigh;
    neigh.reserve(static_cast<std::size_t>(meanK) * 4);
    std::vector<float> dists;
    dists.reserve(64);

    double sumMean = 0.0;
    int counted = 0;

    for (std::size_t i = 0; i < cloud.points.size(); ++i) {
        if (!Visible(cloud, i)) continue;
        grid.Neighbors(cloud.points[i], searchR, neigh);
        dists.clear();
        for (std::size_t j : neigh) {
            if (j == i) continue;
            dists.push_back((cloud.points[j] - cloud.points[i]).Length());
        }
        if (dists.empty()) {
            meanDist[i] = searchR * 2.f;
        } else {
            const int k = std::min(meanK, static_cast<int>(dists.size()));
            std::nth_element(dists.begin(), dists.begin() + k, dists.end());
            double acc = 0.0;
            for (int t = 0; t < k; ++t) acc += dists[static_cast<std::size_t>(t)];
            meanDist[i] = static_cast<float>(acc / static_cast<double>(k));
        }
        sumMean += meanDist[i];
        ++counted;
    }

    if (counted < 3) {
        error = u8"可见点过少，无法统计滤波";
        return false;
    }

    const double mean = sumMean / static_cast<double>(counted);
    double varAcc = 0.0;
    for (std::size_t i = 0; i < cloud.points.size(); ++i) {
        if (!Visible(cloud, i)) continue;
        const double d = static_cast<double>(meanDist[i]) - mean;
        varAcc += d * d;
    }
    const double stdev = std::sqrt(varAcc / static_cast<double>(counted));
    const float thresh = static_cast<float>(mean + static_cast<double>(stdMul) * stdev);

    for (std::size_t i = 0; i < cloud.points.size(); ++i) {
        if (!Visible(cloud, i)) {
            keepMask[i] = 0;
            continue;
        }
        keepMask[i] = (meanDist[i] <= thresh) ? 1 : 0;
    }

    int kept = 0;
    for (uint8_t m : keepMask)
        if (m) ++kept;
    if (outKept) *outKept = kept;
    return true;
}

}  // namespace FilterTools
