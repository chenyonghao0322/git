#include "core/PointCloud.h"

#include <cmath>

void PointCloud::Clear() {
    points.clear();
    colors.clear();
    mask.clear();
    bounds = {};
    originOffset = {0, 0, 0};
    sourcePath.clear();
}

void PointCloud::RecomputeBounds() {
    bounds = {};
    for (const Vec3& p : points) {
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) continue;
        bounds.Expand(p);
    }
}

void PointCloud::CenterToOrigin() {
    if (points.empty() || !bounds.Valid()) return;
    originOffset = bounds.Center();
    for (Vec3& p : points) {
        p = p - originOffset;
    }
    RecomputeBounds();
}

void PointCloud::ApplyHeightColors(float zMin, float zMax) {
    if (points.empty()) return;
    if (zMax <= zMin) {
        zMin = bounds.min.z;
        zMax = bounds.max.z;
        if (zMax <= zMin) {
            zMax = zMin + 1.f;
        }
    }

    colors.resize(points.size());
    const float inv = 1.f / (zMax - zMin);
    for (std::size_t i = 0; i < points.size(); ++i) {
        const float t = (points[i].z - zMin) * inv;
        colors[i] = HeightToColor(t);
    }
}

void PointCloud::ResetMask() {
    mask.assign(points.size(), 1);
}

std::size_t PointCloud::VisibleCount() const {
    if (mask.empty()) return points.size();
    std::size_t n = 0;
    for (uint8_t m : mask) {
        if (m) ++n;
    }
    return n;
}
