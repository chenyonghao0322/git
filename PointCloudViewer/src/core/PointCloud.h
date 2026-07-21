#pragma once

#include "core/MathTypes.h"

#include <cstdint>
#include <string>
#include <vector>

struct PointCloud {
    std::vector<Vec3> points;   // display coordinates (centered)
    std::vector<Vec3> colors;
    std::vector<uint8_t> mask;  // 1 = visible
    Aabb bounds;
    Vec3 originOffset{0, 0, 0};  // world = display + originOffset
    std::string sourcePath;

    void Clear();
    void RecomputeBounds();
    void CenterToOrigin();  // improve float precision for large coordinates
    void ApplyHeightColors(float zMin, float zMax);
    void ResetMask();
    std::size_t VisibleCount() const;
    Vec3 ToWorld(const Vec3& display) const { return display + originOffset; }
};
