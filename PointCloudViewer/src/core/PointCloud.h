#pragma once

// PointCloud — 内存中的点云数据结构
//
// 坐标：points 为以包围盒中心为原点的显示坐标；世界坐标 = points + originOffset。
// 编辑：删除/剖切/滤波等操作通常只改 mask（软删除），不物理删点，便于撤销。
// 颜色：colors 可选；为空时由渲染器按 Z 着色或分析结果着色。

#include "core/MathTypes.h"

#include <cstdint>
#include <string>
#include <vector>

struct PointCloud {
    std::vector<Vec3> points;   // 显示坐标（已中心化）
    std::vector<Vec3> colors;   // 逐点颜色，可与 points 等长
    std::vector<uint8_t> mask;  // 1=可见 0=隐藏（软删除，撤销时恢复）
    Aabb bounds;
    Vec3 originOffset{0, 0, 0};  // world = display + originOffset
    std::string sourcePath;

    void Clear();                              // 清空所有数据
    void RecomputeBounds();                    // 按可见点重算包围盒
    void CenterToOrigin();                     // 平移点云使包围盒中心为原点（提高精度）
    void ApplyHeightColors(float zMin, float zMax);  // 按 Z 映射伪彩色到 colors
    void ResetMask();                          // 全部点设为可见
    std::size_t VisibleCount() const;          // 统计 mask==1 的点数
    Vec3 ToWorld(const Vec3& display) const { return display + originOffset; }
};
