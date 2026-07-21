#include "io/PointCloudGenerator.h"

#include <cmath>
#include <random>

namespace PointCloudGenerator {
namespace {

constexpr float kPi = 3.14159265358979323846f;

void FinalizeCloud(PointCloud& cloud, const char* sourceLabel) {
    cloud.RecomputeBounds();
    cloud.CenterToOrigin();
    cloud.ResetMask();
    cloud.colors.clear();
    cloud.sourcePath = sourceLabel;
}

}  // namespace

bool GenerateSphere(const SphereParams& params, PointCloud& out, std::string& error) {
    if (params.radius <= 1e-6f) {
        error = u8"球半径必须大于 0。";
        return false;
    }
    if (params.pointCount < 16) {
        error = u8"点数至少为 16。";
        return false;
    }

    out.Clear();
    out.points.reserve(static_cast<std::size_t>(params.pointCount));

    std::mt19937 rng(42u);
    std::uniform_real_distribution<float> uni01(0.f, 1.f);
    std::normal_distribution<float> noiseN(0.f, params.noise > 0.f ? params.noise : 0.f);

    const Vec3 center{params.centerX, params.centerY, params.centerZ};
    const float golden = kPi * (3.f - std::sqrt(5.f));
    for (int i = 0; i < params.pointCount; ++i) {
        const float y = 1.f - 2.f * (static_cast<float>(i) + 0.5f) /
                                  static_cast<float>(params.pointCount);
        const float rHoriz = std::sqrt(std::max(0.f, 1.f - y * y));
        const float theta = golden * static_cast<float>(i);
        const Vec3 dir{std::cos(theta) * rHoriz, y, std::sin(theta) * rHoriz};
        Vec3 p = center + dir * params.radius;
        if (params.noise > 0.f) p += dir * noiseN(rng);
        out.points.push_back(p);
    }

    FinalizeCloud(out, u8"<生成:球面>");
    return true;
}

bool GenerateCylinder(const CylinderParams& params, PointCloud& out, std::string& error) {
    if (params.radius <= 1e-6f) {
        error = u8"圆柱半径必须大于 0。";
        return false;
    }
    if (params.height <= 1e-6f) {
        error = u8"圆柱高度必须大于 0。";
        return false;
    }
    if (params.pointCount < 32) {
        error = u8"点数至少为 32。";
        return false;
    }

    out.Clear();
    out.points.reserve(static_cast<std::size_t>(params.pointCount));

    std::mt19937 rng(123u);
    std::uniform_real_distribution<float> uni01(0.f, 1.f);
    std::normal_distribution<float> noiseN(0.f, params.noise > 0.f ? params.noise : 0.f);

    const Vec3 center{params.centerX, params.centerY, params.centerZ};
    const float halfH = 0.5f * params.height;

    // Mix of helical rings for even coverage along height + angle
    for (int i = 0; i < params.pointCount; ++i) {
        const float t = (static_cast<float>(i) + 0.5f) / static_cast<float>(params.pointCount);
        const float h = -halfH + t * params.height;
        const float a = 2.f * kPi * t * std::sqrt(static_cast<float>(params.pointCount)) * 0.37f +
                        uni01(rng) * 0.02f;
        float r = params.radius;
        if (params.noise > 0.f) r += noiseN(rng);
        if (r < 1e-4f) r = 1e-4f;
        // Axis along Z
        out.points.push_back(center + Vec3{r * std::cos(a), r * std::sin(a), h});
    }

    FinalizeCloud(out, u8"<生成:圆柱>");
    return true;
}

bool GenerateDisk(const DiskParams& params, PointCloud& out, std::string& error) {
    if (params.radius <= 1e-6f) {
        error = u8"圆面半径必须大于 0。";
        return false;
    }
    if (params.pointCount < 16) {
        error = u8"点数至少为 16。";
        return false;
    }

    out.Clear();
    out.points.reserve(static_cast<std::size_t>(params.pointCount));

    std::mt19937 rng(77u);
    std::uniform_real_distribution<float> uni01(0.f, 1.f);
    std::normal_distribution<float> noiseN(0.f, params.noise > 0.f ? params.noise : 0.f);

    const Vec3 center{params.centerX, params.centerY, params.centerZ};
    // 黄金角螺旋：面积均匀、确定性排布，旋转视角时不闪烁
    const float golden = kPi * (3.f - std::sqrt(5.f));
    for (int i = 0; i < params.pointCount; ++i) {
        const float t = (static_cast<float>(i) + 0.5f) / static_cast<float>(params.pointCount);
        const float r = params.radius * std::sqrt(t);
        const float theta = golden * static_cast<float>(i);
        float z = 0.f;
        if (params.noise > 0.f) z += noiseN(rng);
        out.points.push_back(center + Vec3{r * std::cos(theta), r * std::sin(theta), z});
    }
    (void)uni01;

    FinalizeCloud(out, u8"<生成:圆面>");
    return true;
}

}  // namespace PointCloudGenerator
