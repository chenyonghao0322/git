#pragma once

#include "core/MathTypes.h"
#include "core/PointCloud.h"
#include "render/Camera.h"

#include <optional>
#include <string>
#include <vector>

enum class ToolMode {
    Navigate = 0,
    Pick,
    Distance,
    PlaneFit,
    SphereFit,
    CircleFit,
    CylinderFit,
    Roi,
    ClipPlane,
    Section,
    StepHeight,
    Flatness,   // 平面度
    StepGap     // 段差（区域A平面 → 区域B距离图）
};

struct PlaneModel {
    Vec3 centroid{0, 0, 0};
    Vec3 normal{0, 0, 1};
    float rms = 0.f;
    int pointCount = 0;
    float halfSize = 1.f;      // max(halfExtentU, halfExtentV)，兼容旧逻辑
    float halfExtentU = 1.f;   // 平面局部 U 向半宽（贴合点集）
    float halfExtentV = 1.f;   // 平面局部 V 向半宽
};

struct SphereModel {
    Vec3 center{0, 0, 0};
    float radius = 0.f;
    float rms = 0.f;
    int pointCount = 0;
};

struct CircleModel {
    Vec3 center{0, 0, 0};
    Vec3 normal{0, 0, 1};
    float radius = 0.f;
    float rms = 0.f;
    int pointCount = 0;
};

struct CylinderModel {
    Vec3 axisPoint{0, 0, 0};  // 轴线上一点（通常为投影中心）
    Vec3 axisDir{0, 0, 1};    // 单位轴方向
    float radius = 0.f;
    float halfHeight = 1.f;   // 沿轴向覆盖半长（用于显示）
    float rms = 0.f;
    int pointCount = 0;
};

struct FlatnessResult {
    bool valid = false;
    PlaneModel plane;
    float minDev = 0.f;
    float maxDev = 0.f;
    float peakToValley = 0.f;  // 平面度 PV = max - min
    float meanAbs = 0.f;
    float rms = 0.f;
    std::vector<std::size_t> indices;
    std::vector<float> signedDist;  // 与 indices 一一对应
};

enum class StepGapPhase { SelectA = 0, FitA, SelectB, Done };

struct StepGapResult {
    StepGapPhase phase = StepGapPhase::SelectA;
    std::vector<std::size_t> regionA;
    std::vector<std::size_t> regionB;
    PlaneModel planeA;
    bool hasPlane = false;
    bool hasDistances = false;
    std::vector<float> signedDistB;  // 与 regionB 一一对应：ΔZ = B.z − mean(A.z)
    float zRefA = 0.f;               // 区域 A 平均高度（水平基准）
    float mean = 0.f;                // 段差主结果：ΔZ 均值（有符号）
    float meanAbs = 0.f;
    float median = 0.f;
    float minDist = 0.f;
    float maxDist = 0.f;
    float rms = 0.f;
};

struct SectionPoint2D {
    float u = 0.f;
    float v = 0.f;
    Vec3 p3{0, 0, 0};
};

struct SectionData {
    bool cutAlongX = true;  // true: X=const -> 2D(Y,Z); false: Y=const -> 2D(X,Z)
    float position = 0.f;
    float thickness = 0.05f;
    std::vector<SectionPoint2D> points;
    float uMin = 0, uMax = 1, vMin = 0, vMax = 1;
    std::optional<std::size_t> pickA;
    std::optional<std::size_t> pickB;
    float lineDistance = 0.f;  // 垂线间距 |ΔU|
    float zDistance = 0.f;     // Z 向距离 |ΔV|
};

struct MeasureState {
    ToolMode mode = ToolMode::Navigate;
    std::optional<Vec3> picked;
    std::optional<Vec3> distA;
    std::optional<Vec3> distB;
    float distance = 0.f;
    std::optional<PlaneModel> plane;
    bool roiDragging = false;
    float roiX0 = 0, roiY0 = 0, roiX1 = 0, roiY1 = 0;
    std::vector<std::size_t> roiIndices;
    bool clipEnabled = false;
    Vec3 clipNormal{0, 0, 1};
    float clipD = 0.f;
    SectionData section;
    // 台阶/高度差：点A为基准，点B为测量点，stepDeltaZ = B.z - A.z（显示坐标，同世界相对差）
    std::optional<Vec3> stepA;
    std::optional<Vec3> stepB;
    float stepDeltaZ = 0.f;
    FlatnessResult flatness;
    StepGapResult stepGap;
    std::optional<SphereModel> sphere;
    std::optional<CircleModel> circle;
    std::optional<CylinderModel> cylinder;
    std::string status;
};

namespace MeasureTools {

std::optional<std::size_t> PickNearest(const PointCloud& cloud, const Camera& camera, int fbW,
                                       int fbH, float mouseX, float mouseY,
                                       float maxPixelDist = 12.f,
                                       const std::vector<std::size_t>* onlyIndices = nullptr);

bool FitPlaneSVD(const PointCloud& cloud, const std::vector<std::size_t>& indices, PlaneModel& out,
                 std::string& error);

bool FitSphere(const PointCloud& cloud, const std::vector<std::size_t>& indices, SphereModel& out,
               std::string& error);

// 先拟合支撑平面，再在平面内代数拟合圆
bool FitCircle3D(const PointCloud& cloud, const std::vector<std::size_t>& indices, CircleModel& out,
                 std::string& error);

// PCA 候选轴 + 垂面圆拟合，取径向残差最小者
bool FitCylinder(const PointCloud& cloud, const std::vector<std::size_t>& indices,
                 CylinderModel& out, std::string& error);

bool ComputeFlatness(const PointCloud& cloud, const std::vector<std::size_t>& indices,
                     FlatnessResult& out, std::string& error);

bool ComputeStepGapDistances(const PointCloud& cloud, const PlaneModel& planeA,
                             const std::vector<std::size_t>& regionB, StepGapResult& out,
                             std::string& error);

// 段差：以区域 A 平均 Z 为水平基准，ΔZ = B.z − mean(A.z)
bool ComputeStepGapZHeight(const PointCloud& cloud, const std::vector<std::size_t>& regionA,
                           const std::vector<std::size_t>& regionB, StepGapResult& out,
                           std::string& error);

void SelectRoi(const PointCloud& cloud, const Camera& camera, int fbW, int fbH, float x0, float y0,
               float x1, float y1, std::vector<std::size_t>& outIndices);
// 框选仅选当前视角可见表面点（深度缓冲遮挡剔除，不含被挡住的背面/下层点）


void ApplyClipMask(PointCloud& cloud, const Vec3& normal, float d, bool enabled);

void ApplyRoiDelete(PointCloud& cloud, const std::vector<std::size_t>& roiIndices, bool deleteInside);

void RestoreAllPoints(PointCloud& cloud);

bool ExtractSection(const PointCloud& cloud, bool cutAlongX, float position, float thickness,
                    SectionData& out, std::string& error, int maxPoints = 200000);

PlaneModel MakeSectionCutPlane(const PointCloud& cloud, bool cutAlongX, float position);

}  // namespace MeasureTools
