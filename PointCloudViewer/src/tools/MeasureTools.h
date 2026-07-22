#pragma once

// MeasureTools — 自研几何/测量算法（无 PCL 依赖）
//
// 提供：屏幕 ROI 框选、平面/圆/球/圆柱拟合、剖切、截面、平面度、段差、
//       ROI 填充（投影/拟合平面/直接 XY）等。Application 与 PclTools 均可调用。

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
    PlaneAlign,  // 平面校准：框选基准面后旋转点云（线扫倾斜校正）
    SphereFit,
    SphereBodyFit,
    CircleFit,
    CylinderFit,
    Roi,
    ClipPlane,
    Section,
    StepHeight,
    Flatness,   // 平面度
    StepGap     // 段差（区域A平面 → 区域B距离图）
};

inline bool IsSphereFitMode(ToolMode mode) {
    return mode == ToolMode::SphereFit || mode == ToolMode::SphereBodyFit;
}

enum class RoiShape {
    Rect = 0,
    Circle,
    FreePolygon,
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

struct HoleMeasureResult {
    bool valid = false;
    float innerRadius = 0.f;
    float outerRadius = 0.f;
    Vec3 holeCenter{0, 0, 0};
    PlaneModel plane;
    int boundaryPointCount = 0;
};

struct SphereModel {
    Vec3 center{0, 0, 0};
    float radius = 0.f;
    float rms = 0.f;
    int pointCount = 0;
    std::vector<std::size_t> inlierIndices;  // 拟合内点（用于对比着色）
};

struct CircleModel {
    Vec3 center{0, 0, 0};
    Vec3 normal{0, 0, 1};
    float radius = 0.f;
    float rms = 0.f;
    int pointCount = 0;
    std::vector<std::size_t> inlierIndices;  // 拟合内点（用于对比着色）
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

enum class RoiFillMode {
    ProjectAxis = 0,  // 投影到固定轴平面后填充
    FitPlane = 1,     // 拟合支撑平面后投影填充
    DirectXY = 2,     // 不投影，XY 平面直接填充（Z 取参考值）
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
    RoiShape roiShape = RoiShape::Rect;
    bool roiUseWorldSize = false;
    float roiWorldRadius = 5.f;
    float roiWorldWidth = 10.f;
    float roiWorldHeight = 10.f;
    Vec3 roiWorldCenter{0, 0, 0};
    bool roiHasWorldCenter = false;
    std::vector<float> roiPolyX;
    std::vector<float> roiPolyY;
    bool roiPolyBuilding = false;
    float roiFillGridStep = 0.f;  // 0 = 自动网格步长 (mm)
    int roiFillMode = 0;          // RoiFillMode
    std::vector<std::size_t> roiFillPlaneIndices;  // 平面拟合填充：区域 A（矩形框选）
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
    HoleMeasureResult holeMeasure;
    std::string status;
};

namespace MeasureTools {

// 屏幕拾取：返回距鼠标最近的可见点索引
std::optional<std::size_t> PickNearest(const PointCloud& cloud, const Camera& camera, int fbW,
                                       int fbH, float mouseX, float mouseY,
                                       float maxPixelDist = 12.f,
                                       const std::vector<std::size_t>* onlyIndices = nullptr);

// SVD 最小二乘拟合平面
bool FitPlaneSVD(const PointCloud& cloud, const std::vector<std::size_t>& indices, PlaneModel& out,
                 std::string& error);

bool FitSphere(const PointCloud& cloud, const std::vector<std::size_t>& indices, SphereModel& out,
               std::string& error);

// 先拟合支撑平面，再在平面内代数拟合圆
bool FitCircle3D(const PointCloud& cloud, const std::vector<std::size_t>& indices, CircleModel& out,
                 std::string& error);

// 已知平面后在平面内拟合圆（支持圆环与填充圆盘）
bool FitCircleOnPlane(const PointCloud& cloud, const std::vector<std::size_t>& indices,
                      const PlaneModel& plane, float inlierBand, CircleModel& out,
                      std::string& error);

// PCA 候选轴 + 垂面圆拟合，取径向残差最小者
bool FitCylinder(const PointCloud& cloud, const std::vector<std::size_t>& indices,
                 CylinderModel& out, std::string& error);

// 拟合平面并计算各点到平面的偏差统计（平面度 PV 等）
bool ComputeFlatness(const PointCloud& cloud, const std::vector<std::size_t>& indices,
                     FlatnessResult& out, std::string& error);

// 已知平面 A 后，计算区域 B 各点到平面的有符号距离
bool ComputeStepGapDistances(const PointCloud& cloud, const PlaneModel& planeA,
                             const std::vector<std::size_t>& regionB, StepGapResult& out,
                             std::string& error);

// 段差：以区域 A 平均 Z 为水平基准，ΔZ = B.z − mean(A.z)
bool ComputeStepGapZHeight(const PointCloud& cloud, const std::vector<std::size_t>& regionA,
                           const std::vector<std::size_t>& regionB, StepGapResult& out,
                           std::string& error);

// 屏幕矩形框选可见点
void SelectRoi(const PointCloud& cloud, const Camera& camera, int fbW, int fbH, float x0, float y0,
               float x1, float y1, std::vector<std::size_t>& outIndices);

// 屏幕圆形框选可见点
void SelectRoiCircle(const PointCloud& cloud, const Camera& camera, int fbW, int fbH, float cx,
                     float cy, float radiusPx, std::vector<std::size_t>& outIndices);

// 屏幕多边形框选可见点
void SelectRoiPolygon(const PointCloud& cloud, const Camera& camera, int fbW, int fbH,
                      const std::vector<float>& polyX, const std::vector<float>& polyY,
                      std::vector<std::size_t>& outIndices);

// 世界坐标 XY 平面圆形/矩形 ROI（适用于水平圆面）
void SelectRoiWorldCircleXY(const PointCloud& cloud, const Camera& camera, int fbW, int fbH,
                            const Vec3& center, float radius,
                            std::vector<std::size_t>& outIndices);

void SelectRoiWorldRectXY(const PointCloud& cloud, const Camera& camera, int fbW, int fbH,
                          const Vec3& center, float halfW, float halfH,
                          std::vector<std::size_t>& outIndices);

// 孔径：在环状/带孔点云上估计内圆半径（平面 + 内边界径向分位）
bool MeasureHoleRadius(const PointCloud& cloud, const std::vector<std::size_t>& indices,
                       HoleMeasureResult& out, std::string& error);

bool MeasureHoleRadiusOnPlane(const PointCloud& cloud, const std::vector<std::size_t>& indices,
                              const PlaneModel& plane, HoleMeasureResult& out,
                              std::string& error);

// ROI 点集 → 按模式填充；filledOut 为填充后点云
bool RoiFill(const PointCloud& cloud, const std::vector<std::size_t>& indices, RoiFillMode mode,
             int axis, float gridStepMm, bool clipCircle, const Vec3& clipCenter, float clipRadius,
             const std::vector<std::size_t>* planeFitIndices, PointCloud& filledOut,
             PlaneModel& planeOut, float& outGridStep, std::string& error);

// ROI 点集 → 投影到垂直于 axis 的平面 → 极坐标/网格填充
bool RoiProjectFill(const PointCloud& cloud, const std::vector<std::size_t>& indices, int axis,
                    float gridStepMm, bool clipCircle, const Vec3& clipCenter, float clipRadius,
                    PointCloud& filledOut, PlaneModel& planeOut, float& outGridStep,
                    std::string& error);

// 投影填充后在结果上圆拟合（旧接口）
bool RoiProjectFillAndFitCircle(const PointCloud& cloud, const std::vector<std::size_t>& indices,
                                int axis, float gridStepMm, bool clipCircle,
                                const Vec3& clipCenter, float clipRadius, PointCloud& filledOut,
                                CircleModel& circleOut, PlaneModel& planeOut, float& outGridStep,
                                std::string& error);

// 按半空间法向剖切，隐藏法向负侧点（改 mask）
void ApplyClipMask(PointCloud& cloud, const Vec3& normal, float d, bool enabled);

// ROI 软删除：deleteInside=true 清除框内，false 只保留框内
void ApplyRoiDelete(PointCloud& cloud, const std::vector<std::size_t>& roiIndices, bool deleteInside);

void RestoreAllPoints(PointCloud& cloud);  // mask 全部置 1

// 提取截面附近点并展开为 2D 轮廓
bool ExtractSection(const PointCloud& cloud, bool cutAlongX, float position, float thickness,
                    SectionData& out, std::string& error, int maxPoints = 200000);

// 构造截面切割平面的可视化模型
PlaneModel MakeSectionCutPlane(const PointCloud& cloud, bool cutAlongX, float position);

// 绕 plane.centroid 旋转点云，使 plane.normal 与 targetNormal 对齐（线扫倾斜校正）
bool AlignCloudToPlaneNormal(PointCloud& cloud, const PlaneModel& plane, const Vec3& targetNormal,
                             PlaneModel& outPlane, std::string& error);

}  // namespace MeasureTools
