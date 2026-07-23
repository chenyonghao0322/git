#pragma once

// PclTools — 基于 PCL 1.x 的算法封装
//
// 与 MeasureTools 接口对齐：滤波、RANSAC 拟合、ROI 填充、孔径测量等。
// Application 通过 AlgorithmBackend 在「自研 / PCL」两套实现间切换。

#include "core/PointCloud.h"
#include "render/Camera.h"
#include "tools/MeasureTools.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace PclTools {

const char* VersionString();  // 返回 PCL 版本字符串

// --- 滤波（输出 keepMask，1=保留）---
bool VoxelDownsample(const PointCloud& cloud, float leafSize, std::vector<uint8_t>& keepMask,
                     std::string& error, int* outKept = nullptr);

bool RadiusOutlier(const PointCloud& cloud, float radius, int minNeighbors,
                   std::vector<uint8_t>& keepMask, std::string& error, int* outKept = nullptr);

bool StatisticalOutlier(const PointCloud& cloud, int meanK, float stdMul,
                        std::vector<uint8_t>& keepMask, std::string& error, int* outKept = nullptr);

// --- RANSAC 拟合 ---
bool FitPlaneRANSAC(const PointCloud& cloud, const std::vector<std::size_t>& indices,
                    float distanceThreshold, int maxIterations, PlaneModel& out,
                    std::string& error);

bool FitSphereRANSAC(const PointCloud& cloud, const std::vector<std::size_t>& indices,
                     float distanceThreshold, int maxIterations, SphereModel& out,
                     std::string& error);

bool FitCircleRANSAC(const PointCloud& cloud, const std::vector<std::size_t>& indices,
                     float distanceThreshold, int maxIterations, CircleModel& out,
                     std::string& error);

bool FitCylinderRANSAC(const PointCloud& cloud, const std::vector<std::size_t>& indices,
                       float distanceThreshold, int maxIterations, CylinderModel& out,
                       std::string& error);

// --- 测量 ---
bool ComputeFlatness(const PointCloud& cloud, const std::vector<std::size_t>& indices,
                     float distanceThreshold, int maxIterations, FlatnessResult& out,
                     std::string& error);

bool ComputeStepGapZHeight(const PointCloud& cloud, const std::vector<std::size_t>& regionA,
                           const std::vector<std::size_t>& regionB, StepGapResult& out,
                           std::string& error);

bool ExtractSection(const PointCloud& cloud, bool cutAlongX, float position, float thickness,
                    SectionData& out, std::string& error, int maxPoints = 200000);

// --- 点云编辑 ---

void ApplyRoiDelete(PointCloud& cloud, const std::vector<std::size_t>& roiIndices,
                    bool deleteInside);  // true=删框内 false=只留框内

void RestoreAllPoints(PointCloud& cloud);  // 重置 mask 为全可见

// --- 拾取与框选 ---
std::optional<std::size_t> PickNearest(const PointCloud& cloud, const Camera& camera, int fbW,
                                       int fbH, float mouseX, float mouseY,
                                       float maxPixelDist = 12.f,
                                       const std::vector<std::size_t>* onlyIndices = nullptr);

void SelectRoi(const PointCloud& cloud, const Camera& camera, int fbW, int fbH, float x0, float y0,
               float x1, float y1, std::vector<std::size_t>& outIndices);  // 屏幕矩形框选

// 将可见点正交投影到垂直于指定轴的平面
bool ProjectOntoAxis(PointCloud& cloud, const Vec3& axisOrigin, const Vec3& axisDir,
                     std::string& error);

// 以拟合平面为基准旋转点云，使法向与 targetNormal 对齐
bool AlignCloudToPlaneNormal(PointCloud& cloud, const PlaneModel& plane, const Vec3& targetNormal,
                             PlaneModel& outPlane, std::string& error);

bool MeasureHoleRadius(const PointCloud& cloud, const std::vector<std::size_t>& indices,
                       float planeDistThresh, int planeMaxIter, HoleMeasureResult& out,
                       std::string& error);

// --- ROI 填充 ---
bool RoiFill(const PointCloud& cloud, const std::vector<std::size_t>& indices, RoiFillMode mode,
             int axis, float gridStepMm, bool clipCircle, const Vec3& clipCenter, float clipRadius,
             const std::vector<std::size_t>* planeFitIndices, PointCloud& filledOut,
             PlaneModel& planeOut, float& outGridStep, std::string& error);

bool RoiProjectFill(const PointCloud& cloud, const std::vector<std::size_t>& indices, int axis,
                    float gridStepMm, bool clipCircle, const Vec3& clipCenter, float clipRadius,
                    PointCloud& filledOut, PlaneModel& planeOut, float& outGridStep,
                    std::string& error);

bool RoiProjectFillAndFitCircle(const PointCloud& cloud, const std::vector<std::size_t>& indices,
                                int axis, float gridStepMm, bool clipCircle,
                                const Vec3& clipCenter, float clipRadius, PointCloud& filledOut,
                                CircleModel& circleOut, PlaneModel& planeOut, float& outGridStep,
                                std::string& error);

// --- ICP 配准 ---
struct IcpParams {
    float maxCorrespondenceDist = 5.f;  // 最大对应距离（世界坐标 mm）
    int maxIterations = 50;
    float transEpsilon = 1e-8f;
    float euclideanEpsilon = 1e-6f;
    float voxelLeaf = 0.f;  // 配准前体素下采样，0=不下采样
    bool useCentroidInit = true;
};

struct IcpResult {
    bool success = false;
    bool converged = false;
    float fitnessScore = 0.f;
    float transform[16]{};  // 行主序 4×4，作用于源点世界坐标
};

bool RunIcp(const PointCloud& target, const PointCloud& source, const IcpParams& params,
            IcpResult& result, PointCloud& alignedSourceOut, std::string& error);

}  // namespace PclTools
