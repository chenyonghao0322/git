#pragma once

#include "core/PointCloud.h"
#include "render/Camera.h"
#include "tools/MeasureTools.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace PclTools {

const char* VersionString();

bool VoxelDownsample(const PointCloud& cloud, float leafSize, std::vector<uint8_t>& keepMask,
                     std::string& error, int* outKept = nullptr);

bool RadiusOutlier(const PointCloud& cloud, float radius, int minNeighbors,
                   std::vector<uint8_t>& keepMask, std::string& error, int* outKept = nullptr);

bool StatisticalOutlier(const PointCloud& cloud, int meanK, float stdMul,
                        std::vector<uint8_t>& keepMask, std::string& error, int* outKept = nullptr);

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

bool ComputeFlatness(const PointCloud& cloud, const std::vector<std::size_t>& indices,
                     float distanceThreshold, int maxIterations, FlatnessResult& out,
                     std::string& error);

bool ComputeStepGapZHeight(const PointCloud& cloud, const std::vector<std::size_t>& regionA,
                           const std::vector<std::size_t>& regionB, StepGapResult& out,
                           std::string& error);

bool ExtractSection(const PointCloud& cloud, bool cutAlongX, float position, float thickness,
                    SectionData& out, std::string& error, int maxPoints = 200000);

void ApplyClipMask(PointCloud& cloud, const Vec3& normal, float d, bool enabled);

void ApplyRoiDelete(PointCloud& cloud, const std::vector<std::size_t>& roiIndices,
                    bool deleteInside);

void RestoreAllPoints(PointCloud& cloud);

std::optional<std::size_t> PickNearest(const PointCloud& cloud, const Camera& camera, int fbW,
                                       int fbH, float mouseX, float mouseY,
                                       float maxPixelDist = 12.f,
                                       const std::vector<std::size_t>* onlyIndices = nullptr);

void SelectRoi(const PointCloud& cloud, const Camera& camera, int fbW, int fbH, float x0, float y0,
               float x1, float y1, std::vector<std::size_t>& outIndices);

// 将可见点正交投影到垂直于指定轴的平面（PCL 管线入口，几何实现）
bool ProjectOntoAxis(PointCloud& cloud, const Vec3& axisOrigin, const Vec3& axisDir,
                     std::string& error);

bool MeasureHoleRadius(const PointCloud& cloud, const std::vector<std::size_t>& indices,
                       float planeDistThresh, int planeMaxIter, HoleMeasureResult& out,
                       std::string& error);

bool RoiProjectFill(const PointCloud& cloud, const std::vector<std::size_t>& indices, int axis,
                    float gridStepMm, bool clipCircle, const Vec3& clipCenter, float clipRadius,
                    PointCloud& filledOut, PlaneModel& planeOut, float& outGridStep,
                    std::string& error);

bool RoiProjectFillAndFitCircle(const PointCloud& cloud, const std::vector<std::size_t>& indices,
                                int axis, float gridStepMm, bool clipCircle,
                                const Vec3& clipCenter, float clipRadius, PointCloud& filledOut,
                                CircleModel& circleOut, PlaneModel& planeOut, float& outGridStep,
                                std::string& error);

}  // namespace PclTools
