#pragma once

#include "core/PointCloud.h"
#include "tools/AlgorithmBackend.h"
#include "tools/MeasureTools.h"

#include <functional>
#include <string>

struct AlgoToolsHost {
    AlgorithmBackend backend = AlgorithmBackend::PCL;
    const PointCloud* cloud = nullptr;
    const MeasureState* measure = nullptr;
    ToolMode currentMode = ToolMode::Navigate;

    bool canUndo = false;
    bool canRedo = false;
    bool filterCompareActive = false;
    int filterLastKept = 0;
    int filterLastRemoved = 0;
    bool* filterHideRemoved = nullptr;

    float* filterVoxelLeaf = nullptr;
    float* filterRadius = nullptr;
    int* filterRadiusMinNeighbors = nullptr;
    int* filterStatMeanK = nullptr;
    float* filterStatStdMul = nullptr;
    float* planeDistThresh = nullptr;
    int* planeMaxIter = nullptr;

    bool* showCreateSphere = nullptr;
    bool* showCreateCylinder = nullptr;
    float* genSphereRadius = nullptr;
    int* genSpherePoints = nullptr;
    float* genSphereNoise = nullptr;
    float* genCylRadius = nullptr;
    float* genCylHeight = nullptr;
    int* genCylPoints = nullptr;
    float* genCylNoise = nullptr;

    std::function<void(ToolMode)> setToolMode;
    std::function<void()> undo;
    std::function<void()> redo;
    std::function<void()> clearVisuals;
    std::function<void(int filterType)> runFilterPreview;
    std::function<void()> applyFilter;
    std::function<void()> clearFilter;
    std::function<bool(const std::vector<std::size_t>&, PlaneModel&, std::string&)> fitPlane;
    std::function<bool(const std::vector<std::size_t>&, SphereModel&, std::string&)> fitSphere;
    std::function<bool(const std::vector<std::size_t>&, CircleModel&, std::string&)> fitCircle;
    std::function<bool(const std::vector<std::size_t>&, CylinderModel&, std::string&)> fitCylinder;
    std::function<void(const PlaneModel&)> showPlane;
    std::function<void()> clearPlane;
    std::function<void(const std::string&)> setStatus;
    std::function<void()> requestRefreshGpu;
};

namespace AlgoToolsPanel {

void Draw(bool& visible, const char* windowTitle, const AlgoToolsHost& host);

}  // namespace AlgoToolsPanel
