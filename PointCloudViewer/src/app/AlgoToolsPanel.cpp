#include "app/AlgoToolsPanel.h"

#include "tools/PclTools.h"

#include <imgui.h>

#include <cstdio>

namespace AlgoToolsPanel {
namespace {

const char* ToolModeLabel(ToolMode mode) {
    switch (mode) {
        case ToolMode::Navigate:
            return u8"漫游";
        case ToolMode::Pick:
            return u8"点选";
        case ToolMode::Distance:
            return u8"测距";
        case ToolMode::PlaneFit:
            return u8"平面拟合";
        case ToolMode::SphereFit:
            return u8"球面拟合";
        case ToolMode::SphereBodyFit:
            return u8"球体拟合";
        case ToolMode::CircleFit:
            return u8"圆拟合";
        case ToolMode::CylinderFit:
            return u8"圆柱拟合";
        case ToolMode::Roi:
            return u8"ROI框选";
        case ToolMode::ClipPlane:
            return u8"剖切平面";
        case ToolMode::Section:
            return u8"截面";
        case ToolMode::StepHeight:
            return u8"台阶高度";
        case ToolMode::Flatness:
            return u8"平面度";
        case ToolMode::StepGap:
            return u8"段差计算";
    }
    return u8"工具";
}

void ToolButton(const AlgoToolsHost& host, ToolMode mode) {
    const bool selected = host.currentMode == mode;
    if (ImGui::Selectable(ToolModeLabel(mode), selected) && host.setToolMode) {
        host.setToolMode(mode);
    }
}

void DrawCreateTab(const AlgoToolsHost& host) {
    if (!host.showCreateSphere || !host.showCreateCylinder) return;
    ImGui::TextWrapped(u8"生成测试点云，用于验证拟合与滤波。");
    ImGui::Spacing();
    if (ImGui::Button(u8"球面点云…", ImVec2(-1, 32.f))) *host.showCreateSphere = true;
    if (ImGui::Button(u8"圆柱点云…", ImVec2(-1, 32.f))) *host.showCreateCylinder = true;
}

void DrawEditTab(const AlgoToolsHost& host) {
    if (!host.canUndo) ImGui::BeginDisabled();
    if (ImGui::Button(u8"撤销", ImVec2(-1, 0)) && host.undo) host.undo();
    if (!host.canUndo) ImGui::EndDisabled();
    if (!host.canRedo) ImGui::BeginDisabled();
    if (ImGui::Button(u8"重做", ImVec2(-1, 0)) && host.redo) host.redo();
    if (!host.canRedo) ImGui::EndDisabled();
    ImGui::Separator();
    if (ImGui::Button(u8"清空显示", ImVec2(-1, 0)) && host.clearVisuals) host.clearVisuals();
}

void DrawToolsTab(const AlgoToolsHost& host) {
    ImGui::TextDisabled(u8"交互工具");
    ToolButton(host, ToolMode::Navigate);
    ToolButton(host, ToolMode::Pick);
    ToolButton(host, ToolMode::Distance);
    ToolButton(host, ToolMode::Roi);
    ToolButton(host, ToolMode::ClipPlane);
    ToolButton(host, ToolMode::Section);
    ToolButton(host, ToolMode::StepHeight);
    ImGui::Separator();
    ImGui::TextDisabled(u8"测量");
    ToolButton(host, ToolMode::Flatness);
    ToolButton(host, ToolMode::StepGap);
}

void DrawFitTab(const AlgoToolsHost& host) {
    const bool isPcl = host.backend == AlgorithmBackend::PCL;
    ImGui::TextDisabled(isPcl ? u8"PCL 几何拟合" : u8"自研几何拟合");
    ToolButton(host, ToolMode::PlaneFit);
    ToolButton(host, ToolMode::SphereFit);
    ToolButton(host, ToolMode::SphereBodyFit);
    ToolButton(host, ToolMode::CircleFit);
    ToolButton(host, ToolMode::CylinderFit);

    const std::size_t roiCount = host.measure ? host.measure->roiIndices.size() : 0;
    ImGui::Spacing();
    ImGui::Text(u8"当前框选: %zu 点", roiCount);

    if (host.fitPlane) {
        ImGui::Spacing();
        if (isPcl && host.planeDistThresh && host.planeMaxIter) {
            ImGui::SetNextItemWidth(-1.f);
            ImGui::DragFloat(u8"RANSAC 距离阈值##fit", host.planeDistThresh, 0.001f, 1e-4f, 10.f,
                             "%.4f");
            ImGui::SetNextItemWidth(-1.f);
            ImGui::DragInt(u8"最大迭代##fit", host.planeMaxIter, 50, 50, 10000);
        }
        if (ImGui::Button(u8"对框选区域拟合平面", ImVec2(-1, 32.f))) {
            if (roiCount == 0) {
                if (host.setStatus) host.setStatus(u8"请先框选区域");
            } else {
                PlaneModel plane;
                std::string error;
                if (host.fitPlane(host.measure->roiIndices, plane, error)) {
                    if (host.showPlane) host.showPlane(plane);
                    if (host.setStatus) host.setStatus(u8"平面拟合完成");
                } else if (host.setStatus) {
                    host.setStatus(error);
                }
            }
        }
        if (ImGui::Button(u8"对全部可见点拟合平面", ImVec2(-1, 0))) {
            PlaneModel plane;
            std::string error;
            std::vector<std::size_t> empty;
            if (host.fitPlane(empty, plane, error)) {
                if (host.showPlane) host.showPlane(plane);
                if (host.setStatus) host.setStatus(u8"全点云平面拟合完成");
            } else if (host.setStatus) {
                host.setStatus(error);
            }
        }
        if (ImGui::Button(u8"清除拟合平面", ImVec2(-1, 0)) && host.clearPlane) host.clearPlane();
    }

    if (host.fitSphere && ImGui::CollapsingHeader(u8"球面拟合", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (ImGui::Button(u8"对框选区域拟合球##fit", ImVec2(-1, 0))) {
            if (roiCount == 0) {
                if (host.setStatus) host.setStatus(u8"请先框选区域");
            } else {
                SphereModel sphere;
                std::string error;
                if (host.fitSphere(host.measure->roiIndices, sphere, error) && host.setStatus) {
                    char buf[160];
                    std::snprintf(buf, sizeof(buf), u8"球面拟合 R=%.6f RMS=%.6f", sphere.radius,
                                  sphere.rms);
                    host.setStatus(buf);
                } else if (host.setStatus) {
                    host.setStatus(error);
                }
            }
        }
    }

    if (host.fitCircle && ImGui::CollapsingHeader(u8"圆拟合")) {
        if (ImGui::Button(u8"对框选区域拟合圆##fit", ImVec2(-1, 0))) {
            if (roiCount == 0) {
                if (host.setStatus) host.setStatus(u8"请先框选区域");
            } else {
                CircleModel circle;
                std::string error;
                if (host.fitCircle(host.measure->roiIndices, circle, error) && host.setStatus) {
                    host.setStatus(u8"圆拟合完成");
                } else if (host.setStatus) {
                    host.setStatus(error);
                }
            }
        }
    }

    if (host.fitCylinder && ImGui::CollapsingHeader(u8"圆柱拟合")) {
        if (ImGui::Button(u8"对框选区域拟合圆柱##fit", ImVec2(-1, 0))) {
            if (roiCount == 0) {
                if (host.setStatus) host.setStatus(u8"请先框选区域");
            } else {
                CylinderModel cylinder;
                std::string error;
                if (host.fitCylinder(host.measure->roiIndices, cylinder, error) && host.setStatus) {
                    host.setStatus(u8"圆柱拟合完成");
                } else if (host.setStatus) {
                    host.setStatus(error);
                }
            }
        }
    }
}

void DrawFilterTab(const AlgoToolsHost& host) {
    if (!host.filterVoxelLeaf || !host.runFilterPreview) return;
    const bool isPcl = host.backend == AlgorithmBackend::PCL;
    ImGui::TextDisabled(isPcl ? u8"PCL 点云滤波" : u8"自研点云滤波");

    ImGui::TextDisabled(u8"体素下采样");
    ImGui::SetNextItemWidth(-1.f);
    ImGui::DragFloat(u8"体素边长##flt", host.filterVoxelLeaf, 0.01f, 1e-4f, 1e6f, "%.4f");
    if (ImGui::Button(u8"预览体素滤波", ImVec2(-1, 0))) host.runFilterPreview(0);

    ImGui::Spacing();
    ImGui::TextDisabled(u8"半径离群点");
    if (host.filterRadius && host.filterRadiusMinNeighbors) {
        ImGui::SetNextItemWidth(-1.f);
        ImGui::DragFloat(u8"搜索半径##flt", host.filterRadius, 0.01f, 1e-4f, 1e6f, "%.4f");
        ImGui::SetNextItemWidth(-1.f);
        ImGui::DragInt(u8"最少邻居##flt", host.filterRadiusMinNeighbors, 1, 1, 200);
    }
    if (ImGui::Button(u8"预览半径滤波", ImVec2(-1, 0))) host.runFilterPreview(1);

    ImGui::Spacing();
    ImGui::TextDisabled(u8"统计离群点");
    if (host.filterStatMeanK && host.filterStatStdMul) {
        ImGui::SetNextItemWidth(-1.f);
        ImGui::DragInt(u8"邻域 K##flt", host.filterStatMeanK, 1, 2, 200);
        ImGui::SetNextItemWidth(-1.f);
        ImGui::DragFloat(u8"标准差倍数##flt", host.filterStatStdMul, 0.05f, 0.1f, 10.f, "%.2f");
    }
    if (ImGui::Button(u8"预览统计滤波", ImVec2(-1, 0))) host.runFilterPreview(2);

    if (host.filterCompareActive) {
        ImGui::Separator();
        ImGui::Text(u8"预览：保留 %d，滤除 %d", host.filterLastKept, host.filterLastRemoved);
        if (host.filterHideRemoved) {
            if (ImGui::Checkbox(u8"仅显示滤波后", host.filterHideRemoved)) {
                if (host.requestRefreshGpu) host.requestRefreshGpu();
            }
        }
        if (ImGui::Button(u8"应用滤波", ImVec2(-1, 0)) && host.applyFilter) host.applyFilter();
        if (ImGui::Button(u8"取消预览", ImVec2(-1, 0)) && host.clearFilter) host.clearFilter();
    } else {
        ImGui::TextDisabled(u8"先预览：青绿=保留，红=滤除");
    }
}

}  // namespace

void Draw(bool& visible, const char* windowTitle, const AlgoToolsHost& host) {
    if (!visible) return;

    ImGui::SetNextWindowSize(ImVec2(380.f, 560.f), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin(windowTitle, &visible, ImGuiWindowFlags_None)) {
        ImGui::End();
        return;
    }

    const bool isPcl = host.backend == AlgorithmBackend::PCL;
    ImGui::TextColored(isPcl ? ImVec4(0.55f, 0.90f, 0.65f, 1.f) : ImVec4(0.75f, 0.80f, 0.90f, 1.f),
                       "%s", AlgorithmBackendLabel(host.backend));
    if (isPcl) ImGui::TextDisabled("%s", PclTools::VersionString());
    if (host.currentMode != ToolMode::Navigate) {
        ImGui::SameLine();
        ImGui::TextDisabled(u8"| 当前工具 %s", ToolModeLabel(host.currentMode));
    }
    ImGui::Separator();

    const bool hasCloud = host.cloud && !host.cloud->points.empty();
    if (!hasCloud) {
        ImGui::TextColored(ImVec4(0.95f, 0.75f, 0.35f, 1.f), u8"请先加载点云");
        ImGui::Separator();
    }

    if (ImGui::BeginTabBar(u8"##algotools_tabs")) {
        if (ImGui::BeginTabItem(u8"创建")) {
            if (!hasCloud) ImGui::BeginDisabled();
            DrawCreateTab(host);
            if (!hasCloud) ImGui::EndDisabled();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem(u8"编辑")) {
            DrawEditTab(host);
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem(u8"工具")) {
            DrawToolsTab(host);
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem(u8"拟合")) {
            if (!hasCloud) ImGui::BeginDisabled();
            DrawFitTab(host);
            if (!hasCloud) ImGui::EndDisabled();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem(u8"滤波")) {
            if (!hasCloud) ImGui::BeginDisabled();
            DrawFilterTab(host);
            if (!hasCloud) ImGui::EndDisabled();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }

    ImGui::End();
}

}  // namespace AlgoToolsPanel
