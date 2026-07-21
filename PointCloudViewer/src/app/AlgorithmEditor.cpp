#include "app/AlgorithmEditor.h"

#include "io/PointCloudGenerator.h"
#include "io/PointCloudIO.h"
#include "tools/FilterTools.h"
#include "tools/MeasureTools.h"

#include <imgui.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

// 三次贝塞尔弧线连接（输出端口 → 输入端口）
void DrawSmoothLink(ImDrawList* dl, float ax, float ay, float bx, float by, ImU32 col,
                    float thickness, float scale = 1.f) {
    scale = std::max(0.35f, scale);
    const float dx = bx - ax;
    const float dy = by - ay;
    float bend = std::max(40.f * scale, std::fabs(dy) * 0.45f);
    bend = std::min(bend, 180.f * scale);
    // 端口朝下出、朝上入：控制点沿竖直方向拉开
    const ImVec2 p0(ax, ay);
    const ImVec2 p1(ax, ay + bend);
    const ImVec2 p2(bx, by - bend);
    const ImVec2 p3(bx, by);
    dl->AddBezierCubic(p0, p1, p2, p3, col, thickness, 0);
    (void)dx;
}

}  // namespace

const AlgorithmEditor::ModuleDef* AlgorithmEditor::Catalog() {
    static const ModuleDef kDefs[] = {
        {ModuleType::InputCloud, u8"输入点云", u8"数据", IM_COL32(64, 160, 200, 255)},
        {ModuleType::FilterVoxel, u8"体素滤波", u8"预处理", IM_COL32(80, 180, 140, 255)},
        {ModuleType::FilterRadius, u8"半径滤波", u8"预处理", IM_COL32(80, 180, 140, 255)},
        {ModuleType::FilterStatistical, u8"统计滤波", u8"预处理", IM_COL32(80, 180, 140, 255)},
        {ModuleType::RoiSelect, u8"ROI 框选", u8"预处理", IM_COL32(90, 170, 220, 255)},
        {ModuleType::FitPlane, u8"平面拟合", u8"拟合", IM_COL32(220, 140, 60, 255)},
        {ModuleType::FitSphere, u8"球面拟合", u8"拟合", IM_COL32(220, 140, 60, 255)},
        {ModuleType::FitCircle, u8"圆拟合", u8"拟合", IM_COL32(220, 140, 60, 255)},
        {ModuleType::FitCylinder, u8"圆柱拟合", u8"拟合", IM_COL32(220, 140, 60, 255)},
        {ModuleType::Flatness, u8"平面度", u8"测量", IM_COL32(180, 120, 220, 255)},
        {ModuleType::StepGap, u8"段差测量", u8"测量", IM_COL32(180, 120, 220, 255)},
        {ModuleType::Section, u8"截面提取", u8"测量", IM_COL32(180, 120, 220, 255)},
        {ModuleType::OutputResult, u8"输出结果", u8"输出", IM_COL32(200, 90, 90, 255)},
    };
    return kDefs;
}

int AlgorithmEditor::CatalogSize() { return static_cast<int>(ModuleType::Count); }

const char* AlgorithmEditor::TypeName(ModuleType t) {
    const ModuleDef* cat = Catalog();
    for (int i = 0; i < CatalogSize(); ++i) {
        if (cat[i].type == t) return cat[i].name;
    }
    return u8"模块";
}

unsigned int AlgorithmEditor::TypeColor(ModuleType t) {
    const ModuleDef* cat = Catalog();
    for (int i = 0; i < CatalogSize(); ++i) {
        if (cat[i].type == t) return cat[i].color;
    }
    return IM_COL32(140, 140, 140, 255);
}

bool AlgorithmEditor::NodeHasInputPort(ModuleType t) {
    return t != ModuleType::InputCloud;
}

bool AlgorithmEditor::NodeHasOutputPort(ModuleType t) {
    return t != ModuleType::OutputResult;
}

AlgorithmEditor::NodeParams AlgorithmEditor::DefaultParams(ModuleType t) {
    NodeParams p;
    std::snprintf(p.outputPath, sizeof(p.outputPath), "result.csv");
    switch (t) {
        case ModuleType::FilterVoxel:
            p.voxelLeaf = 0.5f;
            break;
        case ModuleType::FilterRadius:
            p.outlierRadius = 1.0f;
            p.outlierMinNeighbors = 4;
            break;
        case ModuleType::FilterStatistical:
            p.statMeanK = 20;
            p.statStdMul = 1.0f;
            break;
        case ModuleType::FitSphere:
        case ModuleType::FitCylinder:
            p.minPoints = 50;
            p.maxRms = 0.5f;
            break;
        case ModuleType::FitCircle:
            p.minPoints = 20;
            p.maxRms = 0.3f;
            break;
        case ModuleType::FitPlane:
        case ModuleType::Flatness:
            p.minPoints = 30;
            p.maxRms = 0.2f;
            break;
        case ModuleType::Section:
            p.sectionThickness = 0.05f;
            p.sectionMaxPoints = 200000;
            break;
        case ModuleType::OutputResult:
            p.tolLower = -0.1f;
            p.tolUpper = 0.1f;
            break;
        default:
            break;
    }
    return p;
}

const char* AlgorithmEditor::ParamSummary(const Node& n, char* buf, int bufSize) {
    const NodeParams& p = n.params;
    if (n.hasCloud) {
        std::snprintf(buf, bufSize, u8"%zu 点", n.cloud.points.size());
        return buf;
    }
    switch (n.type) {
        case ModuleType::InputCloud: {
            const char* src[] = {u8"当前点云", u8"文件", u8"生成球", u8"生成圆柱"};
            std::snprintf(buf, bufSize, "%s", src[std::clamp(p.inputSource, 0, 3)]);
            break;
        }
        case ModuleType::FilterVoxel:
            std::snprintf(buf, bufSize, u8"leaf=%.3f", p.voxelLeaf);
            break;
        case ModuleType::FilterRadius:
            std::snprintf(buf, bufSize, u8"R=%.2f N≥%d", p.outlierRadius, p.outlierMinNeighbors);
            break;
        case ModuleType::FilterStatistical:
            std::snprintf(buf, bufSize, u8"K=%d σ×%.2f", p.statMeanK, p.statStdMul);
            break;
        case ModuleType::RoiSelect:
            std::snprintf(buf, bufSize, n.hasCloud ? u8"双击框选编辑" : u8"双击进入框选");
            break;
        case ModuleType::FitPlane:
        case ModuleType::FitSphere:
        case ModuleType::FitCircle:
        case ModuleType::FitCylinder:
        case ModuleType::Flatness:
            std::snprintf(buf, bufSize, u8"min=%d RMS≤%.3f", p.minPoints, p.maxRms);
            break;
        case ModuleType::StepGap:
            std::snprintf(buf, bufSize, "%s", p.useZHeight ? u8"ΔZ" : u8"点面距");
            break;
        case ModuleType::Section:
            std::snprintf(buf, bufSize, u8"%s 厚=%.3f", p.cutAlongX ? "X" : "Y",
                          p.sectionThickness);
            break;
        case ModuleType::OutputResult: {
            const char* fmt[] = {"Status", "CSV", "JSON"};
            std::snprintf(buf, bufSize, "%s", fmt[std::clamp(p.outputFormat, 0, 2)]);
            break;
        }
        default:
            std::snprintf(buf, bufSize, "-");
            break;
    }
    return buf;
}

void AlgorithmEditor::SetVisible(bool v) {
    if (v && !visible_) focusOnOpen_ = true;
    visible_ = v;
    if (!v) {
        draggingNew_ = false;
        paletteArmed_ = false;
        draggingNode_ = false;
        dragNodeId_ = -1;
        linking_ = false;
        previewOpen_ = false;
        previewOrbiting_ = false;
        previewPanning_ = false;
        roiEditOpen_ = false;
        roiDragging_ = false;
        roiOrbiting_ = false;
        roiPanning_ = false;
    }
}

void AlgorithmEditor::ToggleVisible() { SetVisible(!visible_); }

int AlgorithmEditor::AllocId() { return nextId_++; }

AlgorithmEditor::Node* AlgorithmEditor::FindNode(int id) {
    for (Node& n : nodes_) {
        if (n.id == id) return &n;
    }
    return nullptr;
}

const AlgorithmEditor::Node* AlgorithmEditor::FindNode(int id) const {
    for (const Node& n : nodes_) {
        if (n.id == id) return &n;
    }
    return nullptr;
}

void AlgorithmEditor::CanvasToScreen(float cx, float cy, float& sx, float& sy) const {
    sx = canvasScreenX_ + canvasPanX_ + cx * canvasZoom_;
    sy = canvasScreenY_ + canvasPanY_ + cy * canvasZoom_;
}

void AlgorithmEditor::ScreenToCanvas(float sx, float sy, float& cx, float& cy) const {
    const float z = std::max(canvasZoom_, 1e-4f);
    cx = (sx - canvasScreenX_ - canvasPanX_) / z;
    cy = (sy - canvasScreenY_ - canvasPanY_) / z;
}

void AlgorithmEditor::GetPortScreenPos(const Node& n, bool output, float& sx, float& sy) const {
    // 输入在顶部中央，输出在底部中央
    CanvasToScreen(n.x + n.w * 0.5f, n.y + (output ? n.h : 0.f), sx, sy);
}

void AlgorithmEditor::GetNodeActionBtns(const Node& n, float& resetX0, float& resetY0,
                                         float& resetX1, float& resetY1, float& runX0, float& runY0,
                                         float& runX1, float& runY1) const {
    float bx, by, brx, bry;
    CanvasToScreen(n.x, n.y, bx, by);
    CanvasToScreen(n.x + n.w, n.y + n.h, brx, bry);
    const float s = 18.f * canvasZoom_;
    const float pad = 8.f * canvasZoom_;
    // 按钮放在右下角内侧，避开底部中央输出端口
    runX1 = brx - pad;
    runY1 = bry - pad - 10.f * canvasZoom_;
    runX0 = runX1 - s;
    runY0 = runY1 - s;
    resetX1 = runX0 - 4.f * canvasZoom_;
    resetY1 = runY1;
    resetX0 = resetX1 - s;
    resetY0 = runY0;
}

int AlgorithmEditor::HitTestNodeRunBtn(float mx, float my) const {
    for (int i = static_cast<int>(nodes_.size()) - 1; i >= 0; --i) {
        const Node& n = nodes_[static_cast<std::size_t>(i)];
        float rx0, ry0, rx1, ry1, ux0, uy0, ux1, uy1;
        GetNodeActionBtns(n, rx0, ry0, rx1, ry1, ux0, uy0, ux1, uy1);
        if (mx >= ux0 && mx <= ux1 && my >= uy0 && my <= uy1) return n.id;
    }
    return -1;
}

int AlgorithmEditor::HitTestNodeResetBtn(float mx, float my) const {
    for (int i = static_cast<int>(nodes_.size()) - 1; i >= 0; --i) {
        const Node& n = nodes_[static_cast<std::size_t>(i)];
        float rx0, ry0, rx1, ry1, ux0, uy0, ux1, uy1;
        GetNodeActionBtns(n, rx0, ry0, rx1, ry1, ux0, uy0, ux1, uy1);
        if (mx >= rx0 && mx <= rx1 && my >= ry0 && my <= ry1) return n.id;
    }
    return -1;
}

void AlgorithmEditor::RunNode(int nodeId) {
    Node* n = FindNode(nodeId);
    if (!n) return;
    selectedId_ = nodeId;

    PointCloud working;
    std::string err;
    if (n->type != ModuleType::InputCloud) {
        Node* up = FindUpstream(n->id);
        if (up && up->hasCloud) {
            working = up->cloud;
        } else if (NodeHasInputPort(n->type)) {
            runStatus_ = std::string(n->title) + u8"：请先连接并运行上游节点，或先运行上游。";
            runStatusOk_ = false;
            n->runMsg = u8"缺少上游点云";
            n->runOk = false;
            return;
        }
    }

    if (!ExecuteNode(*n, working, err)) {
        n->runMsg = err;
        n->runOk = false;
        runStatus_ = std::string(n->title) + u8"：" + err;
        runStatusOk_ = false;
        return;
    }

    char buf[160];
    std::snprintf(buf, sizeof(buf), u8"[单步] %s 完成", n->title.c_str());
    if (!n->runMsg.empty()) {
        std::snprintf(buf, sizeof(buf), u8"[单步] %s：%s", n->title.c_str(), n->runMsg.c_str());
    }
    runStatus_ = buf;
    runStatusOk_ = true;

    // 输出节点单步运行时同样发布
    if (n->type == ModuleType::OutputResult && n->hasCloud && host_.publishCloud) {
        PointCloud pub = n->cloud;
        host_.publishCloud(std::move(pub), buf);
    }
}

void AlgorithmEditor::ResetNode(int nodeId) {
    Node* n = FindNode(nodeId);
    if (!n) return;
    selectedId_ = nodeId;
    ClearNodeResults(*n);
    n->cloud.Clear();
    n->hasCloud = false;
    n->roiEdited = false;
    n->runMsg.clear();
    n->runOk = false;
    n->params = DefaultParams(n->type);
    if (previewNodeId_ == nodeId) previewOpen_ = false;
    if (roiEditNodeId_ == nodeId) roiEditOpen_ = false;
    runStatus_ = std::string(n->title) + u8"：已重置参数与结果";
    runStatusOk_ = true;
}

void AlgorithmEditor::ClearNodeResults(Node& n) {
    n.plane.reset();
    n.flatness.reset();
    n.sphere.reset();
    n.circle.reset();
    n.cylinder.reset();
}

int AlgorithmEditor::HitTestOutPort(float mx, float my) const {
    const float r = 12.f * canvasZoom_;
    for (int i = static_cast<int>(nodes_.size()) - 1; i >= 0; --i) {
        const Node& n = nodes_[static_cast<std::size_t>(i)];
        if (!NodeHasOutputPort(n.type)) continue;
        float sx, sy;
        GetPortScreenPos(n, true, sx, sy);
        const float dx = mx - sx, dy = my - sy;
        if (dx * dx + dy * dy <= r * r) return n.id;
    }
    return -1;
}

int AlgorithmEditor::HitTestInPort(float mx, float my) const {
    const float r = 12.f * canvasZoom_;
    for (int i = static_cast<int>(nodes_.size()) - 1; i >= 0; --i) {
        const Node& n = nodes_[static_cast<std::size_t>(i)];
        if (!NodeHasInputPort(n.type)) continue;
        float sx, sy;
        GetPortScreenPos(n, false, sx, sy);
        const float dx = mx - sx, dy = my - sy;
        if (dx * dx + dy * dy <= r * r) return n.id;
    }
    return -1;
}

void AlgorithmEditor::ConnectNodes(int fromId, int toId) {
    if (fromId == toId) return;
    Node* a = FindNode(fromId);
    Node* b = FindNode(toId);
    if (!a || !b) return;
    if (!NodeHasOutputPort(a->type) || !NodeHasInputPort(b->type)) return;
    // 每个输入口只保留一条入边
    DisconnectIncoming(toId);
    links_.push_back({fromId, toId});
}

void AlgorithmEditor::DisconnectIncoming(int toId) {
    links_.erase(std::remove_if(links_.begin(), links_.end(),
                                [&](const Link& L) { return L.toNode == toId; }),
                 links_.end());
}

AlgorithmEditor::Node* AlgorithmEditor::FindUpstream(int nodeId) {
    for (const Link& L : links_) {
        if (L.toNode == nodeId) return FindNode(L.fromNode);
    }
    return nullptr;
}

bool AlgorithmEditor::BuildExecOrder(std::vector<int>& outOrder, std::string& error) const {
    outOrder.clear();
    const int n = static_cast<int>(nodes_.size());
    if (n == 0) return true;

    std::vector<int> idToIdx(nextId_ + 1, -1);
    for (int i = 0; i < n; ++i) {
        const int id = nodes_[static_cast<std::size_t>(i)].id;
        if (id >= 0 && id < static_cast<int>(idToIdx.size())) idToIdx[id] = i;
    }

    std::vector<int> indeg(n, 0);
    std::vector<std::vector<int>> adj(n);
    if (!links_.empty()) {
        for (const Link& L : links_) {
            if (L.fromNode < 0 || L.toNode < 0) continue;
            if (L.fromNode >= static_cast<int>(idToIdx.size()) ||
                L.toNode >= static_cast<int>(idToIdx.size()))
                continue;
            const int a = idToIdx[L.fromNode];
            const int b = idToIdx[L.toNode];
            if (a < 0 || b < 0) continue;
            adj[a].push_back(b);
            indeg[b]++;
        }
    } else {
        // 无连线：按从上到下
        std::vector<int> order(n);
        for (int i = 0; i < n; ++i) order[i] = i;
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            if (std::fabs(nodes_[a].y - nodes_[b].y) > 1.f) return nodes_[a].y < nodes_[b].y;
            return nodes_[a].x < nodes_[b].x;
        });
        outOrder = std::move(order);
        return true;
    }

    std::vector<int> q;
    for (int i = 0; i < n; ++i) {
        if (indeg[i] == 0) q.push_back(i);
    }
    while (!q.empty()) {
        const int u = q.back();
        q.pop_back();
        outOrder.push_back(u);
        for (int v : adj[u]) {
            if (--indeg[v] == 0) q.push_back(v);
        }
    }
    if (static_cast<int>(outOrder.size()) != n) {
        error = u8"连线存在环路，请检查。";
        return false;
    }
    return true;
}

void AlgorithmEditor::ClearGraph() {
    nodes_.clear();
    links_.clear();
    selectedId_ = -1;
    draggingNode_ = false;
    dragNodeId_ = -1;
    linking_ = false;
    previewOpen_ = false;
    roiEditOpen_ = false;
    runStatus_.clear();
}

bool AlgorithmEditor::MouseInCanvas(float mx, float my) const {
    return mx >= canvasScreenX_ && mx < canvasScreenX_ + canvasScreenW_ && my >= canvasScreenY_ &&
           my < canvasScreenY_ + canvasScreenH_;
}

int AlgorithmEditor::HitTestNode(float mx, float my) const {
    for (int i = static_cast<int>(nodes_.size()) - 1; i >= 0; --i) {
        const Node& n = nodes_[static_cast<std::size_t>(i)];
        float x0, y0, x1, y1;
        CanvasToScreen(n.x, n.y, x0, y0);
        CanvasToScreen(n.x + n.w, n.y + n.h, x1, y1);
        if (mx >= x0 && mx <= x1 && my >= y0 && my <= y1) return n.id;
    }
    return -1;
}

void AlgorithmEditor::AddNodeAtCanvas(ModuleType type, float canvasX, float canvasY) {
    Node n;
    n.id = AllocId();
    n.type = type;
    n.title = TypeName(type);
    n.params = DefaultParams(type);
    n.w = 200.f;
    n.h = 108.f;
    n.x = canvasX - n.w * 0.5f;
    n.y = canvasY - n.h * 0.5f;
    nodes_.push_back(n);
    selectedId_ = n.id;
}

void AlgorithmEditor::FitPreviewCamera(const PointCloud& cloud) {
    if (!cloud.bounds.Valid()) return;
    const Vec3 e = cloud.bounds.Extent();
    const float maxExtent = std::max({e.x, e.y, e.z, 0.1f});
    previewCam_.SetTarget(cloud.bounds.Center(), maxExtent * 2.0f);
    previewCam_.Reset();
}

void AlgorithmEditor::FitRoiCamera(const PointCloud& cloud) {
    if (!cloud.bounds.Valid()) return;
    const Vec3 e = cloud.bounds.Extent();
    const float maxExtent = std::max({e.x, e.y, e.z, 0.1f});
    roiCam_.SetTarget(cloud.bounds.Center(), maxExtent * 2.0f);
    roiCam_.Reset();
}

bool AlgorithmEditor::EnsureNodeCloudFromUpstream(Node& n, std::string& error) {
    if (n.hasCloud && !n.cloud.points.empty()) return true;
    if (n.type == ModuleType::InputCloud) {
        PointCloud cloud;
        if (!ResolveInputCloud(n.params, cloud, error)) return false;
        n.cloud = std::move(cloud);
        n.hasCloud = true;
        return true;
    }
    Node* up = FindUpstream(n.id);
    if (!up) {
        error = u8"请先连接上游点云节点。";
        return false;
    }
    if (!up->hasCloud || up->cloud.points.empty()) {
        // 尝试自动跑上游输入
        if (up->type == ModuleType::InputCloud) {
            PointCloud cloud;
            if (!ResolveInputCloud(up->params, cloud, error)) return false;
            up->cloud = std::move(cloud);
            up->hasCloud = true;
            up->runOk = true;
        } else {
            error = u8"上游尚无点云，请先运行上游节点。";
            return false;
        }
    }
    n.cloud = up->cloud;
    n.hasCloud = true;
    n.roiEdited = false;
    return true;
}

void AlgorithmEditor::OpenRoiEditor(int nodeId) {
    Node* n = FindNode(nodeId);
    if (!n || n->type != ModuleType::RoiSelect) return;
    std::string err;
    if (!EnsureNodeCloudFromUpstream(*n, err)) {
        runStatus_ = err;
        runStatusOk_ = false;
        return;
    }
    if (n->cloud.mask.size() != n->cloud.points.size()) n->cloud.ResetMask();
    roiEditNodeId_ = nodeId;
    roiEditOpen_ = true;
    roiNeedFocus_ = true;
    roiBoxMode_ = true;
    roiDragging_ = false;
    roiIndices_.clear();
    FitRoiCamera(n->cloud);
    selectedId_ = nodeId;
}

void AlgorithmEditor::OpenCloudPreview(int nodeId) {
    Node* n = FindNode(nodeId);
    if (!n) return;
    if (n->type == ModuleType::RoiSelect) {
        OpenRoiEditor(nodeId);
        return;
    }
    if (!n->hasCloud && n->type == ModuleType::InputCloud) {
        std::string err;
        PointCloud cloud;
        if (ResolveInputCloud(n->params, cloud, err)) {
            n->cloud = std::move(cloud);
            n->hasCloud = true;
            n->runMsg = err.empty() ? u8"预览已加载" : err;
            n->runOk = true;
        } else {
            runStatus_ = err;
            runStatusOk_ = false;
            return;
        }
    }
    if (!n->hasCloud || n->cloud.points.empty()) {
        runStatus_ = u8"该节点尚无点云，请先连线并点击「运行」";
        runStatusOk_ = false;
        return;
    }
    previewNodeId_ = nodeId;
    previewOpen_ = true;
    previewNeedFocus_ = true;
    FitPreviewCamera(n->cloud);
}

bool AlgorithmEditor::ResolveInputCloud(const NodeParams& p, PointCloud& out, std::string& error) {
    out.Clear();
    switch (p.inputSource) {
        case 0: {  // 当前点云
            if (!host_.currentCloud || host_.currentCloud->points.empty()) {
                error = u8"当前主视图没有点云，请先加载/创建，或改用文件/生成。";
                return false;
            }
            out = *host_.currentCloud;
            return true;
        }
        case 1: {
            if (p.filePath[0] == '\0') {
                error = u8"请填写点云文件路径。";
                return false;
            }
            return PointCloudIO::Load(p.filePath, out, error);
        }
        case 2: {
            PointCloudGenerator::SphereParams sp;
            sp.radius = p.genRadius;
            sp.pointCount = p.genPoints;
            sp.noise = p.genNoise;
            return PointCloudGenerator::GenerateSphere(sp, out, error);
        }
        case 3: {
            PointCloudGenerator::CylinderParams cp;
            cp.radius = p.genRadius;
            cp.height = p.genHeight;
            cp.pointCount = p.genPoints;
            cp.noise = p.genNoise;
            return PointCloudGenerator::GenerateCylinder(cp, out, error);
        }
        default:
            error = u8"未知输入源。";
            return false;
    }
}

bool AlgorithmEditor::ApplyFilterMask(PointCloud& cloud, const std::vector<uint8_t>& keep) {
    if (keep.size() != cloud.points.size()) return false;
    if (cloud.mask.size() != cloud.points.size()) cloud.ResetMask();
    for (std::size_t i = 0; i < keep.size(); ++i) {
        if (!keep[i]) cloud.mask[i] = 0;
    }
    return true;
}

bool AlgorithmEditor::ExecuteNode(Node& node, PointCloud& working, std::string& error) {
    node.runOk = false;
    node.runMsg.clear();
    const NodeParams& p = node.params;

    auto storeWorking = [&]() {
        node.cloud = working;
        node.hasCloud = !working.points.empty();
    };

    switch (node.type) {
        case ModuleType::InputCloud: {
            PointCloud cloud;
            if (!ResolveInputCloud(p, cloud, error)) return false;
            working = std::move(cloud);
            storeWorking();
            char buf[96];
            std::snprintf(buf, sizeof(buf), u8"输入 %zu 点", working.points.size());
            node.runMsg = buf;
            node.runOk = true;
            return true;
        }
        case ModuleType::FilterVoxel:
        case ModuleType::FilterRadius:
        case ModuleType::FilterStatistical: {
            if (working.points.empty()) {
                error = u8"滤波前没有点云，请先连接/放置输入点云。";
                return false;
            }
            std::vector<uint8_t> keep;
            int kept = 0;
            bool ok = false;
            if (node.type == ModuleType::FilterVoxel)
                ok = FilterTools::VoxelDownsample(working, p.voxelLeaf, keep, error, &kept);
            else if (node.type == ModuleType::FilterRadius)
                ok = FilterTools::RadiusOutlier(working, p.outlierRadius, p.outlierMinNeighbors, keep,
                                                error, &kept);
            else
                ok = FilterTools::StatisticalOutlier(working, p.statMeanK, p.statStdMul, keep, error,
                                                     &kept);
            if (!ok) return false;
            ApplyFilterMask(working, keep);
            storeWorking();
            char buf[96];
            std::snprintf(buf, sizeof(buf), u8"保留约 %d 可见点", kept);
            node.runMsg = buf;
            node.runOk = true;
            return true;
        }
        case ModuleType::RoiSelect: {
            if (node.roiEdited && node.hasCloud && !node.cloud.points.empty()) {
                working = node.cloud;
            } else {
                if (working.points.empty()) {
                    error = u8"ROI 前没有点云，请连接上游并双击本模块进行框选。";
                    return false;
                }
                if (working.mask.size() != working.points.size()) working.ResetMask();
                storeWorking();
            }
            char buf[96];
            std::snprintf(buf, sizeof(buf), u8"ROI 输出可见 %zu / %zu", working.VisibleCount(),
                          working.points.size());
            node.runMsg = buf;
            node.runOk = true;
            node.hasCloud = true;
            node.cloud = working;
            return true;
        }
        case ModuleType::FitPlane: {
            if (working.VisibleCount() < static_cast<std::size_t>(std::max(3, p.minPoints))) {
                error = u8"平面拟合点数不足。";
                return false;
            }
            PlaneModel plane;
            std::vector<std::size_t> empty;
            if (!MeasureTools::FitPlaneSVD(working, empty, plane, error)) return false;
            if (p.flipNormalUp && plane.normal.z < 0.f) {
                plane.normal = plane.normal * -1.f;
            }
            if (plane.rms > p.maxRms) {
                char buf[128];
                std::snprintf(buf, sizeof(buf), u8"RMS=%.4f 超过阈值 %.4f", plane.rms, p.maxRms);
                error = buf;
                return false;
            }
            storeWorking();
            ClearNodeResults(node);
            node.plane = plane;
            const float d = -plane.normal.Dot(plane.centroid);
            char buf[192];
            std::snprintf(buf, sizeof(buf),
                          u8"平面 RMS=%.4f\n%.4fx%+.4fy%+.4fz%+.4f=0", plane.rms, plane.normal.x,
                          plane.normal.y, plane.normal.z, d);
            node.runMsg = buf;
            node.runOk = true;
            return true;
        }
        case ModuleType::FitSphere: {
            SphereModel s;
            std::vector<std::size_t> empty;
            if (!MeasureTools::FitSphere(working, empty, s, error)) return false;
            if (s.pointCount < p.minPoints) {
                error = u8"球面拟合点数不足。";
                return false;
            }
            if (s.rms > p.maxRms) {
                error = u8"球面拟合 RMS 超限。";
                return false;
            }
            storeWorking();
            ClearNodeResults(node);
            node.sphere = s;
            char buf[128];
            std::snprintf(buf, sizeof(buf), u8"球 R=%.4f RMS=%.4f", s.radius, s.rms);
            node.runMsg = buf;
            node.runOk = true;
            return true;
        }
        case ModuleType::FitCircle: {
            CircleModel c;
            std::vector<std::size_t> empty;
            if (!MeasureTools::FitCircle3D(working, empty, c, error)) return false;
            if (c.pointCount < p.minPoints || c.rms > p.maxRms) {
                error = u8"圆拟合未满足点数/RMS 条件。";
                return false;
            }
            storeWorking();
            ClearNodeResults(node);
            node.circle = c;
            char buf[128];
            std::snprintf(buf, sizeof(buf), u8"圆 R=%.4f RMS=%.4f", c.radius, c.rms);
            node.runMsg = buf;
            node.runOk = true;
            return true;
        }
        case ModuleType::FitCylinder: {
            CylinderModel c;
            std::vector<std::size_t> empty;
            if (!MeasureTools::FitCylinder(working, empty, c, error)) return false;
            if (c.pointCount < p.minPoints || c.rms > p.maxRms) {
                error = u8"圆柱拟合未满足点数/RMS 条件。";
                return false;
            }
            storeWorking();
            ClearNodeResults(node);
            node.cylinder = c;
            char buf[128];
            std::snprintf(buf, sizeof(buf), u8"圆柱 R=%.4f RMS=%.4f", c.radius, c.rms);
            node.runMsg = buf;
            node.runOk = true;
            return true;
        }
        case ModuleType::Flatness: {
            FlatnessResult fr;
            std::vector<std::size_t> empty;
            if (!MeasureTools::ComputeFlatness(working, empty, fr, error)) return false;
            storeWorking();
            ClearNodeResults(node);
            node.flatness = fr;
            node.plane = fr.plane;
            char buf[160];
            std::snprintf(buf, sizeof(buf), u8"平面度 PV=%.4f RMS=%.4f  偏差[%.4f,%.4f]",
                          fr.peakToValley, fr.rms, fr.minDev, fr.maxDev);
            node.runMsg = buf;
            node.runOk = true;
            return true;
        }
        case ModuleType::StepGap:
            storeWorking();
            node.runMsg = u8"段差需在主工具中框选 A/B（图中仅透传点云）";
            node.runOk = true;
            return true;
        case ModuleType::Section: {
            SectionData sec;
            if (!MeasureTools::ExtractSection(working, p.cutAlongX, p.sectionPos, p.sectionThickness,
                                              sec, error, p.sectionMaxPoints))
                return false;
            storeWorking();
            char buf[96];
            std::snprintf(buf, sizeof(buf), u8"截面轮廓 %zu 点", sec.points.size());
            node.runMsg = buf;
            node.runOk = true;
            return true;
        }
        case ModuleType::OutputResult: {
            if (working.points.empty()) {
                error = u8"输出节点没有点云数据。";
                return false;
            }
            storeWorking();
            ClearNodeResults(node);
            // 从上游链收集最近一次拟合/测量结果，复制到本节点供右侧结果栏展示
            Node* up = FindUpstream(node.id);
            int hops = 0;
            while (up && hops < 64) {
                if (up->flatness) {
                    node.flatness = up->flatness;
                    node.plane = up->flatness->plane;
                    break;
                }
                if (up->plane) {
                    node.plane = up->plane;
                    break;
                }
                if (up->sphere) {
                    node.sphere = up->sphere;
                    break;
                }
                if (up->circle) {
                    node.circle = up->circle;
                    break;
                }
                if (up->cylinder) {
                    node.cylinder = up->cylinder;
                    break;
                }
                up = FindUpstream(up->id);
                ++hops;
            }

            std::string summary = u8"输出 ";
            summary += std::to_string(working.points.size()) + u8" 点（可见 ";
            summary += std::to_string(working.VisibleCount()) + u8"）";
            if (node.flatness) {
                char buf[128];
                std::snprintf(buf, sizeof(buf), u8"\n平面度 PV=%.6f  RMS=%.6f",
                              node.flatness->peakToValley, node.flatness->rms);
                summary += buf;
                if (node.params.outputOkNg) {
                    const float v = node.flatness->peakToValley;
                    const bool ok = v >= node.params.tolLower && v <= node.params.tolUpper;
                    summary += ok ? u8"  → OK" : u8"  → NG";
                }
            } else if (node.plane) {
                const float d = -node.plane->normal.Dot(node.plane->centroid);
                char buf[160];
                std::snprintf(buf, sizeof(buf), u8"\n平面方程 %.6fx%+.6fy%+.6fz%+.6f=0\nRMS=%.6f",
                              node.plane->normal.x, node.plane->normal.y, node.plane->normal.z, d,
                              node.plane->rms);
                summary += buf;
            }
            node.runMsg = summary;
            node.runOk = true;
            if (host_.publishCloud) {
                PointCloud pub = working;
                host_.publishCloud(std::move(pub), summary.c_str());
            }
            return true;
        }
        default:
            error = u8"未实现的算子。";
            return false;
    }
}

void AlgorithmEditor::RunGraph() {
    runStatus_.clear();
    runStatusOk_ = true;
    if (nodes_.empty()) {
        runStatus_ = u8"画布为空，请先拖入算子。";
        runStatusOk_ = false;
        return;
    }

    std::vector<int> order;
    std::string orderErr;
    if (!BuildExecOrder(order, orderErr)) {
        runStatus_ = orderErr;
        runStatusOk_ = false;
        return;
    }

    for (Node& n : nodes_) {
        n.hasCloud = false;
        n.cloud.Clear();
        n.runMsg.clear();
        n.runOk = false;
        ClearNodeResults(n);
    }

    PointCloud lastCloud;
    for (int idx : order) {
        Node& n = nodes_[static_cast<std::size_t>(idx)];
        PointCloud working;
        std::string err;

        if (n.type != ModuleType::InputCloud) {
            Node* up = FindUpstream(n.id);
            if (up && up->hasCloud) {
                working = up->cloud;
            } else if (!links_.empty()) {
                err = u8"请将上游节点底部输出拖到本节点顶部输入完成连线。";
                n.runMsg = err;
                n.runOk = false;
                runStatus_ = std::string(n.title) + u8"：" + err;
                runStatusOk_ = false;
                return;
            } else {
                working = lastCloud;
            }
        }

        if (!ExecuteNode(n, working, err)) {
            n.runMsg = err;
            n.runOk = false;
            runStatus_ = std::string(n.title) + u8"：" + err;
            runStatusOk_ = false;
            return;
        }
        if (n.hasCloud) lastCloud = n.cloud;
    }

    bool hasOut = false;
    for (const Node& n : nodes_) {
        if (n.type == ModuleType::OutputResult && n.runOk) hasOut = true;
    }
    if (!hasOut && !lastCloud.points.empty() && host_.publishCloud) {
        char buf[96];
        std::snprintf(buf, sizeof(buf), u8"算法运行完成，%zu 点已同步到主视图",
                      lastCloud.points.size());
        PointCloud pub = lastCloud;
        host_.publishCloud(std::move(pub), buf);
        runStatus_ = buf;
    } else if (links_.empty()) {
        runStatus_ = u8"运行完成（无连线，按左右顺序）。建议用端口连线。";
    } else {
        runStatus_ = u8"运行完成。双击节点可三维预览点云。";
    }
    runStatusOk_ = true;
}

void AlgorithmEditor::Draw(float menuBottomY) {
    if (!visible_) return;

    ImGuiViewport* vp = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(ImVec2(vp->Pos.x, menuBottomY), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(vp->Size.x, vp->Pos.y + vp->Size.y - menuBottomY),
                             ImGuiCond_Always);
    ImGui::SetNextWindowBgAlpha(1.f);
    if (focusOnOpen_) {
        ImGui::SetNextWindowFocus();
        focusOnOpen_ = false;
    }

    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.f);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.09f, 0.10f, 0.12f, 1.f));

    const ImGuiWindowFlags flags =
        ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoSavedSettings;

    bool open = visible_;
    if (!ImGui::Begin(u8"算法编辑器", &open, flags)) {
        ImGui::End();
        ImGui::PopStyleColor();
        ImGui::PopStyleVar(2);
        visible_ = open;
        return;
    }
    visible_ = open;
    if (!visible_) {
        ImGui::End();
        ImGui::PopStyleColor();
        ImGui::PopStyleVar(2);
        return;
    }
    if (ImGui::IsWindowAppearing()) ImGui::SetWindowFocus();

    DrawToolbar();
    ImGui::Separator();

    const float fullH = ImGui::GetContentRegionAvail().y;
    const float paletteW = 200.f;
    const float propsW = 280.f;

    ImGui::BeginChild(u8"##algo_palette", ImVec2(paletteW, fullH), true);
    DrawPalette();
    ImGui::EndChild();

    ImGui::SameLine();
    ImGui::BeginChild(u8"##algo_canvas", ImVec2(-(propsW + 8.f), fullH), true,
                      ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    DrawCanvas();
    ImGui::EndChild();

    ImGui::SameLine();
    ImGui::BeginChild(u8"##algo_props", ImVec2(0.f, fullH), true);
    DrawProperties();
    ImGui::EndChild();

    DrawGhostModule();
    DrawCloudPreviewWindow();
    DrawRoiEditWindow();

    ImGui::End();
    ImGui::PopStyleColor();
    ImGui::PopStyleVar(2);
}

void AlgorithmEditor::DrawToolbar() {
    if (ImGui::Button(u8"关闭编辑器")) SetVisible(false);
    ImGui::SameLine();
    if (ImGui::Button(u8"清空画布")) ClearGraph();
    ImGui::SameLine(0.f, 16.f);

    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.15f, 0.55f, 0.35f, 1.f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.20f, 0.65f, 0.42f, 1.f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.12f, 0.48f, 0.30f, 1.f));
    if (ImGui::Button(u8"▶ 运行", ImVec2(100.f, 0))) RunGraph();
    ImGui::PopStyleColor(3);

    ImGui::SameLine();
    ImGui::TextDisabled(u8"▶单步/↺重置 | 顶→底连线 | 双击预览 | Shift+滚轮缩放");
    ImGui::SameLine(ImGui::GetWindowWidth() - 120.f);
    ImGui::TextColored(ImVec4(0.45f, 0.85f, 0.90f, 1.f), u8"节点 %d",
                       static_cast<int>(nodes_.size()));

    if (!runStatus_.empty()) {
        ImGui::Spacing();
        ImGui::TextColored(runStatusOk_ ? ImVec4(0.45f, 0.85f, 0.55f, 1.f)
                                        : ImVec4(0.95f, 0.45f, 0.40f, 1.f),
                           "%s", runStatus_.c_str());
    }
}

void AlgorithmEditor::DrawPalette() {
    ImGui::TextDisabled(u8"算子库");
    ImGui::Spacing();

    const ModuleDef* cat = Catalog();
    const char* lastCat = nullptr;
    const ImGuiIO& io = ImGui::GetIO();

    for (int i = 0; i < CatalogSize(); ++i) {
        const ModuleDef& def = cat[i];
        if (!lastCat || std::strcmp(lastCat, def.category) != 0) {
            if (lastCat) ImGui::Spacing();
            ImGui::TextDisabled("%s", def.category);
            lastCat = def.category;
        }

        ImGui::PushID(i);
        const ImVec4 base = ImGui::ColorConvertU32ToFloat4(def.color);
        ImGui::PushStyleColor(ImGuiCol_Button, base);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered,
                              ImVec4(base.x + 0.08f, base.y + 0.08f, base.z + 0.08f, 1.f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive,
                              ImVec4(base.x + 0.12f, base.y + 0.12f, base.z + 0.12f, 1.f));
        ImGui::Button(def.name, ImVec2(-1.f, 30.f));
        ImGui::PopStyleColor(3);

        if (ImGui::IsItemActivated()) {
            paletteArmed_ = true;
            armedType_ = def.type;
        }
        if (paletteArmed_ && armedType_ == def.type && ImGui::IsItemActive() &&
            ImGui::IsMouseDragging(ImGuiMouseButton_Left, 4.f)) {
            draggingNew_ = true;
            dragType_ = def.type;
        }
        if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
            float cx = 0.f, cy = 0.f;
            ScreenToCanvas(canvasScreenX_ + canvasScreenW_ * 0.5f,
                           canvasScreenY_ + canvasScreenH_ * 0.5f, cx, cy);
            AddNodeAtCanvas(def.type, cx, cy);
            paletteArmed_ = false;
            draggingNew_ = false;
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip(u8"拖到画布，或双击添加到中心");
        ImGui::PopID();
    }

    if (!io.MouseDown[0] && !draggingNew_) paletteArmed_ = false;
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::TextWrapped(u8"从节点底部端口拖到下一节点顶部端口连线，再点运行。");
}

void AlgorithmEditor::DrawGhostModule() {
    if (!draggingNew_) return;
    const ImGuiIO& io = ImGui::GetIO();
    ImDrawList* dl = ImGui::GetForegroundDrawList();
    const float w = 200.f, h = 108.f;
    const ImVec2 p0(io.MousePos.x - w * 0.5f, io.MousePos.y - h * 0.5f);
    const ImVec2 p1(p0.x + w, p0.y + h);
    const ImU32 col = TypeColor(dragType_);
    dl->AddRectFilled(p0, p1, IM_COL32(28, 32, 38, 200), 6.f);
    dl->AddRectFilled(p0, ImVec2(p1.x, p0.y + 22.f), col, 6.f, ImDrawFlags_RoundCornersTop);
    dl->AddRect(p0, p1, col, 6.f, 0, 2.f);
    dl->AddText(ImVec2(p0.x + 10.f, p0.y + 4.f), IM_COL32(255, 255, 255, 255), TypeName(dragType_));

    if (MouseInCanvas(io.MousePos.x, io.MousePos.y)) {
        dl->AddText(ImVec2(io.MousePos.x + 16.f, io.MousePos.y + 16.f), IM_COL32(120, 230, 210, 255),
                    u8"松开以放置");
    }

    if (!io.MouseDown[0]) {
        if (MouseInCanvas(io.MousePos.x, io.MousePos.y)) {
            float cx = 0.f, cy = 0.f;
            ScreenToCanvas(io.MousePos.x, io.MousePos.y, cx, cy);
            AddNodeAtCanvas(dragType_, cx, cy);
        }
        draggingNew_ = false;
        paletteArmed_ = false;
    }
}

void AlgorithmEditor::DrawNodeVisual(const Node& node, bool selected) {
    ImDrawList* dl = ImGui::GetWindowDrawList();
    float x0, y0, x1, y1;
    CanvasToScreen(node.x, node.y, x0, y0);
    CanvasToScreen(node.x + node.w, node.y + node.h, x1, y1);
    const ImVec2 p0(x0, y0);
    const ImVec2 p1(x1, y1);
    const float z = canvasZoom_;
    const float round = 8.f * z;
    const float headerH = 28.f * z;
    const float footerH = 22.f * z;

    const ImU32 fill = IM_COL32(32, 36, 44, 255);
    const ImU32 border = selected ? IM_COL32(80, 220, 210, 255) : IM_COL32(55, 62, 72, 255);
    const ImU32 header = TypeColor(node.type);
    const ImU32 footer = IM_COL32(24, 28, 34, 255);

    // 主体卡片（类梅卡曼德：头-身-尾）
    dl->AddRectFilled(p0, p1, fill, round);
    dl->AddRectFilled(p0, ImVec2(p1.x, p0.y + headerH), header, round, ImDrawFlags_RoundCornersTop);
    dl->AddRectFilled(ImVec2(p0.x, p1.y - footerH), p1, footer, round, ImDrawFlags_RoundCornersBottom);
    dl->AddRect(p0, p1, border, round, 0, selected ? 2.4f : 1.4f);

    // 标题
    dl->AddText(ImVec2(p0.x + 12.f * z, p0.y + 6.f * z), IM_COL32(255, 255, 255, 255),
                node.title.c_str());

    char summary[64];
    ParamSummary(node, summary, sizeof(summary));
    dl->AddText(ImVec2(p0.x + 12.f * z, p0.y + headerH + 8.f * z), IM_COL32(170, 180, 190, 255),
                summary);

    if (node.hasCloud) {
        dl->AddText(ImVec2(p0.x + 12.f * z, p0.y + headerH + 28.f * z), IM_COL32(90, 210, 160, 255),
                    u8"双击预览");
    } else if (node.runOk) {
        dl->AddText(ImVec2(p0.x + 12.f * z, p0.y + headerH + 28.f * z), IM_COL32(120, 200, 140, 255),
                    u8"已运行");
    } else {
        dl->AddText(ImVec2(p0.x + 12.f * z, p0.y + headerH + 28.f * z), IM_COL32(110, 125, 140, 255),
                    u8"点 ▶ 单步运行");
    }

    // 顶部输入 / 底部输出端口
    float inX, inY, outX, outY;
    GetPortScreenPos(node, false, inX, inY);
    GetPortScreenPos(node, true, outX, outY);
    const float pr = 7.5f * z;
    auto drawPort = [&](float px, float py, bool isOut) {
        // 菱形端口（梅卡曼德风格）
        const float s = pr * 1.15f;
        const ImVec2 diamond[4] = {
            ImVec2(px, py - s),
            ImVec2(px + s, py),
            ImVec2(px, py + s),
            ImVec2(px - s, py),
        };
        const ImU32 fillCol = isOut ? IM_COL32(70, 200, 170, 255) : IM_COL32(90, 190, 220, 255);
        dl->AddConvexPolyFilled(diamond, 4, fillCol);
        dl->AddPolyline(diamond, 4, IM_COL32(20, 35, 40, 255), ImDrawFlags_Closed, 1.5f);
    };
    if (NodeHasInputPort(node.type)) drawPort(inX, inY, false);
    if (NodeHasOutputPort(node.type)) drawPort(outX, outY, true);

    float rx0, ry0, rx1, ry1, ux0, uy0, ux1, uy1;
    GetNodeActionBtns(node, rx0, ry0, rx1, ry1, ux0, uy0, ux1, uy1);

    dl->AddRectFilled(ImVec2(rx0, ry0), ImVec2(rx1, ry1), IM_COL32(55, 62, 72, 255), 3.f * z);
    dl->AddRect(ImVec2(rx0, ry0), ImVec2(rx1, ry1), IM_COL32(140, 150, 160, 255), 3.f * z, 0, 1.f);
    {
        const float cx = 0.5f * (rx0 + rx1);
        const float cy = 0.5f * (ry0 + ry1);
        dl->AddCircle(ImVec2(cx, cy), 4.5f * z, IM_COL32(200, 210, 220, 255), 0, 1.6f);
        dl->AddTriangleFilled(ImVec2(cx + 3.5f * z, cy - 4.5f * z),
                              ImVec2(cx + 6.5f * z, cy - 1.f * z), ImVec2(cx + 1.f * z, cy - 1.f * z),
                              IM_COL32(200, 210, 220, 255));
    }

    const ImU32 runBg = node.runOk ? IM_COL32(35, 120, 75, 255) : IM_COL32(30, 100, 70, 255);
    dl->AddRectFilled(ImVec2(ux0, uy0), ImVec2(ux1, uy1), runBg, 3.f * z);
    dl->AddRect(ImVec2(ux0, uy0), ImVec2(ux1, uy1), IM_COL32(90, 220, 150, 255), 3.f * z, 0, 1.2f);
    {
        const float cx = 0.5f * (ux0 + ux1);
        const float cy = 0.5f * (uy0 + uy1);
        dl->AddTriangleFilled(ImVec2(cx - 3.5f * z, cy - 5.f * z), ImVec2(cx - 3.5f * z, cy + 5.f * z),
                              ImVec2(cx + 5.5f * z, cy), IM_COL32(240, 255, 245, 255));
    }
}

void AlgorithmEditor::DrawCanvas() {
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImVec2 canvasPos = ImGui::GetCursorScreenPos();
    const ImVec2 canvasSize = ImGui::GetContentRegionAvail();
    canvasScreenX_ = canvasPos.x;
    canvasScreenY_ = canvasPos.y;
    canvasScreenW_ = canvasSize.x;
    canvasScreenH_ = canvasSize.y;
    const ImVec2 canvasMax(canvasPos.x + canvasSize.x, canvasPos.y + canvasSize.y);

    dl->AddRectFilled(canvasPos, canvasMax, IM_COL32(14, 16, 20, 255));
    const float grid = 24.f * canvasZoom_;
    const ImU32 gridCol = IM_COL32(36, 40, 48, 255);
    const float ox = std::fmod(canvasPanX_, grid);
    const float oy = std::fmod(canvasPanY_, grid);
    for (float x = ox; x < canvasSize.x; x += grid)
        dl->AddLine(ImVec2(canvasPos.x + x, canvasPos.y), ImVec2(canvasPos.x + x, canvasMax.y),
                    gridCol);
    for (float y = oy; y < canvasSize.y; y += grid)
        dl->AddLine(ImVec2(canvasPos.x, canvasPos.y + y), ImVec2(canvasMax.x, canvasPos.y + y),
                    gridCol);

    ImGui::InvisibleButton(u8"##canvas_bg", canvasSize,
                           ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonMiddle);
    const bool canvasHovered = ImGui::IsItemHovered();
    const ImGuiIO& io = ImGui::GetIO();
    const bool spaceDown = ImGui::IsKeyDown(ImGuiKey_Space);
    const bool shift = io.KeyShift;

    // Shift + 滚轮：以鼠标为中心缩放画布
    if (canvasHovered && shift && io.MouseWheel != 0.f) {
        float beforeX = 0.f, beforeY = 0.f;
        ScreenToCanvas(io.MousePos.x, io.MousePos.y, beforeX, beforeY);
        const float factor = (io.MouseWheel > 0.f) ? 1.12f : (1.f / 1.12f);
        canvasZoom_ = std::clamp(canvasZoom_ * factor, 0.25f, 3.f);
        canvasPanX_ = io.MousePos.x - canvasScreenX_ - beforeX * canvasZoom_;
        canvasPanY_ = io.MousePos.y - canvasScreenY_ - beforeY * canvasZoom_;
    }

    // —— 连线拖拽 ——
    if (linking_) {
        float sx = io.MousePos.x, sy = io.MousePos.y;
        if (const Node* from = FindNode(linkFromId_)) {
            GetPortScreenPos(*from, true, sx, sy);
        }
        DrawSmoothLink(dl, sx, sy, io.MousePos.x, io.MousePos.y, IM_COL32(120, 230, 200, 220),
                       3.f * canvasZoom_, canvasZoom_);
        if (!io.MouseDown[0]) {
            const int toId = HitTestInPort(io.MousePos.x, io.MousePos.y);
            if (toId >= 0) ConnectNodes(linkFromId_, toId);
            linking_ = false;
            linkFromId_ = -1;
        }
    } else if (draggingNode_) {
        if (!io.MouseDown[0]) {
            draggingNode_ = false;
            dragNodeId_ = -1;
            clickOnNode_ = false;
        } else if (Node* n = FindNode(dragNodeId_)) {
            float cx = 0.f, cy = 0.f;
            ScreenToCanvas(io.MousePos.x, io.MousePos.y, cx, cy);
            n->x = cx - dragGrabDX_;
            n->y = cy - dragGrabDY_;
        }
    } else if (!draggingNew_ && canvasHovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left) &&
               !spaceDown) {
        const int runId = HitTestNodeRunBtn(io.MousePos.x, io.MousePos.y);
        const int resetId = HitTestNodeResetBtn(io.MousePos.x, io.MousePos.y);
        if (runId >= 0) {
            RunNode(runId);
        } else if (resetId >= 0) {
            ResetNode(resetId);
        } else {
            const int outPort = HitTestOutPort(io.MousePos.x, io.MousePos.y);
            if (outPort >= 0) {
                linking_ = true;
                linkFromId_ = outPort;
                selectedId_ = outPort;
            } else {
                const int hit = HitTestNode(io.MousePos.x, io.MousePos.y);
                if (hit >= 0) {
                    if (HitTestInPort(io.MousePos.x, io.MousePos.y) < 0) {
                        selectedId_ = hit;
                        draggingNode_ = true;
                        dragNodeId_ = hit;
                        clickOnNode_ = true;
                        clickNodeId_ = hit;
                        clickDownX_ = io.MousePos.x;
                        clickDownY_ = io.MousePos.y;
                        if (Node* n = FindNode(hit)) {
                            float cx = 0.f, cy = 0.f;
                            ScreenToCanvas(io.MousePos.x, io.MousePos.y, cx, cy);
                            dragGrabDX_ = cx - n->x;
                            dragGrabDY_ = cy - n->y;
                            auto it = std::find_if(nodes_.begin(), nodes_.end(),
                                                   [&](const Node& x) { return x.id == hit; });
                            if (it != nodes_.end()) {
                                Node moved = std::move(*it);
                                nodes_.erase(it);
                                nodes_.push_back(std::move(moved));
                            }
                        }
                    } else {
                        selectedId_ = hit;
                    }
                } else {
                    selectedId_ = -1;
                }
            }
        }
    }

    if (!draggingNew_ && !linking_ && canvasHovered &&
        ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left) && !spaceDown) {
        if (HitTestNodeRunBtn(io.MousePos.x, io.MousePos.y) < 0 &&
            HitTestNodeResetBtn(io.MousePos.x, io.MousePos.y) < 0) {
            const int hit = HitTestNode(io.MousePos.x, io.MousePos.y);
            if (hit >= 0) {
                selectedId_ = hit;
                OpenCloudPreview(hit);
                draggingNode_ = false;
                dragNodeId_ = -1;
            }
        }
    }

    if (!draggingNew_ && !draggingNode_ && !linking_ && canvasHovered) {
        if (ImGui::IsMouseDragging(ImGuiMouseButton_Middle, 0.f)) {
            canvasPanX_ += io.MouseDelta.x;
            canvasPanY_ += io.MouseDelta.y;
        } else if (spaceDown && ImGui::IsMouseDragging(ImGuiMouseButton_Left, 0.f)) {
            canvasPanX_ += io.MouseDelta.x;
            canvasPanY_ += io.MouseDelta.y;
        }
    }

    for (const Link& L : links_) {
        const Node* a = FindNode(L.fromNode);
        const Node* b = FindNode(L.toNode);
        if (!a || !b) continue;
        float ax, ay, bx, by;
        GetPortScreenPos(*a, true, ax, ay);
        GetPortScreenPos(*b, false, bx, by);
        DrawSmoothLink(dl, ax, ay, bx, by, IM_COL32(100, 200, 190, 230), 3.2f * canvasZoom_,
                       canvasZoom_);
    }

    for (const Node& n : nodes_) DrawNodeVisual(n, n.id == selectedId_);

    if (nodes_.empty() && !draggingNew_) {
        const char* tip = u8"拖入算子 → 底端口拖到顶端口连线 → 运行";
        const ImVec2 sz = ImGui::CalcTextSize(tip);
        dl->AddText(ImVec2(canvasPos.x + (canvasSize.x - sz.x) * 0.5f,
                           canvasPos.y + canvasSize.y * 0.45f),
                    IM_COL32(110, 125, 140, 200), tip);
    }
}

void AlgorithmEditor::DrawCloudPreviewWindow() {
    if (!previewOpen_) return;
    Node* n = FindNode(previewNodeId_);
    if (!n || !n->hasCloud || n->cloud.points.empty()) {
        previewOpen_ = false;
        return;
    }

    ImGui::SetNextWindowSize(ImVec2(640.f, 480.f), ImGuiCond_FirstUseEver);
    if (previewNeedFocus_) {
        ImGui::SetNextWindowFocus();
        previewNeedFocus_ = false;
    }
    if (!ImGui::Begin(u8"点云预览", &previewOpen_, ImGuiWindowFlags_NoCollapse)) {
        ImGui::End();
        return;
    }

    const PointCloud& cloud = n->cloud;
    ImGui::Text("%s", n->title.c_str());
    ImGui::SameLine(0.f, 16.f);
    ImGui::Text(u8"点数 %zu　可见 %zu", cloud.points.size(), cloud.VisibleCount());
    if (ImGui::Button(u8"复位视角")) FitPreviewCamera(cloud);
    ImGui::SameLine();
    ImGui::TextDisabled(u8"左键旋转 | 中键/Shift+左键平移 | 滚轮缩放");
    if (!n->runMsg.empty()) {
        ImGui::TextColored(n->runOk ? ImVec4(0.5f, 0.85f, 0.6f, 1.f) : ImVec4(0.95f, 0.5f, 0.4f, 1.f),
                           "%s", n->runMsg.c_str());
    }
    ImGui::Separator();

    const ImVec2 plotPos = ImGui::GetCursorScreenPos();
    const ImVec2 plotSize = ImGui::GetContentRegionAvail();
    ImGui::InvisibleButton(u8"##preview3d", plotSize,
                           ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonMiddle);
    const bool hovered = ImGui::IsItemHovered();
    const bool active = ImGui::IsItemActive();
    const ImGuiIO& io = ImGui::GetIO();

    if (hovered) {
        if (ImGui::IsMouseClicked(ImGuiMouseButton_Left) && !io.KeyShift) previewOrbiting_ = true;
        if (ImGui::IsMouseClicked(ImGuiMouseButton_Middle) ||
            (ImGui::IsMouseClicked(ImGuiMouseButton_Left) && io.KeyShift))
            previewPanning_ = true;
        if (io.MouseWheel != 0.f) previewCam_.Zoom(io.MouseWheel);
    }
    if (!io.MouseDown[0]) previewOrbiting_ = false;
    if (!io.MouseDown[2] && !(io.MouseDown[0] && io.KeyShift)) previewPanning_ = false;

    if (previewOrbiting_ && io.MouseDown[0] && !io.KeyShift) {
        previewCam_.Orbit(io.MouseDelta.x * 0.01f, io.MouseDelta.y * 0.01f);
    }
    if (previewPanning_) {
        previewCam_.Pan(io.MouseDelta.x, io.MouseDelta.y, io.MouseDown[2] ? 1.f : 0.35f);
    }
    (void)active;

    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(plotPos, ImVec2(plotPos.x + plotSize.x, plotPos.y + plotSize.y),
                      IM_COL32(12, 14, 18, 255));
    dl->AddRect(plotPos, ImVec2(plotPos.x + plotSize.x, plotPos.y + plotSize.y),
                IM_COL32(50, 60, 70, 255));

    if (plotSize.x > 10.f && plotSize.y > 10.f && cloud.bounds.Valid()) {
        const float aspect = plotSize.x / plotSize.y;
        const Mat4 mvp = previewCam_.ProjMatrix(aspect) * previewCam_.ViewMatrix();
        const float zMin = cloud.bounds.min.z;
        float zMax = cloud.bounds.max.z;
        if (zMax <= zMin) zMax = zMin + 1.f;
        const float invZ = 1.f / (zMax - zMin);

        struct Pix {
            float x, y, d;
            ImU32 col;
        };
        std::vector<Pix> pix;
        const int stride = std::max(1, static_cast<int>(cloud.points.size() / 120000));
        pix.reserve(cloud.points.size() / static_cast<std::size_t>(stride) + 8);

        for (std::size_t i = 0; i < cloud.points.size(); i += static_cast<std::size_t>(stride)) {
            if (!cloud.mask.empty() && !cloud.mask[i]) continue;
            const Vec3& p = cloud.points[i];
            const Vec4 clip = mvp.MulVec4({p.x, p.y, p.z, 1.f});
            if (std::fabs(clip.w) < 1e-12f) continue;
            const float ndcX = clip.x / clip.w;
            const float ndcY = clip.y / clip.w;
            const float ndcZ = clip.z / clip.w;
            if (ndcZ < -1.f || ndcZ > 1.f) continue;
            if (ndcX < -1.2f || ndcX > 1.2f || ndcY < -1.2f || ndcY > 1.2f) continue;
            const float sx = plotPos.x + (ndcX * 0.5f + 0.5f) * plotSize.x;
            const float sy = plotPos.y + (1.f - (ndcY * 0.5f + 0.5f)) * plotSize.y;
            const float t = (p.z - zMin) * invZ;
            const Vec3 c = HeightToColor(t);
            pix.push_back({sx, sy, ndcZ,
                           IM_COL32(static_cast<int>(c.x * 255), static_cast<int>(c.y * 255),
                                    static_cast<int>(c.z * 255), 230)});
        }
        std::sort(pix.begin(), pix.end(), [](const Pix& a, const Pix& b) { return a.d > b.d; });
        for (const Pix& q : pix) {
            dl->AddRectFilled(ImVec2(q.x - 1.2f, q.y - 1.2f), ImVec2(q.x + 1.2f, q.y + 1.2f), q.col);
        }

        // 拟合平面叠加（模块预览）
        if (n->params.showOverlay && n->plane) {
            DrawPlaneOverlay(dl, *n->plane, mvp, plotPos, plotSize);
        }
    }

    ImGui::End();
}

void AlgorithmEditor::DrawPlaneOverlay(ImDrawList* dl, const PlaneModel& plane, const Mat4& mvp,
                                       const ImVec2& plotPos, const ImVec2& plotSize) {
    Vec3 n = plane.normal.Normalized();
    Vec3 tmp = (std::fabs(n.x) < 0.9f) ? Vec3{1, 0, 0} : Vec3{0, 1, 0};
    Vec3 u = n.Cross(tmp).Normalized();
    Vec3 v = n.Cross(u).Normalized();
    const float su = std::max(plane.halfExtentU > 0.f ? plane.halfExtentU : plane.halfSize, 0.01f);
    const float sv = std::max(plane.halfExtentV > 0.f ? plane.halfExtentV : plane.halfSize, 0.01f);

    auto project = [&](const Vec3& p, ImVec2& out) -> bool {
        const Vec4 clip = mvp.MulVec4({p.x, p.y, p.z, 1.f});
        if (std::fabs(clip.w) < 1e-12f) return false;
        const float ndcX = clip.x / clip.w;
        const float ndcY = clip.y / clip.w;
        const float ndcZ = clip.z / clip.w;
        if (ndcZ < -1.f || ndcZ > 1.f) return false;
        out.x = plotPos.x + (ndcX * 0.5f + 0.5f) * plotSize.x;
        out.y = plotPos.y + (1.f - (ndcY * 0.5f + 0.5f)) * plotSize.y;
        return true;
    };

    const Vec3 corners[4] = {
        plane.centroid + u * (-su) + v * (-sv),
        plane.centroid + u * (su) + v * (-sv),
        plane.centroid + u * (su) + v * (sv),
        plane.centroid + u * (-su) + v * (sv),
    };
    ImVec2 sc[4];
    bool ok[4] = {};
    int okCount = 0;
    for (int i = 0; i < 4; ++i) {
        ok[i] = project(corners[i], sc[i]);
        if (ok[i]) ++okCount;
    }
    if (okCount < 3) return;

    const ImU32 fill = IM_COL32(255, 120, 40, 55);
    const ImU32 edge = IM_COL32(255, 170, 70, 230);
    // 填充四边形（若四点都可见）
    if (ok[0] && ok[1] && ok[2] && ok[3]) {
        dl->AddQuadFilled(sc[0], sc[1], sc[2], sc[3], fill);
        dl->AddQuad(sc[0], sc[1], sc[2], sc[3], edge, 2.f);
    } else {
        for (int i = 0; i < 4; ++i) {
            const int j = (i + 1) % 4;
            if (ok[i] && ok[j]) dl->AddLine(sc[i], sc[j], edge, 2.f);
        }
    }

    // 网格线
    constexpr int kDiv = 4;
    for (int i = 1; i < kDiv; ++i) {
        const float t = -1.f + 2.f * static_cast<float>(i) / static_cast<float>(kDiv);
        ImVec2 a, b;
        if (project(plane.centroid + u * (t * su) + v * (-sv), a) &&
            project(plane.centroid + u * (t * su) + v * (sv), b)) {
            dl->AddLine(a, b, IM_COL32(255, 160, 60, 140), 1.f);
        }
        if (project(plane.centroid + u * (-su) + v * (t * sv), a) &&
            project(plane.centroid + u * (su) + v * (t * sv), b)) {
            dl->AddLine(a, b, IM_COL32(255, 160, 60, 140), 1.f);
        }
    }

    // 法向箭头
    ImVec2 c0, c1;
    const Vec3 tip = plane.centroid + n * (std::max(su, sv) * 0.35f);
    if (project(plane.centroid, c0) && project(tip, c1)) {
        dl->AddLine(c0, c1, IM_COL32(80, 200, 255, 255), 2.5f);
        dl->AddCircleFilled(c1, 4.f, IM_COL32(80, 200, 255, 255));
    }
}

void AlgorithmEditor::DrawRoiEditWindow() {
    if (!roiEditOpen_) return;
    Node* n = FindNode(roiEditNodeId_);
    if (!n || n->type != ModuleType::RoiSelect || !n->hasCloud || n->cloud.points.empty()) {
        roiEditOpen_ = false;
        return;
    }

    ImGui::SetNextWindowSize(ImVec2(720.f, 560.f), ImGuiCond_FirstUseEver);
    if (roiNeedFocus_) {
        ImGui::SetNextWindowFocus();
        roiNeedFocus_ = false;
    }
    if (!ImGui::Begin(u8"ROI 框选编辑", &roiEditOpen_, ImGuiWindowFlags_NoCollapse)) {
        ImGui::End();
        return;
    }

    PointCloud& cloud = n->cloud;
    ImGui::Text(u8"可见 %zu / 总 %zu　已选 %zu", cloud.VisibleCount(), cloud.points.size(),
                roiIndices_.size());
    ImGui::SameLine(0.f, 12.f);
    if (ImGui::RadioButton(u8"框选模式", roiBoxMode_)) roiBoxMode_ = true;
    ImGui::SameLine();
    if (ImGui::RadioButton(u8"浏览旋转", !roiBoxMode_)) roiBoxMode_ = false;
    ImGui::SameLine(0.f, 12.f);
    if (ImGui::Button(u8"复位视角")) FitRoiCamera(cloud);

    if (ImGui::Button(u8"删除框内点")) {
        if (!roiIndices_.empty()) {
            MeasureTools::ApplyRoiDelete(cloud, roiIndices_, true);
            n->roiEdited = true;
            n->runOk = true;
            roiIndices_.clear();
            char buf[96];
            std::snprintf(buf, sizeof(buf), u8"已删框内，可见 %zu", cloud.VisibleCount());
            n->runMsg = buf;
        }
    }
    ImGui::SameLine();
    if (ImGui::Button(u8"删除框外点（只留框内）")) {
        if (!roiIndices_.empty()) {
            MeasureTools::ApplyRoiDelete(cloud, roiIndices_, false);
            n->roiEdited = true;
            n->runOk = true;
            roiIndices_.clear();
            char buf[96];
            std::snprintf(buf, sizeof(buf), u8"已删框外，可见 %zu", cloud.VisibleCount());
            n->runMsg = buf;
        }
    }
    ImGui::SameLine();
    if (ImGui::Button(u8"清除框选")) roiIndices_.clear();
    ImGui::SameLine();
    if (ImGui::Button(u8"恢复全部点")) {
        MeasureTools::RestoreAllPoints(cloud);
        n->roiEdited = true;
        roiIndices_.clear();
        n->runMsg = u8"已恢复全部点显示";
    }
    ImGui::SameLine();
    if (ImGui::Button(u8"完成")) {
        n->hasCloud = true;
        n->roiEdited = true;
        char buf[96];
        std::snprintf(buf, sizeof(buf), u8"ROI 完成，可见 %zu", cloud.VisibleCount());
        n->runMsg = buf;
        n->runOk = true;
        runStatus_ = buf;
        runStatusOk_ = true;
        roiEditOpen_ = false;
    }
    ImGui::TextDisabled(u8"框选：左键拖拽 | 浏览：左键旋转 / 中键平移 / 滚轮缩放");
    ImGui::Separator();

    const ImVec2 plotPos = ImGui::GetCursorScreenPos();
    const ImVec2 plotSize = ImGui::GetContentRegionAvail();
    roiPlotX_ = plotPos.x;
    roiPlotY_ = plotPos.y;
    roiPlotW_ = plotSize.x;
    roiPlotH_ = plotSize.y;

    ImGui::InvisibleButton(u8"##roi3d", plotSize,
                           ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonMiddle);
    const bool hovered = ImGui::IsItemHovered();
    const ImGuiIO& io = ImGui::GetIO();

    const float localX = io.MousePos.x - plotPos.x;
    const float localY = io.MousePos.y - plotPos.y;

    if (hovered) {
        if (roiBoxMode_) {
            if (ImGui::IsMouseClicked(ImGuiMouseButton_Left) && !io.KeyShift) {
                roiDragging_ = true;
                roiX0_ = roiX1_ = localX;
                roiY0_ = roiY1_ = localY;
            }
            if (roiDragging_ && io.MouseDown[0]) {
                roiX1_ = localX;
                roiY1_ = localY;
            }
            if (roiDragging_ && !io.MouseDown[0]) {
                roiDragging_ = false;
                const int fbW = std::max(1, static_cast<int>(plotSize.x));
                const int fbH = std::max(1, static_cast<int>(plotSize.y));
                MeasureTools::SelectRoi(cloud, roiCam_, fbW, fbH, roiX0_, roiY0_, roiX1_, roiY1_,
                                        roiIndices_);
            }
            if (ImGui::IsMouseDown(ImGuiMouseButton_Middle) ||
                (io.KeyShift && ImGui::IsMouseDragging(ImGuiMouseButton_Left, 0.f))) {
                roiCam_.Pan(io.MouseDelta.x, io.MouseDelta.y, io.MouseDown[2] ? 1.f : 0.35f);
            }
        } else {
            if (ImGui::IsMouseClicked(ImGuiMouseButton_Left) && !io.KeyShift) roiOrbiting_ = true;
            if (ImGui::IsMouseClicked(ImGuiMouseButton_Middle) ||
                (ImGui::IsMouseClicked(ImGuiMouseButton_Left) && io.KeyShift))
                roiPanning_ = true;
            if (roiOrbiting_ && io.MouseDown[0] && !io.KeyShift)
                roiCam_.Orbit(io.MouseDelta.x * 0.01f, io.MouseDelta.y * 0.01f);
            if (roiPanning_)
                roiCam_.Pan(io.MouseDelta.x, io.MouseDelta.y, io.MouseDown[2] ? 1.f : 0.35f);
        }
        if (io.MouseWheel != 0.f) roiCam_.Zoom(io.MouseWheel);
    }
    if (!io.MouseDown[0]) {
        roiOrbiting_ = false;
        if (!roiBoxMode_) roiDragging_ = false;
    }
    if (!io.MouseDown[2] && !(io.MouseDown[0] && io.KeyShift)) roiPanning_ = false;

    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(plotPos, ImVec2(plotPos.x + plotSize.x, plotPos.y + plotSize.y),
                      IM_COL32(12, 14, 18, 255));

    std::vector<uint8_t> selMask;
    if (!roiIndices_.empty()) {
        selMask.assign(cloud.points.size(), 0);
        for (std::size_t idx : roiIndices_) {
            if (idx < selMask.size()) selMask[idx] = 1;
        }
    }

    if (plotSize.x > 10.f && plotSize.y > 10.f && cloud.bounds.Valid()) {
        const float aspect = plotSize.x / plotSize.y;
        const Mat4 mvp = roiCam_.ProjMatrix(aspect) * roiCam_.ViewMatrix();
        const float zMin = cloud.bounds.min.z;
        float zMax = cloud.bounds.max.z;
        if (zMax <= zMin) zMax = zMin + 1.f;
        const float invZ = 1.f / (zMax - zMin);
        const int stride = std::max(1, static_cast<int>(cloud.points.size() / 100000));

        struct Pix {
            float x, y, d;
            ImU32 col;
        };
        std::vector<Pix> pix;
        pix.reserve(cloud.points.size() / static_cast<std::size_t>(stride) + 8);

        for (std::size_t i = 0; i < cloud.points.size(); i += static_cast<std::size_t>(stride)) {
            if (!cloud.mask.empty() && !cloud.mask[i]) continue;
            const Vec3& p = cloud.points[i];
            const Vec4 clip = mvp.MulVec4({p.x, p.y, p.z, 1.f});
            if (std::fabs(clip.w) < 1e-12f) continue;
            const float ndcX = clip.x / clip.w;
            const float ndcY = clip.y / clip.w;
            const float ndcZ = clip.z / clip.w;
            if (ndcZ < -1.f || ndcZ > 1.f) continue;
            if (ndcX < -1.2f || ndcX > 1.2f || ndcY < -1.2f || ndcY > 1.2f) continue;
            const float sx = plotPos.x + (ndcX * 0.5f + 0.5f) * plotSize.x;
            const float sy = plotPos.y + (1.f - (ndcY * 0.5f + 0.5f)) * plotSize.y;
            ImU32 col;
            if (!selMask.empty() && selMask[i]) {
                col = IM_COL32(40, 255, 120, 255);
            } else {
                const float t = (p.z - zMin) * invZ;
                const Vec3 c = HeightToColor(t);
                col = IM_COL32(static_cast<int>(c.x * 255), static_cast<int>(c.y * 255),
                               static_cast<int>(c.z * 255), 220);
            }
            pix.push_back({sx, sy, ndcZ, col});
        }
        std::sort(pix.begin(), pix.end(), [](const Pix& a, const Pix& b) { return a.d > b.d; });
        for (const Pix& q : pix) {
            dl->AddRectFilled(ImVec2(q.x - 1.2f, q.y - 1.2f), ImVec2(q.x + 1.2f, q.y + 1.2f), q.col);
        }
    }

    if (roiBoxMode_ && (roiDragging_ || (roiX0_ != roiX1_ || roiY0_ != roiY1_))) {
        const float x0 = plotPos.x + std::min(roiX0_, roiX1_);
        const float y0 = plotPos.y + std::min(roiY0_, roiY1_);
        const float x1 = plotPos.x + std::max(roiX0_, roiX1_);
        const float y1 = plotPos.y + std::max(roiY0_, roiY1_);
        if (roiDragging_) {
            dl->AddRectFilled(ImVec2(x0, y0), ImVec2(x1, y1), IM_COL32(64, 200, 180, 40));
            dl->AddRect(ImVec2(x0, y0), ImVec2(x1, y1), IM_COL32(64, 220, 190, 230), 0.f, 0, 2.f);
        }
    }

    ImGui::End();
}

void AlgorithmEditor::DrawProperties() {
    ImGui::TextDisabled(u8"属性");
    ImGui::Spacing();

    Node* n = FindNode(selectedId_);
    if (!n) {
        ImGui::TextWrapped(u8"未选中节点。\n配置参数后点「运行」；双击节点打开点云预览。");
        return;
    }

    ImGui::TextColored(ImVec4(0.45f, 0.85f, 0.90f, 1.f), "%s", n->title.c_str());
    ImGui::TextDisabled(u8"ID: %d", n->id);
    if (FindUpstream(n->id)) {
        ImGui::TextColored(ImVec4(0.5f, 0.85f, 0.7f, 1.f), u8"已连接上游输入");
        if (ImGui::Button(u8"断开输入连线", ImVec2(-1.f, 0))) DisconnectIncoming(n->id);
    } else if (NodeHasInputPort(n->type)) {
        ImGui::TextDisabled(u8"未连接输入（从上游底端口拖到本节点顶端口）");
    }
    if (n->hasCloud) {
        ImGui::TextColored(ImVec4(0.5f, 0.9f, 0.65f, 1.f), u8"已缓存 %zu 点", n->cloud.points.size());
        if (n->type == ModuleType::RoiSelect) {
            if (ImGui::Button(u8"打开 ROI 框选编辑", ImVec2(-1.f, 0))) OpenRoiEditor(n->id);
        } else if (ImGui::Button(u8"打开点云预览", ImVec2(-1.f, 0))) {
            OpenCloudPreview(n->id);
        }
    }
    if (n->type == ModuleType::RoiSelect && !n->hasCloud) {
        if (ImGui::Button(u8"从上游加载并框选", ImVec2(-1.f, 28.f))) OpenRoiEditor(n->id);
    }
    if (ImGui::Button(u8"▶ 单步运行此模块", ImVec2(-1.f, 28.f))) RunNode(n->id);
    if (ImGui::Button(u8"↺ 重置此模块", ImVec2(-1.f, 0))) ResetNode(n->id);
    if (!n->runMsg.empty()) {
        ImGui::TextWrapped("%s", n->runMsg.c_str());
    }
    ImGui::Separator();
    ImGui::Spacing();

    NodeParams& p = n->params;
    ImGui::PushItemWidth(-1.f);

    switch (n->type) {
        case ModuleType::InputCloud: {
            ImGui::TextUnformatted(u8"数据源");
            const char* sources[] = {u8"当前点云", u8"从文件加载", u8"生成球面", u8"生成圆柱"};
            ImGui::Combo(u8"##src", &p.inputSource, sources, 4);
            if (p.inputSource == 1) {
                ImGui::TextUnformatted(u8"文件路径");
                ImGui::InputText(u8"##path", p.filePath, sizeof(p.filePath));
                ImGui::TextDisabled(u8"支持 PLY / PCD / XYZ / OBJ");
            }
            if (p.inputSource == 2 || p.inputSource == 3) {
                ImGui::TextUnformatted(u8"半径");
                ImGui::DragFloat(u8"##genR", &p.genRadius, 0.1f, 0.01f, 1e6f, "%.3f");
                if (p.inputSource == 3) {
                    ImGui::TextUnformatted(u8"高度");
                    ImGui::DragFloat(u8"##genH", &p.genHeight, 0.1f, 0.01f, 1e6f, "%.3f");
                }
                ImGui::TextUnformatted(u8"点数");
                ImGui::DragInt(u8"##genN", &p.genPoints, 100.f, 16, 2000000);
                ImGui::TextUnformatted(u8"径向噪声");
                ImGui::DragFloat(u8"##genNoise", &p.genNoise, 0.001f, 0.f, 100.f, "%.4f");
            }
            if (ImGui::Button(u8"加载并预览", ImVec2(-1.f, 28.f))) {
                OpenCloudPreview(n->id);
            }
            break;
        }
        case ModuleType::FilterVoxel:
            ImGui::TextUnformatted(u8"体素边长 leaf");
            ImGui::DragFloat(u8"##leaf", &p.voxelLeaf, 0.01f, 1e-4f, 1e4f, "%.4f");
            break;
        case ModuleType::FilterRadius:
            ImGui::TextUnformatted(u8"搜索半径");
            ImGui::DragFloat(u8"##rr", &p.outlierRadius, 0.01f, 1e-4f, 1e4f, "%.4f");
            ImGui::TextUnformatted(u8"最少邻域点数");
            ImGui::DragInt(u8"##rmin", &p.outlierMinNeighbors, 1.f, 1, 200);
            break;
        case ModuleType::FilterStatistical:
            ImGui::TextUnformatted(u8"邻域点数 meanK");
            ImGui::DragInt(u8"##k", &p.statMeanK, 1.f, 2, 200);
            ImGui::TextUnformatted(u8"标准差倍数");
            ImGui::DragFloat(u8"##std", &p.statStdMul, 0.05f, 0.1f, 10.f, "%.2f");
            break;
        case ModuleType::RoiSelect:
            ImGui::TextWrapped(
                u8"双击本模块进入框选编辑：\n"
                u8"· 框选模式：左键拖拽框选可见表面\n"
                u8"· 可删除框内/框外点\n"
                u8"· 浏览模式：左键旋转查看");
            if (n->roiEdited) ImGui::TextColored(ImVec4(0.5f, 0.9f, 0.6f, 1.f), u8"已交互编辑");
            break;
        case ModuleType::FitPlane:
            ImGui::Checkbox(u8"仅使用 ROI / 框选点", &p.useRoiOnly);
            ImGui::Checkbox(u8"法向朝上 (N.z≥0)", &p.flipNormalUp);
            ImGui::TextUnformatted(u8"最少点数");
            ImGui::DragInt(u8"##minp", &p.minPoints, 1.f, 3, 1000000);
            ImGui::TextUnformatted(u8"最大允许 RMS");
            ImGui::DragFloat(u8"##rms", &p.maxRms, 0.001f, 0.f, 1e3f, "%.4f");
            ImGui::Checkbox(u8"显示拟合面叠加", &p.showOverlay);
            if (n->plane) {
                ImGui::Separator();
                ImGui::TextDisabled(u8"拟合结果");
                const PlaneModel& pl = *n->plane;
                const float d = -pl.normal.Dot(pl.centroid);
                ImGui::TextWrapped(u8"方程\n%.6fx %+.6fy %+.6fz %+.6f = 0", pl.normal.x, pl.normal.y,
                                   pl.normal.z, d);
                ImGui::Text(u8"法向  (%.4f, %.4f, %.4f)", pl.normal.x, pl.normal.y, pl.normal.z);
                ImGui::Text(u8"中心  (%.4f, %.4f, %.4f)", pl.centroid.x, pl.centroid.y,
                            pl.centroid.z);
                ImGui::Text(u8"RMS = %.6f   点数 %d", pl.rms, pl.pointCount);
                ImGui::Text(u8"半宽 U=%.3f  V=%.3f", pl.halfExtentU, pl.halfExtentV);
                if (ImGui::Button(u8"预览拟合平面", ImVec2(-1.f, 0))) OpenCloudPreview(n->id);
            }
            break;
        case ModuleType::FitSphere:
        case ModuleType::FitCircle:
        case ModuleType::FitCylinder:
            ImGui::Checkbox(u8"仅使用 ROI / 框选点", &p.useRoiOnly);
            ImGui::TextUnformatted(u8"最少点数");
            ImGui::DragInt(u8"##minp", &p.minPoints, 1.f, 3, 1000000);
            ImGui::TextUnformatted(u8"最大允许 RMS");
            ImGui::DragFloat(u8"##rms", &p.maxRms, 0.001f, 0.f, 1e3f, "%.4f");
            if (n->type == ModuleType::FitCylinder) {
                ImGui::TextUnformatted(u8"轴方向初值");
                ImGui::DragFloat3(u8"##axis", &p.seedAxisX, 0.01f, -1.f, 1.f, "%.3f");
            }
            ImGui::Checkbox(u8"显示线框叠加", &p.showOverlay);
            break;
        case ModuleType::Flatness:
            ImGui::Checkbox(u8"仅使用 ROI / 框选点", &p.useRoiOnly);
            ImGui::TextUnformatted(u8"最少点数");
            ImGui::DragInt(u8"##minp", &p.minPoints, 1.f, 3, 1000000);
            ImGui::TextUnformatted(u8"最大允许 RMS");
            ImGui::DragFloat(u8"##rms", &p.maxRms, 0.001f, 0.f, 1e3f, "%.4f");
            ImGui::Checkbox(u8"显示偏差着色", &p.showOverlay);
            if (n->flatness && n->flatness->valid) {
                ImGui::Separator();
                ImGui::TextDisabled(u8"平面度结果");
                const FlatnessResult& fr = *n->flatness;
                ImGui::Text(u8"PV (峰谷) = %.6f", fr.peakToValley);
                ImGui::Text(u8"RMS       = %.6f", fr.rms);
                ImGui::Text(u8"平均|偏差| = %.6f", fr.meanAbs);
                ImGui::Text(u8"偏差范围  [%.6f, %.6f]", fr.minDev, fr.maxDev);
                if (n->plane) {
                    const float d = -n->plane->normal.Dot(n->plane->centroid);
                    ImGui::TextWrapped(u8"基准面 %.4fx%+.4fy%+.4fz%+.4f=0", n->plane->normal.x,
                                       n->plane->normal.y, n->plane->normal.z, d);
                }
            }
            break;
        case ModuleType::StepGap:
            ImGui::Checkbox(u8"使用 Z 高度差 ΔZ", &p.useZHeight);
            ImGui::Checkbox(u8"仅可见表面框选", &p.useRoiOnly);
            ImGui::TextUnformatted(u8"区域最少点数");
            ImGui::DragInt(u8"##minp", &p.minPoints, 1.f, 3, 1000000);
            ImGui::Checkbox(u8"显示着色", &p.showOverlay);
            break;
        case ModuleType::Section: {
            const char* axes[] = {u8"沿 X（剖面 YZ）", u8"沿 Y（剖面 XZ）"};
            int axis = p.cutAlongX ? 0 : 1;
            if (ImGui::Combo(u8"切割方向", &axis, axes, 2)) p.cutAlongX = (axis == 0);
            ImGui::TextUnformatted(u8"切割位置");
            ImGui::DragFloat(u8"##pos", &p.sectionPos, 0.01f, -1e6f, 1e6f, "%.4f");
            ImGui::TextUnformatted(u8"截面厚度");
            ImGui::DragFloat(u8"##th", &p.sectionThickness, 0.001f, 1e-5f, 1e3f, "%.5f");
            ImGui::TextUnformatted(u8"最大采样点数");
            ImGui::DragInt(u8"##max", &p.sectionMaxPoints, 1000.f, 1000, 5000000);
            ImGui::Checkbox(u8"显示切割面", &p.showOverlay);
            break;
        }
        case ModuleType::OutputResult: {
            const char* fmts[] = {u8"状态栏", u8"CSV 文件", u8"JSON 文件"};
            ImGui::Combo(u8"输出方式", &p.outputFormat, fmts, 3);
            if (p.outputFormat == 1 || p.outputFormat == 2) {
                ImGui::TextUnformatted(u8"输出路径");
                ImGui::InputText(u8"##out", p.outputPath, sizeof(p.outputPath));
            }
            ImGui::Checkbox(u8"判定 OK/NG", &p.outputOkNg);
            if (p.outputOkNg) {
                ImGui::TextUnformatted(u8"公差下限");
                ImGui::DragFloat(u8"##lo", &p.tolLower, 0.001f, -1e6f, 1e6f, "%.4f");
                ImGui::TextUnformatted(u8"公差上限");
                ImGui::DragFloat(u8"##hi", &p.tolUpper, 0.001f, -1e6f, 1e6f, "%.4f");
                ImGui::TextDisabled(u8"平面度以 PV 判定；其它量可后续扩展");
            }
            ImGui::Separator();
            DrawOutputResultPanel(*n);
            break;
        }
        default:
            break;
    }

    ImGui::PopItemWidth();
    ImGui::Spacing();
    ImGui::Separator();
    if (ImGui::Button(u8"恢复默认参数", ImVec2(-1.f, 0))) p = DefaultParams(n->type);
    if (ImGui::Button(u8"删除节点", ImVec2(-1.f, 0))) {
        const int delId = selectedId_;
        if (previewNodeId_ == delId) previewOpen_ = false;
        nodes_.erase(std::remove_if(nodes_.begin(), nodes_.end(),
                                    [&](const Node& x) { return x.id == delId; }),
                     nodes_.end());
        links_.erase(std::remove_if(links_.begin(), links_.end(),
                                    [&](const Link& L) {
                                        return L.fromNode == delId || L.toNode == delId;
                                    }),
                     links_.end());
        selectedId_ = -1;
        draggingNode_ = false;
        dragNodeId_ = -1;
    }
}

void AlgorithmEditor::DrawOutputResultPanel(Node& n) {
    ImGui::TextColored(ImVec4(0.45f, 0.85f, 0.90f, 1.f), u8"结果栏");
    if (!n.runOk && !n.hasCloud) {
        ImGui::TextWrapped(u8"运行本模块后，这里会显示上游拟合/测量的输出数据。");
        return;
    }

    if (n.hasCloud) {
        ImGui::Text(u8"点云  总数 %zu　可见 %zu", n.cloud.points.size(), n.cloud.VisibleCount());
    }

    if (n.flatness && n.flatness->valid) {
        const FlatnessResult& fr = *n.flatness;
        ImGui::Spacing();
        ImGui::TextDisabled(u8"平面度");
        ImGui::Text(u8"PV (峰谷)   = %.6f", fr.peakToValley);
        ImGui::Text(u8"RMS         = %.6f", fr.rms);
        ImGui::Text(u8"平均|偏差|  = %.6f", fr.meanAbs);
        ImGui::Text(u8"最小偏差    = %.6f", fr.minDev);
        ImGui::Text(u8"最大偏差    = %.6f", fr.maxDev);
        if (n.params.outputOkNg) {
            const bool ok =
                fr.peakToValley >= n.params.tolLower && fr.peakToValley <= n.params.tolUpper;
            ImGui::TextColored(ok ? ImVec4(0.4f, 0.9f, 0.5f, 1.f) : ImVec4(0.95f, 0.4f, 0.35f, 1.f),
                               ok ? u8"判定：OK" : u8"判定：NG");
        }
    }

    if (n.plane) {
        const PlaneModel& pl = *n.plane;
        const float d = -pl.normal.Dot(pl.centroid);
        ImGui::Spacing();
        ImGui::TextDisabled(u8"拟合平面");
        ImGui::TextWrapped(u8"%.6fx\n%+.6fy\n%+.6fz\n%+.6f = 0", pl.normal.x, pl.normal.y,
                           pl.normal.z, d);
        ImGui::Text(u8"法向 N = (%.6f, %.6f, %.6f)", pl.normal.x, pl.normal.y, pl.normal.z);
        ImGui::Text(u8"中心 C = (%.6f, %.6f, %.6f)", pl.centroid.x, pl.centroid.y, pl.centroid.z);
        ImGui::Text(u8"RMS = %.6f", pl.rms);
        ImGui::Text(u8"点数 = %d", pl.pointCount);
    }

    if (n.sphere) {
        ImGui::Spacing();
        ImGui::TextDisabled(u8"拟合球");
        ImGui::Text(u8"中心 (%.4f, %.4f, %.4f)", n.sphere->center.x, n.sphere->center.y,
                    n.sphere->center.z);
        ImGui::Text(u8"半径 %.6f   RMS %.6f", n.sphere->radius, n.sphere->rms);
    }
    if (n.circle) {
        ImGui::Spacing();
        ImGui::TextDisabled(u8"拟合圆");
        ImGui::Text(u8"中心 (%.4f, %.4f, %.4f)", n.circle->center.x, n.circle->center.y,
                    n.circle->center.z);
        ImGui::Text(u8"半径 %.6f   RMS %.6f", n.circle->radius, n.circle->rms);
    }
    if (n.cylinder) {
        ImGui::Spacing();
        ImGui::TextDisabled(u8"拟合圆柱");
        ImGui::Text(u8"半径 %.6f   RMS %.6f", n.cylinder->radius, n.cylinder->rms);
    }

    if (!n.plane && !n.flatness && !n.sphere && !n.circle && !n.cylinder) {
        ImGui::TextDisabled(u8"上游暂无拟合/测量数值结果（可先接平面拟合或平面度）");
    }
    ImGui::TextDisabled(u8"运行后点云会同步到主视图");
}
