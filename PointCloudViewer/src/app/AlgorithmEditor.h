#pragma once

#include "core/PointCloud.h"
#include "render/Camera.h"
#include "tools/MeasureTools.h"

#include <functional>
#include <optional>
#include <string>
#include <vector>

struct ImDrawList;
struct ImVec2;

struct AlgoHost {
    const PointCloud* currentCloud = nullptr;
    std::function<void(PointCloud&& cloud, const char* status)> publishCloud;
};

class AlgorithmEditor {
public:
    void SetHost(const AlgoHost& host) { host_ = host; }
    void Draw(float menuBottomY);
    void SetVisible(bool v);
    bool IsVisible() const { return visible_; }
    void ToggleVisible();

private:
    enum class ModuleType : int {
        InputCloud = 0,
        FilterVoxel,
        FilterRadius,
        FilterStatistical,
        RoiSelect,
        FitPlane,
        FitSphere,
        FitCircle,
        FitCylinder,
        Flatness,
        StepGap,
        Section,
        OutputResult,
        Count
    };

    struct ModuleDef {
        ModuleType type;
        const char* name;
        const char* category;
        unsigned int color;
    };

    struct NodeParams {
        int inputSource = 0;
        char filePath[260] = {};
        float genRadius = 10.f;
        float genHeight = 30.f;
        int genPoints = 20000;
        float genNoise = 0.05f;

        float voxelLeaf = 0.5f;
        float outlierRadius = 1.0f;
        int outlierMinNeighbors = 4;
        int statMeanK = 20;
        float statStdMul = 1.0f;

        bool useRoiOnly = true;
        int minPoints = 50;
        float maxRms = 0.5f;
        bool showOverlay = true;
        bool flipNormalUp = true;

        float seedAxisX = 0.f;
        float seedAxisY = 0.f;
        float seedAxisZ = 1.f;

        bool useZHeight = true;

        bool cutAlongX = true;
        float sectionPos = 0.f;
        float sectionThickness = 0.05f;
        int sectionMaxPoints = 200000;

        int outputFormat = 0;
        char outputPath[260] = {};
        bool outputOkNg = true;
        float tolLower = -0.1f;
        float tolUpper = 0.1f;
    };

    struct Node {
        int id = 0;
        ModuleType type = ModuleType::InputCloud;
        float x = 0.f;
        float y = 0.f;
        float w = 200.f;
        float h = 108.f;
        std::string title;
        NodeParams params;
        PointCloud cloud;
        bool hasCloud = false;
        bool roiEdited = false;  // ROI 模块交互编辑过
        std::string runMsg;
        bool runOk = false;
        // 拟合 / 测量结果（供预览叠加与输出节点展示）
        std::optional<PlaneModel> plane;
        std::optional<FlatnessResult> flatness;
        std::optional<SphereModel> sphere;
        std::optional<CircleModel> circle;
        std::optional<CylinderModel> cylinder;
    };

    struct Link {
        int fromNode = -1;
        int toNode = -1;
    };

    static const ModuleDef* Catalog();
    static int CatalogSize();
    static const char* TypeName(ModuleType t);
    static unsigned int TypeColor(ModuleType t);
    static NodeParams DefaultParams(ModuleType t);
    static const char* ParamSummary(const Node& n, char* buf, int bufSize);
    static bool NodeHasInputPort(ModuleType t);
    static bool NodeHasOutputPort(ModuleType t);

    void DrawToolbar();
    void DrawPalette();
    void DrawCanvas();
    void DrawProperties();
    void DrawNodeVisual(const Node& node, bool selected);
    void DrawGhostModule();
    void DrawCloudPreviewWindow();
    void DrawRoiEditWindow();
    void ClearGraph();
    void AddNodeAtCanvas(ModuleType type, float canvasX, float canvasY);
    void RunGraph();
    void RunNode(int nodeId);
    void ResetNode(int nodeId);
    bool ExecuteNode(Node& node, PointCloud& working, std::string& error);
    bool ResolveInputCloud(const NodeParams& p, PointCloud& out, std::string& error);
    bool ApplyFilterMask(PointCloud& cloud, const std::vector<uint8_t>& keep);
    void OpenCloudPreview(int nodeId);
    void OpenRoiEditor(int nodeId);
    bool EnsureNodeCloudFromUpstream(Node& n, std::string& error);
    void FitPreviewCamera(const PointCloud& cloud);
    void FitRoiCamera(const PointCloud& cloud);
    void ConnectNodes(int fromId, int toId);
    void DisconnectIncoming(int toId);
    void ClearNodeResults(Node& n);
    void DrawPlaneOverlay(ImDrawList* dl, const PlaneModel& plane, const Mat4& mvp,
                          const ImVec2& plotPos, const ImVec2& plotSize);
    void DrawOutputResultPanel(Node& n);
    Node* FindUpstream(int nodeId);
    bool BuildExecOrder(std::vector<int>& outOrder, std::string& error) const;
    void GetPortScreenPos(const Node& n, bool output, float& sx, float& sy) const;
    void CanvasToScreen(float cx, float cy, float& sx, float& sy) const;
    void ScreenToCanvas(float sx, float sy, float& cx, float& cy) const;
    void GetNodeActionBtns(const Node& n, float& resetX0, float& resetY0, float& resetX1,
                           float& resetY1, float& runX0, float& runY0, float& runX1,
                           float& runY1) const;
    int HitTestOutPort(float mx, float my) const;
    int HitTestInPort(float mx, float my) const;
    int HitTestNodeRunBtn(float mx, float my) const;
    int HitTestNodeResetBtn(float mx, float my) const;
    Node* FindNode(int id);
    const Node* FindNode(int id) const;
    int HitTestNode(float mx, float my) const;
    int AllocId();
    bool MouseInCanvas(float mx, float my) const;

    AlgoHost host_;
    bool visible_ = false;
    bool focusOnOpen_ = false;
    int nextId_ = 1;
    std::vector<Node> nodes_;
    std::vector<Link> links_;
    int selectedId_ = -1;

    float canvasPanX_ = 40.f;
    float canvasPanY_ = 40.f;
    float canvasZoom_ = 1.f;
    float canvasScreenX_ = 0.f;
    float canvasScreenY_ = 0.f;
    float canvasScreenW_ = 0.f;
    float canvasScreenH_ = 0.f;

    bool draggingNew_ = false;
    ModuleType dragType_ = ModuleType::InputCloud;
    bool paletteArmed_ = false;
    ModuleType armedType_ = ModuleType::InputCloud;

    bool draggingNode_ = false;
    int dragNodeId_ = -1;
    float dragGrabDX_ = 0.f;
    float dragGrabDY_ = 0.f;

    // 端口连线拖拽
    bool linking_ = false;
    int linkFromId_ = -1;

    std::string runStatus_;
    bool runStatusOk_ = true;

    bool previewOpen_ = false;
    int previewNodeId_ = -1;
    bool previewNeedFocus_ = false;
    Camera previewCam_;
    bool previewOrbiting_ = false;
    bool previewPanning_ = false;

    // ROI 框选编辑窗
    bool roiEditOpen_ = false;
    int roiEditNodeId_ = -1;
    bool roiNeedFocus_ = false;
    Camera roiCam_;
    bool roiOrbiting_ = false;
    bool roiPanning_ = false;
    bool roiBoxMode_ = true;  // true=框选 false=旋转浏览
    bool roiDragging_ = false;
    float roiX0_ = 0, roiY0_ = 0, roiX1_ = 0, roiY1_ = 0;
    std::vector<std::size_t> roiIndices_;
    float roiPlotX_ = 0, roiPlotY_ = 0, roiPlotW_ = 1, roiPlotH_ = 1;

    float clickDownX_ = 0.f;
    float clickDownY_ = 0.f;
    bool clickOnNode_ = false;
    int clickNodeId_ = -1;
};
