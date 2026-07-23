#pragma once

// Application — 主程序：窗口、UI、输入、点云生命周期与工具调度
//
// 职责概览：
// - 持有主点云 cloud_ 与填充点云 filledCloud_，协调渲染与 GPU 上传
// - 根据 ToolMode 分发点击/框选/测量逻辑
// - 在 MeasureTools（自研）与 PclTools（PCL）之间按设置选择后端
// - 管理撤销栈、滤波预览、双视图（原始+填充叠加）等应用状态

#include "core/PointCloud.h"
#include "render/Camera.h"
#include "render/PointCloudRenderer.h"
#include "tools/MeasureTools.h"
#include "tools/PclTools.h"
#include "tools/UndoHistory.h"
#include "tools/OpenCv2D.h"
#include "app/AlgorithmEditor.h"
#include "app/ShapeTemplateMatchWindow.h"
#include "app/HalconMatchWindow.h"
#include "app/AlgoToolsPanel.h"
#include "app/DbTreePanel.h"
#include "app/PclPanel.h"
#include "app/PclToolsPanel.h"
#include "tools/AlgorithmBackend.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

struct GLFWwindow;
struct ImDrawList;

enum class Image2DTool {
    None = 0,
    CaliperLine,
    CaliperArc,
    LineDistance,
    ArcDistance,
    CircleFit,
    PointDistance,
    LineAngle,
    CircleGap,
    PointLineDistance,
    CaliperPoint,
    CaliperCircle,
    ArcLength,
    ThreePointCircle,
    ParallelLineDistance,
    RectCaliper,
    EllipseFit,
    ProfileWidth,
    PointProjection,
    Concentricity,
    Roundness,
    RegionBlob,
    DepthHeightDiff,
    DepthProfile,
    TemplateMatch,
};

enum class ThreePointPhase : int {
    Pick0 = 0,
    Pick1 = 1,
    Pick2 = 2,
};

enum class PointPickPhase : int {
    PickA = 0,
    PickB = 1,
};

enum class CircleCaliperPhase : int {
    PickCenter = 0,
    DragRadius = 1,
};

enum class TemplateMatchPhase : int {
    DrawTemplate = 0,
    DrawSearch = 1,
};

enum class DepthProfileMode : int {
    ScanRow = 0,     // 扫描行 X-Z（线扫常用）
    ScanColumn = 1,  // 扫描列 Y-Z
    FreeLine = 2,    // 任意线段
};

inline const char* Image2DToolLabel(Image2DTool tool) {
    switch (tool) {
        case Image2DTool::CaliperLine:
            return u8"提取线段（卡尺）";
        case Image2DTool::CaliperArc:
            return u8"提取圆弧（卡尺）";
        case Image2DTool::LineDistance:
            return u8"线线距离（间隙）";
        case Image2DTool::ArcDistance:
            return u8"圆弧线线距离（间隙）";
        case Image2DTool::CircleFit:
            return u8"拟合圆";
        case Image2DTool::PointDistance:
            return u8"点点距离";
        case Image2DTool::LineAngle:
            return u8"两线夹角";
        case Image2DTool::CircleGap:
            return u8"圆心距 / 圆间隙";
        case Image2DTool::PointLineDistance:
            return u8"点线距离";
        case Image2DTool::CaliperPoint:
            return u8"单点卡尺";
        case Image2DTool::CaliperCircle:
            return u8"圆卡尺（整圆）";
        case Image2DTool::ArcLength:
            return u8"弧长 / 弦长";
        case Image2DTool::ThreePointCircle:
            return u8"三点定圆";
        case Image2DTool::ParallelLineDistance:
            return u8"平行线距离";
        case Image2DTool::RectCaliper:
            return u8"矩形卡尺";
        case Image2DTool::EllipseFit:
            return u8"拟合椭圆";
        case Image2DTool::ProfileWidth:
            return u8"剖面测宽";
        case Image2DTool::PointProjection:
            return u8"投影点 / 垂足";
        case Image2DTool::Concentricity:
            return u8"同心度";
        case Image2DTool::Roundness:
            return u8"圆度";
        case Image2DTool::RegionBlob:
            return u8"区域面积 / 质心";
        case Image2DTool::DepthHeightDiff:
            return u8"两点高度差";
        case Image2DTool::DepthProfile:
            return u8"剖面高度曲线";
        case Image2DTool::TemplateMatch:
            return u8"模板匹配";
        default:
            return u8"无";
    }
}

enum class ArcMeasurePhase : int {
    PickA = 0,
    PickB = 1,
    DragBulge = 2,
};

inline const char* ArcMeasurePhaseHint(ArcMeasurePhase phase) {
    switch (phase) {
        case ArcMeasurePhase::PickA:
            return u8"① 点击设置 A 点";
        case ArcMeasurePhase::PickB:
            return u8"② 点击设置 B 点";
        case ArcMeasurePhase::DragBulge:
            return u8"③ 拖拽拱高线段调节弧线，松开预览";
        default:
            return u8"";
    }
}

class Application {
public:
    bool Init();      // 初始化 GLFW、OpenGL、ImGui 与渲染器
    void Run();       // 主循环：处理事件、更新、绘制
    void Shutdown();  // 释放 GPU 与窗口资源

private:
    struct ImageView {
        unsigned int texId = 0;
        int width = 0;
        int height = 0;
        std::string path;
        std::vector<float> gray;       // depth raw values (optional)
        std::vector<uint8_t> rgb;      // display RGB8
        float valueMin = 0.f;
        float valueMax = 1.f;
        bool valid() const { return texId != 0 && width > 0 && height > 0; }
    };

    struct SceneCloudLayer {
        int id = 0;
        PointCloud cloud;
        bool visible = true;
        PointCloudRenderer renderer;
        int gpuPointCount = 0;
        bool needUpload = true;
        std::vector<std::size_t> displayIndices;
        bool rendererReady = false;
    };

    // --- UI 绘制 ---
    void DrawUi();                    // 整帧 ImGui 布局入口
    float DrawMenuBar();              // 顶部菜单栏，返回底边 Y
    void DrawToolbar(float y, float height);
    void DrawToolPanel();             // 左侧工具面板（ROI、拟合等）
    void DrawDbTreePanel(float height);
    void DrawDbTreeSplitter(float maxDbHeight);
    void DrawEmptyCloudPrompt(float contentTop, float contentH);
    DbTreeHost BuildDbTreeHost();
    void DrawPlaneDistanceHistogram(const PlaneDistanceProfile& profile);
    void UpdatePlaneFitProfile(const PlaneModel& plane, const std::vector<std::size_t>& indices);
    void FitCameraToAllClouds();
    void SetActiveLayerById(int layerId);
    void RemoveSceneLayerById(int layerId);
    void InitLayerRenderer(SceneCloudLayer& layer);
    void ShutdownSceneLayers();
    PointCloud& MainCloud();
    const PointCloud& MainCloud() const;
    SceneCloudLayer& ActiveLayer();
    const SceneCloudLayer& ActiveLayer() const;
    PointCloudRenderer& ActiveRenderer();
    bool HasSceneClouds() const;
    void DrawViewAxisWidget(float contentTop, float contentBottom, float leftInset);
    void DrawAboutPopup();
    void DrawNativeAlgoPasswordPopup();
    void DrawSectionPanel();          // 截面 2D 图与测距面板
    void DrawStepGapPanel();          // 段差结果面板
    void DrawDepthProfilePanel();     // 深度 Profile 曲线面板
    void DrawFilterMenuItems();       // 菜单栏滤波子项
    void Draw2DOperatorMenuItems();   // 菜单栏 2D 算子（OpenCV）
    void DrawCreatePopups();          // 创建球/柱/圆盘弹窗
    void DrawImagePanel();            // 右侧深度/亮度图面板
    void HandleInput();               // 键盘快捷键（撤销、工具切换等）

    // --- 文件与点云 ---
    bool LoadPath(const std::string& path, bool append = true);
    bool SaveCloud();
    bool ApplyCloud(PointCloud&& cloud, const char* statusMsg, bool append = false);
    void CreateSphereCloud();
    void CreateCylinderCloud();
    void CreateDiskCloud();
    void CreatePlaneCloud();
    bool OpenDepthImage();
    bool OpenBrightnessImage();
    void DestroyImageView(ImageView& view);
    bool UploadImageTexture(ImageView& view);

    // --- 相机与 3D 视图 ---
    void FitCameraToCloud();
    void ApplyViewPreset(int preset);  // 0顶 1侧X 2侧Z 3沿Y 4包围盒
    void SetToolMode(ToolMode mode);
    void ClearToolVisuals(bool resetStatus = true);
    void RefreshGpu();                 // 上传当前激活点云到 GPU
    void RefreshAllLayerGpu();         // 上传所有场景点云
    void RefreshLayerGpu(SceneCloudLayer& layer, bool isActive);
    void RebuildAnalysisColors();    // 按分析结果（平面度/段差/拟合）着色
    void OnLeftClick(float mouseX, float mouseY);  // 3D 视区左键：点选/测距/剖切等
    void DrawOverlays();
    void UpdateOverlays();           // 同步拟合线框、测距线等到渲染器
    void SyncFittedSphereLayer(const SphereModel& sphere);
    void RemoveFittedSphereLayers();

    // --- ROI 框选 ---
    void BeginRoiDrag(float mouseX, float mouseY);
    void UpdateRoiDrag(float mouseX, float mouseY);
    void EndRoiDrag();
    void FinishRoiPolygon();
    void RunRoiSelection();          // 结束拖拽后计算 roiIndices
    void RefreshWorldRoiAt(float mouseX, float mouseY);  // 世界尺寸 ROI 中心拾取

    // --- 投影与填充 ---
    void ApplyProjectionToAxis(int axis);  // 投影到 XY/YZ/XZ 平面
    void AlignCloudToReferencePlane(const std::vector<std::size_t>* roiIndices,
                                    const PlaneModel* existingPlane);
    bool RunRoiProjectFill(std::string& error);  // 执行 ROI 填充并进入双视区
    void RefreshGpuFilled();           // 上传填充视区合成显示
    void CloseDualCloudView();
    bool DualCloudViewActive() const;
    int CloudPaneAtMouse(float mouseX) const;
    void GetCloudPaneFbRect(int pane, int& x, int& y, int& w, int& h) const;
    void GetCloudPaneGlViewport(int pane, int& x, int& y, int& w, int& h) const;
    float CloudPaneAspect(int pane) const;
    PointCloud& EditableCloud();       // 当前可编辑点云（主云或填充云）
    const PointCloud& EditableCloud() const;
    void DrawDualCloudPaneLabels();
    void DrawIcpOverlayLabel();
    void DrawFitRoiShapeControls();
    void ResetFitRoiSelection();
    void BuildFilledPaneDisplayCloud(PointCloud& out);  // 灰原+青补合成
    bool MeasureHoleRadiusWithBackend(HoleMeasureResult& out, std::string& error);

    // --- ICP 配准 ---
    void LoadIcpSourceCloud();
    void RunIcpRegistration();
    void ClearIcpState();
    void RefreshIcpGpu();
    void ReparentCloudToTargetFrame(PointCloud& src) const;
    void DrawIcpPanel();

    // --- 截面 ---
    void GenerateSection();
    void OnSectionPlotClick(float plotX, float plotY, float plotW, float plotH);
    void UpdateSectionDistances();
    std::optional<std::size_t> FindNearestSectionPoint(float plotX, float plotY, float plotW,
                                                       float plotH, float* outDistPx = nullptr) const;
    bool HitSectionPickMarker(float plotX, float plotY, float plotW, float plotH, bool point1,
                              float hitRadiusPx = 14.f) const;
    void SyncSectionCutPlane();
    void BeginSectionDrag(float mouseX, float mouseY);
    void UpdateSectionDrag(float mouseX, float mouseY);
    void EndSectionDrag();
    bool ProjectWorldToScreen(const Vec3& p, float& sx, float& sy) const;

    // --- 撤销 / 重做 ---
    void PushHistory(const std::string& label, bool captureMainPoints = false);
    void PushMainCloudHistory(const std::string& label, bool captureMainPoints = false);
    CloudSnapshot CaptureCloudSnapshot(HistoryCloudTarget target, const std::string& label,
                                     bool captureMainPoints) const;
    void ApplyCloudSnapshot(const CloudSnapshot& snap, bool closeDualOnMain);
    void Undo();
    void Redo();

    // --- 布局与坐标 ---
    void UpdateAxesLength();
    void UpdateView3dLayout(float contentTop, float contentH, float sidebarW);
    bool MouseInView3d(double mx, double my) const;
    void GetView3dFbRect(int& x, int& y, int& w, int& h) const;       // FB 坐标，原点左上
    void GetView3dGlViewport(int& x, int& y, int& w, int& h) const;  // GL 视口，原点左下
    float View3dAspect() const;
    bool HasImagePanel() const;
    float ImagePanelWidth() const;
    float SidebarWidth() const;
    void DrawSidebarSplitter(float contentTop, float contentH);
    void DrawConsolePanel();
    void AppendConsoleLog(const std::string& msg);
    void SetStatus(const std::string& msg, bool logConsole = true);
    float ConsoleHeight() const;

    // --- 深度/亮度图联动 ---
    bool TryEnableImageSync();
    void ClearImageSyncPick();
    void SetImageSyncPixel(int col, int row);
    void RebuildDepthDisplay();
    void ClearLineMeasure();
    void ClearArcMeasure();
    void ResetArcMeasurePick();
    void ClearMeasuredLinesOnly();
    void ClearMeasuredArcsOnly();
    void ClearAllMeasuredLines();
    void PreviewLineMeasure(ImageView& view);
    void PreviewArcMeasure(ImageView& view);
    void ConfirmLineMeasure();
    void ConfirmArcMeasure();
    void CancelCaliperPending();
    void PreviewCircleFitFromRoi(ImageView& view);
    void PreviewCircleFitFromMeasuredArc(int arcIndex);
    void ConfirmCircleFit();
    void CancelCircleFitPending();
    void UndoLastMeasuredLine();
    bool CanUndoMeasuredLine() const;
    void Undo2DOrCloud();
    int ImageSourceOf(const ImageView& view) const;
    ImageView* ImageViewFromSource(int source);
    const ImageView* ImageViewFromSource(int source) const;
    void ComputeSelectedLineDistance();
    void ClearLineDistance();
    void PickLineForDistance(int lineIndex);
    int FindClosestMeasuredLine(int imageSource, float px, float py, float maxDistPx) const;
    void ComputeSelectedArcDistance();
    void ClearArcDistance();
    void PickArcForDistance(int arcIndex);
    int FindClosestMeasuredArc(int imageSource, float px, float py, float maxDistPx) const;
    int FindClosestMeasuredCircleFit(int imageSource, float px, float py, float maxDistPx) const;
    void AddPointDistance(float ax, float ay, float bx, float by, int imageSource);
    void PickLineForAngle(int lineIndex);
    void ComputeSelectedLineAngle();
    void ClearLineAngle();
    void PickCircleForGap(int circleIndex);
    void ComputeSelectedCircleGap();
    void ClearCircleGap();
    void PickPointForPointLine(float px, float py, int imageSource);
    void PickLineForPointLine(int lineIndex);
    void ComputePointLineDistance();
    void ClearPointLineDistance();
    void PreviewCaliperPoint(ImageView& view);
    void ConfirmCaliperPoint();
    void CancelCaliperPointPending();
    void PreviewCircleCaliper(ImageView& view);
    void ConfirmCircleCaliper();
    void CancelCircleCaliperPending();
    void PickArcForLength(int arcIndex);
    void ComputeSelectedArcLength();
    void ClearArcLength();
    void ConfirmThreePointCircle();
    void ClearThreePointCircle();
    void PickLineForParallelDist(int lineIndex);
    void ComputeParallelLineDistance();
    void ClearParallelLineDistance();
    void PreviewRectCaliper(ImageView& view);
    void ConfirmRectCaliper();
    void CancelRectCaliperPending();
    void PreviewEllipseFitFromRoi(ImageView& view);
    void PreviewEllipseFitFromMeasuredArc(int arcIndex);
    void ConfirmEllipseFit();
    void CancelEllipseFitPending();
    void PreviewProfileWidth(ImageView& view);
    void ConfirmProfileWidth();
    void CancelProfileWidthPending();
    void PickPointForProjection(float px, float py, int imageSource);
    void PickLineForProjection(int lineIndex);
    void ComputePointProjection();
    void ClearPointProjection();
    void PickCircleForConcentricity(int circleIndex);
    void ComputeConcentricity();
    void ClearConcentricity();
    void PickCircleForRoundness(int circleIndex);
    void ComputeRoundness();
    void ClearRoundness();
    int FindClosestMeasuredCircleCaliper(int imageSource, float px, float py, float maxDistPx) const;
    void PreviewRegionBlob(ImageView& view, bool quiet = false);
    void RefreshRegionBlobPreview();
    bool HasRegionBlobRoi() const;
    void ConfirmRegionBlob();
    void CancelRegionBlobPending();
    void AddDepthHeightSample(float px, float py, int imageSource);
    void ClearDepthHeightDiff();
    void PreviewDepthProfile(ImageView& view);
    void PreviewDepthProfileAt(float px, float py, ImageView& view);
    void ClearDepthProfile();
    void PreviewTemplateMatch(ImageView& view);
    void ConfirmTemplateMatch();
    void CancelTemplateMatchPending();
    void ResetTemplateMatchPick();
    void ClearTemplateMatch();
    void ResetImage2dView();
    void EnterView2DMode();
    void EnterView3DMode();
    void RestoreApplicationState();
    void RotateImages2d90CW();
    void RotateImages2d90CCW();
    bool RotateImageView90CW(ImageView& view);
    bool RotateImageView90CCW(ImageView& view);
    void DrawLineMeasureOverlay(ImDrawList* dl, const ImageView& view, float cursorX, float cursorY,
                                float drawW, float drawH);
    void DrawImage2DToolPanel();
    void DrawImageWithSyncMarker(ImageView& view, const char* label);
    void DrawDepthRenderControls();
    void DrawRoiRegionOverlay(ImDrawList* dl, int winW, int winH,
                              const std::vector<std::size_t>& indices, const char* label,
                              unsigned int col, unsigned int textCol);
    void DrawStepGapRegionOverlays(ImDrawList* dl, int winW, int winH);

    // --- 滤波 ---
    void RunFilterPreview(int type, AlgorithmBackend backend);  // 0体素 1半径 2统计
    void ApplyFilterResult();
    void ClearFilterCompare();
    void DrawFilterCompareViewControls();

    // --- 算法后端桥接（自研 / PCL 二选一）---
    bool FitPlaneWithBackend(const std::vector<std::size_t>& indices, PlaneModel& plane,
                             std::string& error, AlgorithmBackend backend);
    bool FitSphereWithBackend(const std::vector<std::size_t>& indices, SphereModel& sphere,
                              std::string& error, AlgorithmBackend backend);
    bool FitCircleWithBackend(const std::vector<std::size_t>& indices, CircleModel& circle,
                              std::string& error, AlgorithmBackend backend);
    bool FitCylinderWithBackend(const std::vector<std::size_t>& indices, CylinderModel& cylinder,
                                std::string& error, AlgorithmBackend backend);
    bool ComputeFlatnessWithBackend(const std::vector<std::size_t>& indices, FlatnessResult& out,
                                    std::string& error, AlgorithmBackend backend);
    bool ComputeStepGapZHeightWithBackend(const std::vector<std::size_t>& regionA,
                                          const std::vector<std::size_t>& regionB,
                                          StepGapResult& out, std::string& error,
                                          AlgorithmBackend backend);
    bool ExtractSectionWithBackend(bool cutAlongX, float position, float thickness,
                                   SectionData& out, std::string& error,
                                   AlgorithmBackend backend);
    void SelectRoiWithBackend(int fbW, int fbH, float x0, float y0, float x1, float y1,
                              std::vector<std::size_t>& outIndices, RoiShape shape,
                              bool useWorldSize, float worldRadius, float worldHalfW,
                              float worldHalfH, const Vec3& worldCenter,
                              const std::vector<float>* polyX, const std::vector<float>* polyY);
    std::optional<std::size_t> PickNearestWithBackend(int fbW, int fbH, float mouseX, float mouseY,
                                                      float maxPixelDist,
                                                      const std::vector<std::size_t>* onlyIndices);
    void ApplyRoiDeleteWithBackend(const std::vector<std::size_t>& roiIndices, bool deleteInside);
    void RestoreAllPointsWithBackend();
    AlgoToolsHost BuildAlgoToolsHost(AlgorithmBackend backend);  // 供侧栏面板回调

    AlgorithmBackend EffectiveAlgoBackend() const;  // 当前生效的算法后端

    GLFWwindow* window_ = nullptr;
    std::vector<SceneCloudLayer> sceneLayers_;
    int activeLayerIndex_ = -1;
    int nextSceneLayerId_ = 1;
    int dbSelectedLayerId_ = 0;
    DbEntityId dbSelectedSpecial_ = DbEntityId::None;
    PointCloud filledCloud_;     // 填充点云：ROI 填充生成的补点（仅掩码可编辑）
    Camera camera_;
    PointCloudRenderer filledRenderer_;  // 填充视区专用渲染器（灰原+青补叠加）
    PointCloudRenderer icpRenderer_;     // ICP 源点云叠加渲染
    MeasureState measure_;
    UndoHistory history_;
    std::vector<std::size_t> displayFilledIndices_;
    bool dualCloudView_ = false;   // true：显示「灰原+青补」填充视区
    int activeCloudPane_ = 0;    // 0=主点云 1=填充点云（双视区时框选/删除作用对象）

    float pointSize_ = 2.5f;
    float opacity_ = 1.f;
    float zMin_ = 0.f;
    float zMax_ = 1.f;
    bool autoZRange_ = true;
    bool sceneGpuDirty_ = false;
    bool needUploadFilled_ = false;
    bool needUploadIcp_ = false;
    int maxDisplayPoints_ = 1200000;
    int gpuFilledPointCount_ = 0;
    float filledZMin_ = 0.f;
    float filledZMax_ = 1.f;
    PointCloud icpSourceCloud_;
    PointCloud icpAlignedSource_;
    PclTools::IcpParams icpParams_;
    PclTools::IcpResult icpResult_;
    bool icpSourceLoaded_ = false;
    bool icpOverlayActive_ = false;
    bool icpShowAligned_ = false;
    float icpZMin_ = 0.f;
    float icpZMax_ = 1.f;
    bool dbFilledRefVisible_ = true;
    bool dbFilledResultVisible_ = true;
    bool showAxes_ = true;
    float axesLength_ = 1.f;
    bool showAbout_ = false;
    bool showNativeAlgoPassword_ = false;
    bool nativeAlgoUnlocked_ = false;
    char nativeAlgoPasswordBuf_[16] = {};
    bool showCreateSphere_ = false;
    bool showCreateCylinder_ = false;
    bool showCreateDisk_ = false;
    bool showCreatePlane_ = false;
    static constexpr const char* kAppVersion = "0.5";

    // 创建点云参数
    float genSphereRadius_ = 10.f;
    int genSpherePoints_ = 20000;
    float genSphereNoise_ = 0.f;
    float genCylRadius_ = 8.f;
    float genCylHeight_ = 30.f;
    int genCylPoints_ = 30000;
    float genCylNoise_ = 0.05f;
    float genDiskRadius_ = 10.f;
    int genDiskPoints_ = 20000;
    float genDiskNoise_ = 0.f;
    float genPlaneExtentX_ = 20.f;
    float genPlaneExtentY_ = 20.f;
    int genPlanePoints_ = 20000;
    float genPlaneNoise_ = 0.f;

    // 2D 深度图 / 亮度图窗口（不转点云）
    ImageView depthImage_;
    ImageView brightnessImage_;
    int imagePanelTab_ = 0;  // 0 深度 1 亮度
    float sidebarPreferredW_ = 360.f;
    float dbTreePreferredH_ = 200.f;
    float imagePanelPreferredW_ = 420.f;
    float image2dZoom_ = 1.f;
    float image2dPanX_ = 0.f;
    float image2dPanY_ = 0.f;
    bool saveVisibleOnly_ = true;
    bool useIntensityColors_ = false;
    bool view2DMode_ = false;
    std::vector<Vec3> intensityColors_;

    // 深度图 / 亮度图同源联动（仅 2D）
    bool imageSyncEnabled_ = false;
    int syncWidth_ = 0;
    int syncHeight_ = 0;
    bool syncHasPick_ = false;
    int syncCol_ = -1;
    int syncRow_ = -1;

    // 深度图伪彩渲染窗口（收窄范围可放大高度差观感）
    float depthDataMin_ = 0.f;
    float depthDataMax_ = 1.f;
    float depthDisplayMin_ = 0.f;
    float depthDisplayMax_ = 1.f;
    bool depthSkipZero_ = true;

    // OpenCV 2D 算子：卡尺提线
    Image2DTool image2DTool_ = Image2DTool::None;
    bool lineMeasureDragging_ = false;
    float lineMeasureRoiX0_ = 0.f;
    float lineMeasureRoiY0_ = 0.f;
    float lineMeasureRoiX1_ = 0.f;
    float lineMeasureRoiY1_ = 0.f;
    int lineMeasureDragSource_ = -1;  // 0 深度 1 亮度
    bool lineMeasurePending_ = false;
    int lineMeasurePendingSource_ = -1;
    OpenCv2D::CaliperLineResult lineMeasurePendingResult_;
    OpenCv2D::CaliperLineParams lineMeasureParams_;
    bool showLineMeasureOverlay_ = true;

    struct MeasuredImageLine {
        int id = 0;
        int imageSource = 0;  // 0 深度 1 亮度
        OpenCv2D::CaliperLineResult result;
    };
    struct MeasuredImageArc {
        int id = 0;
        int imageSource = 0;
        OpenCv2D::CaliperArcResult result;
    };
    std::vector<MeasuredImageLine> measuredLines_;
    std::vector<MeasuredImageArc> measuredArcs_;
    int nextMeasuredLineId_ = 1;
    int nextMeasuredArcId_ = 1;

    ArcMeasurePhase arcMeasurePhase_ = ArcMeasurePhase::PickA;
    int arcMeasureSource_ = -1;
    bool arcBulgeDragging_ = false;
    float arcRoiP0X_ = 0.f;
    float arcRoiP0Y_ = 0.f;
    float arcRoiP1X_ = 0.f;
    float arcRoiP1Y_ = 0.f;
    float arcRoiP2X_ = 0.f;
    float arcRoiP2Y_ = 0.f;
    bool arcMeasurePending_ = false;
    int arcMeasurePendingSource_ = -1;
    OpenCv2D::CaliperArcResult arcMeasurePendingResult_;

    struct MeasuredCircleFit {
        int id = 0;
        int imageSource = 0;
        int sourceArcId = -1;  // 来自已有圆弧时记录圆弧 id，否则 -1
        OpenCv2D::CircleFitResult result;
        std::vector<OpenCv2D::CaliperEdgePoint> edgePoints;
    };
    std::vector<MeasuredCircleFit> measuredCircleFits_;
    int nextCircleFitId_ = 1;
    bool circleFitPending_ = false;
    int circleFitPendingSource_ = -1;
    int circleFitPendingFromArcId_ = -1;
    OpenCv2D::CircleFitResult circleFitPendingResult_;
    std::vector<OpenCv2D::CaliperEdgePoint> circleFitPendingEdgePoints_;

    int lineDistPickA_ = -1;
    int lineDistPickB_ = -1;
    bool lineDistValid_ = false;
    int lineDistSampleCount_ = 32;
    float lineDistPx_ = 0.f;
    float lineDistMinPx_ = 0.f;
    float lineDistMaxPx_ = 0.f;
    std::vector<OpenCv2D::GapSample> lineDistSamples_;

    int arcDistPickA_ = -1;
    int arcDistPickB_ = -1;
    bool arcDistValid_ = false;
    int arcDistSampleCount_ = 32;
    float arcDistPx_ = 0.f;
    float arcDistMinPx_ = 0.f;
    float arcDistMaxPx_ = 0.f;
    std::vector<OpenCv2D::GapSample> arcDistSamples_;

    struct MeasuredPointDist {
        int id = 0;
        int imageSource = 0;
        float ax = 0.f;
        float ay = 0.f;
        float bx = 0.f;
        float by = 0.f;
        float distance = 0.f;
        float dx = 0.f;
        float dy = 0.f;
    };
    std::vector<MeasuredPointDist> measuredPointDists_;
    int nextPointDistId_ = 1;
    PointPickPhase pointDistPhase_ = PointPickPhase::PickA;
    int pointDistSource_ = -1;
    float pointDistAx_ = 0.f;
    float pointDistAy_ = 0.f;

    int lineAnglePickA_ = -1;
    int lineAnglePickB_ = -1;
    bool lineAngleValid_ = false;
    float lineAngleDeg_ = 0.f;

    int circleGapPickA_ = -1;
    int circleGapPickB_ = -1;
    bool circleGapValid_ = false;
    float circleGapCenterDist_ = 0.f;
    float circleGapSurfaceGap_ = 0.f;

    PointPickPhase pointLinePhase_ = PointPickPhase::PickA;
    int pointLineSource_ = -1;
    float pointLinePx_ = 0.f;
    float pointLinePy_ = 0.f;
    int pointLinePick_ = -1;
    bool pointLineValid_ = false;
    float pointLineDistPx_ = 0.f;
    float pointLineFootX_ = 0.f;
    float pointLineFootY_ = 0.f;

    bool caliperPointDragging_ = false;
    int caliperPointDragSource_ = -1;
    float caliperPointRoiX0_ = 0.f;
    float caliperPointRoiY0_ = 0.f;
    float caliperPointRoiX1_ = 0.f;
    float caliperPointRoiY1_ = 0.f;
    bool caliperPointPending_ = false;
    int caliperPointPendingSource_ = -1;
    OpenCv2D::CaliperEdgePoint caliperPointPendingEdge_;
    struct MeasuredCaliperPoint {
        int id = 0;
        int imageSource = 0;
        float x = 0.f;
        float y = 0.f;
        float roiX0 = 0.f;
        float roiY0 = 0.f;
        float roiX1 = 0.f;
        float roiY1 = 0.f;
    };
    std::vector<MeasuredCaliperPoint> measuredCaliperPoints_;
    int nextCaliperPointId_ = 1;

    CircleCaliperPhase circleCaliperPhase_ = CircleCaliperPhase::PickCenter;
    int circleCaliperSource_ = -1;
    bool circleCaliperDragging_ = false;
    float circleCaliperCx_ = 0.f;
    float circleCaliperCy_ = 0.f;
    float circleCaliperR_ = 0.f;
    bool circleCaliperPending_ = false;
    int circleCaliperPendingSource_ = -1;
    OpenCv2D::CaliperCircleResult circleCaliperPendingResult_;
    struct MeasuredCircleCaliper {
        int id = 0;
        int imageSource = 0;
        OpenCv2D::CaliperCircleResult result;
    };
    std::vector<MeasuredCircleCaliper> measuredCircleCalipers_;
    int nextCircleCaliperId_ = 1;

    int arcLengthPick_ = -1;
    bool arcLengthValid_ = false;
    OpenCv2D::ArcMetrics arcLengthMetrics_;

    ThreePointPhase threePointPhase_ = ThreePointPhase::Pick0;
    int threePointSource_ = -1;
    float threePointX_[3] = {};
    float threePointY_[3] = {};
    struct MeasuredThreePointCircle {
        int id = 0;
        int imageSource = 0;
        float centerX = 0.f;
        float centerY = 0.f;
        float radius = 0.f;
    };
    std::vector<MeasuredThreePointCircle> measuredThreePointCircles_;
    int nextThreePointCircleId_ = 1;

    int parallelDistPickA_ = -1;
    int parallelDistPickB_ = -1;
    bool parallelDistValid_ = false;
    float parallelDistPx_ = 0.f;

    bool rectCaliperDragging_ = false;
    int rectCaliperDragSource_ = -1;
    float rectCaliperRoiX0_ = 0.f;
    float rectCaliperRoiY0_ = 0.f;
    float rectCaliperRoiX1_ = 0.f;
    float rectCaliperRoiY1_ = 0.f;
    bool rectCaliperPending_ = false;
    int rectCaliperPendingSource_ = -1;
    OpenCv2D::CaliperRectResult rectCaliperPendingResult_;
    struct MeasuredRectCaliper {
        int id = 0;
        int imageSource = 0;
        OpenCv2D::CaliperRectResult result;
    };
    std::vector<MeasuredRectCaliper> measuredRectCalipers_;
    int nextRectCaliperId_ = 1;

    bool ellipseFitPending_ = false;
    int ellipseFitPendingSource_ = -1;
    int ellipseFitPendingFromArcId_ = -1;
    OpenCv2D::EllipseFitResult ellipseFitPendingResult_;
    std::vector<OpenCv2D::CaliperEdgePoint> ellipseFitPendingEdgePoints_;
    struct MeasuredEllipseFit {
        int id = 0;
        int imageSource = 0;
        int sourceArcId = -1;
        OpenCv2D::EllipseFitResult result;
        std::vector<OpenCv2D::CaliperEdgePoint> edgePoints;
    };
    std::vector<MeasuredEllipseFit> measuredEllipseFits_;
    int nextEllipseFitId_ = 1;

    bool profileWidthDragging_ = false;
    int profileWidthDragSource_ = -1;
    float profileWidthRoiX0_ = 0.f;
    float profileWidthRoiY0_ = 0.f;
    float profileWidthRoiX1_ = 0.f;
    float profileWidthRoiY1_ = 0.f;
    bool profileWidthPending_ = false;
    int profileWidthPendingSource_ = -1;
    OpenCv2D::ProfileWidthResult profileWidthPendingResult_;
    struct MeasuredProfileWidth {
        int id = 0;
        int imageSource = 0;
        OpenCv2D::ProfileWidthResult result;
    };
    std::vector<MeasuredProfileWidth> measuredProfileWidths_;
    int nextProfileWidthId_ = 1;

    PointPickPhase pointProjPhase_ = PointPickPhase::PickA;
    int pointProjSource_ = -1;
    float pointProjPx_ = 0.f;
    float pointProjPy_ = 0.f;
    int pointProjLinePick_ = -1;
    bool pointProjValid_ = false;
    OpenCv2D::PointProjectionResult pointProjResult_;

    int concentricityPickA_ = -1;
    int concentricityPickB_ = -1;
    bool concentricityValid_ = false;
    OpenCv2D::ConcentricityResult concentricityResult_;

    int roundnessPick_ = -1;
    int roundnessCircleSource_ = 0;  // 0=fit 1=caliper
    bool roundnessValid_ = false;
    OpenCv2D::RoundnessResult roundnessResult_;

    bool regionBlobDragging_ = false;
    int regionBlobDragSource_ = -1;
    float regionBlobRoiX0_ = 0.f;
    float regionBlobRoiY0_ = 0.f;
    float regionBlobRoiX1_ = 0.f;
    float regionBlobRoiY1_ = 0.f;
    float regionBlobThreshold_ = 128.f;
    bool regionBlobGreaterThan_ = true;
    bool regionBlobPending_ = false;
    int regionBlobPendingSource_ = -1;
    OpenCv2D::RegionBlobResult regionBlobPendingResult_;
    struct MeasuredRegionBlob {
        int id = 0;
        int imageSource = 0;
        OpenCv2D::RegionBlobResult result;
    };
    std::vector<MeasuredRegionBlob> measuredRegionBlobs_;
    int nextRegionBlobId_ = 1;

    PointPickPhase depthHeightPhase_ = PointPickPhase::PickA;
    int depthHeightSource_ = -1;
    float depthHeightAx_ = 0.f;
    float depthHeightAy_ = 0.f;
    float depthHeightAz_ = 0.f;
    float depthHeightBz_ = 0.f;
    bool depthHeightValid_ = false;
    float depthHeightDelta_ = 0.f;

    bool depthProfileDragging_ = false;
    int depthProfileDragSource_ = -1;
    float depthProfileRoiX0_ = 0.f;
    float depthProfileRoiY0_ = 0.f;
    float depthProfileRoiX1_ = 0.f;
    float depthProfileRoiY1_ = 0.f;
    bool depthProfileValid_ = false;
    bool showDepthProfilePanel_ = true;
    DepthProfileMode depthProfileMode_ = DepthProfileMode::ScanRow;
    int depthProfileSampleCount_ = 256;
    std::vector<OpenCv2D::LineProfileSample> depthProfileSamples_;
    std::string depthProfileLabel_;
    int depthProfileHoverIdx_ = -1;
    int depthProfilePickA_ = -1;
    int depthProfilePickB_ = -1;
    float depthProfileMeasureDelta_ = 0.f;
    bool depthProfileMeasureValid_ = false;

    TemplateMatchPhase templateMatchPhase_ = TemplateMatchPhase::DrawTemplate;
    bool templateMatchTplDragging_ = false;
    bool templateMatchSearchDragging_ = false;
    int templateMatchDragSource_ = -1;
    float templateMatchTplX0_ = 0.f;
    float templateMatchTplY0_ = 0.f;
    float templateMatchTplX1_ = 0.f;
    float templateMatchTplY1_ = 0.f;
    bool templateMatchHasTemplate_ = false;
    float templateMatchSearchX0_ = 0.f;
    float templateMatchSearchY0_ = 0.f;
    float templateMatchSearchX1_ = 0.f;
    float templateMatchSearchY1_ = 0.f;
    bool templateMatchSearchFull_ = true;
    OpenCv2D::TemplateMatchParams templateMatchParams_;
    bool templateMatchPending_ = false;
    int templateMatchPendingSource_ = -1;
    OpenCv2D::TemplateMatchResult templateMatchPendingResult_;
    struct MeasuredTemplateMatch {
        int id = 0;
        int imageSource = 0;
        OpenCv2D::TemplateMatchResult result;
    };
    std::vector<MeasuredTemplateMatch> measuredTemplateMatches_;
    int nextTemplateMatchId_ = 1;

    // 滤波
    float filterVoxelLeaf_ = 0.5f;
    float filterRadius_ = 1.0f;
    int filterRadiusMinNeighbors_ = 4;
    int filterStatMeanK_ = 20;
    float filterStatStdMul_ = 1.0f;
    bool filterCompareActive_ = false;
    FilterCompareViewMode filterCompareViewMode_ = FilterCompareViewMode::Compare;
    std::vector<uint8_t> filterKeepMask_;
    std::vector<uint8_t> filterBackupMask_;
    int filterLastKept_ = 0;
    int filterLastRemoved_ = 0;

    AlgorithmBackend algoBackend_ = AlgorithmBackend::PCL;
    int planeAlignTarget_ = 0;  // 0=+Z 水平  1=+Y  2=+X
    PclPanel pclPanel_;
    PclToolsPanel pclToolsPanel_;

    bool rotating_ = false;
    bool panning_ = false;
    bool sectionDragging_ = false;
    bool showSectionPanel_ = true;
    int sectionPlotDragTarget_ = 0;
    double lastX_ = 0.0;
    double lastY_ = 0.0;
    float lastSectionMouseX_ = 0.f;
    float lastSectionMouseY_ = 0.f;

    int fbW_ = 1280;
    int fbH_ = 800;

    float view3dX_ = 320.f;
    float view3dY_ = 0.f;
    float view3dW_ = 800.f;
    float view3dH_ = 600.f;
    float view3dPane0X_ = 320.f;
    float view3dPane0W_ = 800.f;
    float view3dPane1X_ = 0.f;
    float view3dPane1W_ = 0.f;
    float cachedContentTop_ = 0.f;
    float cachedContentH_ = 600.f;
    float cachedSidebarW_ = 320.f;
    bool cachedViewLayoutValid_ = false;

    struct ConsoleLine {
        std::string time;  // HH:MM:SS
        std::string text;
    };
    std::vector<ConsoleLine> consoleLog_;
    float consoleHeight_ = 168.f;
    bool consoleAutoScroll_ = true;
    static constexpr std::size_t kConsoleMaxLines = 500;

    AlgorithmEditor algoEditor_;
    ShapeTemplateMatchWindow shapeTemplateWindow_;
    HalconMatchWindow halconMatchWindow_;
};
