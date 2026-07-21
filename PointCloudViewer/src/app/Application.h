#pragma once

#include "core/PointCloud.h"
#include "render/Camera.h"
#include "render/PointCloudRenderer.h"
#include "tools/MeasureTools.h"
#include "tools/UndoHistory.h"
#include "app/AlgorithmEditor.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

struct GLFWwindow;
struct ImDrawList;

class Application {
public:
    bool Init();
    void Run();
    void Shutdown();

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

    void DrawUi();
    float DrawMenuBar();  // returns menu bar bottom Y
    void DrawToolbar(float y, float height);
    void DrawToolPanel();
    void DrawViewAxisWidget(float contentTop, float contentBottom, float leftInset);
    void DrawAboutPopup();
    void DrawSectionPanel();
    void DrawStepGapPanel();
    void DrawFilterMenuItems();
    void DrawCreatePopups();
    void DrawImagePanel();
    void HandleInput();
    bool LoadPath(const std::string& path);
    bool SaveCloud();
    bool ApplyCloud(PointCloud&& cloud, const char* statusMsg);
    void CreateSphereCloud();
    void CreateCylinderCloud();
    bool OpenDepthImage();
    bool OpenBrightnessImage();
    void DestroyImageView(ImageView& view);
    bool UploadImageTexture(ImageView& view);
    void FitCameraToCloud();
    void ApplyViewPreset(int preset);  // 0顶 1侧X 2侧Z 3沿Y 4包围盒
    void SetToolMode(ToolMode mode);
    void ClearToolVisuals(bool resetStatus = true);
    void RefreshGpu();
    void RebuildAnalysisColors();
    void OnLeftClick(float mouseX, float mouseY);
    void DrawOverlays();
    void UpdateOverlays();
    void BeginRoiDrag(float mouseX, float mouseY);
    void UpdateRoiDrag(float mouseX, float mouseY);
    void EndRoiDrag();
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
    void PushHistory(const std::string& label);
    void Undo();
    void Redo();
    void UpdateAxesLength();
    void UpdateView3dLayout(float contentTop, float contentH, float sidebarW);
    bool MouseInView3d(double mx, double my) const;
    // 点云视区在 framebuffer 中的矩形（原点左上，与鼠标 FB 坐标一致）
    void GetView3dFbRect(int& x, int& y, int& w, int& h) const;
    // OpenGL viewport（原点左下）
    void GetView3dGlViewport(int& x, int& y, int& w, int& h) const;
    float View3dAspect() const;
    bool HasImagePanel() const;
    float ImagePanelWidth() const;
    bool TryEnableImageSync();
    void ClearImageSyncPick();
    void SetImageSyncPixel(int col, int row);
    void RebuildDepthDisplay();
    void DrawImageWithSyncMarker(ImageView& view, const char* label);
    void DrawDepthRenderControls();
    void DrawRoiRegionOverlay(ImDrawList* dl, int winW, int winH,
                              const std::vector<std::size_t>& indices, const char* label,
                              unsigned int col, unsigned int textCol);
    void DrawStepGapRegionOverlays(ImDrawList* dl, int winW, int winH);
    void RunFilterPreview(int type);  // 0 voxel 1 radius 2 statistical
    void ApplyFilterResult();
    void ClearFilterCompare();

    GLFWwindow* window_ = nullptr;
    PointCloud cloud_;
    Camera camera_;
    PointCloudRenderer renderer_;
    MeasureState measure_;
    UndoHistory history_;
    std::vector<std::size_t> displayIndices_;

    float pointSize_ = 2.5f;
    float opacity_ = 1.f;
    float zMin_ = 0.f;
    float zMax_ = 1.f;
    bool autoZRange_ = true;
    bool needUpload_ = false;
    int maxDisplayPoints_ = 1200000;
    int gpuPointCount_ = 0;
    bool showAxes_ = true;
    float axesLength_ = 1.f;
    bool showAbout_ = false;
    bool showCreateSphere_ = false;
    bool showCreateCylinder_ = false;
    static constexpr const char* kAppVersion = "0.1";

    // 创建点云参数
    float genSphereRadius_ = 10.f;
    int genSpherePoints_ = 20000;
    float genSphereNoise_ = 0.05f;
    float genCylRadius_ = 8.f;
    float genCylHeight_ = 30.f;
    int genCylPoints_ = 30000;
    float genCylNoise_ = 0.05f;

    // 2D 深度图 / 亮度图窗口（不转点云）
    ImageView depthImage_;
    ImageView brightnessImage_;
    bool showImagePanel_ = false;
    int imagePanelTab_ = 0;  // 0 深度 1 亮度
    float imagePanelPreferredW_ = 420.f;
    float image2dZoom_ = 1.f;
    bool saveVisibleOnly_ = true;
    bool useIntensityColors_ = false;
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

    // 滤波
    float filterVoxelLeaf_ = 0.5f;
    float filterRadius_ = 1.0f;
    int filterRadiusMinNeighbors_ = 4;
    int filterStatMeanK_ = 20;
    float filterStatStdMul_ = 1.0f;
    bool filterCompareActive_ = false;
    bool filterHideRemoved_ = false;
    std::vector<uint8_t> filterKeepMask_;
    std::vector<uint8_t> filterBackupMask_;
    int filterLastKept_ = 0;
    int filterLastRemoved_ = 0;

    bool rotating_ = false;
    bool panning_ = false;
    bool sectionDragging_ = false;
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

    AlgorithmEditor algoEditor_;
};
