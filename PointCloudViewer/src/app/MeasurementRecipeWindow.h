#pragma once

#include "tools/HalconShapeMatch.h"

#include <functional>
#include <string>
#include <vector>

struct ImDrawList;
struct ImVec2;

// 测量配方：步骤0 固定为模板匹配，其后为点点/点到线/2D算子（几何存模板坐标系）。
class MeasurementRecipeWindow {
public:
    using StatusCallback = std::function<void(const char*)>;
    using SyncSourceCallback =
        std::function<void(const std::vector<uint8_t>& rgb, const std::vector<float>& gray, int w,
                           int h, const std::string& path)>;
    using DrawHostCanvasCallback = std::function<void()>;
    using DrawHostToolPanelCallback = std::function<void()>;
    using BoolCallback = std::function<bool()>;
    // tool：与 Application::Image2DTool 同值；0=None
    using Activate2DToolCallback = std::function<void(int tool)>;
    using GetActive2DToolCallback = std::function<int()>;
    // 按模板匹配位姿，把主程序里已确认的 2D 几何从旧位姿变换到新位姿，并刷新线距等结果。
    using RemapHostGeometryCallback =
        std::function<void(float fromCx, float fromCy, float fromAngDeg, float fromScale, float toCx,
                           float toCy, float toAngDeg, float toScale)>;
    using VoidCallback = std::function<void()>;

    void SetStatusCallback(StatusCallback cb) { onStatus_ = std::move(cb); }
    void SetSyncSourceCallback(SyncSourceCallback cb) { onSyncSource_ = std::move(cb); }
    void SetDrawHostCanvasCallback(DrawHostCanvasCallback cb) { onDrawHostCanvas_ = std::move(cb); }
    void SetDrawHostToolPanelCallback(DrawHostToolPanelCallback cb) {
        onDrawHostToolPanel_ = std::move(cb);
    }
    void SetIs2DToolActiveCallback(BoolCallback cb) { is2DToolActive_ = std::move(cb); }
    void SetActivate2DToolCallback(Activate2DToolCallback cb) { onActivate2DTool_ = std::move(cb); }
    void SetGetActive2DToolCallback(GetActive2DToolCallback cb) { getActive2DTool_ = std::move(cb); }
    void SetRemapHostGeometryCallback(RemapHostGeometryCallback cb) {
        onRemapHostGeometry_ = std::move(cb);
    }
    void SetInvalidateHostResultsCallback(VoidCallback cb) {
        onInvalidateHostResults_ = std::move(cb);
    }
    void SetVisible(bool visible);
    bool IsVisible() const { return visible_; }
    void ToggleVisible();
    void Draw(float menuBottomY, float bottomInset = 0.f);
    bool HasSourceImage() const { return sourceImage_.valid(); }
    bool HasMatchPose() const { return pose_.valid; }
    void DrawMeasureOverlaysOnHost(ImDrawList* dl, const ImVec2& imgPos, float drawW, float drawH);
    // 是否有匹配轮廓可叠加（供 host 画布使用）
    bool HasMatchOverlay() const { return !lastResult_.hits.empty(); }

private:
    enum class StepType { Match, PointPoint, PointLine, Op2D };
    enum class PlaceMode { None, PointA, PointB, PointP, LineA, LineB };
    enum class ViewMode { Template, Source };
    enum class RoiMode { Template, Search };

    struct ImageSlot {
        int width = 0;
        int height = 0;
        std::vector<uint8_t> rgb;
        std::vector<float> gray;
        std::string path;
        unsigned int texId = 0;
        bool valid() const { return texId != 0 && width > 0 && height > 0; }
    };

    struct Pose2D {
        float cx = 0.f;
        float cy = 0.f;
        float angleDeg = 0.f;
        float scale = 1.f;
        bool valid = false;
    };

    struct RecipeStep {
        StepType type = StepType::Match;
        std::string name;
        bool enabled = true;
        int op2dTool = 0;  // Image2DTool 整型
        float ax = 0.f, ay = 0.f;
        float bx = 0.f, by = 0.f;
        float px = 0.f, py = 0.f;
        float lx0 = 0.f, ly0 = 0.f;
        float lx1 = 0.f, ly1 = 0.f;
        bool aSet = false, bSet = false, pSet = false, lineASet = false, lineBSet = false;
        bool hasResult = false;
        float resultValue = 0.f;
    };

    bool LoadImageSlot(ImageSlot& slot, const std::string& path, std::string& error);
    void DestroyImageSlot(ImageSlot& slot);
    bool UploadTexture(ImageSlot& slot);
    bool ImageToPixel(const ImageSlot& slot, const ImVec2& imgPos, float drawW, float drawH,
                      float mouseX, float mouseY, float& outX, float& outY) const;

    void SetStatus(const char* msg);
    void EnsureMatchStep();
    void ResetMeasureResults();
    void RunMatch();
    void CreateTemplate();
    void ReadTemplateImage();
    void ReadSourceImage();
    void SaveTemplateModel();
    void LoadTemplateModel();
    void ClearTemplate();
    void AddPointPointStep();
    void AddPointLineStep();
    void AddOp2DStep(int tool, const char* name);
    void RemoveSelectedStep();
    void MoveSelectedStep(int delta);
    void RunAllMeasures();
    void EvaluateStep(RecipeStep& step);
    void SaveRecipe();
    void LoadRecipe();
    bool WriteRecipeFile(const std::string& path, std::string& error);
    bool ReadRecipeFile(const std::string& path, std::string& error);
    void OnSelectStep(int index);
    void ActivateOp2D(int tool);

    void LocalToImage(float lx, float ly, float& ix, float& iy) const;
    void ImageToLocal(float ix, float iy, float& lx, float& ly) const;
    bool HasPose() const { return pose_.valid; }

    void DrawLeftPanel(float width);
    void DrawRightPanel();
    void DrawMatchControls();
    void DrawTemplatePreviewPanel();
    void DrawStepList();
    void Draw2DToolLibrary();
    void DrawImageCanvas();
    void DrawOverlays(ImDrawList* dl, const ImVec2& imgPos, float drawW, float drawH);
    void HandleCanvasClick(float imgX, float imgY);
    void ClearSearchRoi();
    void ClearTemplateRoi();
    bool GetTemplateRoiForCreate(float& cx, float& cy, float& hw, float& hh, float& angleDeg) const;
    void SyncSourceToHost();
    bool Is2DToolActive() const { return is2DToolActive_ && is2DToolActive_(); }
    bool UseHostCanvas() const;
    int Active2DTool() const { return getActive2DTool_ ? getActive2DTool_() : 0; }

    bool visible_ = false;
    bool focusOnOpen_ = false;
    StatusCallback onStatus_;
    SyncSourceCallback onSyncSource_;
    DrawHostCanvasCallback onDrawHostCanvas_;
    DrawHostToolPanelCallback onDrawHostToolPanel_;
    BoolCallback is2DToolActive_;
    Activate2DToolCallback onActivate2DTool_;
    GetActive2DToolCallback getActive2DTool_;
    RemapHostGeometryCallback onRemapHostGeometry_;
    VoidCallback onInvalidateHostResults_;
    std::string statusText_;

    ImageSlot templateImage_;
    ImageSlot sourceImage_;
    ViewMode viewMode_ = ViewMode::Source;

    HalconShapeMatch::ShapeModel model_;
    HalconShapeMatch::CreateParams createParams_;
    HalconShapeMatch::FindParams findParams_;
    HalconShapeMatch::MatchResult lastResult_;
    Pose2D pose_;
    // 当前 measuredLines_ 等 2D 几何所对应的匹配位姿（换图重匹配时用于刚体变换）
    Pose2D geometryPose_;
    std::string modelPath_;

    bool templateUseRoi_ = false;
    bool templateRoiValid_ = false;
    float templateRoiX0_ = 0.f, templateRoiY0_ = 0.f;
    float templateRoiX1_ = 0.f, templateRoiY1_ = 0.f;
    bool templateRoiDragging_ = false;

    bool searchRoiEnabled_ = false;
    bool searchRoiValid_ = false;
    float searchRoiX0_ = 0.f, searchRoiY0_ = 0.f;
    float searchRoiX1_ = 0.f, searchRoiY1_ = 0.f;
    bool searchRoiDragging_ = false;

    RoiMode roiMode_ = RoiMode::Template;
    std::vector<RecipeStep> steps_;
    int selectedStep_ = 0;
    PlaceMode placeMode_ = PlaceMode::None;

    float zoom_ = 1.f;
    float panX_ = 0.f;
    float panY_ = 0.f;
    int hoverPx_ = -1;
    int hoverPy_ = -1;

    float leftPanelW_ = 360.f;
    float stepListH_ = 200.f;
    float toolLibH_ = 160.f;
    float hostToolH_ = 240.f;
};
