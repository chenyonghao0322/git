#pragma once

#include "tools/HalconShapeMatch.h"

#include <functional>
#include <string>
#include <vector>

struct ImDrawList;
struct ImVec2;

class HalconMatchWindow {
public:
    using StatusCallback = std::function<void(const char*)>;

    void SetStatusCallback(StatusCallback cb) { onStatus_ = std::move(cb); }
    void SetVisible(bool visible);
    bool IsVisible() const { return visible_; }
    void ToggleVisible();
    void Draw(float menuBottomY, float bottomInset = 0.f);

private:
    struct ImageSlot {
        int width = 0;
        int height = 0;
        std::vector<uint8_t> rgb;
        std::string path;
        unsigned int texId = 0;
        bool valid() const { return texId != 0 && width > 0 && height > 0; }
    };

    enum class RoiMode { Template, Search };
    enum class LeftViewMode { Template, Source };

    enum class TemplateRoiDrag {
        None,
        Create,
        Move,
        ResizeCorner,
        Rotate,
    };

    bool LoadImageSlot(ImageSlot& slot, const std::string& path, std::string& error);
    void DestroyImageSlot(ImageSlot& slot);
    bool UploadTexture(ImageSlot& slot);
    void DestroyPreviewTexture();
    void DrawImageCanvas(ImageSlot& slot, bool allowRoi, RoiMode roiMode, const char* childId);
    void DrawTemplatePreviewPanel();
    void DrawMatchOverlays(ImDrawList* dl, const ImVec2& imgPos, float drawW, float drawH);
    void DrawRotatedTemplateRoi(ImDrawList* dl, const ImVec2& imgPos, float sx, float sy);
    int HitTestTemplateRoi(float imgX, float imgY, float sx, float sy) const;
    bool ImageToPixel(const ImageSlot& slot, const ImVec2& imgPos, float drawW, float drawH,
                      float mouseX, float mouseY, float& outX, float& outY) const;
    void DrawParamColumns();
    void ClearTemplateRoi();
    bool GetTemplateRoiForCreate(float& cx, float& cy, float& hw, float& hh, float& angleDeg) const;

    void ReadTemplateImage();
    void ReadSourceImage();
    void CreateTemplate();
    void RunMatch();
    void SaveTemplate();
    void LoadTemplate();
    void ClearTemplate();
    void ClearSearchRoi();
    void SetStatus(const char* msg);

    bool visible_ = false;
    bool focusOnOpen_ = false;
    StatusCallback onStatus_;

    ImageSlot templateImage_;
    ImageSlot sourceImage_;
    HalconShapeMatch::ShapeModel model_;
    HalconShapeMatch::CreateParams createParams_;
    HalconShapeMatch::FindParams findParams_;
    HalconShapeMatch::MatchResult lastResult_;

    LeftViewMode leftView_ = LeftViewMode::Template;

    bool useSearchRoi_ = false;
    bool roiEnabled_ = true;
    bool templateUseRoi_ = false;
    RoiMode roiMode_ = RoiMode::Template;

    bool templateRoiValid_ = false;
    float templateRoiCx_ = 0.f;
    float templateRoiCy_ = 0.f;
    float templateRoiHalfW_ = 0.f;
    float templateRoiHalfH_ = 0.f;
    float templateRoiAngleDeg_ = 0.f;

    TemplateRoiDrag templateRoiDrag_ = TemplateRoiDrag::None;
    int templateRoiHitCorner_ = -1;
    float templateRoiDragStartX_ = 0.f;
    float templateRoiDragStartY_ = 0.f;
    float templateRoiDragAnchorCx_ = 0.f;
    float templateRoiDragAnchorCy_ = 0.f;
    float templateRoiDragAnchorHw_ = 0.f;
    float templateRoiDragAnchorHh_ = 0.f;
    float templateRoiDragAnchorAngle_ = 0.f;
    float templateRoiCreateX0_ = 0.f;
    float templateRoiCreateY0_ = 0.f;

    bool searchRoiDragging_ = false;
    float searchRoiX0_ = 0.f;
    float searchRoiY0_ = 0.f;
    float searchRoiX1_ = 0.f;
    float searchRoiY1_ = 0.f;

    float zoom_ = 1.f;
    int hoverPx_ = -1;
    int hoverPy_ = -1;
    std::string statusText_;
};
