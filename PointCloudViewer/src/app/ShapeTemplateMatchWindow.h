#pragma once

#include "tools/ShapeTemplateMatch.h"

#include <functional>
#include <string>
#include <vector>

struct ImDrawList;
struct ImVec2;

class ShapeTemplateMatchWindow {
public:
    using StatusCallback = std::function<void(const char*)>;

    void SetStatusCallback(StatusCallback cb) { onStatus_ = std::move(cb); }
    void SetVisible(bool visible);
    bool IsVisible() const { return visible_; }
    void ToggleVisible();
    void Draw();

private:
    struct ImageSlot {
        int width = 0;
        int height = 0;
        std::vector<uint8_t> rgb;
        std::string path;
        unsigned int texId = 0;
        bool valid() const { return texId != 0 && width > 0 && height > 0; }
    };

    enum class RoiMode { None, Template, Search };

    bool LoadImageSlot(ImageSlot& slot, const std::string& path, std::string& error);
    void DestroyImageSlot(ImageSlot& slot);
    bool UploadTexture(ImageSlot& slot);
    void DestroyModelPreviewTexture();
    void RefreshModelPreviewTexture();
    void DrawImageCanvas(ImageSlot& slot, bool allowRoi, RoiMode roiMode, float height,
                         const char* childId);
    void DrawTemplatePreviewPanel();
    void DrawMatchOverlays(ImDrawList* dl, const ImVec2& imgPos, float drawW, float drawH);
    bool ImageToPixel(const ImageSlot& slot, const ImVec2& imgPos, float drawW, float drawH,
                      float mouseX, float mouseY, float& outX, float& outY) const;
    void DrawParamColumns();

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
    StatusCallback onStatus_;

    ImageSlot templateImage_;
    ImageSlot sourceImage_;
    ShapeTemplateMatch::ShapeModel model_;
    ShapeTemplateMatch::CreateParams createParams_;
    ShapeTemplateMatch::FindParams findParams_;
    ShapeTemplateMatch::MatchResult lastResult_;

    unsigned int modelPreviewTexId_ = 0;
    int modelPreviewW_ = 0;
    int modelPreviewH_ = 0;

    bool useSearchRoi_ = false;
    bool roiEnabled_ = true;
    RoiMode roiMode_ = RoiMode::Template;
    bool roiDragging_ = false;
    float roiX0_ = 0.f;
    float roiY0_ = 0.f;
    float roiX1_ = 0.f;
    float roiY1_ = 0.f;
    float searchRoiX0_ = 0.f;
    float searchRoiY0_ = 0.f;
    float searchRoiX1_ = 0.f;
    float searchRoiY1_ = 0.f;

    float zoom_ = 1.f;
    float panX_ = 0.f;
    float panY_ = 0.f;
    int hoverPx_ = -1;
    int hoverPy_ = -1;
    int hoverR_ = 0;
    int hoverG_ = 0;
    int hoverB_ = 0;
    std::string statusText_;
};
