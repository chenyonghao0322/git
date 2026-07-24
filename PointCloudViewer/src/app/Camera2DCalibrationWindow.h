#pragma once

#include "tools/Camera2DCalibration.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

struct ImDrawList;
struct ImVec2;

// 2D 相机九点标定窗口：每张图像对应一组机器人坐标，框选圆点后检测圆心并求解仿射变换。
class Camera2DCalibrationWindow {
public:
    using StatusCallback = std::function<void(const char*)>;

    void SetStatusCallback(StatusCallback cb) { onStatus_ = std::move(cb); }
    void SetVisible(bool visible);
    bool IsVisible() const { return visible_; }
    void ToggleVisible();
    void Draw(float menuBottomY, float bottomInset = 0.f);
    ~Camera2DCalibrationWindow();

private:
    struct ImageSlot {
        int width = 0;
        int height = 0;
        std::vector<uint8_t> rgb;
        std::string path;
        unsigned int texId = 0;
        bool valid() const { return texId != 0 && width > 0 && height > 0; }
    };

    struct CalibEntry {
        ImageSlot image;
        float pixelU = 0.f;
        float pixelV = 0.f;
        bool hasPixel = false;
        float robotX = 0.f;
        float robotY = 0.f;
        bool hasRobot = false;
        float roiX0 = 0.f;
        float roiY0 = 0.f;
        float roiX1 = 0.f;
        float roiY1 = 0.f;
        bool hasRoi = false;
        float detectedRadius = 0.f;
        float calibError = -1.f;
    };

    void EnsureDefaultEntries();
    void DestroyImageSlot(ImageSlot& slot);
    bool LoadImageSlot(ImageSlot& slot, const std::string& path, std::string& error);
    bool UploadTexture(ImageSlot& slot);
    void LoadImageForEntry(int index);
    void LoadImagesBatch();
    void AddEntry();
    void RemoveEntry(int index);
    void ClearAll();
    void ComputeCalibration();
    void DetectCircleForEntry(CalibEntry& entry, bool quiet = false);
    void DrawPointTable(float height);
    void DrawImageCanvas();
    void DrawOverlays(ImDrawList* dl, const ImVec2& imgPos, float drawW, float drawH,
                      const CalibEntry& entry) const;
    bool ImageToPixel(const ImageSlot& slot, float imgPosX, float imgPosY, float drawW, float drawH,
                      float mouseX, float mouseY, float& outX, float& outY) const;
    void SetStatus(const char* msg);
    static std::string BaseName(const std::string& path);
    static void NormalizeRoi(float& x0, float& y0, float& x1, float& y1);

    bool visible_ = false;
    bool focusOnOpen_ = false;
    StatusCallback onStatus_;

    std::vector<CalibEntry> entries_;
    int selectedIndex_ = 0;
    bool roiDragging_ = false;
    float zoom_ = 1.f;
    float panX_ = 0.f;
    float panY_ = 0.f;
    int lastCanvasIndex_ = -1;
    float listPanelPreferredW_ = 420.f;
    Camera2DCalibration::AffineResult lastResult_;
    Camera2DCalibration::ErrorStats lastErrorStats_;
    std::string lastError_;
    std::string statusText_;
};
