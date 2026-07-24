#pragma once

#include "tools/CameraIntrinsicsCalibration.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

struct ImDrawList;
struct ImVec2;

// 2D 相机内参标定窗口：多张棋盘图像、角点检测、calibrateCamera 求解内参。
class CameraIntrinsicsCalibrationWindow {
public:
    using StatusCallback = std::function<void(const char*)>;

    void SetStatusCallback(StatusCallback cb) { onStatus_ = std::move(cb); }
    void SetVisible(bool visible);
    bool IsVisible() const { return visible_; }
    void ToggleVisible();
    void Draw(float menuBottomY, float bottomInset = 0.f);
    ~CameraIntrinsicsCalibrationWindow();

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
        std::vector<float> cornersX;
        std::vector<float> cornersY;
        bool hasCorners = false;
        float reprojError = -1.f;
    };

    void DestroyImageSlot(ImageSlot& slot);
    bool LoadImageSlot(ImageSlot& slot, const std::string& path, std::string& error);
    bool UploadTexture(ImageSlot& slot);
    void LoadImagesBatch();
    void RemoveEntry(int index);
    void ClearAll();
    void DetectCornersForEntry(CalibEntry& entry, bool quiet = false);
    void DetectCornersAll();
    void ComputeCalibration();
    void DrawImageTable(float height);
    void DrawImageCanvas();
    void DrawCornerOverlay(ImDrawList* dl, const ImVec2& imgPos, float drawW, float drawH,
                           const CalibEntry& entry) const;
    void SetStatus(const char* msg);
    static std::string BaseName(const std::string& path);

    bool visible_ = false;
    bool focusOnOpen_ = false;
    StatusCallback onStatus_;

    std::vector<CalibEntry> entries_;
    int selectedIndex_ = -1;
    float zoom_ = 1.f;
    float panX_ = 0.f;
    float panY_ = 0.f;
    int lastCanvasIndex_ = -1;
    float listPanelPreferredW_ = 400.f;

    CameraIntrinsicsCalibration::PatternConfig pattern_;
    CameraIntrinsicsCalibration::IntrinsicsResult lastResult_;
    std::string lastError_;
    std::string statusText_;
};
