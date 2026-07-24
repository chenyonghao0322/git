#pragma once

#include "tools/MultiViewGeometry.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

struct ImDrawList;
struct ImVec2;

// 多视图几何窗口：两视图对应点、基础矩阵 F、极线显示、三角化。
class MultiViewGeometryWindow {
public:
    using StatusCallback = std::function<void(const char*)>;
    using ImportCloudCallback = std::function<void(const std::vector<Vec3>&, const char*)>;

    void SetStatusCallback(StatusCallback cb) { onStatus_ = std::move(cb); }
    void SetImportCloudCallback(ImportCloudCallback cb) { onImportCloud_ = std::move(cb); }
    void SetVisible(bool visible);
    bool IsVisible() const { return visible_; }
    void ToggleVisible();
    void Draw(float menuBottomY, float bottomInset = 0.f);
    ~MultiViewGeometryWindow();

private:
    struct ImageSlot {
        int width = 0;
        int height = 0;
        std::vector<uint8_t> rgb;
        std::string path;
        unsigned int texId = 0;
        bool valid() const { return texId != 0 && width > 0 && height > 0; }
    };

    struct ViewCanvas {
        float zoom = 1.f;
        float panX = 0.f;
        float panY = 0.f;
    };

    void DestroyImageSlot(ImageSlot& slot);
    bool LoadImageSlot(ImageSlot& slot, const std::string& path, std::string& error);
    bool UploadTexture(ImageSlot& slot);
    void LoadImage1();
    void LoadImage2();
    void ClearPairs();
    void RemovePair(int index);
    void DrawLeftPanel(float tableHeight);
    void DrawImagePane(const char* childId, ImageSlot& slot, ViewCanvas& canvas, int viewIndex,
                       float width, float height);
    void DrawEpipolarOverlay(ImDrawList* dl, const ImVec2& imgPos, float drawW, float drawH,
                             int viewIndex) const;
    bool ImageToPixel(const ImageSlot& slot, float imgPosX, float imgPosY, float drawW, float drawH,
                      float mouseX, float mouseY, float& outX, float& outY) const;
    void ComputeFundamental();
    void Triangulate();
    void ImportToScene();
    void SetStatus(const char* msg);
    static std::string BaseName(const std::string& path);

    bool visible_ = false;
    bool focusOnOpen_ = false;
    StatusCallback onStatus_;
    ImportCloudCallback onImportCloud_;

    ImageSlot image1_;
    ImageSlot image2_;
    ViewCanvas canvas1_;
    ViewCanvas canvas2_;
    std::vector<MultiViewGeometry::Correspondence> pairs_;
    int selectedPair_ = -1;
    bool hasPendingView1_ = false;
    float pendingU1_ = 0.f;
    float pendingV1_ = 0.f;

    MultiViewGeometry::CameraIntrinsics intrinsics_;
    MultiViewGeometry::FundamentalResult fundamental_;
    MultiViewGeometry::TriangulationResult triangulation_;
    std::string lastError_;
    std::string statusText_;
    float listPanelPreferredW_ = 380.f;
    float ransacThresh_ = 1.0f;
};
