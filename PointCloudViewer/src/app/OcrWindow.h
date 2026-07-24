#pragma once

#include "tools/OcrTools.h"
#include "tools/PaddleOcrTools.h"

#include <functional>
#include <string>
#include <vector>

struct ImDrawList;
struct ImVec2;

// OCR 识别窗口：读取图像、框选 ROI、预处理并调用 Tesseract 识别。
class OcrWindow {
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

    bool LoadImageSlot(ImageSlot& slot, const std::string& path, std::string& error);
    void DestroyImageSlot(ImageSlot& slot);
    bool UploadTexture(ImageSlot& slot);
    void DestroyPreviewTexture();
    void RefreshPreviewTexture();
    void DrawImageCanvas();
    void DrawOcrOverlays(ImDrawList* dl, const ImVec2& imgPos, float drawW, float drawH);
    bool ImageToPixel(float imgPosX, float imgPosY, float drawW, float drawH, float mouseX,
                      float mouseY, float& outX, float& outY) const;
    void DrawEngineBar();
    void DrawTesseractParamPanel();
    void DrawPaddleParamPanel();
    void DrawParamPanel();
    void DrawResultPanel();

    void ReadImage();
    void RunRecognize();
    void RunCliRecognize();
    void RunPlateRecognize();
    void ClearRoi();
    void ClearResults();
    void SetStatus(const char* msg);
    void NormalizeRoi();
    bool HasValidRoi() const;

    bool visible_ = false;
    bool focusOnOpen_ = false;
    StatusCallback onStatus_;

    ImageSlot image_;
    OcrTools::OcrEngine engine_ = OcrTools::OcrEngine::Tesseract;
    OcrTools::PreprocessParams prepParams_;
    OcrTools::RecognizeParams recParams_;
    PaddleOcrTools::PaddleParams paddleParams_;
    OcrTools::OcrResult lastResult_;

    unsigned int previewTexId_ = 0;
    int previewW_ = 0;
    int previewH_ = 0;
    OcrTools::PreprocessPreview lastPreview_;

    bool useRoi_ = false;
    bool roiDragging_ = false;
    float roiX0_ = 0.f;
    float roiY0_ = 0.f;
    float roiX1_ = 0.f;
    float roiY1_ = 0.f;

    bool showBoxes_ = true;
    bool showConfidence_ = true;
    bool showPreprocessPreview_ = false;

    float zoom_ = 1.f;
    int hoverPx_ = -1;
    int hoverPy_ = -1;
    int hoverR_ = 0;
    int hoverG_ = 0;
    int hoverB_ = 0;
    std::string statusText_;
    char langBuf_[64] = "chi_sim";
    char paddleLangBuf_[16] = "ch";
    char whiteBuf_[128]{};
    char blackBuf_[128]{};
};
