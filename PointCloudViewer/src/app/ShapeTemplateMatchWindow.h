#pragma once

#include "tools/ShapeTemplateMatch.h"

#include <functional>
#include <string>
#include <vector>

struct ImDrawList;
struct ImVec2;

// 2D 形状模板匹配窗口：读取模板/源图、框选 ROI、创建模板、执行匹配并叠加显示结果。
class ShapeTemplateMatchWindow {
public:
    using StatusCallback = std::function<void(const char*)>;

    // 设置状态栏回调，用于将提示信息同步到主窗口状态栏。
    void SetStatusCallback(StatusCallback cb) { onStatus_ = std::move(cb); }
    // 显示或隐藏本窗口。
    void SetVisible(bool visible);
    bool IsVisible() const { return visible_; }
    // 切换窗口显示状态。
    void ToggleVisible();
    // 绘制整个模板匹配窗口（ImGui 每帧调用）。
    void Draw();

private:
    // 图像槽：内存 RGB 数据 + OpenGL 纹理，用于显示与取色。
    struct ImageSlot {
        int width = 0;
        int height = 0;
        std::vector<uint8_t> rgb;
        std::string path;
        unsigned int texId = 0;
        bool valid() const { return texId != 0 && width > 0 && height > 0; }
    };

    // ROI 框选模式：无 / 模板区域 / 搜索区域。
    enum class RoiMode { None, Template, Search };

    // 从磁盘加载图像到 ImageSlot，并上传为 OpenGL 纹理。
    bool LoadImageSlot(ImageSlot& slot, const std::string& path, std::string& error);
    // 释放 ImageSlot 的纹理与像素数据。
    void DestroyImageSlot(ImageSlot& slot);
    // 将 ImageSlot 中的 RGB 数据上传到 GPU 纹理（无纹理则创建）。
    bool UploadTexture(ImageSlot& slot);
    // 释放模板预览区使用的 OpenGL 纹理。
    void DestroyModelPreviewTexture();
    // 根据当前 model_ 重新生成模板预览纹理。
    void RefreshModelPreviewTexture();
    // 绘制可缩放/平移的图像画布，支持 ROI 拖拽与匹配结果叠加。
    void DrawImageCanvas(ImageSlot& slot, bool allowRoi, RoiMode roiMode, float height,
                         const char* childId);
    // 绘制右侧模板预览小图（含模板 ROI 框）。
    void DrawTemplatePreviewPanel();
    // 在源图上叠加搜索 ROI、匹配轮廓与得分标注。
    void DrawMatchOverlays(ImDrawList* dl, const ImVec2& imgPos, float drawW, float drawH);
    // 将屏幕坐标转换为图像像素坐标（考虑缩放后的绘制区域）。
    bool ImageToPixel(const ImageSlot& slot, const ImVec2& imgPos, float drawW, float drawH,
                      float mouseX, float mouseY, float& outX, float& outY) const;
    // 绘制「创建模板」「寻找模板」两组参数表格。
    void DrawParamColumns();

    // 打开文件对话框，读取模板图像并初始化模板 ROI。
    void ReadTemplateImage();
    // 打开文件对话框，读取待匹配的源图像。
    void ReadSourceImage();
    // 根据模板图像与 ROI、创建参数生成形状模板模型。
    void CreateTemplate();
    // 在源图像上执行模板匹配，结果写入 lastResult_。
    void RunMatch();
    // 将当前模板模型保存到磁盘文件。
    void SaveTemplate();
    // 从磁盘文件加载模板模型并刷新预览。
    void LoadTemplate();
    // 清除当前模板模型、匹配结果与预览纹理。
    void ClearTemplate();
    // 将搜索 ROI 重置为整幅源图并关闭搜索区域限制。
    void ClearSearchRoi();
    // 更新本地状态文本，并触发状态栏回调。
    void SetStatus(const char* msg);

    bool visible_ = false;
    StatusCallback onStatus_;

    ImageSlot templateImage_;   // 模板图像
    ImageSlot sourceImage_;     // 源图像（匹配目标）
    ShapeTemplateMatch::ShapeModel model_;
    ShapeTemplateMatch::CreateParams createParams_;
    ShapeTemplateMatch::FindParams findParams_;
    ShapeTemplateMatch::MatchResult lastResult_;

    unsigned int modelPreviewTexId_ = 0;  // 模板预览纹理
    int modelPreviewW_ = 0;
    int modelPreviewH_ = 0;

    bool useSearchRoi_ = false;   // 是否启用搜索 ROI 限制
    bool roiEnabled_ = true;      // 是否允许框选 ROI
    RoiMode roiMode_ = RoiMode::Template;
    bool roiDragging_ = false;
    float roiX0_ = 0.f;           // 模板 ROI（图像像素坐标）
    float roiY0_ = 0.f;
    float roiX1_ = 0.f;
    float roiY1_ = 0.f;
    float searchRoiX0_ = 0.f;     // 搜索 ROI（图像像素坐标）
    float searchRoiY0_ = 0.f;
    float searchRoiX1_ = 0.f;
    float searchRoiY1_ = 0.f;

    float zoom_ = 1.f;            // 主画布缩放
    float panX_ = 0.f;            // 主画布平移
    float panY_ = 0.f;
    int hoverPx_ = -1;            // 鼠标悬停像素坐标与 RGB
    int hoverPy_ = -1;
    int hoverR_ = 0;
    int hoverG_ = 0;
    int hoverB_ = 0;
    std::string statusText_;
};
