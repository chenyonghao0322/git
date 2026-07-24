#include "app/ShapeTemplateMatchWindow.h"

#include "app/FileDialog.h"
#include "io/ImageIO.h"

#include <glad/gl.h>
#include <imgui.h>

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace {
constexpr ImU32 kContourCol = IM_COL32(40, 220, 80, 255);
constexpr ImU32 kScoreCol = IM_COL32(255, 60, 60, 255);
constexpr ImU32 kRoiCol = IM_COL32(80, 180, 255, 220);

// 按轮廓分段将模型边缘折线绘制到 ImGui 画布上（用于匹配结果叠加）。
void DrawContourPolylines(ImDrawList* dl, const std::vector<float>& xs,
                          const std::vector<float>& ys, const std::vector<int>& starts,
                          float originX, float originY, float sx, float sy, ImU32 col,
                          float thickness) {
    if (xs.size() < 2 || ys.size() != xs.size()) return;
    std::vector<int> segs = starts;
    if (segs.empty()) segs.push_back(0);
    segs.push_back(static_cast<int>(xs.size()));
    for (std::size_t s = 0; s + 1 < segs.size(); ++s) {
        const int i0 = segs[s];
        const int i1 = segs[s + 1];
        for (int i = i0 + 1; i < i1; ++i) {
            dl->AddLine(ImVec2(originX + xs[static_cast<std::size_t>(i - 1)] * sx,
                               originY + ys[static_cast<std::size_t>(i - 1)] * sy),
                        ImVec2(originX + xs[static_cast<std::size_t>(i)] * sx,
                               originY + ys[static_cast<std::size_t>(i)] * sy),
                        col, thickness);
        }
    }
}
}  // namespace

// 设置窗口可见性。
void ShapeTemplateMatchWindow::SetVisible(bool visible) { visible_ = visible; }

// 切换窗口显示/隐藏。
void ShapeTemplateMatchWindow::ToggleVisible() { visible_ = !visible_; }

// 更新状态文本，并通知主窗口状态栏。
void ShapeTemplateMatchWindow::SetStatus(const char* msg) {
    statusText_ = msg ? msg : "";
    if (onStatus_) onStatus_(msg);
}

// 释放图像槽的 OpenGL 纹理并清空像素数据。
void ShapeTemplateMatchWindow::DestroyImageSlot(ImageSlot& slot) {
    if (slot.texId) {
        glDeleteTextures(1, &slot.texId);
        slot.texId = 0;
    }
    slot.width = 0;
    slot.height = 0;
    slot.rgb.clear();
    slot.path.clear();
}

// 释放模板预览纹理。
void ShapeTemplateMatchWindow::DestroyModelPreviewTexture() {
    if (modelPreviewTexId_) {
        glDeleteTextures(1, &modelPreviewTexId_);
        modelPreviewTexId_ = 0;
    }
    modelPreviewW_ = 0;
    modelPreviewH_ = 0;
}

// 从 model_ 的预览图或模板灰度图生成 OpenGL 预览纹理。
void ShapeTemplateMatchWindow::RefreshModelPreviewTexture() {
    DestroyModelPreviewTexture();
    if (!model_.valid) return;

    const uint8_t* data = nullptr;
    int w = 0;
    int h = 0;
    if (!model_.previewRgb.empty() && model_.templateW > 0 && model_.templateH > 0) {
        data = model_.previewRgb.data();
        w = model_.templateW;
        h = model_.templateH;
    } else if (!model_.templateRgb.empty() && model_.templateW > 0 && model_.templateH > 0) {
        data = model_.templateRgb.data();
        w = model_.templateW;
        h = model_.templateH;
    }
    if (!data || w <= 0 || h <= 0) return;

    glGenTextures(1, &modelPreviewTexId_);
    glBindTexture(GL_TEXTURE_2D, modelPreviewTexId_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
    glBindTexture(GL_TEXTURE_2D, 0);
    modelPreviewW_ = w;
    modelPreviewH_ = h;
}

// 将 ImageSlot 的 RGB 像素上传到 GPU 纹理。
bool ShapeTemplateMatchWindow::UploadTexture(ImageSlot& slot) {
    if (slot.rgb.empty() || slot.width <= 0 || slot.height <= 0) return false;
    if (slot.texId == 0) glGenTextures(1, &slot.texId);
    glBindTexture(GL_TEXTURE_2D, slot.texId);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, slot.width, slot.height, 0, GL_RGB, GL_UNSIGNED_BYTE,
                 slot.rgb.data());
    glBindTexture(GL_TEXTURE_2D, 0);
    return true;
}

// 从文件路径加载图像，替换 slot 内容并创建显示纹理。
bool ShapeTemplateMatchWindow::LoadImageSlot(ImageSlot& slot, const std::string& path,
                                             std::string& error) {
    ImageIO::RgbImage img;
    if (!ImageIO::LoadRgb(path, img, error)) return false;
    DestroyImageSlot(slot);
    slot.width = img.width;
    slot.height = img.height;
    slot.rgb = std::move(img.rgb);
    slot.path = path;
    return UploadTexture(slot);
}

// 鼠标屏幕坐标 → 图像像素坐标（仅在绘制区域内有效）。
bool ShapeTemplateMatchWindow::ImageToPixel(const ImageSlot& slot, const ImVec2& imgPos, float drawW,
                                            float drawH, float mouseX, float mouseY, float& outX,
                                            float& outY) const {
    if (!slot.valid()) return false;
    if (mouseX < imgPos.x || mouseY < imgPos.y || mouseX > imgPos.x + drawW ||
        mouseY > imgPos.y + drawH) {
        return false;
    }
    outX = (mouseX - imgPos.x) / drawW * static_cast<float>(slot.width);
    outY = (mouseY - imgPos.y) / drawH * static_cast<float>(slot.height);
    outX = std::clamp(outX, 0.f, static_cast<float>(slot.width - 1));
    outY = std::clamp(outY, 0.f, static_cast<float>(slot.height - 1));
    return true;
}

// 在源图上绘制搜索 ROI 矩形、匹配轮廓线与得分/缩放文字。
void ShapeTemplateMatchWindow::DrawMatchOverlays(ImDrawList* dl, const ImVec2& imgPos, float drawW,
                                                 float drawH) {
    if (!sourceImage_.valid()) return;
    const float sx = drawW / static_cast<float>(sourceImage_.width);
    const float sy = drawH / static_cast<float>(sourceImage_.height);

    if (useSearchRoi_) {
        const float x0 = imgPos.x + searchRoiX0_ * sx;
        const float y0 = imgPos.y + searchRoiY0_ * sy;
        const float x1 = imgPos.x + searchRoiX1_ * sx;
        const float y1 = imgPos.y + searchRoiY1_ * sy;
        dl->AddRect(ImVec2(std::min(x0, x1), std::min(y0, y1)),
                    ImVec2(std::max(x0, x1), std::max(y0, y1)), IM_COL32(255, 200, 60, 200), 0.f,
                    0, 1.5f);
    }

    for (std::size_t hi = 0; hi < lastResult_.hits.size(); ++hi) {
        if (findParams_.displayMode == 1 && hi > 0) continue;
        const ShapeTemplateMatch::MatchHit& hit = lastResult_.hits[hi];
        DrawContourPolylines(dl, hit.contourX, hit.contourY, model_.contourStarts, imgPos.x,
                             imgPos.y, sx, sy, kContourCol, 2.f);
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.3f", hit.score);
        dl->AddText(ImVec2(imgPos.x + hit.centerX * sx - 12.f, imgPos.y + hit.centerY * sy - 22.f),
                    kScoreCol, buf);
        std::snprintf(buf, sizeof(buf), "%.2fx%.2fx", hit.scale, hit.scale2);
        dl->AddText(ImVec2(imgPos.x + hit.centerX * sx - 20.f, imgPos.y + hit.centerY * sy + 6.f),
                    IM_COL32(20, 20, 20, 255), buf);
    }
}

// 主图像显示区：自适应缩放、Shift+滚轮缩放、ROI 拖拽、悬停取色、匹配叠加。
void ShapeTemplateMatchWindow::DrawImageCanvas(ImageSlot& slot, bool allowRoi, RoiMode roiMode,
                                               float height, const char* childId) {
    if (!slot.valid()) {
        ImGui::TextDisabled(u8"尚未加载图像");
        return;
    }

    const ImVec2 avail = ImGui::GetContentRegionAvail();
    const float areaH = height > 0.f ? height : std::max(avail.y, 200.f);
    ImGui::BeginChild(childId, ImVec2(avail.x, areaH), ImGuiChildFlags_Borders);

    const float imgAspect =
        (slot.height > 0) ? static_cast<float>(slot.width) / static_cast<float>(slot.height) : 1.f;
    float fitW = std::max(avail.x - 8.f, 1.f);
    float fitH = fitW / imgAspect;
    if (fitH > areaH - 8.f) {
        fitH = std::max(areaH - 8.f, 1.f);
        fitW = fitH * imgAspect;
    }
    const float drawW = std::max(fitW * zoom_, 1.f);
    const float drawH = std::max(fitH * zoom_, 1.f);
    const float basePanX = std::max((avail.x - drawW) * 0.5f, 0.f);
    const float basePanY = std::max((areaH - drawH) * 0.5f, 0.f);

    ImGuiIO& io = ImGui::GetIO();
    if (ImGui::IsWindowHovered() && io.KeyShift && io.MouseWheel != 0.f) {
        zoom_ = std::clamp(zoom_ * (1.f + io.MouseWheel * 0.12f), 0.1f, 16.f);
    }

    ImGui::SetCursorPos(ImVec2(basePanX + panX_, basePanY + panY_));
    const ImVec2 imgPos = ImGui::GetCursorScreenPos();
    ImGui::Image((ImTextureID)(intptr_t)slot.texId, ImVec2(drawW, drawH));

    ImDrawList* dl = ImGui::GetWindowDrawList();
    if (allowRoi && roiEnabled_ && roiMode_ == roiMode) {
        const float sx = drawW / static_cast<float>(slot.width);
        const float sy = drawH / static_cast<float>(slot.height);
        float* rx0 = (roiMode == RoiMode::Template) ? &roiX0_ : &searchRoiX0_;
        float* ry0 = (roiMode == RoiMode::Template) ? &roiY0_ : &searchRoiY0_;
        float* rx1 = (roiMode == RoiMode::Template) ? &roiX1_ : &searchRoiX1_;
        float* ry1 = (roiMode == RoiMode::Template) ? &roiY1_ : &searchRoiY1_;
        if (*rx1 > *rx0 && *ry1 > *ry0) {
            dl->AddRect(ImVec2(imgPos.x + *rx0 * sx, imgPos.y + *ry0 * sy),
                        ImVec2(imgPos.x + *rx1 * sx, imgPos.y + *ry1 * sy), kRoiCol, 0.f, 0, 2.f);
        }
    }

    if (slot.texId == sourceImage_.texId) {
        DrawMatchOverlays(dl, imgPos, drawW, drawH);
    }

    if (ImGui::IsItemHovered()) {
        float px = 0.f;
        float py = 0.f;
        if (ImageToPixel(slot, imgPos, drawW, drawH, io.MousePos.x, io.MousePos.y, px, py)) {
            hoverPx_ = static_cast<int>(px);
            hoverPy_ = static_cast<int>(py);
            const std::size_t idx =
                (static_cast<std::size_t>(hoverPy_) * static_cast<std::size_t>(slot.width) +
                 static_cast<std::size_t>(hoverPx_)) *
                3u;
            if (idx + 2 < slot.rgb.size()) {
                hoverR_ = slot.rgb[idx];
                hoverG_ = slot.rgb[idx + 1];
                hoverB_ = slot.rgb[idx + 2];
            }
        }
        if (allowRoi && roiEnabled_ && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            roiMode_ = roiMode;
            roiDragging_ = true;
            float px0 = 0.f;
            float py0 = 0.f;
            ImageToPixel(slot, imgPos, drawW, drawH, io.MousePos.x, io.MousePos.y, px0, py0);
            if (roiMode == RoiMode::Template) {
                roiX0_ = roiX1_ = px0;
                roiY0_ = roiY1_ = py0;
            } else {
                searchRoiX0_ = searchRoiX1_ = px0;
                searchRoiY0_ = searchRoiY1_ = py0;
                useSearchRoi_ = true;
            }
        }
        if (roiDragging_ && roiMode_ == roiMode && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
            float px1 = 0.f;
            float py1 = 0.f;
            ImageToPixel(slot, imgPos, drawW, drawH, io.MousePos.x, io.MousePos.y, px1, py1);
            if (roiMode == RoiMode::Template) {
                roiX1_ = px1;
                roiY1_ = py1;
            } else {
                searchRoiX1_ = px1;
                searchRoiY1_ = py1;
            }
        }
    }
    if (roiDragging_ && ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
        roiDragging_ = false;
    }

    ImGui::EndChild();
}

// 右下角的模板预览：优先显示已创建模型的预览图，否则显示原模板图并支持框选 ROI。
void ShapeTemplateMatchWindow::DrawTemplatePreviewPanel() {
    ImGui::TextDisabled(u8"模板预览");
    const float panelH = 168.f;
    ImGui::BeginChild("##tpl_prev", ImVec2(0, panelH), ImGuiChildFlags_Borders);

    if (model_.valid && modelPreviewTexId_ != 0) {
        const float prevW = ImGui::GetContentRegionAvail().x;
        const float aspect =
            static_cast<float>(modelPreviewW_) / static_cast<float>(std::max(modelPreviewH_, 1));
        float drawW = prevW;
        float drawH = drawW / aspect;
        if (drawH > panelH - 12.f) {
            drawH = panelH - 12.f;
            drawW = drawH * aspect;
        }
        ImGui::Image((ImTextureID)(intptr_t)modelPreviewTexId_, ImVec2(drawW, drawH));
        ImGui::EndChild();
        return;
    }

    if (templateImage_.valid()) {
        const float prevW = ImGui::GetContentRegionAvail().x;
        const float aspect = static_cast<float>(templateImage_.width) /
                             static_cast<float>(std::max(templateImage_.height, 1));
        float drawW = prevW;
        float drawH = drawW / aspect;
        if (drawH > panelH - 12.f) {
            drawH = panelH - 12.f;
            drawW = drawH * aspect;
        }
        const ImVec2 imgPos = ImGui::GetCursorScreenPos();
        ImGui::Image((ImTextureID)(intptr_t)templateImage_.texId, ImVec2(drawW, drawH));

        ImDrawList* dl = ImGui::GetWindowDrawList();
        const float sx = drawW / static_cast<float>(templateImage_.width);
        const float sy = drawH / static_cast<float>(templateImage_.height);
        if (roiEnabled_ && roiMode_ == RoiMode::Template && roiX1_ > roiX0_ && roiY1_ > roiY0_) {
            dl->AddRect(ImVec2(imgPos.x + roiX0_ * sx, imgPos.y + roiY0_ * sy),
                        ImVec2(imgPos.x + roiX1_ * sx, imgPos.y + roiY1_ * sy), kRoiCol, 0.f, 0,
                        2.f);
        }

        ImGuiIO& io = ImGui::GetIO();
        if (ImGui::IsItemHovered() && roiEnabled_) {
            if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                roiMode_ = RoiMode::Template;
                roiDragging_ = true;
                float px0 = 0.f;
                float py0 = 0.f;
                ImageToPixel(templateImage_, imgPos, drawW, drawH, io.MousePos.x, io.MousePos.y, px0,
                             py0);
                roiX0_ = roiX1_ = px0;
                roiY0_ = roiY1_ = py0;
            }
            if (roiDragging_ && roiMode_ == RoiMode::Template &&
                ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
                float px1 = 0.f;
                float py1 = 0.f;
                ImageToPixel(templateImage_, imgPos, drawW, drawH, io.MousePos.x, io.MousePos.y, px1,
                             py1);
                roiX1_ = px1;
                roiY1_ = py1;
            }
        }
        ImGui::EndChild();
        return;
    }

    ImGui::TextDisabled(u8"（读取模板图像后在此框选 ROI）");
    ImGui::EndChild();
}

// 绘制创建模板与寻找模板两组参数控件（表格布局）。
void ShapeTemplateMatchWindow::DrawParamColumns() {
    ImGui::TextDisabled(u8"创建模板");
    if (ImGui::BeginTable("##create_params", 4,
                          ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_PadOuterX)) {
        ImGui::TableSetupColumn("lbl1", ImGuiTableColumnFlags_WidthFixed, 78.f);
        ImGui::TableSetupColumn("val1", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("lbl2", ImGuiTableColumnFlags_WidthFixed, 78.f);
        ImGui::TableSetupColumn("val2", ImGuiTableColumnFlags_WidthStretch);

        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted(u8"对比度(低)");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip(u8"Canny 边缘检测低阈值，越小提取的边缘越多");
        }
        ImGui::TableNextColumn();
        ImGui::SetNextItemWidth(-1);
        ImGui::DragInt("##create_clow", &createParams_.contrastLow, 1, 1, 255);
        ImGui::TableNextColumn();
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted(u8"对比度(高)");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip(u8"Canny 边缘检测高阈值，与低阈值共同控制边缘灵敏度");
        }
        ImGui::TableNextColumn();
        ImGui::SetNextItemWidth(-1);
        ImGui::DragInt("##create_chigh", &createParams_.contrastHigh, 1, 1, 255);

        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted(u8"金字塔");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip(u8"创建模板金字塔层数；0=自动。若「终止金字塔」为 0 则用于搜索");
        }
        ImGui::TableNextColumn();
        ImGui::SetNextItemWidth(-1);
        ImGui::DragInt("##create_pyr", &createParams_.pyramid, 1, 0, 6);
        ImGui::TableNextColumn();
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted(u8"最小对比度");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip(u8"忽略低于此灰度对比度的边缘点");
        ImGui::TableNextColumn();
        ImGui::SetNextItemWidth(-1);
        ImGui::DragInt("##create_minc", &createParams_.minContrast, 1, 0, 255);

        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted(u8"长半轴");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip(u8"创建模板前的高斯平滑长半轴，0=默认 3×3");
        }
        ImGui::TableNextColumn();
        ImGui::SetNextItemWidth(-1);
        ImGui::DragFloat("##create_sal", &createParams_.semiAxisLong, 0.1f, 0.f, 20.f, "%.1f");
        ImGui::TableNextColumn();
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted(u8"短半轴");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip(u8"高斯平滑短半轴，0=与长半轴相同");
        }
        ImGui::TableNextColumn();
        ImGui::SetNextItemWidth(-1);
        ImGui::DragFloat("##create_sas", &createParams_.semiAxisShort, 0.1f, 0.f, 20.f, "%.1f");

        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted(u8"最小组件");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip(u8"连通域面积小于此值的边缘区域将被滤除");
        }
        ImGui::TableNextColumn();
        ImGui::SetNextItemWidth(-1);
        ImGui::DragInt("##create_minsz", &createParams_.minComponentSize, 1, 1, 100);
        ImGui::TableNextColumn();
        ImGui::TableNextColumn();
        ImGui::AlignTextToFramePadding();
        ImGui::Checkbox(u8"使用极性##create_pol", &createParams_.usePolarity);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip(u8"匹配时考虑边缘梯度方向（亮到暗的极性）");
        }
        ImGui::EndTable();
    }

    ImGui::Spacing();
    ImGui::TextDisabled(u8"寻找模板");
    if (ImGui::BeginTable("##find_params", 4,
                          ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_PadOuterX)) {
        ImGui::TableSetupColumn("lbl1", ImGuiTableColumnFlags_WidthFixed, 78.f);
        ImGui::TableSetupColumn("val1", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("lbl2", ImGuiTableColumnFlags_WidthFixed, 78.f);
        ImGui::TableSetupColumn("val2", ImGuiTableColumnFlags_WidthStretch);

        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted(u8"起始角度");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip(u8"搜索旋转的起始角度（度）");
        ImGui::TableNextColumn();
        ImGui::SetNextItemWidth(-1);
        ImGui::DragFloat("##find_ang0", &findParams_.angleStart, 1.f, -360.f, 360.f, "%.0f");
        ImGui::TableNextColumn();
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted(u8"角度范围");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip(u8"从起始角度开始搜索的角度跨度（度）");
        ImGui::TableNextColumn();
        ImGui::SetNextItemWidth(-1);
        ImGui::DragFloat("##find_angr", &findParams_.angleRange, 1.f, 0.f, 360.f, "%.0f");

        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted(u8"最小比例");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip(u8"模板最小缩放比例");
        ImGui::TableNextColumn();
        ImGui::SetNextItemWidth(-1);
        ImGui::DragFloat("##find_smin", &findParams_.scaleMin, 0.01f, 0.1f, 4.f, "%.2f");
        ImGui::TableNextColumn();
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted(u8"最大比例");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip(u8"模板最大缩放比例");
        ImGui::TableNextColumn();
        ImGui::SetNextItemWidth(-1);
        ImGui::DragFloat("##find_smax", &findParams_.scaleMax, 0.01f, 0.1f, 4.f, "%.2f");

        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted(u8"最小得分");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip(u8"匹配得分低于此值的结果将被丢弃（0~1）");
        ImGui::TableNextColumn();
        ImGui::SetNextItemWidth(-1);
        ImGui::DragFloat("##find_score", &findParams_.minScore, 0.01f, 0.f, 1.f, "%.2f");
        ImGui::TableNextColumn();
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted(u8"最大重叠");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip(u8"多个匹配结果之间的最大允许重叠比例");
        ImGui::TableNextColumn();
        ImGui::SetNextItemWidth(-1);
        ImGui::DragFloat("##find_overlap", &findParams_.maxOverlap, 0.01f, 0.f, 1.f, "%.2f");

        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted(u8"贪婪度");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip(u8"越大粗搜剪枝越强、速度越快，但可能漏检");
        }
        ImGui::TableNextColumn();
        ImGui::SetNextItemWidth(-1);
        ImGui::DragFloat("##find_greed", &findParams_.greediness, 0.01f, 0.f, 1.f, "%.2f");
        ImGui::TableNextColumn();
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted(u8"目标数量");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip(u8"最多返回的匹配个数，0 表示自动（最多 32 个）");
        ImGui::TableNextColumn();
        ImGui::SetNextItemWidth(-1);
        ImGui::DragInt("##find_num", &findParams_.numMatches, 1, 0, 32);

        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted(u8"终止金字塔");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip(u8"搜索金字塔层数；创建金字塔为 0 时作为备用");
        ImGui::TableNextColumn();
        ImGui::SetNextItemWidth(-1);
        ImGui::DragInt("##find_pyr", &findParams_.endPyramid, 1, 0, 4);
        ImGui::TableNextColumn();
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted(u8"显示");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip(u8"叠加显示全部匹配或仅最佳");
        ImGui::TableNextColumn();
        ImGui::SetNextItemWidth(-1);
        const char* displayItems[] = {u8"全部", u8"仅最佳"};
        ImGui::Combo("##find_disp", &findParams_.displayMode, displayItems, 2);

        if (!findParams_.isotropicScale) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::AlignTextToFramePadding();
            ImGui::TextUnformatted(u8"最小比例Y");
            ImGui::TableNextColumn();
            ImGui::SetNextItemWidth(-1);
            ImGui::DragFloat("##find_s2min", &findParams_.scale2Min, 0.01f, 0.1f, 4.f, "%.2f");
            ImGui::TableNextColumn();
            ImGui::AlignTextToFramePadding();
            ImGui::TextUnformatted(u8"最大比例Y");
            ImGui::TableNextColumn();
            ImGui::SetNextItemWidth(-1);
            ImGui::DragFloat("##find_s2max", &findParams_.scale2Max, 0.01f, 0.1f, 4.f, "%.2f");
        }

        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::AlignTextToFramePadding();
        ImGui::Checkbox(u8"亚像素##find_sub", &findParams_.subPixel);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip(u8"对匹配位置做亚像素级精修");
        ImGui::TableNextColumn();
        ImGui::TableNextColumn();
        ImGui::AlignTextToFramePadding();
        const char* scaleModes[] = {u8"各向同性", u8"各向异性"};
        int scaleMode = findParams_.isotropicScale ? 0 : 1;
        ImGui::SetNextItemWidth(-1);
        if (ImGui::Combo(u8"缩放方式##find_scale_mode", &scaleMode, scaleModes, 2)) {
            findParams_.isotropicScale = (scaleMode == 0);
        }
        ImGui::TableNextColumn();

        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::AlignTextToFramePadding();
        ImGui::Checkbox(u8"与边缘相交##find_border", &findParams_.borderIntersect);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip(u8"允许匹配结果贴搜索区域边缘；关闭可抑制边缘误检");
        }
        ImGui::TableNextColumn();
        ImGui::EndTable();
    }
}

// 读取模板图像，默认 ROI 为整图，并清除已有模型。
void ShapeTemplateMatchWindow::ReadTemplateImage() {
    const std::string path = FileDialog::OpenImageFile(u8"读取模板图像");
    if (path.empty()) return;
    std::string error;
    if (!LoadImageSlot(templateImage_, path, error)) {
        SetStatus(error.c_str());
        return;
    }
    roiX0_ = roiY0_ = 0.f;
    roiX1_ = static_cast<float>(templateImage_.width - 1);
    roiY1_ = static_cast<float>(templateImage_.height - 1);
    roiMode_ = RoiMode::Template;
    model_ = {};
    DestroyModelPreviewTexture();
    char buf[256];
    std::snprintf(buf, sizeof(buf), u8"已读取模板图像 %dx%d", templateImage_.width,
                  templateImage_.height);
    SetStatus(buf);
}

// 读取源图像，重置搜索 ROI 与上次匹配结果。
void ShapeTemplateMatchWindow::ReadSourceImage() {
    const std::string path = FileDialog::OpenImageFile(u8"读取源图像");
    if (path.empty()) return;
    std::string error;
    if (!LoadImageSlot(sourceImage_, path, error)) {
        SetStatus(error.c_str());
        return;
    }
    searchRoiX0_ = searchRoiY0_ = 0.f;
    searchRoiX1_ = static_cast<float>(sourceImage_.width - 1);
    searchRoiY1_ = static_cast<float>(sourceImage_.height - 1);
    useSearchRoi_ = false;
    lastResult_ = {};
    char buf[256];
    std::snprintf(buf, sizeof(buf), u8"已读取源图像 %dx%d", sourceImage_.width, sourceImage_.height);
    SetStatus(buf);
}

// 调用 ShapeTemplateMatch 根据模板图 ROI 与参数创建形状模型。
void ShapeTemplateMatchWindow::CreateTemplate() {
    if (!templateImage_.valid()) {
        SetStatus(u8"请先读取模板图像");
        return;
    }
    std::string error;
    if (!ShapeTemplateMatch::CreateModel(templateImage_.rgb, templateImage_.width,
                                         templateImage_.height, roiX0_, roiY0_, roiX1_, roiY1_,
                                         createParams_, model_, error)) {
        SetStatus(error.c_str());
        return;
    }
    RefreshModelPreviewTexture();
    char buf[128];
    std::snprintf(buf, sizeof(buf), u8"模板创建成功（%d×%d，轮廓点 %zu）", model_.templateW,
                  model_.templateH, model_.contourX.size());
    SetStatus(buf);
}

// 在源图上执行模板匹配，结果保存到 lastResult_ 供叠加显示。
void ShapeTemplateMatchWindow::RunMatch() {
    if (!sourceImage_.valid()) {
        SetStatus(u8"请先读取源图像");
        return;
    }
    if (!model_.valid) {
        SetStatus(u8"请先创建或加载模板");
        return;
    }
    std::string error;
    if (!ShapeTemplateMatch::FindModel(
            sourceImage_.rgb, sourceImage_.width, sourceImage_.height, searchRoiX0_, searchRoiY0_,
            searchRoiX1_, searchRoiY1_, useSearchRoi_, model_, findParams_, lastResult_, error)) {
        lastResult_ = {};
        SetStatus(error.c_str());
        return;
    }
    char buf[128];
    std::snprintf(buf, sizeof(buf), u8"匹配完成：%zu 个目标，耗时 %.1f ms", lastResult_.hits.size(),
                  lastResult_.elapsedMs);
    SetStatus(buf);
}

// 将当前模型序列化保存到用户选择的文件。
void ShapeTemplateMatchWindow::SaveTemplate() {
    const std::string path = FileDialog::SaveShapeTemplateFile();
    if (path.empty()) return;
    std::string error;
    if (!ShapeTemplateMatch::SaveModel(path, model_, error)) {
        SetStatus(error.c_str());
        return;
    }
    char buf[256];
    std::snprintf(buf, sizeof(buf), u8"模板已保存：%s", path.c_str());
    SetStatus(buf);
}

// 从文件反序列化加载模板模型并刷新预览。
void ShapeTemplateMatchWindow::LoadTemplate() {
    const std::string path = FileDialog::OpenShapeTemplateFile();
    if (path.empty()) return;
    std::string error;
    if (!ShapeTemplateMatch::LoadModel(path, model_, error)) {
        SetStatus(error.c_str());
        return;
    }
    RefreshModelPreviewTexture();
    SetStatus(u8"模板加载成功");
}

// 清除模型、匹配结果与预览纹理，不影响已加载的图像。
void ShapeTemplateMatchWindow::ClearTemplate() {
    model_ = {};
    lastResult_ = {};
    DestroyModelPreviewTexture();
    SetStatus(u8"已清除模板");
}

// 搜索 ROI 恢复为整幅源图，并关闭 useSearchRoi_ 限制。
void ShapeTemplateMatchWindow::ClearSearchRoi() {
    if (sourceImage_.valid()) {
        searchRoiX0_ = searchRoiY0_ = 0.f;
        searchRoiX1_ = static_cast<float>(sourceImage_.width - 1);
        searchRoiY1_ = static_cast<float>(sourceImage_.height - 1);
    }
    useSearchRoi_ = false;
}

// 窗口主入口：左侧大图 + 右侧参数/按钮/预览，底部状态与像素信息。
void ShapeTemplateMatchWindow::Draw() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(1200.f, 780.f), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin(u8"2D 模板匹配", &visible_)) {
        ImGui::End();
        return;
    }

    const float rightW = 420.f;
    const float bottomH = 24.f;
    const ImVec2 contentAvail = ImGui::GetContentRegionAvail();

    ImGui::BeginChild("##main_row", ImVec2(0, contentAvail.y - bottomH), false);
    ImGui::BeginChild("##left_view", ImVec2(ImGui::GetContentRegionAvail().x - rightW, 0), false);

    ImageSlot* mainSlot = nullptr;
    bool allowRoi = false;
    RoiMode mainRoi = RoiMode::Search;
    const char* mainLabel = u8"源图像";
    if (sourceImage_.valid()) {
        mainSlot = &sourceImage_;
        allowRoi = true;
        mainRoi = RoiMode::Search;
        mainLabel = u8"源图像（匹配结果）";
    } else if (templateImage_.valid()) {
        mainSlot = &templateImage_;
        allowRoi = true;
        mainRoi = RoiMode::Template;
        mainLabel = u8"模板图像（框选 ROI）";
    }

    ImGui::TextDisabled("%s", mainLabel);
    if (mainSlot) {
        DrawImageCanvas(*mainSlot, allowRoi, mainRoi, 0.f, "##main_canvas");
    } else {
        ImGui::BeginChild("##main_canvas", ImVec2(0, std::max(ImGui::GetContentRegionAvail().y, 200.f)),
                          ImGuiChildFlags_Borders);
        ImGui::TextDisabled(u8"请读取源图像或模板图像");
        ImGui::EndChild();
    }
    ImGui::EndChild();

    ImGui::SameLine();
    ImGui::BeginChild("##right_panel", ImVec2(rightW, 0), ImGuiChildFlags_Borders);
    DrawParamColumns();

    ImGui::Spacing();
    ImGui::Checkbox(u8"ROI", &roiEnabled_);
    ImGui::SameLine();
    if (ImGui::RadioButton(u8"模板", roiMode_ == RoiMode::Template)) {
        roiMode_ = RoiMode::Template;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton(u8"搜索", roiMode_ == RoiMode::Search)) {
        roiMode_ = RoiMode::Search;
    }
    ImGui::SameLine();
    if (ImGui::Button(u8"CLS")) {
        if (roiMode_ == RoiMode::Template && templateImage_.valid()) {
            roiX0_ = roiY0_ = 0.f;
            roiX1_ = static_cast<float>(templateImage_.width - 1);
            roiY1_ = static_cast<float>(templateImage_.height - 1);
        } else {
            ClearSearchRoi();
        }
    }

    ImGui::Spacing();
    const float btnW = (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x * 2.f) / 3.f;
    const ImVec2 btnSize(btnW, 28.f);
    if (ImGui::Button(u8"读取模板图像", btnSize)) ReadTemplateImage();
    ImGui::SameLine();
    if (ImGui::Button(u8"创建模板", btnSize)) CreateTemplate();
    ImGui::SameLine();
    if (ImGui::Button(u8"加载模板", btnSize)) LoadTemplate();

    if (ImGui::Button(u8"读取源图像", btnSize)) ReadSourceImage();
    ImGui::SameLine();
    if (ImGui::Button(u8"模板匹配", btnSize)) RunMatch();
    ImGui::SameLine();
    if (ImGui::Button(u8"保存模板", btnSize)) SaveTemplate();

    if (ImGui::Button(u8"清除搜索ROI", btnSize)) ClearSearchRoi();
    ImGui::SameLine();
    if (ImGui::Button(u8"清除模板", btnSize)) ClearTemplate();
    ImGui::SameLine();
    ImGui::BeginDisabled();
    ImGui::Button(u8"—", btnSize);
    ImGui::EndDisabled();

    ImGui::Spacing();
    DrawTemplatePreviewPanel();
    ImGui::EndChild();
    ImGui::EndChild();

    ImGui::Separator();
    if (sourceImage_.valid() || templateImage_.valid()) {
        const ImageSlot& infoSlot = sourceImage_.valid() ? sourceImage_ : templateImage_;
        if (hoverPx_ >= 0 && hoverPy_ >= 0) {
            ImGui::TextDisabled(u8"W:%d  H:%d   X:%d  Y:%d   R:%d G:%d B:%d", infoSlot.width,
                                infoSlot.height, hoverPx_, hoverPy_, hoverR_, hoverG_, hoverB_);
        } else {
            ImGui::TextDisabled(u8"W:%d  H:%d", infoSlot.width, infoSlot.height);
        }
        ImGui::SameLine(ImGui::GetWindowWidth() - 280.f);
    }
    if (!statusText_.empty()) {
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(0.35f, 0.9f, 0.55f, 1.f), "%s", statusText_.c_str());
    }

    ImGui::End();
}
