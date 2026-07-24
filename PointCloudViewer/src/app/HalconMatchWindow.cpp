#include "app/HalconMatchWindow.h"

#include "app/FileDialog.h"
#include "io/ImageIO.h"

#include <glad/gl.h>
#include <imgui.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>

namespace {
constexpr ImU32 kContourCol = IM_COL32(40, 220, 80, 255);
constexpr ImU32 kScoreCol = IM_COL32(255, 60, 60, 255);
constexpr ImU32 kRoiCol = IM_COL32(80, 180, 255, 220);
constexpr ImU32 kRoiHandleCol = IM_COL32(255, 220, 60, 255);
constexpr float kPi = 3.14159265358979323846f;
constexpr float kDegToRad = kPi / 180.f;
constexpr float kRadToDeg = 180.f / kPi;

bool DragDouble(const char* label, double* v, float speed, double vMin, double vMax,
                const char* fmt) {
    return ImGui::DragScalar(label, ImGuiDataType_Double, v, speed, &vMin, &vMax, fmt);
}

void LocalToWorld(float lx, float ly, float cx, float cy, float angleDeg, float& wx, float& wy) {
    const float rad = angleDeg * kDegToRad;
    const float c = std::cos(rad);
    const float s = std::sin(rad);
    wx = cx + lx * c - ly * s;
    wy = cy + lx * s + ly * c;
}

void WorldToLocal(float wx, float wy, float cx, float cy, float angleDeg, float& lx, float& ly) {
    const float dx = wx - cx;
    const float dy = wy - cy;
    const float rad = -angleDeg * kDegToRad;
    const float c = std::cos(rad);
    const float s = std::sin(rad);
    lx = dx * c - dy * s;
    ly = dx * s + dy * c;
}

void DrawContourPolylines(ImDrawList* dl, const std::vector<float>& xs, const std::vector<float>& ys,
                          const std::vector<int>& starts, float originX, float originY, float sx,
                          float sy, ImU32 col, float thickness) {
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

void SetRoiFromAxisBox(float x0, float y0, float x1, float y1, float& cx, float& cy, float& hw,
                       float& hh, float& angleDeg) {
    const float minX = std::min(x0, x1);
    const float maxX = std::max(x0, x1);
    const float minY = std::min(y0, y1);
    const float maxY = std::max(y0, y1);
    cx = (minX + maxX) * 0.5f;
    cy = (minY + maxY) * 0.5f;
    hw = std::max((maxX - minX) * 0.5f, 1.f);
    hh = std::max((maxY - minY) * 0.5f, 1.f);
    angleDeg = 0.f;
}
}  // namespace

void HalconMatchWindow::SetVisible(bool visible) {
    if (visible && !visible_) focusOnOpen_ = true;
    visible_ = visible;
}

void HalconMatchWindow::ToggleVisible() { SetVisible(!visible_); }

void HalconMatchWindow::SetStatus(const char* msg) {
    statusText_ = msg ? msg : "";
    if (onStatus_) onStatus_(msg);
}

void HalconMatchWindow::DestroyImageSlot(ImageSlot& slot) {
    if (slot.texId) {
        glDeleteTextures(1, &slot.texId);
        slot.texId = 0;
    }
    slot.width = 0;
    slot.height = 0;
    slot.rgb.clear();
    slot.path.clear();
}

void HalconMatchWindow::DestroyPreviewTexture() {}

bool HalconMatchWindow::UploadTexture(ImageSlot& slot) {
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

bool HalconMatchWindow::LoadImageSlot(ImageSlot& slot, const std::string& path, std::string& error) {
    ImageIO::RgbImage img;
    if (!ImageIO::LoadRgb(path, img, error)) return false;
    DestroyImageSlot(slot);
    slot.width = img.width;
    slot.height = img.height;
    slot.rgb = std::move(img.rgb);
    slot.path = path;
    return UploadTexture(slot);
}

bool HalconMatchWindow::ImageToPixel(const ImageSlot& slot, const ImVec2& imgPos, float drawW,
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

void HalconMatchWindow::ClearTemplateRoi() {
    templateRoiValid_ = false;
    templateRoiDrag_ = TemplateRoiDrag::None;
}

bool HalconMatchWindow::GetTemplateRoiForCreate(float& cx, float& cy, float& hw, float& hh,
                                                float& angleDeg) const {
    if (!templateImage_.valid()) return false;
    if (templateUseRoi_) {
        if (!templateRoiValid_) return false;
        cx = templateRoiCx_;
        cy = templateRoiCy_;
        hw = templateRoiHalfW_;
        hh = templateRoiHalfH_;
        angleDeg = templateRoiAngleDeg_;
        return hw >= 4.f && hh >= 4.f;
    }
    cx = static_cast<float>(templateImage_.width) * 0.5f;
    cy = static_cast<float>(templateImage_.height) * 0.5f;
    hw = static_cast<float>(templateImage_.width) * 0.5f;
    hh = static_cast<float>(templateImage_.height) * 0.5f;
    angleDeg = 0.f;
    return true;
}

int HalconMatchWindow::HitTestTemplateRoi(float imgX, float imgY, float sx, float sy) const {
    if (!templateRoiValid_) return 0;

    const float hitPx = 10.f / std::max(std::min(sx, sy), 0.001f);
    float lx = 0.f;
    float ly = 0.f;
    WorldToLocal(imgX, imgY, templateRoiCx_, templateRoiCy_, templateRoiAngleDeg_, lx, ly);

    const std::array<std::pair<float, float>, 4> localCorners = {
        std::pair<float, float>{-templateRoiHalfW_, -templateRoiHalfH_},
        {templateRoiHalfW_, -templateRoiHalfH_},
        {templateRoiHalfW_, templateRoiHalfH_},
        {-templateRoiHalfW_, templateRoiHalfH_},
    };
    for (int i = 0; i < 4; ++i) {
        const float dx = lx - localCorners[static_cast<std::size_t>(i)].first;
        const float dy = ly - localCorners[static_cast<std::size_t>(i)].second;
        if (std::sqrt(dx * dx + dy * dy) <= hitPx) return 10 + i;
    }

    float rotHx = 0.f;
    float rotHy = 0.f;
    LocalToWorld(0.f, -templateRoiHalfH_ - 24.f / std::max(sy, 0.001f), templateRoiCx_,
                 templateRoiCy_, templateRoiAngleDeg_, rotHx, rotHy);
    const float rdx = imgX - rotHx;
    const float rdy = imgY - rotHy;
    if (std::sqrt(rdx * rdx + rdy * rdy) <= hitPx) return 20;

    if (std::fabs(lx) <= templateRoiHalfW_ && std::fabs(ly) <= templateRoiHalfH_) return 1;
    return 0;
}

void HalconMatchWindow::DrawRotatedTemplateRoi(ImDrawList* dl, const ImVec2& imgPos, float sx,
                                               float sy) {
    if (!templateRoiValid_) return;

    std::array<ImVec2, 4> corners{};
    const std::array<std::pair<float, float>, 4> localCorners = {
        std::pair<float, float>{-templateRoiHalfW_, -templateRoiHalfH_},
        {templateRoiHalfW_, -templateRoiHalfH_},
        {templateRoiHalfW_, templateRoiHalfH_},
        {-templateRoiHalfW_, templateRoiHalfH_},
    };
    for (int i = 0; i < 4; ++i) {
        float wx = 0.f;
        float wy = 0.f;
        LocalToWorld(localCorners[static_cast<std::size_t>(i)].first,
                     localCorners[static_cast<std::size_t>(i)].second, templateRoiCx_,
                     templateRoiCy_, templateRoiAngleDeg_, wx, wy);
        corners[static_cast<std::size_t>(i)] =
            ImVec2(imgPos.x + wx * sx, imgPos.y + wy * sy);
    }

    for (int i = 0; i < 4; ++i) {
        dl->AddLine(corners[static_cast<std::size_t>(i)],
                    corners[static_cast<std::size_t>((i + 1) % 4)], kRoiCol, 2.f);
        dl->AddCircleFilled(corners[static_cast<std::size_t>(i)], 5.f, kRoiHandleCol);
    }

    float rotWx = 0.f;
    float rotWy = 0.f;
    LocalToWorld(0.f, -templateRoiHalfH_ - 24.f / std::max(sy, 0.001f), templateRoiCx_,
                 templateRoiCy_, templateRoiAngleDeg_, rotWx, rotWy);
    const ImVec2 rotScr(imgPos.x + rotWx * sx, imgPos.y + rotWy * sy);
    const ImVec2 topMid((corners[0].x + corners[1].x) * 0.5f, (corners[0].y + corners[1].y) * 0.5f);
    dl->AddLine(topMid, rotScr, kRoiHandleCol, 1.5f);
    dl->AddCircleFilled(rotScr, 6.f, kRoiHandleCol);

    char buf[64];
    std::snprintf(buf, sizeof(buf), u8"%.1f°", templateRoiAngleDeg_);
    dl->AddText(ImVec2(rotScr.x + 8.f, rotScr.y - 8.f), kRoiHandleCol, buf);
}

void HalconMatchWindow::DrawMatchOverlays(ImDrawList* dl, const ImVec2& imgPos, float drawW,
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

    for (const HalconShapeMatch::MatchHit& hit : lastResult_.hits) {
        DrawContourPolylines(dl, hit.contourX, hit.contourY, hit.contourStarts, imgPos.x, imgPos.y,
                             sx, sy, kContourCol, 2.f);
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.3f", hit.score);
        dl->AddText(ImVec2(imgPos.x + hit.centerX * sx - 12.f, imgPos.y + hit.centerY * sy - 22.f),
                    kScoreCol, buf);
        std::snprintf(buf, sizeof(buf), "%.2fx", hit.scale);
        dl->AddText(ImVec2(imgPos.x + hit.centerX * sx - 16.f, imgPos.y + hit.centerY * sy + 6.f),
                    IM_COL32(20, 20, 20, 255), buf);
    }
}

void HalconMatchWindow::DrawImageCanvas(ImageSlot& slot, bool allowRoi, RoiMode roiMode,
                                        const char* childId) {
    if (!slot.valid()) {
        ImGui::TextDisabled(u8"尚未加载图像");
        return;
    }

    const ImVec2 avail = ImGui::GetContentRegionAvail();
    const float areaH = std::max(avail.y, 200.f);
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

    ImGui::SetCursorPos(ImVec2(basePanX, basePanY));
    const ImVec2 imgPos = ImGui::GetCursorScreenPos();
    ImGui::InvisibleButton(childId, ImVec2(drawW, drawH));
    const bool imgHovered = ImGui::IsItemHovered();
    const bool imgClicked = ImGui::IsItemClicked(ImGuiMouseButton_Left);
    const bool imgDragging = ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Left);
    const bool imgReleased = ImGui::IsItemDeactivated();

    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddImage((ImTextureID)(intptr_t)slot.texId, imgPos,
                 ImVec2(imgPos.x + drawW, imgPos.y + drawH));

    const float sx = drawW / static_cast<float>(slot.width);
    const float sy = drawH / static_cast<float>(slot.height);

    const bool isTemplateView = (&slot == &templateImage_);
    const bool isSourceView = (&slot == &sourceImage_);

    const bool templateRoiActive =
        allowRoi && roiEnabled_ && templateUseRoi_ && isTemplateView && roiMode == RoiMode::Template;

    if (templateRoiActive) {
        DrawRotatedTemplateRoi(dl, imgPos, sx, sy);
    } else if (allowRoi && isTemplateView && !templateUseRoi_) {
        dl->AddRect(imgPos, ImVec2(imgPos.x + drawW, imgPos.y + drawH),
                    IM_COL32(120, 120, 120, 160), 0.f, 0, 1.f);
    }
    if (allowRoi && roiEnabled_ && isSourceView && roiMode == RoiMode::Search && useSearchRoi_) {
        if (searchRoiX1_ > searchRoiX0_ && searchRoiY1_ > searchRoiY0_) {
            dl->AddRect(ImVec2(imgPos.x + searchRoiX0_ * sx, imgPos.y + searchRoiY0_ * sy),
                        ImVec2(imgPos.x + searchRoiX1_ * sx, imgPos.y + searchRoiY1_ * sy), kRoiCol,
                        0.f, 0, 2.f);
        }
    }

    if (isSourceView) {
        DrawMatchOverlays(dl, imgPos, drawW, drawH);
    }

    if (imgHovered) {
        float px = 0.f;
        float py = 0.f;
        if (ImageToPixel(slot, imgPos, drawW, drawH, io.MousePos.x, io.MousePos.y, px, py)) {
            hoverPx_ = static_cast<int>(px);
            hoverPy_ = static_cast<int>(py);
        }

        if (templateRoiActive) {
            if (imgClicked) {
                const int hit = HitTestTemplateRoi(px, py, sx, sy);
                templateRoiDragStartX_ = px;
                templateRoiDragStartY_ = py;
                templateRoiDragAnchorCx_ = templateRoiCx_;
                templateRoiDragAnchorCy_ = templateRoiCy_;
                templateRoiDragAnchorHw_ = templateRoiHalfW_;
                templateRoiDragAnchorHh_ = templateRoiHalfH_;
                templateRoiDragAnchorAngle_ = templateRoiAngleDeg_;

                if (hit == 20) {
                    templateRoiDrag_ = TemplateRoiDrag::Rotate;
                } else if (hit >= 10 && hit < 14) {
                    templateRoiDrag_ = TemplateRoiDrag::ResizeCorner;
                    templateRoiHitCorner_ = hit - 10;
                } else if (hit == 1) {
                    templateRoiDrag_ = TemplateRoiDrag::Move;
                } else {
                    templateRoiDrag_ = TemplateRoiDrag::Create;
                    templateRoiCreateX0_ = px;
                    templateRoiCreateY0_ = py;
                    templateRoiValid_ = false;
                }
            }

            if (templateRoiDrag_ == TemplateRoiDrag::Create && imgDragging) {
                SetRoiFromAxisBox(templateRoiCreateX0_, templateRoiCreateY0_, px, py, templateRoiCx_,
                                  templateRoiCy_, templateRoiHalfW_, templateRoiHalfH_,
                                  templateRoiAngleDeg_);
                templateRoiValid_ = templateRoiHalfW_ >= 4.f && templateRoiHalfH_ >= 4.f;
            } else if (templateRoiDrag_ == TemplateRoiDrag::Move && imgDragging) {
                templateRoiCx_ = templateRoiDragAnchorCx_ + (px - templateRoiDragStartX_);
                templateRoiCy_ = templateRoiDragAnchorCy_ + (py - templateRoiDragStartY_);
            } else if (templateRoiDrag_ == TemplateRoiDrag::Rotate && imgDragging) {
                const float angle =
                    std::atan2(py - templateRoiDragAnchorCy_, px - templateRoiDragAnchorCx_) *
                        kRadToDeg +
                    90.f;
                templateRoiAngleDeg_ = angle;
            } else if (templateRoiDrag_ == TemplateRoiDrag::ResizeCorner && imgDragging) {
                float lx = 0.f;
                float ly = 0.f;
                WorldToLocal(px, py, templateRoiDragAnchorCx_, templateRoiDragAnchorCy_,
                             templateRoiDragAnchorAngle_, lx, ly);

                float anchorLx = 0.f;
                float anchorLy = 0.f;
                const std::array<std::pair<float, float>, 4> localCorners = {
                    std::pair<float, float>{-templateRoiDragAnchorHw_, -templateRoiDragAnchorHh_},
                    {templateRoiDragAnchorHw_, -templateRoiDragAnchorHh_},
                    {templateRoiDragAnchorHw_, templateRoiDragAnchorHh_},
                    {-templateRoiDragAnchorHw_, templateRoiDragAnchorHh_},
                };
                const int opp = (templateRoiHitCorner_ + 2) % 4;
                anchorLx = localCorners[static_cast<std::size_t>(opp)].first;
                anchorLy = localCorners[static_cast<std::size_t>(opp)].second;

                const float newHw = std::max(std::fabs(lx - anchorLx) * 0.5f, 4.f);
                const float newHh = std::max(std::fabs(ly - anchorLy) * 0.5f, 4.f);
                const float midLx = (lx + anchorLx) * 0.5f;
                const float midLy = (ly + anchorLy) * 0.5f;
                float newCx = 0.f;
                float newCy = 0.f;
                LocalToWorld(midLx, midLy, templateRoiDragAnchorCx_, templateRoiDragAnchorCy_,
                             templateRoiDragAnchorAngle_, newCx, newCy);
                templateRoiCx_ = newCx;
                templateRoiCy_ = newCy;
                templateRoiHalfW_ = newHw;
                templateRoiHalfH_ = newHh;
                templateRoiAngleDeg_ = templateRoiDragAnchorAngle_;
            }
        }

        if (allowRoi && roiEnabled_ && isSourceView && roiMode == RoiMode::Search) {
            if (imgClicked) {
                searchRoiDragging_ = true;
                searchRoiX0_ = searchRoiX1_ = px;
                searchRoiY0_ = searchRoiY1_ = py;
                useSearchRoi_ = true;
            }
            if (searchRoiDragging_ && imgDragging) {
                searchRoiX1_ = px;
                searchRoiY1_ = py;
            }
        }
    }

    if (imgReleased) {
        templateRoiDrag_ = TemplateRoiDrag::None;
        searchRoiDragging_ = false;
    }

    if (isTemplateView && templateUseRoi_ && !templateRoiValid_) {
        dl->AddText(ImVec2(imgPos.x + 8.f, imgPos.y + 8.f), IM_COL32(255, 220, 90, 255),
                    u8"左键拖拽框选模板 ROI");
        dl->AddText(ImVec2(imgPos.x + 8.f, imgPos.y + 28.f), IM_COL32(190, 190, 190, 255),
                    u8"黄点拖角点缩放，顶部黄点旋转");
    } else if (isTemplateView && !templateUseRoi_) {
        dl->AddText(ImVec2(imgPos.x + 8.f, imgPos.y + 8.f), IM_COL32(160, 220, 160, 255),
                    u8"整图创建模板（勾选「框选模板ROI」后可框选）");
    }

    ImGui::EndChild();
}

void HalconMatchWindow::DrawTemplatePreviewPanel() {
    ImGui::TextDisabled(u8"模板轮廓预览");
    const float panelH = 168.f;
    ImGui::BeginChild("##halcon_tpl_prev", ImVec2(0, panelH), ImGuiChildFlags_Borders);

    if (model_.valid && !model_.previewContourX.empty()) {
        const ImVec2 panelPos = ImGui::GetCursorScreenPos();
        const float panelW = ImGui::GetContentRegionAvail().x;
        const float panelInnerH = panelH - 8.f;
        ImDrawList* dl = ImGui::GetWindowDrawList();
        dl->AddRectFilled(panelPos, ImVec2(panelPos.x + panelW, panelPos.y + panelInnerH),
                          IM_COL32(24, 24, 24, 255));

        float minX = model_.previewContourX[0];
        float maxX = model_.previewContourX[0];
        float minY = model_.previewContourY[0];
        float maxY = model_.previewContourY[0];
        for (std::size_t i = 1; i < model_.previewContourX.size(); ++i) {
            minX = std::min(minX, model_.previewContourX[i]);
            maxX = std::max(maxX, model_.previewContourX[i]);
            minY = std::min(minY, model_.previewContourY[i]);
            maxY = std::max(maxY, model_.previewContourY[i]);
        }
        const float cw = std::max(maxX - minX, 1.f);
        const float ch = std::max(maxY - minY, 1.f);
        const float margin = 10.f;
        const float padX = cw * 0.12f;
        const float padY = ch * 0.12f;
        const float fitW = cw + padX * 2.f;
        const float fitH = ch + padY * 2.f;
        const float scaleX = (panelW - margin * 2.f) / fitW;
        const float scaleY = (panelInnerH - margin * 2.f) / fitH;
        const float s = std::min(scaleX, scaleY);
        const float offX = panelPos.x + (panelW - fitW * s) * 0.5f - (minX - padX) * s;
        const float offY = panelPos.y + (panelInnerH - fitH * s) * 0.5f - (minY - padY) * s;
        DrawContourPolylines(dl, model_.previewContourX, model_.previewContourY,
                             model_.previewContourStarts, offX, offY, s, s, kContourCol, 1.5f);
        ImGui::Dummy(ImVec2(panelW, panelInnerH));
        ImGui::EndChild();
        return;
    }

    ImGui::TextDisabled(u8"创建模板后在此显示轮廓线");
    ImGui::EndChild();
}

void HalconMatchWindow::DrawParamColumns() {
    if (!HalconShapeMatch::IsHalconAvailable()) {
        ImGui::TextColored(ImVec4(1.f, 0.4f, 0.4f, 1.f), u8"当前构建未启用 Halcon");
        return;
    }

    ImGui::TextDisabled(u8"创建模板");
    if (ImGui::BeginTable("##halcon_create", 4,
                          ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_PadOuterX)) {
        ImGui::TableSetupColumn("lbl1", ImGuiTableColumnFlags_WidthFixed, 78.f);
        ImGui::TableSetupColumn("val1", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("lbl2", ImGuiTableColumnFlags_WidthFixed, 78.f);
        ImGui::TableSetupColumn("val2", ImGuiTableColumnFlags_WidthStretch);

        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(u8"对比度(低)");
        ImGui::TableNextColumn();
        ImGui::SetNextItemWidth(-1);
        ImGui::DragInt("##hc_clow", &createParams_.contrastLow, 1, 1, 255);
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(u8"对比度(高)");
        ImGui::TableNextColumn();
        ImGui::SetNextItemWidth(-1);
        ImGui::DragInt("##hc_chigh", &createParams_.contrastHigh, 1, 1, 255);

        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(u8"最小组件");
        ImGui::TableNextColumn();
        ImGui::SetNextItemWidth(-1);
        ImGui::DragInt("##hc_minsz", &createParams_.minComponentSize, 1, 1, 100);
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(u8"最小对比度");
        ImGui::TableNextColumn();
        ImGui::SetNextItemWidth(-1);
        ImGui::DragInt("##hc_mincon", &createParams_.minContrast, 1, 1, 255);

        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(u8"金字塔");
        ImGui::TableNextColumn();
        ImGui::SetNextItemWidth(-1);
        ImGui::DragInt("##hc_pyr", &createParams_.numLevels, 1, 0, 6);
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(u8"起始角度");
        ImGui::TableNextColumn();
        ImGui::SetNextItemWidth(-1);
        ImGui::DragScalar("##hc_ang0", ImGuiDataType_Double, &createParams_.angleStartDeg, 1.0,
                          nullptr, nullptr, "%.0f");

        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(u8"角度范围");
        ImGui::TableNextColumn();
        ImGui::SetNextItemWidth(-1);
        ImGui::DragScalar("##hc_ang_ext", ImGuiDataType_Double, &createParams_.angleExtentDeg, 1.0,
                          nullptr, nullptr, "%.0f");
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(u8"最小缩放");
        ImGui::TableNextColumn();
        ImGui::SetNextItemWidth(-1);
        DragDouble("##hc_smin", &createParams_.scaleMin, 0.01f, 0.1, 4.0, "%.2f");

        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(u8"最大缩放");
        ImGui::TableNextColumn();
        ImGui::SetNextItemWidth(-1);
        DragDouble("##hc_smax", &createParams_.scaleMax, 0.01f, 0.1, 4.0, "%.2f");
        ImGui::TableNextColumn();
        ImGui::TableNextColumn();
        ImGui::EndTable();
    }

    ImGui::Spacing();
    ImGui::TextDisabled(u8"寻找模板");
    if (ImGui::BeginTable("##halcon_find", 4,
                          ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_PadOuterX)) {
        ImGui::TableSetupColumn("lbl1", ImGuiTableColumnFlags_WidthFixed, 78.f);
        ImGui::TableSetupColumn("val1", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("lbl2", ImGuiTableColumnFlags_WidthFixed, 78.f);
        ImGui::TableSetupColumn("val2", ImGuiTableColumnFlags_WidthStretch);

        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(u8"起始角度");
        ImGui::TableNextColumn();
        ImGui::SetNextItemWidth(-1);
        DragDouble("##hf_ang0", &findParams_.angleStartDeg, 1.0, -360.0, 360.0, "%.0f");
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(u8"角度范围");
        ImGui::TableNextColumn();
        ImGui::SetNextItemWidth(-1);
        DragDouble("##hf_angr", &findParams_.angleExtentDeg, 1.0, 0.0, 360.0, "%.0f");

        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(u8"最小缩放");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip(u8"不能小于创建模板时的最小缩放（当前模型 %.2f）", model_.modelScaleMin);
        }
        ImGui::TableNextColumn();
        ImGui::SetNextItemWidth(-1);
        DragDouble("##hf_smin", &findParams_.scaleMin, 0.01f, 0.1, 4.0, "%.2f");
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(u8"最大缩放");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip(u8"不能大于创建模板时的最大缩放（当前模型 %.2f）", model_.modelScaleMax);
        }
        ImGui::TableNextColumn();
        ImGui::SetNextItemWidth(-1);
        DragDouble("##hf_smax", &findParams_.scaleMax, 0.01f, 0.1, 4.0, "%.2f");

        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(u8"最小得分");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip(
                u8"最终显示得分按此过滤。\n"
                u8"Halcon 金字塔搜索阶段得分偏低，内部会自动放宽搜索阈值，\n"
                u8"避免 0.75 漏检、0.4 却显示 0.84 的情况。");
        }
        ImGui::TableNextColumn();
        ImGui::SetNextItemWidth(-1);
        DragDouble("##hf_score", &findParams_.minScore, 0.01f, 0.0, 1.0, "%.2f");
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(u8"最大重叠");
        ImGui::TableNextColumn();
        ImGui::SetNextItemWidth(-1);
        DragDouble("##hf_overlap", &findParams_.maxOverlap, 0.01f, 0.0, 1.0, "%.2f");

        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(u8"贪婪度");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip(u8"越大越快但越容易漏检；要求最小得分≥0.7 时建议≤0.3");
        }
        ImGui::TableNextColumn();
        ImGui::SetNextItemWidth(-1);
        DragDouble("##hf_greed", &findParams_.greediness, 0.01f, 0.0, 1.0, "%.2f");
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(u8"目标数量");
        ImGui::TableNextColumn();
        ImGui::SetNextItemWidth(-1);
        ImGui::DragInt("##hf_num", &findParams_.numMatches, 1, 0, 64);

        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(u8"金字塔");
        ImGui::TableNextColumn();
        ImGui::SetNextItemWidth(-1);
        ImGui::DragInt("##hf_pyr", &findParams_.numLevels, 1, 0, 6);
        ImGui::TableNextColumn();
        ImGui::Checkbox(u8"亚像素##hf_sub", &findParams_.subPixel);
        ImGui::TableNextColumn();
        ImGui::EndTable();
    }
}

void HalconMatchWindow::ReadTemplateImage() {
    const std::string path = FileDialog::OpenImageFile(u8"读取模板图像");
    if (path.empty()) return;
    std::string error;
    if (!LoadImageSlot(templateImage_, path, error)) {
        SetStatus(error.c_str());
        return;
    }
    leftView_ = LeftViewMode::Template;
    roiMode_ = RoiMode::Template;
    ClearTemplateRoi();
    templateUseRoi_ = false;
    HalconShapeMatch::DestroyModel(model_);
    model_ = {};
    zoom_ = 1.f;
    char buf[256];
    std::snprintf(buf, sizeof(buf), u8"已读取模板图像 %dx%d（默认整图创建，可勾选框选ROI）",
                  templateImage_.width, templateImage_.height);
    SetStatus(buf);
}

void HalconMatchWindow::ReadSourceImage() {
    const std::string path = FileDialog::OpenImageFile(u8"读取源图像");
    if (path.empty()) return;
    std::string error;
    if (!LoadImageSlot(sourceImage_, path, error)) {
        SetStatus(error.c_str());
        return;
    }
    leftView_ = LeftViewMode::Source;
    roiMode_ = RoiMode::Search;
    searchRoiX0_ = searchRoiY0_ = 0.f;
    searchRoiX1_ = static_cast<float>(sourceImage_.width - 1);
    searchRoiY1_ = static_cast<float>(sourceImage_.height - 1);
    useSearchRoi_ = false;
    lastResult_ = {};
    char buf[256];
    std::snprintf(buf, sizeof(buf), u8"已读取源图像 %dx%d，可进行模板匹配", sourceImage_.width,
                  sourceImage_.height);
    SetStatus(buf);
}

void HalconMatchWindow::CreateTemplate() {
    if (!HalconShapeMatch::IsHalconAvailable()) {
        SetStatus(u8"Halcon 未启用");
        return;
    }
    if (!templateImage_.valid()) {
        SetStatus(u8"请先读取模板图像");
        return;
    }
    float cx = 0.f;
    float cy = 0.f;
    float hw = 0.f;
    float hh = 0.f;
    float angle = 0.f;
    if (!GetTemplateRoiForCreate(cx, cy, hw, hh, angle)) {
        SetStatus(templateUseRoi_ ? u8"请先框选模板 ROI，或取消「框选模板ROI」直接整图创建"
                                 : u8"模板 ROI 无效");
        return;
    }
    std::string error;
    if (!HalconShapeMatch::CreateModel(templateImage_.rgb, templateImage_.width,
                                       templateImage_.height, cx, cy, hw, hh, angle, createParams_,
                                       model_, error)) {
        SetStatus(error.c_str());
        return;
    }
    char buf[192];
    std::snprintf(buf, sizeof(buf), u8"Halcon 模板创建成功（%s %dx%d，轮廓点 %zu）",
                  templateUseRoi_ ? u8"ROI" : u8"整图", model_.templateW, model_.templateH,
                  model_.previewContourX.size());
    SetStatus(buf);
    findParams_.scaleMin = model_.modelScaleMin;
    findParams_.scaleMax = model_.modelScaleMax;
}

void HalconMatchWindow::RunMatch() {
    if (!HalconShapeMatch::IsHalconAvailable()) {
        SetStatus(u8"Halcon 未启用");
        return;
    }
    if (!sourceImage_.valid()) {
        SetStatus(u8"请先读取源图像");
        return;
    }
    if (!model_.valid) {
        SetStatus(u8"请先创建或加载 Halcon 模板");
        return;
    }
    std::string error;
    if (!HalconShapeMatch::FindModel(sourceImage_.rgb, sourceImage_.width, sourceImage_.height,
                                     searchRoiX0_, searchRoiY0_, searchRoiX1_, searchRoiY1_,
                                     useSearchRoi_, model_, findParams_, lastResult_, error)) {
        lastResult_ = {};
        SetStatus(error.c_str());
        return;
    }
    leftView_ = LeftViewMode::Source;
    char buf[256];
    std::snprintf(buf, sizeof(buf), u8"Halcon 匹配完成：%zu 个目标，耗时 %.1f ms",
                  lastResult_.hits.size(), lastResult_.elapsedMs);
    if (!lastResult_.note.empty()) {
        SetStatus((std::string(buf) + u8"  |  " + lastResult_.note).c_str());
    } else {
        SetStatus(buf);
    }
}

void HalconMatchWindow::SaveTemplate() {
    const std::string path = FileDialog::SaveHalconShapeModelFile();
    if (path.empty()) return;
    std::string error;
    if (!HalconShapeMatch::SaveModel(path, model_, error)) {
        SetStatus(error.c_str());
        return;
    }
    char buf[256];
    std::snprintf(buf, sizeof(buf), u8"Halcon 模板已保存：%s", path.c_str());
    SetStatus(buf);
}

void HalconMatchWindow::LoadTemplate() {
    const std::string path = FileDialog::OpenHalconShapeModelFile();
    if (path.empty()) return;
    std::string error;
    if (!HalconShapeMatch::LoadModel(path, model_, error)) {
        SetStatus(error.c_str());
        return;
    }
    findParams_.scaleMin = model_.modelScaleMin;
    findParams_.scaleMax = model_.modelScaleMax;
    SetStatus(u8"Halcon 模板加载成功");
}

void HalconMatchWindow::ClearTemplate() {
    HalconShapeMatch::DestroyModel(model_);
    model_ = {};
    lastResult_ = {};
    SetStatus(u8"已清除 Halcon 模板");
}

void HalconMatchWindow::ClearSearchRoi() {
    if (sourceImage_.valid()) {
        searchRoiX0_ = searchRoiY0_ = 0.f;
        searchRoiX1_ = static_cast<float>(sourceImage_.width - 1);
        searchRoiY1_ = static_cast<float>(sourceImage_.height - 1);
    }
    useSearchRoi_ = false;
}

void HalconMatchWindow::Draw(float menuBottomY, float bottomInset) {
    if (!visible_) return;

    ImGuiViewport* vp = ImGui::GetMainViewport();
    const float windowH = std::max(vp->Size.y - menuBottomY - bottomInset, 1.f);
    ImGui::SetNextWindowPos(ImVec2(vp->Pos.x, menuBottomY), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(vp->Size.x, windowH), ImGuiCond_Always);
    ImGui::SetNextWindowBgAlpha(1.f);
    if (focusOnOpen_) {
        ImGui::SetNextWindowFocus();
        focusOnOpen_ = false;
    }

    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.f);

    const ImGuiWindowFlags flags =
        ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoSavedSettings;

    bool open = visible_;
    if (!ImGui::Begin(u8"模板匹配", &open, flags)) {
        ImGui::End();
        ImGui::PopStyleVar(2);
        visible_ = open;
        return;
    }
    visible_ = open;
    if (!visible_) {
        ImGui::End();
        ImGui::PopStyleVar(2);
        return;
    }

    if (ImGui::Button(u8"关闭")) {
        SetVisible(false);
    }
    ImGui::SameLine();
    ImGui::TextDisabled(u8"Shift+滚轮缩放图像 | 对称齿轮建议缩小角度范围并提高最小得分");

    const float rightW =
        std::clamp(ImGui::GetContentRegionAvail().x * 0.32f, 320.f, 480.f);
    const float bottomH = 24.f;
    const ImVec2 contentAvail = ImGui::GetContentRegionAvail();

    ImGui::BeginChild("##halcon_main_row", ImVec2(0, contentAvail.y - bottomH), false);
    ImGui::BeginChild("##halcon_left", ImVec2(ImGui::GetContentRegionAvail().x - rightW, 0), false);

    ImageSlot* mainSlot = nullptr;
    bool allowRoi = false;
    RoiMode mainRoi = RoiMode::Template;
    const char* mainLabel = u8"图像";
    if (leftView_ == LeftViewMode::Template && templateImage_.valid()) {
        mainSlot = &templateImage_;
        allowRoi = true;
        mainRoi = RoiMode::Template;
        mainLabel = u8"模板图像（框选可旋转 ROI）";
    } else if (leftView_ == LeftViewMode::Source && sourceImage_.valid()) {
        mainSlot = &sourceImage_;
        allowRoi = true;
        mainRoi = RoiMode::Search;
        mainLabel = u8"源图像（匹配结果）";
    } else if (templateImage_.valid()) {
        mainSlot = &templateImage_;
        allowRoi = true;
        mainRoi = RoiMode::Template;
        mainLabel = u8"模板图像（框选可旋转 ROI）";
        leftView_ = LeftViewMode::Template;
    } else if (sourceImage_.valid()) {
        mainSlot = &sourceImage_;
        allowRoi = true;
        mainRoi = RoiMode::Search;
        mainLabel = u8"源图像（匹配结果）";
        leftView_ = LeftViewMode::Source;
    }

    ImGui::TextDisabled("%s", mainLabel);
    if (templateImage_.valid() && sourceImage_.valid()) {
        ImGui::SameLine();
        if (ImGui::SmallButton(u8"显示模板")) {
            leftView_ = LeftViewMode::Template;
            roiMode_ = RoiMode::Template;
        }
        ImGui::SameLine();
        if (ImGui::SmallButton(u8"显示源图")) {
            leftView_ = LeftViewMode::Source;
            roiMode_ = RoiMode::Search;
        }
    }

    if (mainSlot) {
        DrawImageCanvas(*mainSlot, allowRoi, mainRoi, "##halcon_main_canvas");
    } else {
        ImGui::BeginChild("##halcon_main_canvas",
                          ImVec2(0, std::max(ImGui::GetContentRegionAvail().y, 200.f)),
                          ImGuiChildFlags_Borders);
        ImGui::TextDisabled(u8"请先读取模板图像");
        ImGui::EndChild();
    }
    ImGui::EndChild();

    ImGui::SameLine();
    ImGui::BeginChild("##halcon_right", ImVec2(rightW, 0), ImGuiChildFlags_Borders);
    DrawParamColumns();

    ImGui::Spacing();
    if (ImGui::Checkbox(u8"框选模板ROI", &templateUseRoi_)) {
        if (templateUseRoi_) {
            leftView_ = LeftViewMode::Template;
            roiMode_ = RoiMode::Template;
        } else {
            ClearTemplateRoi();
        }
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip(u8"勾选后在左侧拖拽框选；不勾选则「创建模板」使用整张图");
    }

    ImGui::Spacing();
    ImGui::Checkbox(u8"搜索ROI", &roiEnabled_);
    ImGui::SameLine();
    if (ImGui::RadioButton(u8"模板", roiMode_ == RoiMode::Template)) {
        roiMode_ = RoiMode::Template;
        if (templateImage_.valid()) leftView_ = LeftViewMode::Template;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton(u8"搜索", roiMode_ == RoiMode::Search)) {
        roiMode_ = RoiMode::Search;
        if (sourceImage_.valid()) leftView_ = LeftViewMode::Source;
    }
    ImGui::SameLine();
    if (ImGui::Button(u8"CLS")) {
        if (roiMode_ == RoiMode::Template) {
            ClearTemplateRoi();
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
    const ImageSlot* infoSlot = nullptr;
    if (leftView_ == LeftViewMode::Template && templateImage_.valid()) {
        infoSlot = &templateImage_;
    } else if (sourceImage_.valid()) {
        infoSlot = &sourceImage_;
    } else if (templateImage_.valid()) {
        infoSlot = &templateImage_;
    }
    if (infoSlot) {
        if (hoverPx_ >= 0 && hoverPy_ >= 0) {
            ImGui::TextDisabled(u8"W:%d  H:%d   X:%d  Y:%d", infoSlot->width, infoSlot->height,
                                hoverPx_, hoverPy_);
        } else {
            ImGui::TextDisabled(u8"W:%d  H:%d", infoSlot->width, infoSlot->height);
        }
    }
    if (!statusText_.empty()) {
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(0.35f, 0.9f, 0.55f, 1.f), "%s", statusText_.c_str());
    }

    ImGui::End();
    ImGui::PopStyleVar(2);
}
