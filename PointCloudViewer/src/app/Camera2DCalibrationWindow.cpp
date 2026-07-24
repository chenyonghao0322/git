#include "app/Camera2DCalibrationWindow.h"

#include "app/FileDialog.h"
#include "io/ImageIO.h"

#include <glad/gl.h>
#include <imgui.h>

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace {
constexpr int kDefaultPointCount = 9;
constexpr int kMinPointCount = 3;
constexpr ImU32 kRoiCol = IM_COL32(80, 180, 255, 220);
constexpr ImU32 kCircleCol = IM_COL32(40, 220, 80, 255);
constexpr ImU32 kCenterCol = IM_COL32(255, 80, 80, 255);
constexpr ImU32 kCenterFillCol = IM_COL32(255, 80, 80, 80);
}  // namespace

void Camera2DCalibrationWindow::SetVisible(bool visible) {
    if (visible && !visible_) {
        focusOnOpen_ = true;
        EnsureDefaultEntries();
    }
    visible_ = visible;
}

void Camera2DCalibrationWindow::ToggleVisible() { SetVisible(!visible_); }

void Camera2DCalibrationWindow::SetStatus(const char* msg) {
    statusText_ = msg ? msg : "";
    if (onStatus_) onStatus_(msg);
}

std::string Camera2DCalibrationWindow::BaseName(const std::string& path) {
    const std::size_t p1 = path.find_last_of("/\\");
    return (p1 == std::string::npos) ? path : path.substr(p1 + 1);
}

void Camera2DCalibrationWindow::NormalizeRoi(float& x0, float& y0, float& x1, float& y1) {
    if (x0 > x1) std::swap(x0, x1);
    if (y0 > y1) std::swap(y0, y1);
}

void Camera2DCalibrationWindow::EnsureDefaultEntries() {
    if (!entries_.empty()) return;
    entries_.resize(static_cast<std::size_t>(kDefaultPointCount));
    selectedIndex_ = 0;
    lastResult_ = {};
    lastError_.clear();
}

void Camera2DCalibrationWindow::DestroyImageSlot(ImageSlot& slot) {
    if (slot.texId) {
        glDeleteTextures(1, &slot.texId);
        slot.texId = 0;
    }
    slot.width = 0;
    slot.height = 0;
    slot.rgb.clear();
    slot.path.clear();
}

bool Camera2DCalibrationWindow::UploadTexture(ImageSlot& slot) {
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

bool Camera2DCalibrationWindow::LoadImageSlot(ImageSlot& slot, const std::string& path,
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

bool Camera2DCalibrationWindow::ImageToPixel(const ImageSlot& slot, float imgPosX, float imgPosY,
                                             float drawW, float drawH, float mouseX, float mouseY,
                                             float& outX, float& outY) const {
    if (!slot.valid()) return false;
    if (mouseX < imgPosX || mouseY < imgPosY || mouseX > imgPosX + drawW ||
        mouseY > imgPosY + drawH) {
        return false;
    }
    outX = (mouseX - imgPosX) / drawW * static_cast<float>(slot.width);
    outY = (mouseY - imgPosY) / drawH * static_cast<float>(slot.height);
    outX = std::clamp(outX, 0.f, static_cast<float>(slot.width - 1));
    outY = std::clamp(outY, 0.f, static_cast<float>(slot.height - 1));
    return true;
}

void Camera2DCalibrationWindow::LoadImageForEntry(int index) {
    if (index < 0 || index >= static_cast<int>(entries_.size())) return;
    const std::string path = FileDialog::OpenImageFile(u8"选择标定图像");
    if (path.empty()) return;
    CalibEntry& entry = entries_[static_cast<std::size_t>(index)];
    std::string error;
    if (!LoadImageSlot(entry.image, path, error)) {
        SetStatus(error.c_str());
        return;
    }
    entry.hasPixel = false;
    entry.hasRoi = false;
    entry.detectedRadius = 0.f;
    selectedIndex_ = index;
    roiDragging_ = false;
    char buf[256];
    std::snprintf(buf, sizeof(buf), u8"已加载第 %d 张图像，请框选圆点区域", index + 1);
    SetStatus(buf);
}

void Camera2DCalibrationWindow::LoadImagesBatch() {
    const std::vector<std::string> paths = FileDialog::OpenMultipleImageFiles(u8"批量选择标定图像");
    if (paths.empty()) return;

    const int needed = static_cast<int>(paths.size());
    while (static_cast<int>(entries_.size()) < needed) entries_.push_back({});

    std::string error;
    int loaded = 0;
    for (int i = 0; i < needed; ++i) {
        CalibEntry& entry = entries_[static_cast<std::size_t>(i)];
        if (LoadImageSlot(entry.image, paths[static_cast<std::size_t>(i)], error)) {
            entry.hasPixel = false;
            entry.hasRoi = false;
            entry.detectedRadius = 0.f;
            ++loaded;
        } else {
            SetStatus(error.c_str());
            break;
        }
    }
    selectedIndex_ = 0;
    roiDragging_ = false;
    char buf[128];
    std::snprintf(buf, sizeof(buf), u8"已批量加载 %d 张图像（共需 %d 组机器人坐标）", loaded, needed);
    SetStatus(buf);
}

void Camera2DCalibrationWindow::AddEntry() {
    entries_.push_back({});
    selectedIndex_ = static_cast<int>(entries_.size()) - 1;
}

void Camera2DCalibrationWindow::RemoveEntry(int index) {
    if (static_cast<int>(entries_.size()) <= kMinPointCount) {
        SetStatus(u8"至少保留 3 组标定点");
        return;
    }
    if (index < 0 || index >= static_cast<int>(entries_.size())) return;
    DestroyImageSlot(entries_[static_cast<std::size_t>(index)].image);
    entries_.erase(entries_.begin() + index);
    selectedIndex_ = std::clamp(selectedIndex_, 0, static_cast<int>(entries_.size()) - 1);
    lastResult_.valid = false;
    char buf[64];
    std::snprintf(buf, sizeof(buf), u8"已删除第 %d 组标定点", index + 1);
    SetStatus(buf);
}

void Camera2DCalibrationWindow::ClearAll() {
    for (CalibEntry& e : entries_) DestroyImageSlot(e.image);
    entries_.clear();
    EnsureDefaultEntries();
    lastResult_ = {};
    lastError_.clear();
    lastErrorStats_ = {};
    roiDragging_ = false;
    SetStatus(u8"已清空标定数据");
}

void Camera2DCalibrationWindow::DetectCircleForEntry(CalibEntry& entry, bool quiet) {
    if (!entry.image.valid() || !entry.hasRoi) {
        if (!quiet) SetStatus(u8"请先框选圆点区域");
        return;
    }

    Camera2DCalibration::DotDetectResult detect;
    std::string error;
    if (!Camera2DCalibration::DetectCalibrationDotCenterRgb(
            entry.image.rgb, entry.image.width, entry.image.height, entry.roiX0, entry.roiY0,
            entry.roiX1, entry.roiY1, detect, error)) {
        entry.hasPixel = false;
        entry.detectedRadius = 0.f;
        if (!quiet) SetStatus(error.c_str());
        return;
    }

    entry.pixelU = detect.centerX;
    entry.pixelV = detect.centerY;
    entry.hasPixel = true;
    entry.detectedRadius = detect.radius;
    entry.calibError = -1.f;
    lastResult_.valid = false;
    lastErrorStats_ = {};

    if (!quiet) {
        char buf[160];
        std::snprintf(buf, sizeof(buf), u8"圆心 (%.2f, %.2f)，半径 %.2f px", entry.pixelU,
                      entry.pixelV, entry.detectedRadius);
        SetStatus(buf);
    }
}

void Camera2DCalibrationWindow::ComputeCalibration() {
    std::vector<Camera2DCalibration::PointPair> pairs;
    pairs.reserve(entries_.size());
    for (const CalibEntry& e : entries_) {
        Camera2DCalibration::PointPair p;
        p.imageU = e.pixelU;
        p.imageV = e.pixelV;
        p.robotX = e.robotX;
        p.robotY = e.robotY;
        p.hasImagePoint = e.hasPixel;
        p.hasRobotCoord = e.hasRobot;
        pairs.push_back(p);
    }

    Camera2DCalibration::AffineResult result;
    std::string error;
    if (!Camera2DCalibration::ComputeAffine(pairs, result, error)) {
        lastResult_ = {};
        lastErrorStats_ = {};
        lastError_ = error;
        for (CalibEntry& e : entries_) e.calibError = -1.f;
        SetStatus(error.c_str());
        return;
    }

    lastResult_ = result;
    lastError_.clear();
    Camera2DCalibration::ComputeErrorStats(pairs, result, lastErrorStats_, error);

    for (CalibEntry& e : entries_) {
        e.calibError = -1.f;
        if (!e.hasPixel || !e.hasRobot) continue;
        float qx = 0.f;
        float qy = 0.f;
        Camera2DCalibration::ImageToRobot(e.pixelU, e.pixelV, result, qx, qy);
        const double dx = static_cast<double>(e.robotX) - static_cast<double>(qx);
        const double dy = static_cast<double>(e.robotY) - static_cast<double>(qy);
        e.calibError = static_cast<float>(std::sqrt(dx * dx + dy * dy));
    }

    char buf[256];
    if (lastErrorStats_.valid) {
        std::snprintf(buf, sizeof(buf), u8"标定完成：Errmax = %.4f mm，Errmean = %.4f mm",
                      lastErrorStats_.errMax, lastErrorStats_.errMean);
    } else {
        std::snprintf(buf, sizeof(buf), u8"标定完成：%d 组点，RMS = %.4f mm", result.pointCount,
                      result.rms);
    }
    SetStatus(buf);
}

void Camera2DCalibrationWindow::DrawPointTable(float height) {
    ImGui::TextDisabled(u8"标定点列表（每张图像对应一组机器人坐标）");
    ImGui::Spacing();

    if (ImGui::Button(u8"添加标定点")) AddEntry();
    ImGui::SameLine();
    if (ImGui::Button(u8"批量加载图像…")) LoadImagesBatch();
    ImGui::SameLine();
    ImGui::TextDisabled(u8"共 %zu 组", entries_.size());

    const ImGuiTableFlags tableFlags =
        ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY |
        ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingStretchProp;
    if (!ImGui::BeginTable(u8"##calib_points", lastErrorStats_.valid ? 8 : 7, tableFlags,
                           ImVec2(0.f, height))) return;

    ImGui::TableSetupColumn(u8"#", ImGuiTableColumnFlags_WidthFixed, 28.f);
    ImGui::TableSetupColumn(u8"图像", ImGuiTableColumnFlags_WidthStretch, 1.6f);
    ImGui::TableSetupColumn(u8"像素 u", ImGuiTableColumnFlags_WidthFixed, 68.f);
    ImGui::TableSetupColumn(u8"像素 v", ImGuiTableColumnFlags_WidthFixed, 68.f);
    ImGui::TableSetupColumn(u8"机器人 X", ImGuiTableColumnFlags_WidthFixed, 80.f);
    ImGui::TableSetupColumn(u8"机器人 Y", ImGuiTableColumnFlags_WidthFixed, 80.f);
    if (lastErrorStats_.valid) {
        ImGui::TableSetupColumn(u8"误差", ImGuiTableColumnFlags_WidthFixed, 64.f);
    }
    ImGui::TableSetupColumn(u8"操作", ImGuiTableColumnFlags_WidthFixed, 64.f);
    ImGui::TableHeadersRow();

    for (int i = 0; i < static_cast<int>(entries_.size()); ++i) {
        CalibEntry& entry = entries_[static_cast<std::size_t>(i)];
        ImGui::TableNextRow();
        ImGui::PushID(i);

        ImGui::TableSetColumnIndex(0);
        const bool selected = (selectedIndex_ == i);
        if (ImGui::Selectable(std::to_string(i + 1).c_str(), selected,
                              ImGuiSelectableFlags_None)) {
            selectedIndex_ = i;
            roiDragging_ = false;
        }

        ImGui::TableSetColumnIndex(1);
        if (entry.image.valid()) {
            if (ImGui::Selectable(BaseName(entry.image.path).c_str(), false,
                                  ImGuiSelectableFlags_None)) {
                selectedIndex_ = i;
                roiDragging_ = false;
            }
        } else {
            ImGui::TextDisabled(u8"(未加载)");
        }
        ImGui::SameLine();
        if (ImGui::SmallButton(u8"加载")) LoadImageForEntry(i);

        ImGui::TableSetColumnIndex(2);
        if (entry.hasPixel) {
            ImGui::Text("%.2f", entry.pixelU);
        } else {
            ImGui::TextDisabled(u8"-");
        }

        ImGui::TableSetColumnIndex(3);
        if (entry.hasPixel) {
            ImGui::Text("%.2f", entry.pixelV);
        } else {
            ImGui::TextDisabled(u8"-");
        }

        ImGui::TableSetColumnIndex(4);
        ImGui::SetNextItemWidth(-1.f);
        ImGui::InputFloat(u8"##rx", &entry.robotX, 0.f, 0.f, "%.4f");
        if (ImGui::IsItemDeactivatedAfterEdit() || ImGui::IsItemActive()) {
            entry.hasRobot = true;
        }

        ImGui::TableSetColumnIndex(5);
        ImGui::SetNextItemWidth(-1.f);
        ImGui::InputFloat(u8"##ry", &entry.robotY, 0.f, 0.f, "%.4f");
        if (ImGui::IsItemDeactivatedAfterEdit() || ImGui::IsItemActive()) {
            entry.hasRobot = true;
        }

        if (lastErrorStats_.valid) {
            ImGui::TableSetColumnIndex(6);
            if (entry.calibError >= 0.f) {
                ImGui::Text("%.3f", entry.calibError);
            } else {
                ImGui::TextDisabled(u8"-");
            }
            ImGui::TableSetColumnIndex(7);
        } else {
            ImGui::TableSetColumnIndex(6);
        }
        if (ImGui::SmallButton(u8"删除")) RemoveEntry(i);

        ImGui::PopID();
    }
    ImGui::EndTable();
}

void Camera2DCalibrationWindow::DrawOverlays(ImDrawList* dl, const ImVec2& imgPos, float drawW,
                                             float drawH, const CalibEntry& entry) const {
    const float sx = drawW / static_cast<float>(std::max(entry.image.width, 1));
    const float sy = drawH / static_cast<float>(std::max(entry.image.height, 1));

    if (entry.hasRoi) {
        float x0 = entry.roiX0;
        float y0 = entry.roiY0;
        float x1 = entry.roiX1;
        float y1 = entry.roiY1;
        NormalizeRoi(x0, y0, x1, y1);
        const ImVec2 p0(imgPos.x + x0 * sx, imgPos.y + y0 * sy);
        const ImVec2 p1(imgPos.x + x1 * sx, imgPos.y + y1 * sy);
        dl->AddRect(p0, p1, kRoiCol, 0.f, 0, 2.f);
    }

    if (entry.hasPixel) {
        const float px = imgPos.x + entry.pixelU * sx;
        const float py = imgPos.y + entry.pixelV * sy;
        if (entry.detectedRadius > 0.f) {
            dl->AddCircle(ImVec2(px, py), entry.detectedRadius * sx, kCircleCol, 0, 2.f);
        }
        dl->AddCircleFilled(ImVec2(px, py), 6.f, kCenterFillCol);
        dl->AddCircle(ImVec2(px, py), 6.f, kCenterCol, 0, 2.f);
        dl->AddLine(ImVec2(px - 12.f, py), ImVec2(px + 12.f, py), kCenterCol, 1.5f);
        dl->AddLine(ImVec2(px, py - 12.f), ImVec2(px, py + 12.f), kCenterCol, 1.5f);
    }
}

void Camera2DCalibrationWindow::DrawImageCanvas() {
    if (selectedIndex_ < 0 || selectedIndex_ >= static_cast<int>(entries_.size())) {
        ImGui::TextDisabled(u8"请选择标定点");
        return;
    }

    CalibEntry& entry = entries_[static_cast<std::size_t>(selectedIndex_)];
    if (!entry.image.valid()) {
        ImGui::TextDisabled(u8"请先为当前行加载图像");
        return;
    }

    const ImVec2 avail = ImGui::GetContentRegionAvail();
    const float areaH = std::max(avail.y, 200.f);
    ImGui::BeginChild(u8"##calib_canvas", ImVec2(avail.x, areaH), ImGuiChildFlags_Borders);

    if (selectedIndex_ != lastCanvasIndex_) {
        panX_ = 0.f;
        panY_ = 0.f;
        zoom_ = 1.f;
        roiDragging_ = false;
        lastCanvasIndex_ = selectedIndex_;
    }

    const float imgAspect =
        (entry.image.height > 0)
            ? static_cast<float>(entry.image.width) / static_cast<float>(entry.image.height)
            : 1.f;
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

    const ImVec2 canvasPos = ImGui::GetCursorScreenPos();
    const ImVec2 imgPos(canvasPos.x + basePanX + panX_, canvasPos.y + basePanY + panY_);

    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddImage((ImTextureID)(intptr_t)entry.image.texId, imgPos,
                 ImVec2(imgPos.x + drawW, imgPos.y + drawH));
    DrawOverlays(dl, imgPos, drawW, drawH, entry);

    ImGuiIO& io = ImGui::GetIO();
    const bool canvasHovered = ImGui::IsWindowHovered();
    const bool onImage = io.MousePos.x >= imgPos.x && io.MousePos.y >= imgPos.y &&
                         io.MousePos.x <= imgPos.x + drawW && io.MousePos.y <= imgPos.y + drawH;

    if (canvasHovered && io.KeyShift && io.MouseWheel != 0.f) {
        zoom_ = std::clamp(zoom_ * (1.f + io.MouseWheel * 0.12f), 0.1f, 16.f);
    }
    if (canvasHovered && io.KeyShift &&
        ImGui::IsMouseDragging(ImGuiMouseButton_Left, 0.f)) {
        panX_ += io.MouseDelta.x;
        panY_ += io.MouseDelta.y;
    }

    if (onImage && !io.KeyShift) {
        const float mx = std::clamp(io.MousePos.x, imgPos.x, imgPos.x + drawW);
        const float my = std::clamp(io.MousePos.y, imgPos.y, imgPos.y + drawH);

        if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            roiDragging_ = true;
            float px0 = 0.f;
            float py0 = 0.f;
            ImageToPixel(entry.image, imgPos.x, imgPos.y, drawW, drawH, mx, my, px0, py0);
            entry.roiX0 = entry.roiX1 = px0;
            entry.roiY0 = entry.roiY1 = py0;
            entry.hasRoi = true;
            entry.hasPixel = false;
            entry.detectedRadius = 0.f;
        }

        if (roiDragging_) {
            float px1 = 0.f;
            float py1 = 0.f;
            ImageToPixel(entry.image, imgPos.x, imgPos.y, drawW, drawH, mx, my, px1, py1);
            entry.roiX1 = px1;
            entry.roiY1 = py1;
            float x0 = entry.roiX0;
            float y0 = entry.roiY0;
            float x1 = entry.roiX1;
            float y1 = entry.roiY1;
            NormalizeRoi(x0, y0, x1, y1);
            const ImVec2 p0(imgPos.x + x0 * drawW / static_cast<float>(entry.image.width),
                            imgPos.y + y0 * drawH / static_cast<float>(entry.image.height));
            const ImVec2 p1(imgPos.x + x1 * drawW / static_cast<float>(entry.image.width),
                            imgPos.y + y1 * drawH / static_cast<float>(entry.image.height));
            dl->AddRect(p0, p1, kRoiCol, 0.f, 0, 2.f);
        }
    }

    if (roiDragging_ && !io.KeyShift && ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
        roiDragging_ = false;
        DetectCircleForEntry(entry);
    }
    if (roiDragging_ && io.KeyShift) {
        roiDragging_ = false;
    }

    ImGui::Dummy(ImVec2(avail.x, areaH));

    ImGui::EndChild();
}

void Camera2DCalibrationWindow::Draw(float menuBottomY, float bottomInset) {
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
    if (!ImGui::Begin(u8"2D 相机九点标定", &open, flags)) {
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

    if (ImGui::Button(u8"关闭")) SetVisible(false);
    ImGui::SameLine();
    ImGui::TextDisabled(u8"拖动列表与图像之间的分割条调整宽度 | Shift+左键平移 | Shift+滚轮缩放 | 左键框选圆点");

    const float bottomH = 28.f;
    const ImVec2 contentAvail = ImGui::GetContentRegionAvail();
    const float mainH = std::max(contentAvail.y - bottomH, 200.f);
    const float availW = contentAvail.x;
    const float maxLeftW = std::max(availW - 320.f, 280.f);
    const float leftW = std::clamp(listPanelPreferredW_, 280.f, maxLeftW);

    ImGui::BeginChild(u8"##calib_main_row", ImVec2(0.f, mainH), false);
    ImGui::BeginChild(u8"##calib_left", ImVec2(leftW, 0.f), ImGuiChildFlags_Borders);
    DrawPointTable(std::max(ImGui::GetContentRegionAvail().y - 220.f, 160.f));

    ImGui::Spacing();
    if (selectedIndex_ >= 0 && selectedIndex_ < static_cast<int>(entries_.size())) {
        CalibEntry& entry = entries_[static_cast<std::size_t>(selectedIndex_)];
        if (ImGui::Button(u8"重新检测圆心", ImVec2(-1.f, 32.f))) {
            DetectCircleForEntry(entry);
        }
        if (entry.hasPixel) {
            ImGui::Text(u8"圆心: (%.2f, %.2f)", entry.pixelU, entry.pixelV);
            if (entry.detectedRadius > 0.f) {
                ImGui::Text(u8"半径: %.2f px", entry.detectedRadius);
            }
        } else if (entry.hasRoi) {
            ImGui::TextDisabled(u8"已框选，等待检测圆心");
        } else {
            ImGui::TextDisabled(u8"请在右侧图像上框选圆点");
        }
    }

    ImGui::Spacing();
    if (ImGui::Button(u8"计算标定", ImVec2(-1.f, 36.f))) ComputeCalibration();
    if (ImGui::Button(u8"清空全部", ImVec2(-1.f, 0))) ClearAll();

    ImGui::Spacing();
    ImGui::Separator();
    if (lastResult_.valid) {
        ImGui::TextWrapped(u8"X = %.6f·u + %.6f·v + %.6f", lastResult_.a, lastResult_.b,
                          lastResult_.c);
        ImGui::TextWrapped(u8"Y = %.6f·u + %.6f·v + %.6f", lastResult_.d, lastResult_.e,
                          lastResult_.f);
        if (lastErrorStats_.valid) {
            ImGui::TextColored(ImVec4(0.45f, 0.95f, 0.70f, 1.f), u8"Errmax = %.4f mm",
                               lastErrorStats_.errMax);
            ImGui::Text(u8"Errmean = %.4f mm    RMS = %.4f mm", lastErrorStats_.errMean,
                       lastResult_.rms);
        } else {
            ImGui::Text(u8"RMS: %.4f mm (%d 点)", lastResult_.rms, lastResult_.pointCount);
        }
    } else if (!lastError_.empty()) {
        ImGui::TextColored(ImVec4(1.f, 0.5f, 0.45f, 1.f), "%s", lastError_.c_str());
    } else {
        ImGui::TextDisabled(u8"完成取点并填写机器人坐标后计算");
    }
    ImGui::EndChild();

    ImGui::SameLine(0.f, 0.f);
    ImGui::InvisibleButton(u8"##calib_list_split", ImVec2(6.f, mainH));
    if (ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
        listPanelPreferredW_ =
            std::clamp(ImGui::GetIO().MousePos.x - ImGui::GetWindowPos().x, 280.f, maxLeftW);
    }
    if (ImGui::IsItemHovered() || ImGui::IsItemActive()) {
        ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
    }

    ImGui::SameLine(0.f, 0.f);
    ImGui::BeginChild(u8"##calib_right", ImVec2(0.f, 0.f), ImGuiChildFlags_Borders);
    if (selectedIndex_ >= 0 && selectedIndex_ < static_cast<int>(entries_.size())) {
        const CalibEntry& entry = entries_[static_cast<std::size_t>(selectedIndex_)];
        if (entry.image.valid()) {
            ImGui::Text(u8"第 %d 组：%s", selectedIndex_ + 1, BaseName(entry.image.path).c_str());
        } else {
            ImGui::Text(u8"第 %d 组预览", selectedIndex_ + 1);
        }
    }
    DrawImageCanvas();
    ImGui::EndChild();
    ImGui::EndChild();

    if (!statusText_.empty()) {
        ImGui::TextDisabled("%s", statusText_.c_str());
    }

    ImGui::End();
    ImGui::PopStyleVar(2);
}

Camera2DCalibrationWindow::~Camera2DCalibrationWindow() {
    for (CalibEntry& e : entries_) DestroyImageSlot(e.image);
}
