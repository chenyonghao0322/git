#include "app/CameraIntrinsicsCalibrationWindow.h"

#include "app/FileDialog.h"
#include "io/ImageIO.h"

#include <glad/gl.h>
#include <imgui.h>

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace {
constexpr ImU32 kCornerCol = IM_COL32(40, 220, 120, 255);
constexpr ImU32 kCornerFillCol = IM_COL32(40, 220, 120, 160);
}  // namespace

void CameraIntrinsicsCalibrationWindow::SetVisible(bool visible) {
    if (visible && !visible_) focusOnOpen_ = true;
    visible_ = visible;
}

void CameraIntrinsicsCalibrationWindow::ToggleVisible() { SetVisible(!visible_); }

void CameraIntrinsicsCalibrationWindow::SetStatus(const char* msg) {
    statusText_ = msg ? msg : "";
    if (onStatus_) onStatus_(msg);
}

std::string CameraIntrinsicsCalibrationWindow::BaseName(const std::string& path) {
    const std::size_t p1 = path.find_last_of("/\\");
    return (p1 == std::string::npos) ? path : path.substr(p1 + 1);
}

void CameraIntrinsicsCalibrationWindow::DestroyImageSlot(ImageSlot& slot) {
    if (slot.texId) {
        glDeleteTextures(1, &slot.texId);
        slot.texId = 0;
    }
    slot.width = 0;
    slot.height = 0;
    slot.rgb.clear();
    slot.path.clear();
}

bool CameraIntrinsicsCalibrationWindow::UploadTexture(ImageSlot& slot) {
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

bool CameraIntrinsicsCalibrationWindow::LoadImageSlot(ImageSlot& slot, const std::string& path,
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

void CameraIntrinsicsCalibrationWindow::LoadImagesBatch() {
    const std::vector<std::string> paths =
        FileDialog::OpenMultipleImageFiles(u8"批量选择棋盘标定图像");
    if (paths.empty()) return;

    entries_.clear();
    entries_.reserve(paths.size());
    std::string error;
    int loaded = 0;
    int detected = 0;
    for (const std::string& path : paths) {
        CalibEntry entry;
        if (!LoadImageSlot(entry.image, path, error)) {
            SetStatus(error.c_str());
            break;
        }
        DetectCornersForEntry(entry, true);
        if (entry.hasCorners) ++detected;
        entries_.push_back(std::move(entry));
        ++loaded;
    }
    selectedIndex_ = entries_.empty() ? -1 : 0;
    lastResult_ = {};
    lastError_.clear();
    char buf[192];
    std::snprintf(buf, sizeof(buf), u8"已加载 %d 张图像，成功检测角点 %d 张", loaded, detected);
    SetStatus(buf);
}

void CameraIntrinsicsCalibrationWindow::RemoveEntry(int index) {
    if (index < 0 || index >= static_cast<int>(entries_.size())) return;
    DestroyImageSlot(entries_[static_cast<std::size_t>(index)].image);
    entries_.erase(entries_.begin() + index);
    if (entries_.empty()) {
        selectedIndex_ = -1;
    } else if (selectedIndex_ >= static_cast<int>(entries_.size())) {
        selectedIndex_ = static_cast<int>(entries_.size()) - 1;
    }
    lastResult_ = {};
}

void CameraIntrinsicsCalibrationWindow::ClearAll() {
    for (CalibEntry& e : entries_) DestroyImageSlot(e.image);
    entries_.clear();
    selectedIndex_ = -1;
    lastResult_ = {};
    lastError_.clear();
    SetStatus(u8"已清空");
}

void CameraIntrinsicsCalibrationWindow::DetectCornersForEntry(CalibEntry& entry, bool quiet) {
    entry.hasCorners = false;
    entry.cornersX.clear();
    entry.cornersY.clear();
    entry.reprojError = -1.f;
    if (!entry.image.valid()) {
        if (!quiet) SetStatus(u8"请先加载图像");
        return;
    }

    CameraIntrinsicsCalibration::CornerDetectResult detect;
    std::string error;
    if (!CameraIntrinsicsCalibration::DetectChessboardCorners(
            entry.image.rgb, entry.image.width, entry.image.height, pattern_, detect, error)) {
        if (!quiet) SetStatus(error.c_str());
        return;
    }

    entry.cornersX = std::move(detect.cornersX);
    entry.cornersY = std::move(detect.cornersY);
    entry.hasCorners = true;
    lastResult_ = {};
    if (!quiet) SetStatus(u8"角点检测成功");
}

void CameraIntrinsicsCalibrationWindow::DetectCornersAll() {
    int ok = 0;
    for (CalibEntry& e : entries_) {
        DetectCornersForEntry(e, true);
        if (e.hasCorners) ++ok;
    }
    lastResult_ = {};
    char buf[128];
    std::snprintf(buf, sizeof(buf), u8"角点检测完成：%d / %zu 张成功", ok, entries_.size());
    SetStatus(buf);
}

void CameraIntrinsicsCalibrationWindow::ComputeCalibration() {
    std::vector<CameraIntrinsicsCalibration::ImageObservation> obs;
    obs.reserve(entries_.size());
    for (const CalibEntry& e : entries_) {
        if (!e.hasCorners) continue;
        CameraIntrinsicsCalibration::ImageObservation o;
        o.rgb = &e.image.rgb;
        o.width = e.image.width;
        o.height = e.image.height;
        o.cornersX = &e.cornersX;
        o.cornersY = &e.cornersY;
        o.hasCorners = true;
        obs.push_back(o);
    }

    CameraIntrinsicsCalibration::IntrinsicsResult result;
    std::string error;
    if (!CameraIntrinsicsCalibration::CalibrateIntrinsics(obs, pattern_, result, error)) {
        lastResult_ = {};
        lastError_ = error;
        SetStatus(error.c_str());
        return;
    }

    lastResult_ = result;
    lastError_.clear();

    int obsIdx = 0;
    for (CalibEntry& e : entries_) {
        e.reprojError = -1.f;
        if (!e.hasCorners) continue;
        if (obsIdx < static_cast<int>(result.perImageReprojError.size())) {
            e.reprojError =
                static_cast<float>(result.perImageReprojError[static_cast<std::size_t>(obsIdx)]);
        }
        ++obsIdx;
    }

    char buf[256];
    std::snprintf(buf, sizeof(buf), u8"内参标定完成：%d 张图，重投影 Errmax=%.4f px，Errmean=%.4f px",
                  result.imageCount, result.reprojMax, result.reprojMean);
    SetStatus(buf);
}

void CameraIntrinsicsCalibrationWindow::DrawImageTable(float height) {
    ImGui::TextDisabled(u8"标定图像列表（建议 10 张以上、多角度拍摄）");
    ImGui::Spacing();

    if (ImGui::Button(u8"批量加载图像…")) LoadImagesBatch();
    ImGui::SameLine();
    if (ImGui::Button(u8"检测全部角点")) DetectCornersAll();
    ImGui::SameLine();
    ImGui::TextDisabled(u8"共 %zu 张", entries_.size());

    const ImGuiTableFlags tableFlags =
        ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY |
        ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingStretchProp;
    const int colCount = lastResult_.valid ? 4 : 3;
    if (!ImGui::BeginTable(u8"##intr_calib_images", colCount, tableFlags,
                           ImVec2(0.f, height))) return;

    ImGui::TableSetupColumn(u8"#", ImGuiTableColumnFlags_WidthFixed, 28.f);
    ImGui::TableSetupColumn(u8"图像", ImGuiTableColumnFlags_WidthStretch, 1.6f);
    ImGui::TableSetupColumn(u8"角点", ImGuiTableColumnFlags_WidthFixed, 52.f);
    if (lastResult_.valid) {
        ImGui::TableSetupColumn(u8"重投影", ImGuiTableColumnFlags_WidthFixed, 64.f);
    }
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
        }

        ImGui::TableSetColumnIndex(1);
        if (entry.image.valid()) {
            if (ImGui::Selectable(BaseName(entry.image.path).c_str(), false,
                                  ImGuiSelectableFlags_None)) {
                selectedIndex_ = i;
            }
        } else {
            ImGui::TextDisabled(u8"(无效)");
        }

        ImGui::TableSetColumnIndex(2);
        if (entry.hasCorners) {
            ImGui::TextColored(ImVec4(0.4f, 0.9f, 0.6f, 1.f), u8"OK");
        } else {
            ImGui::TextColored(ImVec4(1.f, 0.5f, 0.45f, 1.f), u8"—");
        }

        if (lastResult_.valid) {
            ImGui::TableSetColumnIndex(3);
            if (entry.reprojError >= 0.f) {
                ImGui::Text("%.3f", entry.reprojError);
            } else {
                ImGui::TextDisabled(u8"-");
            }
        }

        ImGui::PopID();
    }
    ImGui::EndTable();
}

void CameraIntrinsicsCalibrationWindow::DrawCornerOverlay(ImDrawList* dl, const ImVec2& imgPos,
                                                          float drawW, float drawH,
                                                          const CalibEntry& entry) const {
    if (!entry.hasCorners) return;
    const float sx = drawW / static_cast<float>(std::max(entry.image.width, 1));
    const float sy = drawH / static_cast<float>(std::max(entry.image.height, 1));
    for (std::size_t i = 0; i < entry.cornersX.size(); ++i) {
        const float px = imgPos.x + entry.cornersX[i] * sx;
        const float py = imgPos.y + entry.cornersY[i] * sy;
        dl->AddCircleFilled(ImVec2(px, py), 4.f, kCornerFillCol);
        dl->AddCircle(ImVec2(px, py), 4.f, kCornerCol, 0, 1.5f);
    }
}

void CameraIntrinsicsCalibrationWindow::DrawImageCanvas() {
    if (selectedIndex_ < 0 || selectedIndex_ >= static_cast<int>(entries_.size())) {
        ImGui::TextDisabled(u8"请选择标定图像");
        return;
    }

    CalibEntry& entry = entries_[static_cast<std::size_t>(selectedIndex_)];
    if (!entry.image.valid()) {
        ImGui::TextDisabled(u8"图像无效");
        return;
    }

    const ImVec2 avail = ImGui::GetContentRegionAvail();
    const float areaH = std::max(avail.y, 200.f);
    ImGui::BeginChild(u8"##intr_calib_canvas", ImVec2(avail.x, areaH), ImGuiChildFlags_Borders);

    if (selectedIndex_ != lastCanvasIndex_) {
        panX_ = 0.f;
        panY_ = 0.f;
        zoom_ = 1.f;
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
    DrawCornerOverlay(dl, imgPos, drawW, drawH, entry);

    ImGuiIO& io = ImGui::GetIO();
    const bool canvasHovered = ImGui::IsWindowHovered();
    if (canvasHovered && io.KeyShift && io.MouseWheel != 0.f) {
        zoom_ = std::clamp(zoom_ * (1.f + io.MouseWheel * 0.12f), 0.1f, 16.f);
    }
    if (canvasHovered && io.KeyShift && ImGui::IsMouseDragging(ImGuiMouseButton_Left, 0.f)) {
        panX_ += io.MouseDelta.x;
        panY_ += io.MouseDelta.y;
    }

    ImGui::Dummy(ImVec2(avail.x, areaH));
    ImGui::EndChild();
}

void CameraIntrinsicsCalibrationWindow::Draw(float menuBottomY, float bottomInset) {
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
    if (!ImGui::Begin(u8"相机内参标定", &open, flags)) {
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
    ImGui::TextDisabled(
        u8"拖动分割条调整宽度 | Shift+左键平移 | Shift+滚轮缩放 | 棋盘内角点行列数需与标定板一致");

    const float bottomH = 28.f;
    const ImVec2 contentAvail = ImGui::GetContentRegionAvail();
    const float mainH = std::max(contentAvail.y - bottomH, 200.f);
    const float availW = contentAvail.x;
    const float maxLeftW = std::max(availW - 320.f, 280.f);
    const float leftW = std::clamp(listPanelPreferredW_, 280.f, maxLeftW);

    ImGui::BeginChild(u8"##intr_calib_main_row", ImVec2(0.f, mainH), false);
    ImGui::BeginChild(u8"##intr_calib_left", ImVec2(leftW, 0.f), ImGuiChildFlags_Borders);

    ImGui::TextDisabled(u8"棋盘参数");
    ImGui::SetNextItemWidth(100.f);
    ImGui::InputInt(u8"内角点列数", &pattern_.innerCols, 0, 0);
    pattern_.innerCols = std::max(2, pattern_.innerCols);
    ImGui::SetNextItemWidth(100.f);
    ImGui::InputInt(u8"内角点行数", &pattern_.innerRows, 0, 0);
    pattern_.innerRows = std::max(2, pattern_.innerRows);
    ImGui::SetNextItemWidth(120.f);
    ImGui::InputDouble(u8"方格边长 (mm)", &pattern_.squareSizeMm, 0.0, 0.0, "%.3f");
    if (pattern_.squareSizeMm <= 0.0) pattern_.squareSizeMm = 1.0;

    ImGui::Spacing();
    DrawImageTable(std::max(ImGui::GetContentRegionAvail().y - 300.f, 120.f));

    ImGui::Spacing();
    if (selectedIndex_ >= 0 && selectedIndex_ < static_cast<int>(entries_.size())) {
        if (ImGui::Button(u8"检测当前图角点", ImVec2(-1.f, 32.f))) {
            DetectCornersForEntry(entries_[static_cast<std::size_t>(selectedIndex_)]);
        }
        if (ImGui::Button(u8"删除当前图", ImVec2(-1.f, 0))) {
            RemoveEntry(selectedIndex_);
        }
    }

    ImGui::Spacing();
    if (ImGui::Button(u8"计算内参", ImVec2(-1.f, 36.f))) ComputeCalibration();
    if (ImGui::Button(u8"清空全部", ImVec2(-1.f, 0))) ClearAll();

    ImGui::Spacing();
    ImGui::Separator();
    if (lastResult_.valid) {
        ImGui::Text(u8"fx = %.4f    fy = %.4f", lastResult_.fx, lastResult_.fy);
        ImGui::Text(u8"cx = %.4f    cy = %.4f", lastResult_.cx, lastResult_.cy);
        ImGui::Text(u8"k1 = %.6f  k2 = %.6f  k3 = %.6f", lastResult_.k1, lastResult_.k2,
                   lastResult_.k3);
        ImGui::Text(u8"p1 = %.6f  p2 = %.6f", lastResult_.p1, lastResult_.p2);
        ImGui::TextColored(ImVec4(0.45f, 0.95f, 0.70f, 1.f), u8"Errmax = %.4f px",
                           lastResult_.reprojMax);
        ImGui::Text(u8"Errmean = %.4f px (%d 张图)", lastResult_.reprojMean,
                   lastResult_.imageCount);
    } else if (!lastError_.empty()) {
        ImGui::TextColored(ImVec4(1.f, 0.5f, 0.45f, 1.f), "%s", lastError_.c_str());
    } else {
        ImGui::TextDisabled(u8"加载棋盘图像并检测角点后计算");
    }
    ImGui::EndChild();

    ImGui::SameLine(0.f, 0.f);
    ImGui::InvisibleButton(u8"##intr_calib_split", ImVec2(6.f, mainH));
    if (ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
        listPanelPreferredW_ =
            std::clamp(ImGui::GetIO().MousePos.x - ImGui::GetWindowPos().x, 280.f, maxLeftW);
    }
    if (ImGui::IsItemHovered() || ImGui::IsItemActive()) {
        ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
    }

    ImGui::SameLine(0.f, 0.f);
    ImGui::BeginChild(u8"##intr_calib_right", ImVec2(0.f, 0.f), ImGuiChildFlags_Borders);
    if (selectedIndex_ >= 0 && selectedIndex_ < static_cast<int>(entries_.size())) {
        const CalibEntry& entry = entries_[static_cast<std::size_t>(selectedIndex_)];
        if (entry.image.valid()) {
            ImGui::Text(u8"第 %d 张：%s", selectedIndex_ + 1, BaseName(entry.image.path).c_str());
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

CameraIntrinsicsCalibrationWindow::~CameraIntrinsicsCalibrationWindow() {
    for (CalibEntry& e : entries_) DestroyImageSlot(e.image);
}
