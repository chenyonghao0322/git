#include "app/MultiViewGeometryWindow.h"

#include "app/FileDialog.h"
#include "io/ImageIO.h"

#include <glad/gl.h>
#include <imgui.h>

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace {
constexpr ImU32 kPtCol = IM_COL32(255, 210, 60, 255);
constexpr ImU32 kSelCol = IM_COL32(80, 220, 255, 255);
constexpr ImU32 kPendingCol = IM_COL32(255, 120, 60, 255);
constexpr ImU32 kEpiCol = IM_COL32(40, 220, 120, 220);
}  // namespace

void MultiViewGeometryWindow::SetVisible(bool visible) {
    if (visible && !visible_) focusOnOpen_ = true;
    visible_ = visible;
}

void MultiViewGeometryWindow::ToggleVisible() { SetVisible(!visible_); }

void MultiViewGeometryWindow::SetStatus(const char* msg) {
    statusText_ = msg ? msg : "";
    if (onStatus_) onStatus_(msg);
}

std::string MultiViewGeometryWindow::BaseName(const std::string& path) {
    const std::size_t p1 = path.find_last_of("/\\");
    return (p1 == std::string::npos) ? path : path.substr(p1 + 1);
}

void MultiViewGeometryWindow::DestroyImageSlot(ImageSlot& slot) {
    if (slot.texId) {
        glDeleteTextures(1, &slot.texId);
        slot.texId = 0;
    }
    slot.width = 0;
    slot.height = 0;
    slot.rgb.clear();
    slot.path.clear();
}

bool MultiViewGeometryWindow::UploadTexture(ImageSlot& slot) {
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

bool MultiViewGeometryWindow::LoadImageSlot(ImageSlot& slot, const std::string& path,
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

bool MultiViewGeometryWindow::ImageToPixel(const ImageSlot& slot, float imgPosX, float imgPosY,
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

void MultiViewGeometryWindow::LoadImage1() {
    const std::string path = FileDialog::OpenImageFile(u8"选择视图 1 图像");
    if (path.empty()) return;
    std::string error;
    if (!LoadImageSlot(image1_, path, error)) {
        SetStatus(error.c_str());
        return;
    }
    if (intrinsics_.cx <= 0.0) intrinsics_.cx = image1_.width * 0.5;
    if (intrinsics_.cy <= 0.0) intrinsics_.cy = image1_.height * 0.5;
    fundamental_.valid = false;
    triangulation_.valid = false;
    SetStatus(u8"已加载视图 1");
}

void MultiViewGeometryWindow::LoadImage2() {
    const std::string path = FileDialog::OpenImageFile(u8"选择视图 2 图像");
    if (path.empty()) return;
    std::string error;
    if (!LoadImageSlot(image2_, path, error)) {
        SetStatus(error.c_str());
        return;
    }
    fundamental_.valid = false;
    triangulation_.valid = false;
    SetStatus(u8"已加载视图 2");
}

void MultiViewGeometryWindow::ClearPairs() {
    pairs_.clear();
    selectedPair_ = -1;
    hasPendingView1_ = false;
    fundamental_.valid = false;
    triangulation_.valid = false;
    SetStatus(u8"已清空对应点");
}

void MultiViewGeometryWindow::RemovePair(int index) {
    if (index < 0 || index >= static_cast<int>(pairs_.size())) return;
    pairs_.erase(pairs_.begin() + index);
    if (pairs_.empty()) {
        selectedPair_ = -1;
    } else {
        selectedPair_ = std::clamp(selectedPair_, 0, static_cast<int>(pairs_.size()) - 1);
    }
    fundamental_.valid = false;
    triangulation_.valid = false;
}

void MultiViewGeometryWindow::ComputeFundamental() {
    std::string error;
    if (!MultiViewGeometry::ComputeFundamental(pairs_, fundamental_, error, ransacThresh_)) {
        fundamental_.valid = false;
        triangulation_.valid = false;
        lastError_ = error;
        SetStatus(error.c_str());
        return;
    }
    triangulation_.valid = false;
    lastError_.clear();
    char buf[160];
    std::snprintf(buf, sizeof(buf), u8"基础矩阵完成：内点 %d，平均极线误差 %.3f px",
                  fundamental_.inlierCount, fundamental_.meanEpipolarError);
    SetStatus(buf);
}

void MultiViewGeometryWindow::Triangulate() {
    std::string error;
    if (!MultiViewGeometry::TriangulateWithIntrinsics(pairs_, intrinsics_, fundamental_,
                                                      triangulation_, error)) {
        triangulation_.valid = false;
        lastError_ = error;
        SetStatus(error.c_str());
        return;
    }
    lastError_.clear();
    char buf[160];
    std::snprintf(buf, sizeof(buf), u8"三角化完成：%zu 点，平均重投影误差 %.3f px",
                  triangulation_.points.size(), triangulation_.meanReprojError);
    SetStatus(buf);
}

void MultiViewGeometryWindow::ImportToScene() {
    if (!triangulation_.valid || triangulation_.points.empty()) {
        SetStatus(u8"请先完成三角化");
        return;
    }
    if (onImportCloud_) onImportCloud_(triangulation_.points, u8"多视图几何三角化点云");
}

void MultiViewGeometryWindow::DrawEpipolarOverlay(ImDrawList* dl, const ImVec2& imgPos, float drawW,
                                                  float drawH, int viewIndex) const {
    (void)drawH;
    if (!fundamental_.valid || selectedPair_ < 0 ||
        selectedPair_ >= static_cast<int>(pairs_.size()) || viewIndex != 2) {
        return;
    }
    const MultiViewGeometry::Correspondence& p = pairs_[static_cast<std::size_t>(selectedPair_)];
    double a = 0.0;
    double b = 0.0;
    double c = 0.0;
    if (!MultiViewGeometry::EpipolarLineInImage2(fundamental_, p.u1, p.v1, a, b, c)) return;
    if (std::abs(b) < 1e-8) return;
    const float x0 = imgPos.x;
    const float x1 = imgPos.x + drawW;
    const float y0 = static_cast<float>(-(a * x0 + c) / b);
    const float y1 = static_cast<float>(-(a * x1 + c) / b);
    dl->AddLine(ImVec2(x0, y0), ImVec2(x1, y1), kEpiCol, 2.f);
}

void MultiViewGeometryWindow::DrawImagePane(const char* childId, ImageSlot& slot, ViewCanvas& canvas,
                                            int viewIndex, float width, float height) {
    ImGui::BeginChild(childId, ImVec2(width, height), ImGuiChildFlags_Borders);
    if (!slot.valid()) {
        ImGui::TextDisabled(viewIndex == 1 ? u8"请加载视图 1" : u8"请加载视图 2");
        ImGui::EndChild();
        return;
    }

    ImGui::Text("%s", BaseName(slot.path).c_str());
    const ImVec2 avail = ImGui::GetContentRegionAvail();
    const float areaH = std::max(avail.y, 120.f);

    const float imgAspect =
        (slot.height > 0) ? static_cast<float>(slot.width) / static_cast<float>(slot.height) : 1.f;
    float fitW = std::max(avail.x - 4.f, 1.f);
    float fitH = fitW / imgAspect;
    if (fitH > areaH - 4.f) {
        fitH = std::max(areaH - 4.f, 1.f);
        fitW = fitH * imgAspect;
    }
    const float drawW = std::max(fitW * canvas.zoom, 1.f);
    const float drawH = std::max(fitH * canvas.zoom, 1.f);
    const float basePanX = std::max((avail.x - drawW) * 0.5f, 0.f);
    const float basePanY = std::max((areaH - drawH) * 0.5f, 0.f);

    const ImVec2 canvasPos = ImGui::GetCursorScreenPos();
    const ImVec2 imgPos(canvasPos.x + basePanX + canvas.panX,
                        canvasPos.y + basePanY + canvas.panY);

    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddImage((ImTextureID)(intptr_t)slot.texId, imgPos,
                 ImVec2(imgPos.x + drawW, imgPos.y + drawH));

    const float sx = drawW / static_cast<float>(slot.width);
    const float sy = drawH / static_cast<float>(slot.height);
    for (int i = 0; i < static_cast<int>(pairs_.size()); ++i) {
        const MultiViewGeometry::Correspondence& p = pairs_[static_cast<std::size_t>(i)];
        const float u = (viewIndex == 1) ? p.u1 : p.u2;
        const float v = (viewIndex == 1) ? p.v1 : p.v2;
        const float px = imgPos.x + u * sx;
        const float py = imgPos.y + v * sy;
        const ImU32 col = (i == selectedPair_) ? kSelCol : kPtCol;
        dl->AddCircleFilled(ImVec2(px, py), 5.f, col);
        char label[8];
        std::snprintf(label, sizeof(label), "%d", i + 1);
        dl->AddText(ImVec2(px + 6.f, py - 6.f), col, label);
    }
    if (viewIndex == 1 && hasPendingView1_) {
        const float px = imgPos.x + pendingU1_ * sx;
        const float py = imgPos.y + pendingV1_ * sy;
        dl->AddCircle(ImVec2(px, py), 7.f, kPendingCol, 0, 2.f);
    }
    DrawEpipolarOverlay(dl, imgPos, drawW, drawH, viewIndex);

    ImGuiIO& io = ImGui::GetIO();
    const bool canvasHovered = ImGui::IsWindowHovered();
    if (canvasHovered && io.KeyShift && io.MouseWheel != 0.f) {
        canvas.zoom = std::clamp(canvas.zoom * (1.f + io.MouseWheel * 0.12f), 0.1f, 16.f);
    }
    if (canvasHovered && io.KeyShift &&
        ImGui::IsMouseDragging(ImGuiMouseButton_Left, 0.f)) {
        canvas.panX += io.MouseDelta.x;
        canvas.panY += io.MouseDelta.y;
    }

    const bool onImage = io.MousePos.x >= imgPos.x && io.MousePos.y >= imgPos.y &&
                         io.MousePos.x <= imgPos.x + drawW && io.MousePos.y <= imgPos.y + drawH;
    if (onImage && !io.KeyShift && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
        float u = 0.f;
        float v = 0.f;
        ImageToPixel(slot, imgPos.x, imgPos.y, drawW, drawH, io.MousePos.x, io.MousePos.y, u, v);
        if (viewIndex == 1) {
            hasPendingView1_ = true;
            pendingU1_ = u;
            pendingV1_ = v;
            SetStatus(u8"已在视图 1 取点，请在视图 2 点击对应点");
        } else if (hasPendingView1_) {
            MultiViewGeometry::Correspondence p;
            p.u1 = pendingU1_;
            p.v1 = pendingV1_;
            p.u2 = u;
            p.v2 = v;
            pairs_.push_back(p);
            selectedPair_ = static_cast<int>(pairs_.size()) - 1;
            hasPendingView1_ = false;
            fundamental_.valid = false;
            triangulation_.valid = false;
            char buf[64];
            std::snprintf(buf, sizeof(buf), u8"已添加第 %d 组对应点", selectedPair_ + 1);
            SetStatus(buf);
        }
    }

    ImGui::Dummy(ImVec2(avail.x, areaH));
    ImGui::EndChild();
}

void MultiViewGeometryWindow::DrawLeftPanel(float tableHeight) {
    ImGui::TextDisabled(u8"1. 加载两视图  2. 依次点击对应点  3. 计算 F  4. 填内参后三角化");
    ImGui::Spacing();
    if (ImGui::Button(u8"加载视图 1…")) LoadImage1();
    ImGui::SameLine();
    if (ImGui::Button(u8"加载视图 2…")) LoadImage2();
    ImGui::SameLine();
    if (ImGui::Button(u8"清空对应点")) ClearPairs();

    if (!ImGui::BeginTable(u8"##mv_pairs", 5, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                                                 ImGuiTableFlags_ScrollY | ImGuiTableFlags_Resizable,
                           ImVec2(0.f, tableHeight))) {
        return;
    }
    ImGui::TableSetupColumn(u8"#", ImGuiTableColumnFlags_WidthFixed, 24.f);
    ImGui::TableSetupColumn(u8"视图1 (u,v)", ImGuiTableColumnFlags_WidthStretch);
    ImGui::TableSetupColumn(u8"视图2 (u,v)", ImGuiTableColumnFlags_WidthStretch);
    ImGui::TableSetupColumn(u8"内点", ImGuiTableColumnFlags_WidthFixed, 36.f);
    ImGui::TableSetupColumn(u8"操作", ImGuiTableColumnFlags_WidthFixed, 48.f);
    ImGui::TableHeadersRow();
    for (int i = 0; i < static_cast<int>(pairs_.size()); ++i) {
        const MultiViewGeometry::Correspondence& p = pairs_[static_cast<std::size_t>(i)];
        ImGui::TableNextRow();
        ImGui::PushID(i);
        ImGui::TableSetColumnIndex(0);
        if (ImGui::Selectable(std::to_string(i + 1).c_str(), selectedPair_ == i)) {
            selectedPair_ = i;
        }
        ImGui::TableSetColumnIndex(1);
        ImGui::Text("(%.1f, %.1f)", p.u1, p.v1);
        ImGui::TableSetColumnIndex(2);
        ImGui::Text("(%.1f, %.1f)", p.u2, p.v2);
        ImGui::TableSetColumnIndex(3);
        if (fundamental_.valid && i < static_cast<int>(fundamental_.inlierMask.size()) &&
            fundamental_.inlierMask[static_cast<std::size_t>(i)]) {
            ImGui::TextColored(ImVec4(0.4f, 0.9f, 0.6f, 1.f), u8"是");
        } else if (fundamental_.valid) {
            ImGui::TextDisabled(u8"否");
        } else {
            ImGui::TextDisabled(u8"-");
        }
        ImGui::TableSetColumnIndex(4);
        if (ImGui::SmallButton(u8"删")) RemovePair(i);
        ImGui::PopID();
    }
    ImGui::EndTable();

    ImGui::Spacing();
    ImGui::Text(u8"RANSAC 极线阈值 (px)");
    ImGui::SetNextItemWidth(-1.f);
    ImGui::InputFloat(u8"##ransac", &ransacThresh_, 0.1f, 1.f, "%.2f");
    if (ImGui::Button(u8"计算基础矩阵 F", ImVec2(-1.f, 32.f))) ComputeFundamental();

    ImGui::Separator();
    ImGui::Text(u8"相机内参（两视图共用，像素）");
    ImGui::SetNextItemWidth(-1.f);
    ImGui::InputDouble(u8"fx", &intrinsics_.fx, 0.0, 0.0, "%.2f");
    ImGui::SetNextItemWidth(-1.f);
    ImGui::InputDouble(u8"fy", &intrinsics_.fy, 0.0, 0.0, "%.2f");
    ImGui::SetNextItemWidth(-1.f);
    ImGui::InputDouble(u8"cx", &intrinsics_.cx, 0.0, 0.0, "%.2f");
    ImGui::SetNextItemWidth(-1.f);
    ImGui::InputDouble(u8"cy", &intrinsics_.cy, 0.0, 0.0, "%.2f");
    if (ImGui::Button(u8"三角化 3D 点", ImVec2(-1.f, 32.f))) Triangulate();
    if (triangulation_.valid && ImGui::Button(u8"导入点云到场景", ImVec2(-1.f, 0.f))) {
        ImportToScene();
    }

    ImGui::Spacing();
    ImGui::Separator();
    if (fundamental_.valid) {
        ImGui::Text(u8"极线误差: 平均 %.3f px  最大 %.3f px", fundamental_.meanEpipolarError,
                    fundamental_.maxEpipolarError);
    }
    if (triangulation_.valid) {
        ImGui::TextColored(ImVec4(0.45f, 0.95f, 0.70f, 1.f), u8"3D 点数: %zu",
                           triangulation_.points.size());
        ImGui::Text(u8"重投影误差: 平均 %.3f px  最大 %.3f px", triangulation_.meanReprojError,
                    triangulation_.maxReprojError);
    } else if (!lastError_.empty()) {
        ImGui::TextColored(ImVec4(1.f, 0.5f, 0.45f, 1.f), "%s", lastError_.c_str());
    }
}

void MultiViewGeometryWindow::Draw(float menuBottomY, float bottomInset) {
    if (!visible_) return;

    ImGuiViewport* vp = ImGui::GetMainViewport();
    const float windowH = std::max(vp->Size.y - menuBottomY - bottomInset, 1.f);
    ImGui::SetNextWindowPos(ImVec2(vp->Pos.x, menuBottomY), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(vp->Size.x, windowH), ImGuiCond_Always);
    if (focusOnOpen_) {
        ImGui::SetNextWindowFocus();
        focusOnOpen_ = false;
    }

    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.f);
    const ImGuiWindowFlags flags =
        ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoSavedSettings;

    bool open = visible_;
    if (!ImGui::Begin(u8"多视图几何", &open, flags)) {
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
    ImGui::TextDisabled(u8"视图1取点→视图2取点成对 | Shift+左键平移 | Shift+滚轮缩放");

    const float bottomH = 24.f;
    const ImVec2 contentAvail = ImGui::GetContentRegionAvail();
    const float mainH = std::max(contentAvail.y - bottomH, 200.f);
    const float availW = contentAvail.x;
    const float maxLeftW = std::max(availW - 400.f, 300.f);
    const float leftW = std::clamp(listPanelPreferredW_, 300.f, maxLeftW);

    ImGui::BeginChild(u8"##mv_main", ImVec2(0.f, mainH), false);
    ImGui::BeginChild(u8"##mv_left", ImVec2(leftW, 0.f), ImGuiChildFlags_Borders);
    DrawLeftPanel(std::max(ImGui::GetContentRegionAvail().y - 280.f, 140.f));
    ImGui::EndChild();

    ImGui::SameLine(0.f, 0.f);
    ImGui::InvisibleButton(u8"##mv_split", ImVec2(6.f, mainH));
    if (ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
        listPanelPreferredW_ =
            std::clamp(ImGui::GetIO().MousePos.x - ImGui::GetWindowPos().x, 300.f, maxLeftW);
    }
    if (ImGui::IsItemHovered() || ImGui::IsItemActive()) {
        ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
    }

    ImGui::SameLine(0.f, 0.f);
    ImGui::BeginChild(u8"##mv_right", ImVec2(0.f, 0.f), ImGuiChildFlags_Borders);
    const float paneW = std::max((ImGui::GetContentRegionAvail().x - 6.f) * 0.5f, 120.f);
    const float paneH = ImGui::GetContentRegionAvail().y;
    DrawImagePane(u8"##mv_view1", image1_, canvas1_, 1, paneW, paneH);
    ImGui::SameLine();
    DrawImagePane(u8"##mv_view2", image2_, canvas2_, 2, paneW, paneH);
    ImGui::EndChild();
    ImGui::EndChild();

    if (!statusText_.empty()) ImGui::TextDisabled("%s", statusText_.c_str());
    ImGui::End();
    ImGui::PopStyleVar(2);
}

MultiViewGeometryWindow::~MultiViewGeometryWindow() {
    DestroyImageSlot(image1_);
    DestroyImageSlot(image2_);
}
