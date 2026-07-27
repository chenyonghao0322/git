#include "app/MeasurementRecipeWindow.h"

#include "app/FileDialog.h"
#include "io/ImageIO.h"

#include <glad/gl.h>
#include <imgui.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>

namespace {
constexpr float kPi = 3.14159265358979323846f;
constexpr float kDegToRad = kPi / 180.f;
constexpr ImU32 kContourCol = IM_COL32(40, 220, 80, 255);
constexpr ImU32 kMeasureCol = IM_COL32(255, 200, 40, 255);
constexpr ImU32 kRoiCol = IM_COL32(80, 180, 255, 220);
constexpr ImU32 kTextBg = IM_COL32(0, 0, 0, 160);

bool DragDouble(const char* label, double* v, float speed, double vMin, double vMax,
                const char* fmt) {
    return ImGui::DragScalar(label, ImGuiDataType_Double, v, speed, &vMin, &vMax, fmt);
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

std::string EscapeField(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        if (c == '\\' || c == '|' || c == '\n' || c == '\r') out.push_back('\\');
        if (c == '\n')
            out.push_back('n');
        else if (c == '\r')
            out.push_back('r');
        else
            out.push_back(c);
    }
    return out;
}

std::string UnescapeField(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (std::size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '\\' && i + 1 < s.size()) {
            const char n = s[++i];
            if (n == 'n')
                out.push_back('\n');
            else if (n == 'r')
                out.push_back('\r');
            else
                out.push_back(n);
        } else {
            out.push_back(s[i]);
        }
    }
    return out;
}

std::vector<std::string> SplitPipe(const std::string& line) {
    std::vector<std::string> parts;
    std::string cur;
    for (std::size_t i = 0; i < line.size(); ++i) {
        if (line[i] == '\\' && i + 1 < line.size()) {
            cur.push_back(line[i]);
            cur.push_back(line[++i]);
        } else if (line[i] == '|') {
            parts.push_back(cur);
            cur.clear();
        } else {
            cur.push_back(line[i]);
        }
    }
    parts.push_back(cur);
    return parts;
}
}  // namespace

void MeasurementRecipeWindow::SetVisible(bool visible) {
    if (visible && !visible_) {
        focusOnOpen_ = true;
        EnsureMatchStep();
    }
    visible_ = visible;
}

void MeasurementRecipeWindow::ToggleVisible() { SetVisible(!visible_); }

void MeasurementRecipeWindow::SetStatus(const char* msg) {
    statusText_ = msg ? msg : "";
    if (onStatus_) onStatus_(msg);
}

void MeasurementRecipeWindow::EnsureMatchStep() {
    if (steps_.empty() || steps_[0].type != StepType::Match) {
        RecipeStep match;
        match.type = StepType::Match;
        match.name = u8"模板匹配";
        match.enabled = true;
        steps_.insert(steps_.begin(), match);
    } else {
        steps_[0].name = u8"模板匹配";
        steps_[0].enabled = true;
    }
    if (selectedStep_ < 0 || selectedStep_ >= static_cast<int>(steps_.size())) selectedStep_ = 0;
}

void MeasurementRecipeWindow::DestroyImageSlot(ImageSlot& slot) {
    if (slot.texId) {
        glDeleteTextures(1, &slot.texId);
        slot.texId = 0;
    }
    slot.width = 0;
    slot.height = 0;
    slot.rgb.clear();
    slot.gray.clear();
    slot.path.clear();
}

bool MeasurementRecipeWindow::UploadTexture(ImageSlot& slot) {
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

bool MeasurementRecipeWindow::LoadImageSlot(ImageSlot& slot, const std::string& path,
                                            std::string& error) {
    ImageIO::RgbImage img;
    if (!ImageIO::LoadRgb(path, img, error)) return false;
    DestroyImageSlot(slot);
    slot.width = img.width;
    slot.height = img.height;
    slot.rgb = std::move(img.rgb);
    slot.path = path;
    slot.gray.clear();
    ImageIO::GrayImage gray;
    std::string grayErr;
    if (ImageIO::LoadGray(path, gray, grayErr) && gray.width == slot.width &&
        gray.height == slot.height) {
        slot.gray = std::move(gray.pixels);
    }
    return UploadTexture(slot);
}

void MeasurementRecipeWindow::SyncSourceToHost() {
    if (!onSyncSource_ || !sourceImage_.valid()) return;
    onSyncSource_(sourceImage_.rgb, sourceImage_.gray, sourceImage_.width, sourceImage_.height,
                  sourceImage_.path);
}

bool MeasurementRecipeWindow::UseHostCanvas() const {
    return Is2DToolActive() && viewMode_ == ViewMode::Source && sourceImage_.valid() &&
           static_cast<bool>(onDrawHostCanvas_);
}

bool MeasurementRecipeWindow::ImageToPixel(const ImageSlot& slot, const ImVec2& imgPos, float drawW,
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

void MeasurementRecipeWindow::LocalToImage(float lx, float ly, float& ix, float& iy) const {
    const float rad = pose_.angleDeg * kDegToRad;
    const float c = std::cos(rad);
    const float s = std::sin(rad);
    const float sx = lx * pose_.scale;
    const float sy = ly * pose_.scale;
    ix = pose_.cx + sx * c - sy * s;
    iy = pose_.cy + sx * s + sy * c;
}

void MeasurementRecipeWindow::ImageToLocal(float ix, float iy, float& lx, float& ly) const {
    const float dx = ix - pose_.cx;
    const float dy = iy - pose_.cy;
    const float rad = -pose_.angleDeg * kDegToRad;
    const float c = std::cos(rad);
    const float s = std::sin(rad);
    const float inv = (pose_.scale > 1e-6f) ? (1.f / pose_.scale) : 1.f;
    lx = (dx * c - dy * s) * inv;
    ly = (dx * s + dy * c) * inv;
}

void MeasurementRecipeWindow::ResetMeasureResults() {
    for (RecipeStep& s : steps_) {
        if (s.type != StepType::Match) {
            s.hasResult = false;
            s.resultValue = 0.f;
        }
    }
}

void MeasurementRecipeWindow::ClearTemplateRoi() {
    templateRoiValid_ = false;
    templateRoiDragging_ = false;
}

void MeasurementRecipeWindow::ClearSearchRoi() {
    searchRoiValid_ = false;
    searchRoiDragging_ = false;
    searchRoiEnabled_ = false;
}

bool MeasurementRecipeWindow::GetTemplateRoiForCreate(float& cx, float& cy, float& hw, float& hh,
                                                      float& angleDeg) const {
    if (!templateImage_.valid()) return false;
    if (templateUseRoi_) {
        if (!templateRoiValid_) return false;
        const float minX = std::min(templateRoiX0_, templateRoiX1_);
        const float maxX = std::max(templateRoiX0_, templateRoiX1_);
        const float minY = std::min(templateRoiY0_, templateRoiY1_);
        const float maxY = std::max(templateRoiY0_, templateRoiY1_);
        cx = (minX + maxX) * 0.5f;
        cy = (minY + maxY) * 0.5f;
        hw = std::max((maxX - minX) * 0.5f, 1.f);
        hh = std::max((maxY - minY) * 0.5f, 1.f);
        angleDeg = 0.f;
        return hw >= 4.f && hh >= 4.f;
    }
    cx = static_cast<float>(templateImage_.width) * 0.5f;
    cy = static_cast<float>(templateImage_.height) * 0.5f;
    hw = static_cast<float>(templateImage_.width) * 0.5f;
    hh = static_cast<float>(templateImage_.height) * 0.5f;
    angleDeg = 0.f;
    return true;
}

void MeasurementRecipeWindow::ReadTemplateImage() {
    const std::string path = FileDialog::OpenImageFile(u8"读取模板图像");
    if (path.empty()) return;
    std::string error;
    if (!LoadImageSlot(templateImage_, path, error)) {
        SetStatus(error.c_str());
        return;
    }
    viewMode_ = ViewMode::Template;
    roiMode_ = RoiMode::Template;
    ClearTemplateRoi();
    templateUseRoi_ = false;
    HalconShapeMatch::DestroyModel(model_);
    model_ = {};
    modelPath_.clear();
    pose_ = {};
    lastResult_ = {};
    ResetMeasureResults();
    zoom_ = 1.f;
    panX_ = 0.f;
    panY_ = 0.f;
    char buf[256];
    std::snprintf(buf, sizeof(buf), u8"已读取模板图像 %dx%d", templateImage_.width,
                  templateImage_.height);
    SetStatus(buf);
}

void MeasurementRecipeWindow::ReadSourceImage() {
    const std::string path = FileDialog::OpenImageFile(u8"读取源图像");
    if (path.empty()) return;
    std::string error;
    if (!LoadImageSlot(sourceImage_, path, error)) {
        SetStatus(error.c_str());
        return;
    }
    viewMode_ = ViewMode::Source;
    roiMode_ = RoiMode::Search;
    pose_ = {};
    lastResult_ = {};
    ResetMeasureResults();
    zoom_ = 1.f;
    panX_ = 0.f;
    panY_ = 0.f;
    if (onInvalidateHostResults_) onInvalidateHostResults_();
    SyncSourceToHost();
    char buf[256];
    std::snprintf(buf, sizeof(buf), u8"已读取源图像 %dx%d（请重新模板匹配以更新测量）",
                  sourceImage_.width, sourceImage_.height);
    SetStatus(buf);
}

void MeasurementRecipeWindow::CreateTemplate() {
    if (!templateImage_.valid()) {
        SetStatus(u8"请先读取模板图像");
        return;
    }
    float cx = 0.f, cy = 0.f, hw = 0.f, hh = 0.f, ang = 0.f;
    if (!GetTemplateRoiForCreate(cx, cy, hw, hh, ang)) {
        SetStatus(u8"请勾选并框选有效的模板 ROI");
        return;
    }
    std::string error;
    if (!HalconShapeMatch::CreateModel(templateImage_.rgb, templateImage_.width,
                                       templateImage_.height, cx, cy, hw, hh, ang, createParams_,
                                       model_, error)) {
        SetStatus(error.c_str());
        return;
    }
    char buf[256];
    std::snprintf(buf, sizeof(buf), u8"模板创建成功（%d×%d）", model_.templateW, model_.templateH);
    SetStatus(buf);
}

void MeasurementRecipeWindow::RunMatch() {
    EnsureMatchStep();
    if (!sourceImage_.valid()) {
        SetStatus(u8"请先读取源图像");
        return;
    }
    if (!model_.valid) {
        SetStatus(u8"请先创建或加载模板");
        return;
    }
    std::string error;
    const bool useRoi = searchRoiEnabled_ && searchRoiValid_;
    if (!HalconShapeMatch::FindModel(sourceImage_.rgb, sourceImage_.width, sourceImage_.height,
                                     searchRoiX0_, searchRoiY0_, searchRoiX1_, searchRoiY1_, useRoi,
                                     model_, findParams_, lastResult_, error)) {
        pose_ = {};
        ResetMeasureResults();
        SetStatus(error.c_str());
        return;
    }
    if (!lastResult_.ok || lastResult_.hits.empty()) {
        pose_ = {};
        ResetMeasureResults();
        SetStatus(u8"匹配失败：未找到目标，已跳过后续测量");
        return;
    }
    const auto& hit = lastResult_.hits.front();
    Pose2D newPose;
    newPose.cx = hit.centerX;
    newPose.cy = hit.centerY;
    newPose.angleDeg = hit.angleDeg;
    newPose.scale = hit.scale;
    newPose.valid = true;

    // 换图重匹配：把上一次位姿下的 2D 线段/点等刚体变换到新位姿，并重算线距
    if (geometryPose_.valid && onRemapHostGeometry_) {
        onRemapHostGeometry_(geometryPose_.cx, geometryPose_.cy, geometryPose_.angleDeg,
                             geometryPose_.scale, newPose.cx, newPose.cy, newPose.angleDeg,
                             newPose.scale);
    }
    geometryPose_ = newPose;
    pose_ = newPose;
    viewMode_ = ViewMode::Source;
    SyncSourceToHost();
    RunAllMeasures();
    char buf[256];
    std::snprintf(buf, sizeof(buf), u8"匹配成功：得分 %.3f，角度 %.1f°，缩放 %.2f（已更新测量位姿）",
                  hit.score, hit.angleDeg, hit.scale);
    SetStatus(buf);
}

void MeasurementRecipeWindow::SaveTemplateModel() {
    if (!model_.valid) {
        SetStatus(u8"没有可保存的模板");
        return;
    }
    const std::string path = FileDialog::SaveHalconShapeModelFile();
    if (path.empty()) return;
    std::string error;
    if (!HalconShapeMatch::SaveModel(path, model_, error)) {
        SetStatus(error.c_str());
        return;
    }
    modelPath_ = path;
    char buf[512];
    std::snprintf(buf, sizeof(buf), u8"模板已保存：%s", path.c_str());
    SetStatus(buf);
}

void MeasurementRecipeWindow::LoadTemplateModel() {
    const std::string path = FileDialog::OpenHalconShapeModelFile();
    if (path.empty()) return;
    std::string error;
    if (!HalconShapeMatch::LoadModel(path, model_, error)) {
        SetStatus(error.c_str());
        return;
    }
    modelPath_ = path;
    createParams_ = model_.createParams;
    findParams_.scaleMin = model_.modelScaleMin;
    findParams_.scaleMax = model_.modelScaleMax;
    SetStatus(u8"模板加载成功");
}

void MeasurementRecipeWindow::ClearTemplate() {
    HalconShapeMatch::DestroyModel(model_);
    model_ = {};
    modelPath_.clear();
    pose_ = {};
    geometryPose_ = {};
    lastResult_ = {};
    ResetMeasureResults();
    SetStatus(u8"已清除模板");
}

void MeasurementRecipeWindow::AddPointPointStep() {
    EnsureMatchStep();
    if (onActivate2DTool_) onActivate2DTool_(0);
    RecipeStep s;
    s.type = StepType::PointPoint;
    char buf[64];
    std::snprintf(buf, sizeof(buf), u8"点点距离%d", static_cast<int>(steps_.size()));
    s.name = buf;
    steps_.push_back(s);
    selectedStep_ = static_cast<int>(steps_.size()) - 1;
    placeMode_ = PlaceMode::PointA;
    viewMode_ = ViewMode::Source;
    SetStatus(u8"已添加点点距离：请在源图上依次点击两点（需先匹配成功）");
}

void MeasurementRecipeWindow::AddPointLineStep() {
    EnsureMatchStep();
    if (onActivate2DTool_) onActivate2DTool_(0);
    RecipeStep s;
    s.type = StepType::PointLine;
    char buf[64];
    std::snprintf(buf, sizeof(buf), u8"点到线%d", static_cast<int>(steps_.size()));
    s.name = buf;
    steps_.push_back(s);
    selectedStep_ = static_cast<int>(steps_.size()) - 1;
    placeMode_ = PlaceMode::PointP;
    viewMode_ = ViewMode::Source;
    SetStatus(u8"已添加点到线：先点测量点，再点线段两端（需先匹配成功）");
}

void MeasurementRecipeWindow::AddOp2DStep(int tool, const char* name) {
    EnsureMatchStep();
    RecipeStep s;
    s.type = StepType::Op2D;
    s.op2dTool = tool;
    s.name = name ? name : u8"2D算子";
    steps_.push_back(s);
    selectedStep_ = static_cast<int>(steps_.size()) - 1;
    placeMode_ = PlaceMode::None;
    viewMode_ = ViewMode::Source;
    SyncSourceToHost();
    ActivateOp2D(tool);
    char buf[256];
    std::snprintf(buf, sizeof(buf), u8"已添加步骤：%s（请在右侧源图操作）", s.name.c_str());
    SetStatus(buf);
}

void MeasurementRecipeWindow::ActivateOp2D(int tool) {
    if (onActivate2DTool_) onActivate2DTool_(tool);
}

void MeasurementRecipeWindow::OnSelectStep(int index) {
    if (index < 0 || index >= static_cast<int>(steps_.size())) return;
    selectedStep_ = index;
    placeMode_ = PlaceMode::None;
    RecipeStep& s = steps_[static_cast<std::size_t>(index)];
    if (s.type == StepType::Match) {
        if (onActivate2DTool_) onActivate2DTool_(0);
    } else if (s.type == StepType::PointPoint) {
        if (onActivate2DTool_) onActivate2DTool_(0);
        placeMode_ = (!s.aSet) ? PlaceMode::PointA : ((!s.bSet) ? PlaceMode::PointB : PlaceMode::None);
        viewMode_ = ViewMode::Source;
    } else if (s.type == StepType::PointLine) {
        if (onActivate2DTool_) onActivate2DTool_(0);
        if (!s.pSet)
            placeMode_ = PlaceMode::PointP;
        else if (!s.lineASet)
            placeMode_ = PlaceMode::LineA;
        else if (!s.lineBSet)
            placeMode_ = PlaceMode::LineB;
        viewMode_ = ViewMode::Source;
    } else if (s.type == StepType::Op2D) {
        viewMode_ = ViewMode::Source;
        SyncSourceToHost();
        ActivateOp2D(s.op2dTool);
    }
}

void MeasurementRecipeWindow::RemoveSelectedStep() {
    EnsureMatchStep();
    if (selectedStep_ <= 0 || selectedStep_ >= static_cast<int>(steps_.size())) {
        SetStatus(u8"模板匹配步骤不可删除");
        return;
    }
    steps_.erase(steps_.begin() + selectedStep_);
    selectedStep_ = std::min(selectedStep_, static_cast<int>(steps_.size()) - 1);
    placeMode_ = PlaceMode::None;
    SetStatus(u8"已删除步骤");
}

void MeasurementRecipeWindow::MoveSelectedStep(int delta) {
    EnsureMatchStep();
    if (selectedStep_ <= 0) return;
    const int dst = selectedStep_ + delta;
    if (dst <= 0 || dst >= static_cast<int>(steps_.size())) return;
    std::swap(steps_[static_cast<std::size_t>(selectedStep_)],
              steps_[static_cast<std::size_t>(dst)]);
    selectedStep_ = dst;
}

void MeasurementRecipeWindow::EvaluateStep(RecipeStep& step) {
    step.hasResult = false;
    step.resultValue = 0.f;
    if (!step.enabled || !HasPose()) return;

    if (step.type == StepType::PointPoint) {
        if (!step.aSet || !step.bSet) return;
        float ax = 0.f, ay = 0.f, bx = 0.f, by = 0.f;
        LocalToImage(step.ax, step.ay, ax, ay);
        LocalToImage(step.bx, step.by, bx, by);
        const float dx = bx - ax;
        const float dy = by - ay;
        step.resultValue = std::sqrt(dx * dx + dy * dy);
        step.hasResult = true;
    } else if (step.type == StepType::PointLine) {
        if (!step.pSet || !step.lineASet || !step.lineBSet) return;
        float px = 0.f, py = 0.f, x0 = 0.f, y0 = 0.f, x1 = 0.f, y1 = 0.f;
        LocalToImage(step.px, step.py, px, py);
        LocalToImage(step.lx0, step.ly0, x0, y0);
        LocalToImage(step.lx1, step.ly1, x1, y1);
        const float vx = x1 - x0;
        const float vy = y1 - y0;
        const float len2 = vx * vx + vy * vy;
        if (len2 < 1e-6f) return;
        const float dist = std::abs(vx * (y0 - py) - (x0 - px) * vy) / std::sqrt(len2);
        step.resultValue = dist;
        step.hasResult = true;
    }
}

void MeasurementRecipeWindow::RunAllMeasures() {
    EnsureMatchStep();
    if (!HasPose()) {
        ResetMeasureResults();
        return;
    }
    for (std::size_t i = 1; i < steps_.size(); ++i) EvaluateStep(steps_[i]);
}

void MeasurementRecipeWindow::HandleCanvasClick(float imgX, float imgY) {
    if (selectedStep_ <= 0 || selectedStep_ >= static_cast<int>(steps_.size())) return;
    if (!HasPose()) {
        SetStatus(u8"请先完成模板匹配，再布置测量点");
        return;
    }
    if (viewMode_ != ViewMode::Source) {
        SetStatus(u8"请切换到源图后再布置测量点");
        return;
    }

    RecipeStep& step = steps_[static_cast<std::size_t>(selectedStep_)];
    float lx = 0.f, ly = 0.f;
    ImageToLocal(imgX, imgY, lx, ly);

    if (step.type == StepType::PointPoint) {
        if (placeMode_ == PlaceMode::PointA || (!step.aSet && placeMode_ == PlaceMode::None)) {
            step.ax = lx;
            step.ay = ly;
            step.aSet = true;
            placeMode_ = PlaceMode::PointB;
            SetStatus(u8"已设点 A，请点击点 B");
        } else {
            step.bx = lx;
            step.by = ly;
            step.bSet = true;
            placeMode_ = PlaceMode::None;
            EvaluateStep(step);
            SetStatus(u8"点点距离已设置");
        }
    } else if (step.type == StepType::PointLine) {
        if (placeMode_ == PlaceMode::PointP || (!step.pSet && placeMode_ == PlaceMode::None)) {
            step.px = lx;
            step.py = ly;
            step.pSet = true;
            placeMode_ = PlaceMode::LineA;
            SetStatus(u8"已设测量点，请点击线段端点 1");
        } else if (placeMode_ == PlaceMode::LineA || (step.pSet && !step.lineASet)) {
            step.lx0 = lx;
            step.ly0 = ly;
            step.lineASet = true;
            placeMode_ = PlaceMode::LineB;
            SetStatus(u8"已设端点 1，请点击端点 2");
        } else {
            step.lx1 = lx;
            step.ly1 = ly;
            step.lineBSet = true;
            placeMode_ = PlaceMode::None;
            EvaluateStep(step);
            SetStatus(u8"点到线已设置");
        }
    }
}

bool MeasurementRecipeWindow::WriteRecipeFile(const std::string& path, std::string& error) {
    EnsureMatchStep();
    if (modelPath_.empty() && model_.valid) {
        error = u8"请先保存 Halcon 模板（.shm），再保存配方";
        return false;
    }
    if (modelPath_.empty()) {
        error = u8"配方需要关联模板路径，请先保存/加载模板";
        return false;
    }

    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) {
        error = u8"无法写入配方文件";
        return false;
    }
    ofs << "MRCP1\n";
    ofs << "model|" << EscapeField(modelPath_) << "\n";
    ofs << "template_img|" << EscapeField(templateImage_.path) << "\n";
    ofs << "source_img|" << EscapeField(sourceImage_.path) << "\n";
    ofs << "create|" << createParams_.numLevels << "|" << createParams_.angleStartDeg << "|"
        << createParams_.angleExtentDeg << "|" << createParams_.scaleMin << "|"
        << createParams_.scaleMax << "|" << createParams_.contrastLow << "|"
        << createParams_.contrastHigh << "|" << createParams_.minComponentSize << "|"
        << createParams_.minContrast << "\n";
    ofs << "find|" << findParams_.angleStartDeg << "|" << findParams_.angleExtentDeg << "|"
        << findParams_.scaleMin << "|" << findParams_.scaleMax << "|" << findParams_.minScore << "|"
        << findParams_.maxOverlap << "|" << findParams_.greediness << "|" << findParams_.numMatches
        << "|" << findParams_.numLevels << "|" << (findParams_.subPixel ? 1 : 0) << "\n";

    for (const RecipeStep& s : steps_) {
        if (s.type == StepType::Match) {
            ofs << "step|match|" << EscapeField(s.name) << "|" << (s.enabled ? 1 : 0) << "\n";
        } else if (s.type == StepType::PointPoint) {
            ofs << "step|pp|" << EscapeField(s.name) << "|" << (s.enabled ? 1 : 0) << "|"
                << (s.aSet ? 1 : 0) << "|" << s.ax << "|" << s.ay << "|" << (s.bSet ? 1 : 0) << "|"
                << s.bx << "|" << s.by << "\n";
        } else if (s.type == StepType::PointLine) {
            ofs << "step|pl|" << EscapeField(s.name) << "|" << (s.enabled ? 1 : 0) << "|"
                << (s.pSet ? 1 : 0) << "|" << s.px << "|" << s.py << "|" << (s.lineASet ? 1 : 0)
                << "|" << s.lx0 << "|" << s.ly0 << "|" << (s.lineBSet ? 1 : 0) << "|" << s.lx1 << "|"
                << s.ly1 << "\n";
        } else if (s.type == StepType::Op2D) {
            ofs << "step|op2d|" << EscapeField(s.name) << "|" << (s.enabled ? 1 : 0) << "|"
                << s.op2dTool << "\n";
        }
    }
    return true;
}

bool MeasurementRecipeWindow::ReadRecipeFile(const std::string& path, std::string& error) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        error = u8"无法打开配方文件";
        return false;
    }
    std::string line;
    if (!std::getline(ifs, line) || line != "MRCP1") {
        error = u8"不是有效的测量配方文件";
        return false;
    }

    std::vector<RecipeStep> loaded;
    std::string modelPath;
    std::string templatePath;
    std::string sourcePath;
    HalconShapeMatch::CreateParams create = createParams_;
    HalconShapeMatch::FindParams find = findParams_;

    while (std::getline(ifs, line)) {
        if (line.empty()) continue;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        auto parts = SplitPipe(line);
        if (parts.empty()) continue;
        if (parts[0] == "model" && parts.size() >= 2) {
            modelPath = UnescapeField(parts[1]);
        } else if (parts[0] == "template_img" && parts.size() >= 2) {
            templatePath = UnescapeField(parts[1]);
        } else if (parts[0] == "source_img" && parts.size() >= 2) {
            sourcePath = UnescapeField(parts[1]);
        } else if (parts[0] == "create" && parts.size() >= 10) {
            create.numLevels = std::stoi(parts[1]);
            create.angleStartDeg = std::stod(parts[2]);
            create.angleExtentDeg = std::stod(parts[3]);
            create.scaleMin = std::stod(parts[4]);
            create.scaleMax = std::stod(parts[5]);
            create.contrastLow = std::stoi(parts[6]);
            create.contrastHigh = std::stoi(parts[7]);
            create.minComponentSize = std::stoi(parts[8]);
            create.minContrast = std::stoi(parts[9]);
        } else if (parts[0] == "find" && parts.size() >= 11) {
            find.angleStartDeg = std::stod(parts[1]);
            find.angleExtentDeg = std::stod(parts[2]);
            find.scaleMin = std::stod(parts[3]);
            find.scaleMax = std::stod(parts[4]);
            find.minScore = std::stod(parts[5]);
            find.maxOverlap = std::stod(parts[6]);
            find.greediness = std::stod(parts[7]);
            find.numMatches = std::stoi(parts[8]);
            find.numLevels = std::stoi(parts[9]);
            find.subPixel = std::stoi(parts[10]) != 0;
        } else if (parts[0] == "step" && parts.size() >= 3) {
            RecipeStep s;
            s.name = UnescapeField(parts[2]);
            s.enabled = parts.size() > 3 ? (parts[3] == "1") : true;
            if (parts[1] == "match") {
                s.type = StepType::Match;
            } else if (parts[1] == "pp" && parts.size() >= 10) {
                s.type = StepType::PointPoint;
                s.aSet = parts[4] == "1";
                s.ax = std::stof(parts[5]);
                s.ay = std::stof(parts[6]);
                s.bSet = parts[7] == "1";
                s.bx = std::stof(parts[8]);
                s.by = std::stof(parts[9]);
            } else if (parts[1] == "pl" && parts.size() >= 13) {
                s.type = StepType::PointLine;
                s.pSet = parts[4] == "1";
                s.px = std::stof(parts[5]);
                s.py = std::stof(parts[6]);
                s.lineASet = parts[7] == "1";
                s.lx0 = std::stof(parts[8]);
                s.ly0 = std::stof(parts[9]);
                s.lineBSet = parts[10] == "1";
                s.lx1 = std::stof(parts[11]);
                s.ly1 = std::stof(parts[12]);
            } else if (parts[1] == "op2d" && parts.size() >= 5) {
                s.type = StepType::Op2D;
                s.op2dTool = std::stoi(parts[4]);
            } else {
                continue;
            }
            loaded.push_back(s);
        }
    }

    if (modelPath.empty()) {
        error = u8"配方缺少模板路径";
        return false;
    }
    if (!HalconShapeMatch::LoadModel(modelPath, model_, error)) return false;

    createParams_ = create;
    findParams_ = find;
    modelPath_ = modelPath;
    steps_ = std::move(loaded);
    EnsureMatchStep();
    selectedStep_ = 0;
    placeMode_ = PlaceMode::None;
    pose_ = {};
    lastResult_ = {};
    ResetMeasureResults();

    if (!templatePath.empty()) {
        std::string imgErr;
        LoadImageSlot(templateImage_, templatePath, imgErr);
    }
    if (!sourcePath.empty()) {
        std::string imgErr;
        if (LoadImageSlot(sourceImage_, sourcePath, imgErr)) SyncSourceToHost();
    }
    return true;
}

void MeasurementRecipeWindow::SaveRecipe() {
    const std::string path = FileDialog::SaveMeasurementRecipeFile();
    if (path.empty()) return;
    std::string error;
    if (!WriteRecipeFile(path, error)) {
        SetStatus(error.c_str());
        return;
    }
    char buf[512];
    std::snprintf(buf, sizeof(buf), u8"配方已保存：%s", path.c_str());
    SetStatus(buf);
}

void MeasurementRecipeWindow::LoadRecipe() {
    const std::string path = FileDialog::OpenMeasurementRecipeFile();
    if (path.empty()) return;
    std::string error;
    if (!ReadRecipeFile(path, error)) {
        SetStatus(error.c_str());
        return;
    }
    SetStatus(u8"配方加载成功（请读取/确认源图后执行模板匹配）");
}

void MeasurementRecipeWindow::DrawTemplatePreviewPanel() {
    ImGui::TextDisabled(u8"模板轮廓预览");
    const float panelH = 140.f;
    ImGui::BeginChild("##recipe_tpl_prev", ImVec2(0, panelH), ImGuiChildFlags_Borders);
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
    } else if (templateImage_.valid()) {
        const ImVec2 avail = ImGui::GetContentRegionAvail();
        const float imgAspect =
            static_cast<float>(templateImage_.width) / static_cast<float>(templateImage_.height);
        float drawW = avail.x;
        float drawH = drawW / imgAspect;
        if (drawH > avail.y) {
            drawH = avail.y;
            drawW = drawH * imgAspect;
        }
        const ImVec2 p = ImGui::GetCursorScreenPos();
        ImGui::GetWindowDrawList()->AddImage(
            static_cast<ImTextureID>(static_cast<intptr_t>(templateImage_.texId)), p,
            ImVec2(p.x + drawW, p.y + drawH));
        ImGui::Dummy(ImVec2(drawW, drawH));
        ImGui::TextDisabled(u8"模板图（创建后显示轮廓）");
    } else {
        ImGui::TextDisabled(u8"创建模板后在此显示轮廓 / 模板图");
    }
    ImGui::EndChild();
}

void MeasurementRecipeWindow::DrawMatchControls() {
    if (!HalconShapeMatch::IsHalconAvailable()) {
        ImGui::TextColored(ImVec4(1.f, 0.4f, 0.4f, 1.f), u8"当前构建未启用 Halcon");
        return;
    }

    auto labeledInt = [](const char* label, int* v, float speed, int vMin, int vMax) {
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted(label);
        ImGui::SameLine(96.f);
        ImGui::SetNextItemWidth(-1.f);
        ImGui::PushID(label);
        ImGui::DragInt("##v", v, speed, vMin, vMax);
        ImGui::PopID();
    };
    auto labeledDouble = [](const char* label, double* v, float speed, double vMin, double vMax,
                            const char* fmt) {
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted(label);
        ImGui::SameLine(96.f);
        ImGui::SetNextItemWidth(-1.f);
        ImGui::PushID(label);
        ImGui::DragScalar("##v", ImGuiDataType_Double, v, speed, &vMin, &vMax, fmt);
        ImGui::PopID();
    };

    ImGui::TextDisabled(u8"创建模板");
    labeledInt(u8"对比度低", &createParams_.contrastLow, 1.f, 1, 255);
    labeledInt(u8"对比度高", &createParams_.contrastHigh, 1.f, 1, 255);
    labeledDouble(u8"角度范围", &createParams_.angleExtentDeg, 1.0, 0.0, 360.0, "%.0f");

    ImGui::Spacing();
    ImGui::TextDisabled(u8"寻找模板");
    labeledDouble(u8"最小得分", &findParams_.minScore, 0.01f, 0.0, 1.0, "%.2f");
    labeledDouble(u8"角度范围", &findParams_.angleExtentDeg, 1.0, 0.0, 360.0, "%.0f");
    labeledDouble(u8"贪婪度", &findParams_.greediness, 0.01f, 0.0, 1.0, "%.2f");

    ImGui::Spacing();
    ImGui::Checkbox(u8"框选模板ROI", &templateUseRoi_);
    ImGui::Checkbox(u8"搜索ROI", &searchRoiEnabled_);

    const float btnW = (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x) * 0.5f;
    const ImVec2 btn(btnW, 26.f);
    if (ImGui::Button(u8"读取模板", btn)) ReadTemplateImage();
    ImGui::SameLine();
    if (ImGui::Button(u8"创建模板", btn)) CreateTemplate();
    if (ImGui::Button(u8"读取源图", btn)) ReadSourceImage();
    ImGui::SameLine();
    if (ImGui::Button(u8"模板匹配", btn)) RunMatch();
    if (ImGui::Button(u8"加载模板", btn)) LoadTemplateModel();
    ImGui::SameLine();
    if (ImGui::Button(u8"保存模板", btn)) SaveTemplateModel();
    if (ImGui::Button(u8"清除模板", btn)) ClearTemplate();
    ImGui::SameLine();
    if (ImGui::Button(u8"清除搜索ROI", btn)) ClearSearchRoi();

    ImGui::Spacing();
    if (ImGui::Button(u8"保存配方", ImVec2(btnW, 26.f))) SaveRecipe();
    ImGui::SameLine();
    if (ImGui::Button(u8"加载配方", ImVec2(btnW, 26.f))) LoadRecipe();

    ImGui::Spacing();
    DrawTemplatePreviewPanel();
}

void MeasurementRecipeWindow::Draw2DToolLibrary() {
    // 与 Application::Image2DTool 枚举值保持一致
    struct Item {
        int tool;
        const char* name;
    };
    static const Item kCaliper[] = {
        {1, u8"提取线段"}, {2, u8"提取圆弧"}, {10, u8"单点卡尺"}, {11, u8"圆卡尺"},
        {15, u8"矩形卡尺"}, {17, u8"剖面测宽"},
    };
    static const Item kFit[] = {
        {5, u8"拟合圆"}, {16, u8"拟合椭圆"}, {13, u8"三点定圆"},
    };
    static const Item kDist[] = {
        {6, u8"点点距离"}, {9, u8"点线距离"}, {3, u8"线线距离"}, {4, u8"圆弧间隙"},
        {14, u8"平行线距"}, {7, u8"两线夹角"}, {8, u8"圆心距"}, {12, u8"弧长/弦长"},
        {18, u8"投影点"},
    };
    static const Item kShape[] = {
        {19, u8"同心度"}, {20, u8"圆度"}, {21, u8"区域面积"},
    };
    static const Item kDepth[] = {
        {22, u8"两点高度差"}, {23, u8"剖面高度"}, {24, u8"模板匹配(2D)"},
    };

    auto drawGroup = [&](const char* title, const Item* items, int n) {
        ImGui::TextDisabled("%s", title);
        const float w = (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x) * 0.5f;
        for (int i = 0; i < n; ++i) {
            const bool on = Active2DTool() == items[i].tool;
            if (on) {
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.45f, 0.7f, 1.f));
            }
            if (ImGui::Button(items[i].name, ImVec2(w, 0))) {
                AddOp2DStep(items[i].tool, items[i].name);
            }
            if (on) ImGui::PopStyleColor();
            if ((i % 2) == 0 && i + 1 < n) ImGui::SameLine();
        }
    };

    drawGroup(u8"卡尺提取", kCaliper, 6);
    drawGroup(u8"几何拟合", kFit, 3);
    drawGroup(u8"距离角度", kDist, 9);
    drawGroup(u8"形位区域", kShape, 3);
    drawGroup(u8"深度匹配", kDepth, 3);
}

void MeasurementRecipeWindow::DrawStepList() {
    EnsureMatchStep();
    ImGui::TextDisabled(u8"配方步骤");
    ImGui::BeginChild("##recipe_steps", ImVec2(0, stepListH_), ImGuiChildFlags_Borders);
    for (int i = 0; i < static_cast<int>(steps_.size()); ++i) {
        RecipeStep& s = steps_[static_cast<std::size_t>(i)];
        ImGui::PushID(i);
        const bool isMatch = (s.type == StepType::Match);
        if (isMatch) ImGui::BeginDisabled();
        ImGui::Checkbox("##en", &s.enabled);
        if (isMatch) ImGui::EndDisabled();
        ImGui::SameLine();
        char label[160];
        const char* typeTag = u8"?";
        if (s.type == StepType::Match)
            typeTag = u8"匹配";
        else if (s.type == StepType::PointPoint)
            typeTag = u8"点点";
        else if (s.type == StepType::PointLine)
            typeTag = u8"点线";
        else if (s.type == StepType::Op2D)
            typeTag = u8"2D";
        if (s.hasResult) {
            std::snprintf(label, sizeof(label), u8"%d. [%s] %s  = %.2f px", i, typeTag,
                          s.name.c_str(), s.resultValue);
        } else {
            std::snprintf(label, sizeof(label), u8"%d. [%s] %s", i, typeTag, s.name.c_str());
        }
        if (ImGui::Selectable(label, selectedStep_ == i)) OnSelectStep(i);
        if (!isMatch && ImGui::BeginPopupContextItem("##step_ctx")) {
            if (ImGui::MenuItem(u8"删除步骤")) {
                selectedStep_ = i;
                RemoveSelectedStep();
            }
            ImGui::EndPopup();
        }
        ImGui::PopID();
    }
    ImGui::EndChild();

    ImGui::InvisibleButton("##recipe_step_split", ImVec2(-1, 6.f));
    if (ImGui::IsItemActive()) {
        stepListH_ = std::clamp(stepListH_ + ImGui::GetIO().MouseDelta.y, 80.f, 480.f);
    }
    if (ImGui::IsItemHovered() || ImGui::IsItemActive()) {
        ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);
        ImGui::GetWindowDrawList()->AddRectFilled(ImGui::GetItemRectMin(), ImGui::GetItemRectMax(),
                                                  IM_COL32(80, 140, 200, 120));
    }
}

void MeasurementRecipeWindow::DrawLeftPanel(float width) {
    ImGui::BeginChild("##recipe_left", ImVec2(width, 0), ImGuiChildFlags_Borders,
                      ImGuiWindowFlags_AlwaysVerticalScrollbar);

    if (Is2DToolActive() && onDrawHostToolPanel_) {
        ImGui::TextColored(ImVec4(0.45f, 0.85f, 1.f, 1.f), u8"2D 算子参数");
        ImGui::BeginChild("##recipe_host_tool", ImVec2(0, hostToolH_), ImGuiChildFlags_Borders,
                          ImGuiWindowFlags_AlwaysVerticalScrollbar);
        onDrawHostToolPanel_();
        ImGui::EndChild();
        ImGui::InvisibleButton("##recipe_host_tool_split", ImVec2(-1, 6.f));
        if (ImGui::IsItemActive()) {
            hostToolH_ = std::clamp(hostToolH_ + ImGui::GetIO().MouseDelta.y, 100.f, 600.f);
        }
        if (ImGui::IsItemHovered() || ImGui::IsItemActive()) {
            ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);
            ImGui::GetWindowDrawList()->AddRectFilled(
                ImGui::GetItemRectMin(), ImGui::GetItemRectMax(), IM_COL32(80, 140, 200, 120));
        }
        ImGui::Separator();
    }

    DrawStepList();
    ImGui::Separator();
    ImGui::TextDisabled(u8"2D 算子库（点选加入步骤）");
    ImGui::BeginChild("##recipe_2d_lib", ImVec2(0, toolLibH_), ImGuiChildFlags_Borders,
                      ImGuiWindowFlags_AlwaysVerticalScrollbar);
    Draw2DToolLibrary();
    ImGui::EndChild();
    ImGui::InvisibleButton("##recipe_lib_split", ImVec2(-1, 6.f));
    if (ImGui::IsItemActive()) {
        toolLibH_ = std::clamp(toolLibH_ + ImGui::GetIO().MouseDelta.y, 80.f, 420.f);
    }
    if (ImGui::IsItemHovered() || ImGui::IsItemActive()) {
        ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);
        ImGui::GetWindowDrawList()->AddRectFilled(ImGui::GetItemRectMin(), ImGui::GetItemRectMax(),
                                                  IM_COL32(80, 140, 200, 120));
    }

    ImGui::Separator();
    DrawMatchControls();
    ImGui::EndChild();
}

void MeasurementRecipeWindow::DrawOverlays(ImDrawList* dl, const ImVec2& imgPos, float drawW,
                                           float drawH) {
    const ImageSlot* slot =
        (viewMode_ == ViewMode::Template && templateImage_.valid()) ? &templateImage_ : &sourceImage_;
    if (!slot || !slot->valid()) return;
    const float sx = drawW / static_cast<float>(slot->width);
    const float sy = drawH / static_cast<float>(slot->height);

    auto toScreen = [&](float ix, float iy) {
        return ImVec2(imgPos.x + ix * sx, imgPos.y + iy * sy);
    };

    if (viewMode_ == ViewMode::Template && templateUseRoi_ && templateRoiValid_) {
        const float x0 = std::min(templateRoiX0_, templateRoiX1_);
        const float x1 = std::max(templateRoiX0_, templateRoiX1_);
        const float y0 = std::min(templateRoiY0_, templateRoiY1_);
        const float y1 = std::max(templateRoiY0_, templateRoiY1_);
        dl->AddRect(toScreen(x0, y0), toScreen(x1, y1), kRoiCol, 0.f, 0, 2.f);
    }
    if (viewMode_ == ViewMode::Source && searchRoiEnabled_ && searchRoiValid_) {
        const float x0 = std::min(searchRoiX0_, searchRoiX1_);
        const float x1 = std::max(searchRoiX0_, searchRoiX1_);
        const float y0 = std::min(searchRoiY0_, searchRoiY1_);
        const float y1 = std::max(searchRoiY0_, searchRoiY1_);
        dl->AddRect(toScreen(x0, y0), toScreen(x1, y1), kRoiCol, 0.f, 0, 2.f);
    }

    if (viewMode_ == ViewMode::Source && !lastResult_.hits.empty()) {
        for (const auto& hit : lastResult_.hits) {
            DrawContourPolylines(dl, hit.contourX, hit.contourY, hit.contourStarts, imgPos.x,
                                 imgPos.y, sx, sy, kContourCol, 1.5f);
        }
    }

    if (viewMode_ != ViewMode::Source || !HasPose()) return;

    for (std::size_t i = 1; i < steps_.size(); ++i) {
        const RecipeStep& s = steps_[i];
        if (!s.enabled) continue;
        if (s.type == StepType::PointPoint && s.aSet && s.bSet) {
            float ax, ay, bx, by;
            LocalToImage(s.ax, s.ay, ax, ay);
            LocalToImage(s.bx, s.by, bx, by);
            const ImVec2 pa = toScreen(ax, ay);
            const ImVec2 pb = toScreen(bx, by);
            dl->AddLine(pa, pb, kMeasureCol, 2.f);
            dl->AddCircleFilled(pa, 4.f, kMeasureCol);
            dl->AddCircleFilled(pb, 4.f, kMeasureCol);
            if (s.hasResult) {
                char buf[64];
                std::snprintf(buf, sizeof(buf), u8"%s: %.2f px", s.name.c_str(), s.resultValue);
                const ImVec2 mid((pa.x + pb.x) * 0.5f, (pa.y + pb.y) * 0.5f);
                const ImVec2 ts = ImGui::CalcTextSize(buf);
                dl->AddRectFilled(ImVec2(mid.x - 2, mid.y - ts.y - 2),
                                  ImVec2(mid.x + ts.x + 2, mid.y + 2), kTextBg);
                dl->AddText(mid, IM_COL32(255, 255, 120, 255), buf);
            }
        } else if (s.type == StepType::PointLine && s.pSet && s.lineASet && s.lineBSet) {
            float px, py, x0, y0, x1, y1;
            LocalToImage(s.px, s.py, px, py);
            LocalToImage(s.lx0, s.ly0, x0, y0);
            LocalToImage(s.lx1, s.ly1, x1, y1);
            const ImVec2 pp = toScreen(px, py);
            const ImVec2 pa = toScreen(x0, y0);
            const ImVec2 pb = toScreen(x1, y1);
            dl->AddLine(pa, pb, kMeasureCol, 2.f);
            dl->AddCircleFilled(pp, 4.f, IM_COL32(80, 200, 255, 255));
            dl->AddCircleFilled(pa, 3.f, kMeasureCol);
            dl->AddCircleFilled(pb, 3.f, kMeasureCol);
            if (s.hasResult) {
                char buf[64];
                std::snprintf(buf, sizeof(buf), u8"%s: %.2f px", s.name.c_str(), s.resultValue);
                const ImVec2 ts = ImGui::CalcTextSize(buf);
                dl->AddRectFilled(ImVec2(pp.x + 6, pp.y - ts.y - 2),
                                  ImVec2(pp.x + 8 + ts.x, pp.y + 2), kTextBg);
                dl->AddText(ImVec2(pp.x + 8, pp.y - ts.y), IM_COL32(255, 255, 120, 255), buf);
            }
        }
    }
}

void MeasurementRecipeWindow::DrawMeasureOverlaysOnHost(ImDrawList* dl, const ImVec2& imgPos,
                                                        float drawW, float drawH) {
    if (!sourceImage_.valid()) return;
    // host 画布上强制按源图叠加匹配轮廓与配方测量
    const ViewMode old = viewMode_;
    viewMode_ = ViewMode::Source;
    DrawOverlays(dl, imgPos, drawW, drawH);
    viewMode_ = old;
}

void MeasurementRecipeWindow::DrawImageCanvas() {
    if (UseHostCanvas()) {
        ImGui::BeginChild("##recipe_host_canvas", ImVec2(0, 0), false);
        onDrawHostCanvas_();
        ImGui::EndChild();
        return;
    }

    ImageSlot* slot = nullptr;
    if (viewMode_ == ViewMode::Template && templateImage_.valid())
        slot = &templateImage_;
    else if (sourceImage_.valid())
        slot = &sourceImage_;
    else if (templateImage_.valid())
        slot = &templateImage_;

    if (!slot) {
        ImGui::BeginChild("##recipe_canvas", ImVec2(0, 0), ImGuiChildFlags_Borders);
        ImGui::TextDisabled(u8"请读取模板图像或源图像");
        ImGui::EndChild();
        return;
    }

    ImGui::BeginChild("##recipe_canvas", ImVec2(0, 0), ImGuiChildFlags_Borders,
                      ImGuiWindowFlags_NoScrollWithMouse);
    ImGuiIO& io = ImGui::GetIO();
    const ImVec2 avail = ImGui::GetContentRegionAvail();
    const float imgAspect = static_cast<float>(slot->width) / static_cast<float>(slot->height);
    float fitW = std::max(avail.x, 1.f);
    float fitH = fitW / imgAspect;
    if (fitH > avail.y && avail.y > 1.f) {
        fitH = avail.y;
        fitW = fitH * imgAspect;
    }
    const float drawW = std::max(fitW * zoom_, 1.f);
    const float drawH = std::max(fitH * zoom_, 1.f);
    const float basePanX = std::max((avail.x - drawW) * 0.5f, 0.f);
    const float basePanY = std::max((avail.y - drawH) * 0.5f, 0.f);

    if (ImGui::IsWindowHovered() && io.KeyShift && io.MouseWheel != 0.f) {
        zoom_ = std::clamp(zoom_ * (1.f + io.MouseWheel * 0.12f), 0.2f, 32.f);
    }
    if (ImGui::IsWindowHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
        zoom_ = 1.f;
        panX_ = 0.f;
        panY_ = 0.f;
    }

    ImGui::SetCursorPos(ImVec2(basePanX + panX_, basePanY + panY_));
    const ImVec2 imgPos = ImGui::GetCursorScreenPos();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImGui::Image(static_cast<ImTextureID>(static_cast<intptr_t>(slot->texId)),
                 ImVec2(drawW, drawH));
    const bool hovered = ImGui::IsItemHovered();

    float mx = 0.f, my = 0.f;
    const ImVec2 mouse = io.MousePos;
    if (hovered && ImageToPixel(*slot, imgPos, drawW, drawH, mouse.x, mouse.y, mx, my)) {
        hoverPx_ = static_cast<int>(mx);
        hoverPy_ = static_cast<int>(my);
    } else {
        hoverPx_ = hoverPy_ = -1;
    }

    // Shift + 左键拖拽平移
    if (hovered && io.KeyShift && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
        panX_ += io.MouseDelta.x;
        panY_ += io.MouseDelta.y;
    }

    const bool allowTemplateRoi =
        viewMode_ == ViewMode::Template && templateUseRoi_ && slot == &templateImage_ && !io.KeyShift;
    const bool allowSearchRoi =
        viewMode_ == ViewMode::Source && searchRoiEnabled_ && slot == &sourceImage_ &&
        selectedStep_ == 0 && !io.KeyShift;

    if (allowTemplateRoi) {
        if (hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            templateRoiDragging_ = true;
            templateRoiValid_ = true;
            templateRoiX0_ = templateRoiX1_ = mx;
            templateRoiY0_ = templateRoiY1_ = my;
        }
        if (templateRoiDragging_ && ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            templateRoiX1_ = mx;
            templateRoiY1_ = my;
        }
        if (templateRoiDragging_ && ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
            templateRoiDragging_ = false;
        }
    } else if (allowSearchRoi) {
        if (hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            searchRoiDragging_ = true;
            searchRoiValid_ = true;
            searchRoiX0_ = searchRoiX1_ = mx;
            searchRoiY0_ = searchRoiY1_ = my;
        }
        if (searchRoiDragging_ && ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            searchRoiX1_ = mx;
            searchRoiY1_ = my;
        }
        if (searchRoiDragging_ && ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
            searchRoiDragging_ = false;
        }
    } else if (hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left) && selectedStep_ > 0 &&
               !io.KeyShift) {
        HandleCanvasClick(mx, my);
    }

    DrawOverlays(dl, imgPos, drawW, drawH);
    ImGui::EndChild();
}

void MeasurementRecipeWindow::DrawRightPanel() {
    ImGui::BeginChild("##recipe_right", ImVec2(0, 0), false);
    if (templateImage_.valid() && sourceImage_.valid()) {
        if (ImGui::SmallButton(u8"显示模板")) {
            viewMode_ = ViewMode::Template;
            roiMode_ = RoiMode::Template;
        }
        ImGui::SameLine();
        if (ImGui::SmallButton(u8"显示源图")) {
            viewMode_ = ViewMode::Source;
            roiMode_ = RoiMode::Search;
        }
    }
    ImGui::SameLine();
    if (HasPose()) {
        ImGui::TextColored(ImVec4(0.4f, 0.9f, 0.5f, 1.f), u8"已匹配");
    } else {
        ImGui::TextDisabled(u8"未匹配");
    }
    if (Is2DToolActive()) {
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(0.45f, 0.85f, 1.f, 1.f), u8"| 2D算子模式");
        if (viewMode_ != ViewMode::Source && sourceImage_.valid()) {
            viewMode_ = ViewMode::Source;
        }
        SyncSourceToHost();
    }
    DrawImageCanvas();
    ImGui::EndChild();
}

void MeasurementRecipeWindow::Draw(float menuBottomY, float bottomInset) {
    if (!visible_) return;
    EnsureMatchStep();

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
    if (!ImGui::Begin(u8"测量配方", &open, flags)) {
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
    ImGui::TextDisabled(u8"Shift+左键平移 | Shift+滚轮缩放 | 双击复位 | 左侧可拖拽分割条");

    const float bottomH = 24.f;
    const ImVec2 avail = ImGui::GetContentRegionAvail();
    const float maxLeft = std::max(avail.x - 200.f, 280.f);
    leftPanelW_ = std::clamp(leftPanelW_, 280.f, maxLeft);

    ImGui::BeginChild("##recipe_main", ImVec2(0, avail.y - bottomH), false);
    DrawLeftPanel(leftPanelW_);
    ImGui::SameLine(0.f, 0.f);
    ImGui::InvisibleButton("##recipe_vsplit", ImVec2(6.f, -1.f));
    if (ImGui::IsItemActive()) {
        leftPanelW_ = std::clamp(leftPanelW_ + ImGui::GetIO().MouseDelta.x, 280.f, maxLeft);
    }
    if (ImGui::IsItemHovered() || ImGui::IsItemActive()) {
        ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
        ImGui::GetWindowDrawList()->AddRectFilled(ImGui::GetItemRectMin(), ImGui::GetItemRectMax(),
                                                  IM_COL32(80, 140, 200, 140));
    }
    ImGui::SameLine(0.f, 0.f);
    DrawRightPanel();
    ImGui::EndChild();

    ImGui::Separator();
    if (hoverPx_ >= 0) {
        ImGui::TextDisabled(u8"X:%d  Y:%d", hoverPx_, hoverPy_);
        ImGui::SameLine();
    }
    if (!statusText_.empty()) {
        ImGui::TextColored(ImVec4(0.35f, 0.9f, 0.55f, 1.f), "%s", statusText_.c_str());
    }

    ImGui::End();
    ImGui::PopStyleVar(2);
}
