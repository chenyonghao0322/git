#include "app/Application.h"

#include "app/FileDialog.h"
#include "app/UiTheme.h"
#include "io/ImageIO.h"
#include "io/PointCloudGenerator.h"
#include "io/PointCloudIO.h"
#include "tools/FilterTools.h"

#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

void GlfwErrorCallback(int /*code*/, const char* /*desc*/) {}

bool LoadChineseFont() {
    ImGuiIO& io = ImGui::GetIO();
    const char* candidates[] = {
        "C:\\Windows\\Fonts\\msyh.ttc",
        "C:\\Windows\\Fonts\\msyh.ttf",
        "C:\\Windows\\Fonts\\simhei.ttf",
        "C:\\Windows\\Fonts\\simsun.ttc",
        "C:\\Windows\\Fonts\\msyhbd.ttc",
    };
    for (const char* path : candidates) {
        FILE* fp = nullptr;
#if defined(_MSC_VER)
        fopen_s(&fp, path, "rb");
#else
        fp = std::fopen(path, "rb");
#endif
        if (!fp) continue;
        std::fclose(fp);

        ImFontConfig cfg;
        cfg.OversampleH = 2;
        cfg.OversampleV = 2;
        ImFont* font = io.Fonts->AddFontFromFileTTF(
            path, 19.0f, &cfg, io.Fonts->GetGlyphRangesChineseSimplifiedCommon());
        if (font) return true;
    }
    return false;
}

}  // namespace

bool Application::Init() {
    glfwSetErrorCallback(GlfwErrorCallback);
    if (!glfwInit()) return false;

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window_ = glfwCreateWindow(1440, 900, u8"点云查看器", nullptr, nullptr);
    if (!window_) {
        glfwTerminate();
        return false;
    }
    glfwMakeContextCurrent(window_);
    glfwSwapInterval(1);

    if (!gladLoadGL(reinterpret_cast<GLADloadfunc>(glfwGetProcAddress))) {
        return false;
    }

    std::string err;
    if (!renderer_.Init(err)) {
        return false;
    }

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.IniFilename = nullptr;

    ApplyAppTheme();
    LoadChineseFont();

    ImGui_ImplGlfw_InitForOpenGL(window_, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    measure_.status = u8"请打开点云文件（PLY / PCD / XYZ / OBJ），或打开深度/亮度图对照查看";
    return true;
}

bool Application::ApplyCloud(PointCloud&& cloud, const char* statusMsg) {
    cloud_ = std::move(cloud);
    useIntensityColors_ = false;
    intensityColors_.clear();
    zMin_ = cloud_.bounds.min.z;
    zMax_ = cloud_.bounds.max.z;
    autoZRange_ = true;
    measure_ = {};
    measure_.section.cutAlongX = true;
    measure_.section.position = cloud_.bounds.Center().x;
    measure_.section.thickness =
        std::max(cloud_.bounds.Extent().x, cloud_.bounds.Extent().y) * 0.002f;
    if (measure_.section.thickness < 1e-4f) measure_.section.thickness = 0.01f;
    history_.Clear();
    ClearFilterCompare();
    UpdateAxesLength();

    if (cloud_.points.size() > 5000000) {
        maxDisplayPoints_ = 800000;
    } else if (cloud_.points.size() > 2000000) {
        maxDisplayPoints_ = 1200000;
    } else {
        maxDisplayPoints_ = 1500000;
    }

    if (statusMsg && statusMsg[0]) {
        measure_.status = statusMsg;
    } else {
        char buf[256];
        std::snprintf(buf, sizeof(buf), u8"已加载 %zu 个点（显示上限约 %d）", cloud_.points.size(),
                      maxDisplayPoints_);
        measure_.status = buf;
    }
    FitCameraToCloud();
    needUpload_ = true;
    return true;
}

bool Application::SaveCloud() {
    if (cloud_.points.empty()) {
        measure_.status = u8"当前没有点云可保存";
        return false;
    }
    const std::string path = FileDialog::SavePointCloudFile();
    if (path.empty()) return false;
    std::string error;
    if (!PointCloudIO::Save(path, cloud_, error, saveVisibleOnly_)) {
        measure_.status = error;
        return false;
    }
    char buf[320];
    std::snprintf(buf, sizeof(buf), u8"已保存 %s（%s）", path.c_str(),
                  saveVisibleOnly_ ? u8"仅可见点" : u8"全部点");
    measure_.status = buf;
    return true;
}

void Application::DestroyImageView(ImageView& view) {
    if (view.texId) {
        glDeleteTextures(1, &view.texId);
        view.texId = 0;
    }
    view.width = 0;
    view.height = 0;
    view.path.clear();
    view.gray.clear();
    view.rgb.clear();
    view.valueMin = 0.f;
    view.valueMax = 1.f;
}

bool Application::UploadImageTexture(ImageView& view) {
    if (view.rgb.empty() || view.width <= 0 || view.height <= 0) return false;
    if (view.texId == 0) glGenTextures(1, &view.texId);
    glBindTexture(GL_TEXTURE_2D, view.texId);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, view.width, view.height, 0, GL_RGB, GL_UNSIGNED_BYTE,
                 view.rgb.data());
    glBindTexture(GL_TEXTURE_2D, 0);
    return true;
}

namespace {

void ComputeGrayRange(const std::vector<float>& pixels, bool skipZero, float& valueMin,
                      float& valueMax) {
    float vmin = 1e30f, vmax = -1e30f;
    for (float v : pixels) {
        if (!std::isfinite(v)) continue;
        if (skipZero && std::fabs(v) <= 1e-12f) continue;
        vmin = std::min(vmin, v);
        vmax = std::max(vmax, v);
    }
    if (!(vmax > vmin)) {
        vmin = 0.f;
        vmax = 1.f;
    }
    valueMin = vmin;
    valueMax = vmax;
}

void BuildDepthRgb(const std::vector<float>& gray, std::vector<uint8_t>& rgbOut, float displayMin,
                   float displayMax, bool skipZero) {
    rgbOut.resize(gray.size() * 3u);
    float lo = displayMin;
    float hi = displayMax;
    if (!(hi > lo)) {
        hi = lo + 1.f;
    }
    const float inv = 1.f / (hi - lo);
    for (std::size_t i = 0; i < gray.size(); ++i) {
        const float v = gray[i];
        float t = 0.f;
        if (std::isfinite(v)) {
            if (skipZero && std::fabs(v) <= 1e-12f) {
                t = 0.f;
            } else {
                t = std::clamp((v - lo) * inv, 0.f, 1.f);
            }
        }
        const Vec3 c = HeightToColor(t);
        rgbOut[i * 3 + 0] = static_cast<uint8_t>(std::clamp(c.x, 0.f, 1.f) * 255.f + 0.5f);
        rgbOut[i * 3 + 1] = static_cast<uint8_t>(std::clamp(c.y, 0.f, 1.f) * 255.f + 0.5f);
        rgbOut[i * 3 + 2] = static_cast<uint8_t>(std::clamp(c.z, 0.f, 1.f) * 255.f + 0.5f);
    }
}

std::string FileNameOf(const std::string& path) {
    const auto pos = path.find_last_of("/\\");
    if (pos == std::string::npos) return path;
    return path.substr(pos + 1);
}

}  // namespace

bool Application::OpenDepthImage() {
    const std::string path = FileDialog::OpenImageFile("打开深度图");
    if (path.empty()) return false;

    ImageIO::GrayImage gray;
    std::string error;
    if (!ImageIO::LoadGray(path, gray, error)) {
        measure_.status = error;
        return false;
    }

    DestroyImageView(depthImage_);
    depthImage_.path = path;
    depthImage_.width = gray.width;
    depthImage_.height = gray.height;
    depthImage_.gray = std::move(gray.pixels);
    ComputeGrayRange(depthImage_.gray, depthSkipZero_, depthDataMin_, depthDataMax_);
    depthDisplayMin_ = depthDataMin_;
    depthDisplayMax_ = depthDataMax_;
    depthImage_.valueMin = depthDataMin_;
    depthImage_.valueMax = depthDataMax_;
    RebuildDepthDisplay();
    if (!depthImage_.valid()) {
        measure_.status = u8"深度图纹理上传失败";
        DestroyImageView(depthImage_);
        return false;
    }

    showImagePanel_ = true;
    imagePanelTab_ = 0;
    if (imageSyncEnabled_ && brightnessImage_.valid() &&
        (brightnessImage_.width != depthImage_.width ||
         brightnessImage_.height != depthImage_.height)) {
        imageSyncEnabled_ = false;
        ClearImageSyncPick();
    }
    char buf[320];
    std::snprintf(buf, sizeof(buf), u8"已打开深度图 %s（%dx%d）", FileNameOf(path).c_str(),
                  depthImage_.width, depthImage_.height);
    measure_.status = buf;
    return true;
}

void Application::RebuildDepthDisplay() {
    if (depthImage_.gray.empty() || depthImage_.width <= 0 || depthImage_.height <= 0) return;
    BuildDepthRgb(depthImage_.gray, depthImage_.rgb, depthDisplayMin_, depthDisplayMax_,
                  depthSkipZero_);
    depthImage_.valueMin = depthDisplayMin_;
    depthImage_.valueMax = depthDisplayMax_;
    UploadImageTexture(depthImage_);
}

bool Application::OpenBrightnessImage() {
    const std::string path = FileDialog::OpenImageFile("打开亮度图");
    if (path.empty()) return false;

    ImageIO::RgbImage rgb;
    std::string error;
    if (!ImageIO::LoadRgb(path, rgb, error)) {
        measure_.status = error;
        return false;
    }

    DestroyImageView(brightnessImage_);
    brightnessImage_.path = path;
    brightnessImage_.width = rgb.width;
    brightnessImage_.height = rgb.height;
    brightnessImage_.rgb = std::move(rgb.rgb);
    brightnessImage_.gray.clear();
    brightnessImage_.valueMin = 0.f;
    brightnessImage_.valueMax = 255.f;
    if (!UploadImageTexture(brightnessImage_)) {
        measure_.status = u8"亮度图纹理上传失败";
        DestroyImageView(brightnessImage_);
        return false;
    }

    showImagePanel_ = true;
    imagePanelTab_ = 1;
    if (imageSyncEnabled_ && depthImage_.valid() &&
        (depthImage_.width != brightnessImage_.width ||
         depthImage_.height != brightnessImage_.height)) {
        imageSyncEnabled_ = false;
        ClearImageSyncPick();
    }
    char buf[320];
    std::snprintf(buf, sizeof(buf), u8"已打开亮度图 %s（%dx%d）", FileNameOf(path).c_str(),
                  brightnessImage_.width, brightnessImage_.height);
    measure_.status = buf;
    return true;
}

bool Application::HasImagePanel() const {
    return showImagePanel_ && (depthImage_.valid() || brightnessImage_.valid());
}

float Application::ImagePanelWidth() const {
    if (!HasImagePanel()) return 0.f;
    const ImGuiViewport* vp = ImGui::GetMainViewport();
    // 至少留给左侧栏 + 点云视区约 600px
    const float maxW = std::max(vp->Size.x - 600.f, 240.f);
    return std::clamp(imagePanelPreferredW_, 240.f, maxW);
}

void Application::ClearImageSyncPick() {
    syncHasPick_ = false;
    syncCol_ = -1;
    syncRow_ = -1;
}

bool Application::TryEnableImageSync() {
    if (!depthImage_.valid() || !brightnessImage_.valid()) {
        measure_.status = u8"请先同时打开深度图和亮度图";
        return false;
    }
    if (depthImage_.width != brightnessImage_.width ||
        depthImage_.height != brightnessImage_.height) {
        measure_.status = u8"深度图与亮度图尺寸不一致，无法联动";
        return false;
    }

    syncWidth_ = depthImage_.width;
    syncHeight_ = depthImage_.height;
    imageSyncEnabled_ = true;
    showImagePanel_ = true;
    ClearImageSyncPick();
    char buf[192];
    std::snprintf(buf, sizeof(buf),
                  u8"已启用深度/亮度联动 %dx%d：在任一图上单击，另一图同步十字线", syncWidth_,
                  syncHeight_);
    measure_.status = buf;
    return true;
}

void Application::SetImageSyncPixel(int col, int row) {
    if (!imageSyncEnabled_ || syncWidth_ <= 0 || syncHeight_ <= 0) return;
    if (col < 0 || row < 0 || col >= syncWidth_ || row >= syncHeight_) return;

    syncCol_ = col;
    syncRow_ = row;
    syncHasPick_ = true;

    float depthVal = 0.f;
    bool hasDepth = false;
    if (depthImage_.valid() && !depthImage_.gray.empty()) {
        const std::size_t di =
            static_cast<std::size_t>(row) * static_cast<std::size_t>(depthImage_.width) +
            static_cast<std::size_t>(col);
        if (di < depthImage_.gray.size()) {
            depthVal = depthImage_.gray[di];
            hasDepth = true;
        }
    }
    int br = 0, bg = 0, bb = 0;
    bool hasBright = false;
    if (brightnessImage_.valid() && !brightnessImage_.rgb.empty()) {
        const std::size_t bi =
            (static_cast<std::size_t>(row) * static_cast<std::size_t>(brightnessImage_.width) +
             static_cast<std::size_t>(col)) *
            3u;
        if (bi + 2 < brightnessImage_.rgb.size()) {
            br = brightnessImage_.rgb[bi];
            bg = brightnessImage_.rgb[bi + 1];
            bb = brightnessImage_.rgb[bi + 2];
            hasBright = true;
        }
    }

    char buf[256];
    if (hasDepth && hasBright) {
        std::snprintf(buf, sizeof(buf), u8"联动 (%d,%d)  深度=%.4f  亮度RGB=%d,%d,%d", col, row,
                      depthVal, br, bg, bb);
    } else if (hasDepth) {
        std::snprintf(buf, sizeof(buf), u8"联动 (%d,%d)  深度=%.4f", col, row, depthVal);
    } else if (hasBright) {
        std::snprintf(buf, sizeof(buf), u8"联动 (%d,%d)  亮度RGB=%d,%d,%d", col, row, br, bg, bb);
    } else {
        std::snprintf(buf, sizeof(buf), u8"联动 (%d,%d)", col, row);
    }
    measure_.status = buf;
}

bool Application::LoadPath(const std::string& path) {
    std::string error;
    PointCloud cloud;
    if (!PointCloudIO::Load(path, cloud, error)) {
        measure_.status = error;
        return false;
    }
    char buf[256];
    std::snprintf(buf, sizeof(buf), u8"已加载 %zu 个点（显示上限约 %d，可在侧栏调节）",
                  cloud.points.size(),
                  cloud.points.size() > 5000000
                      ? 800000
                      : (cloud.points.size() > 2000000 ? 1200000 : 1500000));
    return ApplyCloud(std::move(cloud), buf);
}

void Application::CreateSphereCloud() {
    PointCloudGenerator::SphereParams p;
    p.radius = genSphereRadius_;
    p.pointCount = genSpherePoints_;
    p.noise = genSphereNoise_;
    std::string error;
    PointCloud cloud;
    if (!PointCloudGenerator::GenerateSphere(p, cloud, error)) {
        measure_.status = error;
        return;
    }
    char buf[192];
    std::snprintf(buf, sizeof(buf), u8"已创建球面点云 %zu 点（R=%.3f，噪声=%.3f）",
                  cloud.points.size(), p.radius, p.noise);
    ApplyCloud(std::move(cloud), buf);
}

void Application::CreateCylinderCloud() {
    PointCloudGenerator::CylinderParams p;
    p.radius = genCylRadius_;
    p.height = genCylHeight_;
    p.pointCount = genCylPoints_;
    p.noise = genCylNoise_;
    std::string error;
    PointCloud cloud;
    if (!PointCloudGenerator::GenerateCylinder(p, cloud, error)) {
        measure_.status = error;
        return;
    }
    char buf[192];
    std::snprintf(buf, sizeof(buf), u8"已创建圆柱点云 %zu 点（R=%.3f，H=%.3f，噪声=%.3f）",
                  cloud.points.size(), p.radius, p.height, p.noise);
    ApplyCloud(std::move(cloud), buf);
}

void Application::FitCameraToCloud() {
    if (!cloud_.bounds.Valid()) return;
    const Vec3 e = cloud_.bounds.Extent();
    const float maxExtent = std::max({e.x, e.y, e.z, 0.1f});
    const float dist = maxExtent * 2.0f;
    camera_.SetTarget(cloud_.bounds.Center(), dist);
    camera_.Reset();
    UpdateAxesLength();
}

void Application::ApplyViewPreset(int preset) {
    if (!cloud_.bounds.Valid()) return;
    const Vec3 e = cloud_.bounds.Extent();
    const float maxExtent = std::max({e.x, e.y, e.z, 0.1f});
    const float dist = maxExtent * 2.0f;
    camera_.SetTarget(cloud_.bounds.Center(), dist);

    constexpr float pi = 3.14159265f;
    switch (preset) {
        case 0:  // 俯视 +Y
            camera_.SetYawPitch(0.f, 1.52f);
            measure_.status = u8"视角: 俯视 (+Y → XZ 平面)";
            break;
        case 1:  // 侧视 从 +X 看 YZ
            camera_.SetYawPitch(0.f, 0.f);
            measure_.status = u8"视角: 侧视 (沿 +X 看 YZ)";
            break;
        case 2:  // 侧视 从 +Z 看 XY
            camera_.SetYawPitch(pi * 0.5f, 0.f);
            measure_.status = u8"视角: 侧视 (沿 +Z 看 XY)";
            break;
        case 3:  // 沿运动方向 Y
            camera_.SetYawPitch(0.f, -1.35f);
            measure_.status = u8"视角: 沿运动方向 (沿 Y)";
            break;
        default:
            camera_.Reset();
            measure_.status = u8"视角: 复位到包围盒";
            break;
    }
    UpdateAxesLength();
}

void Application::UpdateAxesLength() {
    if (!cloud_.bounds.Valid()) {
        axesLength_ = 1.f;
    } else {
        const float d = cloud_.bounds.Diagonal();
        axesLength_ = std::max(d * 0.15f, 0.01f);
    }
    renderer_.SetAxes(showAxes_, axesLength_);
}

void Application::PushHistory(const std::string& label) {
    if (cloud_.mask.size() != cloud_.points.size()) cloud_.ResetMask();
    history_.Push(cloud_.mask, label);
}

void Application::Undo() {
    if (cloud_.mask.size() != cloud_.points.size()) cloud_.ResetMask();
    std::string label;
    if (!history_.Undo(cloud_.mask, label)) {
        measure_.status = u8"没有可撤销的操作";
        return;
    }
    measure_.clipEnabled = false;
    needUpload_ = true;
    measure_.status = std::string(u8"已撤销: ") + label;
}

void Application::Redo() {
    if (cloud_.mask.size() != cloud_.points.size()) cloud_.ResetMask();
    std::string label;
    if (!history_.Redo(cloud_.mask, label)) {
        measure_.status = u8"没有可重做的操作";
        return;
    }
    measure_.clipEnabled = false;
    needUpload_ = true;
    measure_.status = std::string(u8"已重做");
}

void Application::RefreshGpu() {
    if (autoZRange_ && cloud_.bounds.Valid()) {
        zMin_ = cloud_.bounds.min.z;
        zMax_ = cloud_.bounds.max.z;
    }

    RebuildAnalysisColors();

    UploadParams params;
    params.maxDisplayPoints = maxDisplayPoints_;
    params.zMin = zMin_;
    params.zMax = zMax_;
    const bool showRoi =
        !measure_.roiIndices.empty() &&
        (measure_.mode == ToolMode::Roi || measure_.mode == ToolMode::PlaneFit ||
         measure_.mode == ToolMode::SphereFit || measure_.mode == ToolMode::CircleFit ||
         measure_.mode == ToolMode::CylinderFit || measure_.mode == ToolMode::Flatness ||
         measure_.mode == ToolMode::StepGap) &&
        !measure_.flatness.valid && !measure_.stepGap.hasDistances;
    params.highlightRoi = showRoi ? &measure_.roiIndices : nullptr;
    params.usePointColors = (cloud_.colors.size() == cloud_.points.size()) &&
                            (filterCompareActive_ || measure_.flatness.valid ||
                             measure_.stepGap.hasDistances || useIntensityColors_);
    params.ignoreMask = filterCompareActive_ && !filterHideRemoved_;

    std::vector<uint8_t> maskBackup;
    if (filterCompareActive_ && filterHideRemoved_ &&
        filterKeepMask_.size() == cloud_.points.size()) {
        maskBackup = cloud_.mask;
        cloud_.mask = filterKeepMask_;
        params.ignoreMask = false;
    }

    gpuPointCount_ = renderer_.Upload(cloud_, params, &displayIndices_);
    if (!maskBackup.empty()) cloud_.mask = std::move(maskBackup);

    UpdateOverlays();
    needUpload_ = false;
}

void Application::RebuildAnalysisColors() {
    if (cloud_.points.empty()) {
        cloud_.colors.clear();
        return;
    }

    if (filterCompareActive_ && filterKeepMask_.size() == cloud_.points.size()) {
        cloud_.colors.resize(cloud_.points.size());
        for (std::size_t i = 0; i < cloud_.points.size(); ++i) {
            if (filterKeepMask_[i]) {
                cloud_.colors[i] = {0.20f, 0.85f, 0.95f};  // 保留：青绿
            } else {
                cloud_.colors[i] = {0.95f, 0.28f, 0.22f};  // 滤除：红
            }
        }
        return;
    }

    if (measure_.mode == ToolMode::Flatness && measure_.flatness.valid) {
        cloud_.colors.assign(cloud_.points.size(), Vec3{0.22f, 0.24f, 0.26f});
        const float span = std::max(measure_.flatness.peakToValley, 1e-6f);
        const float mid = 0.5f * (measure_.flatness.minDev + measure_.flatness.maxDev);
        for (std::size_t k = 0; k < measure_.flatness.indices.size(); ++k) {
            const std::size_t idx = measure_.flatness.indices[k];
            if (idx >= cloud_.points.size()) continue;
            const float t = 0.5f + (measure_.flatness.signedDist[k] - mid) / span;
            cloud_.colors[idx] = DivergingColor(t);
        }
        return;
    }

    if (measure_.mode == ToolMode::StepGap && measure_.stepGap.hasDistances) {
        cloud_.colors.assign(cloud_.points.size(), Vec3{0.20f, 0.22f, 0.24f});
        for (std::size_t idx : measure_.stepGap.regionA) {
            if (idx < cloud_.colors.size()) cloud_.colors[idx] = {0.95f, 0.85f, 0.20f};  // A 黄
        }
        const float amax = std::max(std::fabs(measure_.stepGap.minDist),
                                    std::fabs(measure_.stepGap.maxDist));
        const float span = std::max(amax * 2.f, 1e-6f);
        for (std::size_t k = 0; k < measure_.stepGap.regionB.size(); ++k) {
            const std::size_t idx = measure_.stepGap.regionB[k];
            if (idx >= cloud_.colors.size()) continue;
            const float t = 0.5f + measure_.stepGap.signedDistB[k] / span;
            cloud_.colors[idx] = DivergingColor(t);
        }
        return;
    }

    if (useIntensityColors_ && intensityColors_.size() == cloud_.points.size()) {
        cloud_.colors = intensityColors_;
        return;
    }

    // 默认高度色
    cloud_.ApplyHeightColors(zMin_, zMax_);
}

void Application::RunFilterPreview(int type) {
    if (cloud_.points.empty()) {
        measure_.status = u8"请先加载点云再滤波";
        return;
    }
    if (cloud_.mask.size() != cloud_.points.size()) cloud_.ResetMask();

    filterBackupMask_ = cloud_.mask;
    std::string error;
    int kept = 0;
    bool ok = false;
    if (type == 0) {
        ok = FilterTools::VoxelDownsample(cloud_, filterVoxelLeaf_, filterKeepMask_, error, &kept);
    } else if (type == 1) {
        ok = FilterTools::RadiusOutlier(cloud_, filterRadius_, filterRadiusMinNeighbors_,
                                        filterKeepMask_, error, &kept);
    } else {
        ok = FilterTools::StatisticalOutlier(cloud_, filterStatMeanK_, filterStatStdMul_,
                                             filterKeepMask_, error, &kept);
    }
    if (!ok) {
        measure_.status = error;
        return;
    }

    filterLastKept_ = kept;
    filterLastRemoved_ = static_cast<int>(cloud_.VisibleCount()) - kept;
    if (filterLastRemoved_ < 0) filterLastRemoved_ = 0;
    filterCompareActive_ = true;
    filterHideRemoved_ = false;
    needUpload_ = true;

    char buf[160];
    std::snprintf(buf, sizeof(buf), u8"滤波预览：保留 %d，滤除 %d（青绿=保留，红=滤除）",
                  filterLastKept_, filterLastRemoved_);
    measure_.status = buf;
}

void Application::ApplyFilterResult() {
    if (!filterCompareActive_ || filterKeepMask_.size() != cloud_.points.size()) {
        measure_.status = u8"没有可应用的滤波结果";
        return;
    }
    PushHistory(u8"滤波");
    cloud_.mask = filterKeepMask_;
    filterCompareActive_ = false;
    filterKeepMask_.clear();
    filterBackupMask_.clear();
    needUpload_ = true;
    measure_.status = std::string(u8"已应用滤波，可见 ") + std::to_string(cloud_.VisibleCount());
}

void Application::ClearFilterCompare() {
    if (!filterBackupMask_.empty() && filterBackupMask_.size() == cloud_.points.size()) {
        cloud_.mask = filterBackupMask_;
    }
    filterCompareActive_ = false;
    filterKeepMask_.clear();
    filterBackupMask_.clear();
    needUpload_ = true;
    measure_.status = u8"已取消滤波预览";
}

void Application::UpdateOverlays() {
    if (measure_.mode == ToolMode::StepHeight && (measure_.stepA || measure_.stepB)) {
        renderer_.SetDistanceOverlay(measure_.stepA, measure_.stepB);
    } else if (measure_.distA || measure_.distB) {
        renderer_.SetDistanceOverlay(measure_.distA, measure_.distB);
    } else {
        renderer_.SetPickOverlay(measure_.picked);
    }
    if (measure_.mode == ToolMode::Section) {
        SyncSectionCutPlane();
    } else if (measure_.mode == ToolMode::Flatness && measure_.flatness.valid) {
        renderer_.SetPlaneOverlay(measure_.flatness.plane);
    } else if (measure_.mode == ToolMode::StepGap && measure_.stepGap.hasPlane) {
        renderer_.SetPlaneOverlay(measure_.stepGap.planeA);
    } else if (measure_.mode == ToolMode::PlaneFit) {
        renderer_.SetPlaneOverlay(measure_.plane);
    } else {
        renderer_.SetPlaneOverlay(std::nullopt);
    }

    if (measure_.mode == ToolMode::SphereFit && measure_.sphere) {
        renderer_.SetSphereOverlay(measure_.sphere);
    } else if (measure_.mode == ToolMode::CircleFit && measure_.circle) {
        renderer_.SetCircleOverlay(measure_.circle);
    } else if (measure_.mode == ToolMode::CylinderFit && measure_.cylinder) {
        renderer_.SetCylinderOverlay(measure_.cylinder);
    } else {
        renderer_.ClearFitWireOverlay();
    }
    renderer_.SetAxes(showAxes_, axesLength_);
}

void Application::SyncSectionCutPlane() {
    if (!cloud_.bounds.Valid()) {
        renderer_.SetPlaneOverlay(std::nullopt);
        return;
    }
    measure_.plane = MeasureTools::MakeSectionCutPlane(
        cloud_, measure_.section.cutAlongX, measure_.section.position);
    // Slightly more opaque / larger hint while dragging
    renderer_.SetPlaneOverlay(measure_.plane);
}

bool Application::ProjectWorldToScreen(const Vec3& p, float& sx, float& sy) const {
    int vx = 0, vy = 0, vw = 0, vh = 0;
    GetView3dFbRect(vx, vy, vw, vh);
    if (vw <= 0 || vh <= 0) return false;
    const float aspect = static_cast<float>(vw) / static_cast<float>(vh);
    const Mat4 mvp = camera_.ProjMatrix(aspect) * camera_.ViewMatrix();
    const Vec4 clip = mvp.MulVec4({p.x, p.y, p.z, 1.f});
    if (std::fabs(clip.w) < 1e-12f) return false;
    const float ndcX = clip.x / clip.w;
    const float ndcY = clip.y / clip.w;
    const float ndcZ = clip.z / clip.w;
    if (ndcZ < -1.f || ndcZ > 1.f) return false;
    sx = static_cast<float>(vx) + (ndcX * 0.5f + 0.5f) * static_cast<float>(vw);
    sy = static_cast<float>(vy) + (1.f - (ndcY * 0.5f + 0.5f)) * static_cast<float>(vh);
    return true;
}

void Application::BeginSectionDrag(float mouseX, float mouseY) {
    sectionDragging_ = true;
    lastSectionMouseX_ = mouseX;
    lastSectionMouseY_ = mouseY;
    SyncSectionCutPlane();
    measure_.status = u8"拖拽截面中…松开鼠标后自动生成 2D 轮廓";
}

void Application::UpdateSectionDrag(float mouseX, float mouseY) {
    if (!sectionDragging_ || !cloud_.bounds.Valid()) return;

    const Vec3 axis = measure_.section.cutAlongX ? Vec3{1, 0, 0} : Vec3{0, 1, 0};
    PlaneModel plane = MeasureTools::MakeSectionCutPlane(
        cloud_, measure_.section.cutAlongX, measure_.section.position);
    const Vec3 c = plane.centroid;

    float s0x = 0, s0y = 0, s1x = 0, s1y = 0;
    const bool ok0 = ProjectWorldToScreen(c, s0x, s0y);
    const bool ok1 = ProjectWorldToScreen(c + axis, s1x, s1y);
    float worldPerPixel = 0.f;
    float nx = 1.f, ny = 0.f;

    if (ok0 && ok1) {
        const float ax = s1x - s0x;
        const float ay = s1y - s0y;
        const float len = std::sqrt(ax * ax + ay * ay);
        if (len > 1e-3f) {
            nx = ax / len;
            ny = ay / len;
            worldPerPixel = 1.f / len;  // axis length = 1
        }
    }

    if (worldPerPixel <= 0.f) {
        // Fallback when axis is nearly along view direction: use horizontal mouse for X, vertical for Y
        worldPerPixel = camera_.Distance() * 0.002f;
        nx = measure_.section.cutAlongX ? 1.f : 0.f;
        ny = measure_.section.cutAlongX ? 0.f : 1.f;
    }

    const float ddx = mouseX - lastSectionMouseX_;
    const float ddy = mouseY - lastSectionMouseY_;
    const float along = ddx * nx + ddy * ny;
    measure_.section.position += along * worldPerPixel;

    float amin = measure_.section.cutAlongX ? cloud_.bounds.min.x : cloud_.bounds.min.y;
    float amax = measure_.section.cutAlongX ? cloud_.bounds.max.x : cloud_.bounds.max.y;
    measure_.section.position = std::clamp(measure_.section.position, amin, amax);

    lastSectionMouseX_ = mouseX;
    lastSectionMouseY_ = mouseY;
    SyncSectionCutPlane();

    char buf[96];
    std::snprintf(buf, sizeof(buf), u8"截面位置 = %.4f（拖拽中）", measure_.section.position);
    measure_.status = buf;
}

void Application::EndSectionDrag() {
    if (!sectionDragging_) return;
    sectionDragging_ = false;
    GenerateSection();
}

void Application::BeginRoiDrag(float mouseX, float mouseY) {
    measure_.roiDragging = true;
    measure_.roiX0 = measure_.roiX1 = mouseX;
    measure_.roiY0 = measure_.roiY1 = mouseY;
}

void Application::UpdateRoiDrag(float mouseX, float mouseY) {
    if (!measure_.roiDragging) return;
    measure_.roiX1 = mouseX;
    measure_.roiY1 = mouseY;
}

void Application::EndRoiDrag() {
    if (!measure_.roiDragging) return;
    measure_.roiDragging = false;
    int vx = 0, vy = 0, vw = 0, vh = 0;
    GetView3dFbRect(vx, vy, vw, vh);
    MeasureTools::SelectRoi(cloud_, camera_, vw, vh, measure_.roiX0 - static_cast<float>(vx),
                            measure_.roiY0 - static_cast<float>(vy),
                            measure_.roiX1 - static_cast<float>(vx),
                            measure_.roiY1 - static_cast<float>(vy), measure_.roiIndices);

    if (measure_.mode == ToolMode::StepGap) {
        auto& sg = measure_.stepGap;
        // 已拟合 A 或已进入选 B：后续框选写入 B；否则写入 A
        if (sg.hasPlane || sg.phase == StepGapPhase::SelectB || sg.phase == StepGapPhase::Done) {
            sg.regionB = measure_.roiIndices;
            sg.hasDistances = false;
            sg.signedDistB.clear();
            sg.phase = StepGapPhase::SelectB;
            measure_.status = std::string(u8"段差区域 B：") +
                              std::to_string(sg.regionB.size()) + u8" 点，请计算段差";
        } else {
            sg.regionA = measure_.roiIndices;
            sg.hasPlane = false;
            sg.hasDistances = false;
            sg.regionB.clear();
            sg.signedDistB.clear();
            sg.phase = StepGapPhase::SelectA;
            measure_.status = std::string(u8"段差区域 A：") +
                              std::to_string(sg.regionA.size()) + u8" 点，请拟合平面";
        }
    } else if (measure_.mode == ToolMode::Flatness) {
        measure_.flatness = {};
        measure_.status = std::string(u8"平面度框选 ") +
                          std::to_string(measure_.roiIndices.size()) + u8" 个点";
    } else {
        measure_.status = std::string(u8"已框选 ") + std::to_string(measure_.roiIndices.size()) +
                          u8" 个点";
    }

    // 框选完成后清除屏幕矩形，改用 3D 投影框（随视角移动）
    measure_.roiX0 = measure_.roiX1 = 0.f;
    measure_.roiY0 = measure_.roiY1 = 0.f;
    needUpload_ = true;
}

void Application::GenerateSection() {
    std::string error;
    if (!MeasureTools::ExtractSection(cloud_, measure_.section.cutAlongX, measure_.section.position,
                                      measure_.section.thickness, measure_.section, error)) {
        measure_.status = error;
        SyncSectionCutPlane();
        return;
    }
    measure_.section.pickA.reset();
    measure_.section.pickB.reset();
    measure_.section.lineDistance = 0.f;
    measure_.section.zDistance = 0.f;
    SyncSectionCutPlane();
    char buf[160];
    std::snprintf(buf, sizeof(buf), u8"截面生成成功：%zu 个轮廓点（位置=%.4f）",
                  measure_.section.points.size(), measure_.section.position);
    measure_.status = buf;
}

void Application::UpdateSectionDistances() {
    auto& sec = measure_.section;
    if (!sec.pickA || !sec.pickB || sec.points.empty()) {
        sec.lineDistance = 0.f;
        sec.zDistance = 0.f;
        return;
    }
    if (*sec.pickA >= sec.points.size() || *sec.pickB >= sec.points.size()) return;
    const auto& a = sec.points[*sec.pickA];
    const auto& b = sec.points[*sec.pickB];
    sec.lineDistance = std::fabs(a.u - b.u);
    sec.zDistance = std::fabs(a.v - b.v);
    char buf[160];
    std::snprintf(buf, sizeof(buf), u8"垂线间距=%.6f，Z向距离=%.6f", sec.lineDistance,
                  sec.zDistance);
    measure_.status = buf;
}

std::optional<std::size_t> Application::FindNearestSectionPoint(float plotX, float plotY,
                                                                float plotW, float plotH,
                                                                float* outDistPx) const {
    const auto& sec = measure_.section;
    if (sec.points.empty() || plotW < 1.f || plotH < 1.f) return std::nullopt;

    const float pad = 0.08f;
    const float du = (sec.uMax - sec.uMin) * (1.f + pad * 2.f);
    const float dv = (sec.vMax - sec.vMin) * (1.f + pad * 2.f);
    const float uStart = sec.uMin - (sec.uMax - sec.uMin) * pad;
    const float vStart = sec.vMin - (sec.vMax - sec.vMin) * pad;
    const float u = uStart + (plotX / plotW) * du;
    const float v = vStart + (1.f - plotY / plotH) * dv;
    const float pxPerU = plotW / du;
    const float pxPerV = plotH / dv;

    float best = 1e30f;
    std::size_t bestIdx = 0;
    for (std::size_t i = 0; i < sec.points.size(); ++i) {
        const float ddx = (sec.points[i].u - u) * pxPerU;
        const float ddy = (sec.points[i].v - v) * pxPerV;
        const float d2 = ddx * ddx + ddy * ddy;
        if (d2 < best) {
            best = d2;
            bestIdx = i;
        }
    }
    if (outDistPx) *outDistPx = std::sqrt(best);
    return bestIdx;
}

bool Application::HitSectionPickMarker(float plotX, float plotY, float plotW, float plotH,
                                       bool point1, float hitRadiusPx) const {
    const auto& sec = measure_.section;
    const auto& pick = point1 ? sec.pickA : sec.pickB;
    if (!pick || *pick >= sec.points.size() || plotW < 1.f || plotH < 1.f) return false;

    const float pad = 0.08f;
    const float du = (sec.uMax - sec.uMin) * (1.f + pad * 2.f);
    const float dv = (sec.vMax - sec.vMin) * (1.f + pad * 2.f);
    const float uStart = sec.uMin - (sec.uMax - sec.uMin) * pad;
    const float vStart = sec.vMin - (sec.vMax - sec.vMin) * pad;
    const auto& sp = sec.points[*pick];
    const float sx = ((sp.u - uStart) / du) * plotW;
    const float sy = (1.f - (sp.v - vStart) / dv) * plotH;
    const float dx = plotX - sx;
    const float dy = plotY - sy;
    return (dx * dx + dy * dy) <= hitRadiusPx * hitRadiusPx;
}

void Application::OnSectionPlotClick(float plotX, float plotY, float plotW, float plotH) {
    auto& sec = measure_.section;
    if (sec.points.empty() || plotW < 1.f || plotH < 1.f) return;

    // Prefer grabbing existing markers for drag.
    if (HitSectionPickMarker(plotX, plotY, plotW, plotH, true)) {
        sectionPlotDragTarget_ = 1;
        measure_.status = u8"拖动点1中…";
        return;
    }
    if (HitSectionPickMarker(plotX, plotY, plotW, plotH, false)) {
        sectionPlotDragTarget_ = 2;
        measure_.status = u8"拖动点2中…";
        return;
    }

    float distPx = 0.f;
    const auto best = FindNearestSectionPoint(plotX, plotY, plotW, plotH, &distPx);
    if (!best || distPx > 12.f) {
        measure_.status = u8"截面图上未点到轮廓点";
        return;
    }

    if (!sec.pickA || sec.pickB) {
        sec.pickA = *best;
        sec.pickB.reset();
        sec.lineDistance = 0.f;
        sec.zDistance = 0.f;
        sectionPlotDragTarget_ = 1;
        measure_.status = u8"已选点1，可拖动调整；再点选点2";
    } else {
        sec.pickB = *best;
        sectionPlotDragTarget_ = 2;
        UpdateSectionDistances();
        measure_.status = std::string(measure_.status) + u8"（可拖动点1/点2实时调整）";
    }
}

void Application::OnLeftClick(float mouseX, float mouseY) {
    if (cloud_.points.empty()) return;
    if (measure_.mode == ToolMode::Navigate) return;
    if (measure_.mode == ToolMode::Roi || measure_.mode == ToolMode::PlaneFit ||
        measure_.mode == ToolMode::SphereFit || measure_.mode == ToolMode::CircleFit ||
        measure_.mode == ToolMode::CylinderFit || measure_.mode == ToolMode::Flatness ||
        measure_.mode == ToolMode::StepGap)
        return;
    if (measure_.mode == ToolMode::Section) return;

    int vx = 0, vy = 0, vw = 0, vh = 0;
    GetView3dFbRect(vx, vy, vw, vh);
    const auto idx = MeasureTools::PickNearest(
        cloud_, camera_, vw, vh, mouseX - static_cast<float>(vx), mouseY - static_cast<float>(vy),
        12.f, displayIndices_.empty() ? nullptr : &displayIndices_);
    if (!idx) {
        measure_.status = u8"光标附近没有点";
        return;
    }
    const Vec3 pLocal = cloud_.points[*idx];
    const Vec3 p = cloud_.ToWorld(pLocal);

    if (measure_.mode == ToolMode::Pick) {
        measure_.picked = pLocal;
        measure_.distA.reset();
        measure_.distB.reset();
        char buf[160];
        std::snprintf(buf, sizeof(buf), u8"选中点 [#%zu]  X=%.4f  Y=%.4f  Z=%.4f", *idx, p.x, p.y,
                      p.z);
        measure_.status = buf;
        UpdateOverlays();
    } else if (measure_.mode == ToolMode::Distance) {
        measure_.picked.reset();
        if (!measure_.distA || measure_.distB) {
            measure_.distA = pLocal;
            measure_.distB.reset();
            measure_.status = u8"测距: 已标记第 1 点（黄），请再点第 2 点";
        } else {
            measure_.distB = pLocal;
            measure_.distance = (*measure_.distB - *measure_.distA).Length();
            char buf[128];
            std::snprintf(buf, sizeof(buf), u8"距离 = %.6f", measure_.distance);
            measure_.status = buf;
        }
        UpdateOverlays();
    } else if (measure_.mode == ToolMode::ClipPlane) {
        Vec3 n = measure_.plane ? measure_.plane->normal : Vec3{0, 0, 1};
        n = n.Normalized();
        measure_.clipNormal = n;
        measure_.clipD = -n.Dot(pLocal);
        measure_.clipEnabled = true;
        PushHistory(u8"剖切平面");
        MeasureTools::ApplyClipMask(cloud_, measure_.clipNormal, measure_.clipD, true);
        needUpload_ = true;
        measure_.status = u8"已通过该点设置剖切平面";
    } else if (measure_.mode == ToolMode::StepHeight) {
        measure_.picked.reset();
        measure_.distA.reset();
        measure_.distB.reset();
        if (!measure_.stepA || measure_.stepB) {
            measure_.stepA = pLocal;
            measure_.stepB.reset();
            measure_.stepDeltaZ = 0.f;
            measure_.status = u8"台阶: 已选基准点A，请再选测量点B";
        } else {
            measure_.stepB = pLocal;
            measure_.stepDeltaZ = measure_.stepB->z - measure_.stepA->z;
            const Vec3 wa = cloud_.ToWorld(*measure_.stepA);
            const Vec3 wb = cloud_.ToWorld(*measure_.stepB);
            char buf[192];
            std::snprintf(buf, sizeof(buf),
                          u8"台阶高度 ΔZ = %.4f mm  (A.Z=%.4f, B.Z=%.4f)", measure_.stepDeltaZ,
                          wa.z, wb.z);
            measure_.status = buf;
        }
        UpdateOverlays();
    }
}

void Application::UpdateView3dLayout(float contentTop, float contentH, float sidebarW) {
    const ImGuiViewport* vp = ImGui::GetMainViewport();
    const float imageW = ImagePanelWidth();
    view3dX_ = vp->Pos.x + sidebarW;
    view3dY_ = contentTop;
    view3dW_ = std::max(vp->Size.x - sidebarW - imageW, 1.f);
    view3dH_ = std::max(contentH, 1.f);
}

bool Application::MouseInView3d(double mx, double my) const {
    return mx >= static_cast<double>(view3dX_) && mx < static_cast<double>(view3dX_ + view3dW_) &&
           my >= static_cast<double>(view3dY_) && my < static_cast<double>(view3dY_ + view3dH_);
}

void Application::GetView3dFbRect(int& x, int& y, int& w, int& h) const {
    int winW = 0, winH = 0;
    if (window_) glfwGetWindowSize(window_, &winW, &winH);
    const float sx = (winW > 0) ? static_cast<float>(fbW_) / static_cast<float>(winW) : 1.f;
    const float sy = (winH > 0) ? static_cast<float>(fbH_) / static_cast<float>(winH) : 1.f;
    x = static_cast<int>(std::lround(view3dX_ * sx));
    y = static_cast<int>(std::lround(view3dY_ * sy));
    w = std::max(1, static_cast<int>(std::lround(view3dW_ * sx)));
    h = std::max(1, static_cast<int>(std::lround(view3dH_ * sy)));
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x + w > fbW_) w = std::max(1, fbW_ - x);
    if (y + h > fbH_) h = std::max(1, fbH_ - y);
}

void Application::GetView3dGlViewport(int& x, int& y, int& w, int& h) const {
    int fx = 0, fy = 0;
    GetView3dFbRect(fx, fy, w, h);
    // GLFW/FB Y 向下；OpenGL viewport Y 向上
    x = fx;
    y = fbH_ - fy - h;
    if (y < 0) y = 0;
}

float Application::View3dAspect() const {
    int x = 0, y = 0, w = 0, h = 0;
    GetView3dFbRect(x, y, w, h);
    return static_cast<float>(w) / static_cast<float>(std::max(h, 1));
}

void Application::DrawRoiRegionOverlay(ImDrawList* dl, int winW, int winH,
                                       const std::vector<std::size_t>& indices, const char* label,
                                       unsigned int col, unsigned int textCol) {
    if (indices.empty()) return;

    const float scaleX = (winW > 0) ? static_cast<float>(winW) / static_cast<float>(fbW_) : 1.f;
    const float scaleY = (winH > 0) ? static_cast<float>(winH) / static_cast<float>(fbH_) : 1.f;

    // 投影选中点 → 屏幕凸包轮廓（不是轴对齐矩形，旋转后外形跟着选区走）
    const int stride = std::max(1, static_cast<int>(indices.size()) / 12000);
    std::vector<ImVec2> pts;
    pts.reserve(std::min(indices.size(), static_cast<std::size_t>(12000)) + 8);
    for (std::size_t k = 0; k < indices.size(); k += static_cast<std::size_t>(stride)) {
        const std::size_t idx = indices[k];
        if (idx >= cloud_.points.size()) continue;
        if (!cloud_.mask.empty() && !cloud_.mask[idx]) continue;
        float sx = 0.f, sy = 0.f;
        if (!ProjectWorldToScreen(cloud_.points[idx], sx, sy)) continue;
        pts.push_back(ImVec2(sx * scaleX, sy * scaleY));
    }
    if (pts.size() < 2) return;

    // Monotone chain convex hull
    auto cross = [](const ImVec2& o, const ImVec2& a, const ImVec2& b) {
        return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
    };
    std::sort(pts.begin(), pts.end(), [](const ImVec2& a, const ImVec2& b) {
        return (a.x < b.x) || (a.x == b.x && a.y < b.y);
    });
    // 去重过近点，减轻凸包毛刺
    {
        std::vector<ImVec2> uniq;
        uniq.reserve(pts.size());
        for (const ImVec2& p : pts) {
            if (uniq.empty()) {
                uniq.push_back(p);
                continue;
            }
            const float dx = p.x - uniq.back().x;
            const float dy = p.y - uniq.back().y;
            if (dx * dx + dy * dy > 2.25f) uniq.push_back(p);  // >1.5px
        }
        pts.swap(uniq);
    }
    if (pts.size() < 2) return;

    std::vector<ImVec2> hull;
    hull.reserve(pts.size() * 2);
    for (const ImVec2& p : pts) {
        while (hull.size() >= 2 &&
               cross(hull[hull.size() - 2], hull[hull.size() - 1], p) <= 0.f) {
            hull.pop_back();
        }
        hull.push_back(p);
    }
    const std::size_t lower = hull.size();
    for (int i = static_cast<int>(pts.size()) - 2; i >= 0; --i) {
        const ImVec2& p = pts[static_cast<std::size_t>(i)];
        while (hull.size() > lower &&
               cross(hull[hull.size() - 2], hull[hull.size() - 1], p) <= 0.f) {
            hull.pop_back();
        }
        hull.push_back(p);
    }
    if (!hull.empty()) hull.pop_back();  // 首尾重复点
    if (hull.size() < 2) return;

    // 半透明填充 + 轮廓线
    const unsigned int fillCol = (col & 0x00FFFFFFu) | 0x28000000u;
    if (hull.size() >= 3) {
        dl->AddConvexPolyFilled(hull.data(), static_cast<int>(hull.size()), fillCol);
        dl->AddPolyline(hull.data(), static_cast<int>(hull.size()), col, ImDrawFlags_Closed, 2.5f);
    } else {
        dl->AddLine(hull[0], hull[1], col, 2.5f);
    }

    // 标签放在凸包顶部中心
    float minX = hull[0].x, maxX = hull[0].x, minY = hull[0].y;
    for (const ImVec2& p : hull) {
        minX = std::min(minX, p.x);
        maxX = std::max(maxX, p.x);
        minY = std::min(minY, p.y);
    }
    char buf[80];
    std::snprintf(buf, sizeof(buf), u8"%s (%zu点)", label, indices.size());
    const ImVec2 ts = ImGui::CalcTextSize(buf);
    const float tx = (minX + maxX) * 0.5f - ts.x * 0.5f;
    const float ty = minY - ts.y - 4.f;
    dl->AddRectFilled(ImVec2(tx - 4.f, ty - 2.f), ImVec2(tx + ts.x + 4.f, ty + ts.y + 2.f),
                      IM_COL32(12, 16, 20, 190));
    dl->AddText(ImVec2(tx, ty), textCol, buf);
}

void Application::DrawStepGapRegionOverlays(ImDrawList* dl, int winW, int winH) {
    const auto& sg = measure_.stepGap;
    DrawRoiRegionOverlay(dl, winW, winH, sg.regionA, u8"区域 A", IM_COL32(240, 210, 60, 240),
                         IM_COL32(255, 230, 80, 255));
    DrawRoiRegionOverlay(dl, winW, winH, sg.regionB, u8"区域 B", IM_COL32(60, 190, 240, 240),
                         IM_COL32(100, 220, 255, 255));
}

void Application::HandleInput() {
    ImGuiIO& io = ImGui::GetIO();
    glfwGetFramebufferSize(window_, &fbW_, &fbH_);

    // 算法编辑器打开时不处理点云视区交互
    if (algoEditor_.IsVisible()) return;

    // HandleInput 在 DrawUi 之前执行，此处同步点云视区范围
    {
        const ImGuiViewport* vp = ImGui::GetMainViewport();
        const float sidebarW = 320.f;
        const float statusH = 42.f;
        const float toolbarH = 40.f;
        const float menuH = ImGui::GetFrameHeight();
        const float contentTop = vp->Pos.y + menuH + toolbarH;
        const float contentH = vp->Pos.y + vp->Size.y - contentTop - statusH;
        UpdateView3dLayout(contentTop, contentH, sidebarW);
    }

    double mx = 0.0, my = 0.0;
    glfwGetCursorPos(window_, &mx, &my);

    int winW = 0, winH = 0;
    glfwGetWindowSize(window_, &winW, &winH);
    const float scaleX = (winW > 0) ? static_cast<float>(fbW_) / static_cast<float>(winW) : 1.f;
    const float scaleY = (winH > 0) ? static_cast<float>(fbH_) / static_cast<float>(winH) : 1.f;
    const float mouseX = static_cast<float>(mx) * scaleX;
    const float mouseY = static_cast<float>(my) * scaleY;

    const bool left = glfwGetMouseButton(window_, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
    const bool right = glfwGetMouseButton(window_, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
    const bool middle = glfwGetMouseButton(window_, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS;
    const bool alt = glfwGetKey(window_, GLFW_KEY_LEFT_ALT) == GLFW_PRESS ||
                     glfwGetKey(window_, GLFW_KEY_RIGHT_ALT) == GLFW_PRESS;
    const bool shift = glfwGetKey(window_, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
                       glfwGetKey(window_, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS;

    static bool leftWasDown = false;
    const bool roiStyle =
        (measure_.mode == ToolMode::Roi || measure_.mode == ToolMode::PlaneFit ||
         measure_.mode == ToolMode::SphereFit || measure_.mode == ToolMode::CircleFit ||
         measure_.mode == ToolMode::CylinderFit || measure_.mode == ToolMode::Flatness ||
         measure_.mode == ToolMode::StepGap);
    const bool sectionStyle = (measure_.mode == ToolMode::Section);

    const bool inView3d = MouseInView3d(mx, my);
    const bool uiCapture = io.WantCaptureMouse;
    // 点云视区内：中键 / Shift+左键 平移优先，不受 UI 抢鼠标影响
    const bool allowPan = inView3d && (!uiCapture || middle || (left && shift));
    const bool allow3d = inView3d && !uiCapture;

    // 双击点云：将旋转中心设为最近点
    if (allow3d && !cloud_.points.empty() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left) &&
        !shift && !alt) {
        int vx = 0, vy = 0, vw = 0, vh = 0;
        GetView3dFbRect(vx, vy, vw, vh);
        const auto idx = MeasureTools::PickNearest(
            cloud_, camera_, vw, vh, mouseX - static_cast<float>(vx),
            mouseY - static_cast<float>(vy), 14.f,
            displayIndices_.empty() ? nullptr : &displayIndices_);
        if (idx) {
            camera_.SetOrbitTarget(cloud_.points[*idx]);
            const Vec3 w = cloud_.ToWorld(cloud_.points[*idx]);
            char buf[160];
            std::snprintf(buf, sizeof(buf), u8"旋转中心已设为点 #%zu  (%.4f, %.4f, %.4f)", *idx, w.x,
                          w.y, w.z);
            measure_.status = buf;
            rotating_ = false;
            panning_ = false;
        } else {
            measure_.status = u8"双击附近没有点，无法设置旋转中心";
        }
    }

    if (allowPan) {
        const bool wantPan = middle || (left && shift);
        if (wantPan) {
            if (!panning_) {
                panning_ = true;
                lastX_ = mx;
                lastY_ = my;
            } else {
                const float panSens = middle ? 1.f : 0.35f;
                camera_.Pan(static_cast<float>(mx - lastX_), static_cast<float>(my - lastY_),
                            panSens);
                lastX_ = mx;
                lastY_ = my;
            }
            rotating_ = false;
            if (sectionDragging_) EndSectionDrag();
            if (measure_.roiDragging) EndRoiDrag();
        } else if (allow3d) {
            panning_ = false;

            if (sectionStyle) {
                if (left && !leftWasDown && !shift && !alt) BeginSectionDrag(mouseX, mouseY);
                if (sectionDragging_ && left) UpdateSectionDrag(mouseX, mouseY);
                if (sectionDragging_ && !left) EndSectionDrag();

                if (right) {
                    if (!rotating_) {
                        rotating_ = true;
                        lastX_ = mx;
                        lastY_ = my;
                    } else {
                        camera_.Orbit(static_cast<float>((mx - lastX_) * 0.005),
                                      static_cast<float>((my - lastY_) * 0.005));
                        lastX_ = mx;
                        lastY_ = my;
                    }
                } else {
                    rotating_ = false;
                }
            } else if (roiStyle) {
                if (left && !leftWasDown && !shift) BeginRoiDrag(mouseX, mouseY);
                if (measure_.roiDragging && left) UpdateRoiDrag(mouseX, mouseY);
                if (measure_.roiDragging && !left) EndRoiDrag();

                if (right) {
                    if (!rotating_) {
                        rotating_ = true;
                        lastX_ = mx;
                        lastY_ = my;
                    } else {
                        camera_.Orbit(static_cast<float>((mx - lastX_) * 0.005),
                                      static_cast<float>((my - lastY_) * 0.005));
                        lastX_ = mx;
                        lastY_ = my;
                    }
                } else {
                    rotating_ = false;
                }
            } else {
                if (left && !leftWasDown && !alt && !shift &&
                    measure_.mode != ToolMode::Navigate) {
                    OnLeftClick(mouseX, mouseY);
                }

                const bool orbit =
                    right || (left && measure_.mode == ToolMode::Navigate && !alt && !shift);
                if (orbit) {
                    if (!rotating_) {
                        rotating_ = true;
                        lastX_ = mx;
                        lastY_ = my;
                    } else {
                        camera_.Orbit(static_cast<float>((mx - lastX_) * 0.005),
                                      static_cast<float>((my - lastY_) * 0.005));
                        lastX_ = mx;
                        lastY_ = my;
                    }
                } else {
                    rotating_ = false;
                }
            }

            if (io.MouseWheel != 0.f) {
                camera_.Zoom(io.MouseWheel);
            }
        } else {
            panning_ = false;
        }
    } else {
        panning_ = false;
        rotating_ = false;
        if (sectionDragging_) EndSectionDrag();
    }

    leftWasDown = left;

    // Ctrl+Z / Ctrl+Y undo redo; Ctrl+S save; Ctrl+O open
    const bool ctrl = glfwGetKey(window_, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS ||
                      glfwGetKey(window_, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS;
    static bool zWas = false, yWas = false, sWas = false, oWas = false;
    const bool zDown = glfwGetKey(window_, GLFW_KEY_Z) == GLFW_PRESS;
    const bool yDown = glfwGetKey(window_, GLFW_KEY_Y) == GLFW_PRESS;
    const bool sDown = glfwGetKey(window_, GLFW_KEY_S) == GLFW_PRESS;
    const bool oDown = glfwGetKey(window_, GLFW_KEY_O) == GLFW_PRESS;
    if (ctrl && zDown && !zWas) Undo();
    if (ctrl && yDown && !yWas) Redo();
    if (ctrl && sDown && !sWas) SaveCloud();
    if (ctrl && oDown && !oWas) {
        const std::string path = FileDialog::OpenPointCloudFile();
        if (!path.empty()) LoadPath(path);
    }
    zWas = zDown;
    yWas = yDown;
    sWas = sDown;
    oWas = oDown;
}

void Application::DrawOverlays() {
    int winW = 0, winH = 0;
    glfwGetWindowSize(window_, &winW, &winH);
    const float sx = (fbW_ > 0) ? static_cast<float>(winW) / static_cast<float>(fbW_) : 1.f;
    const float sy = (fbH_ > 0) ? static_cast<float>(winH) / static_cast<float>(fbH_) : 1.f;
    // 区域框画在 Background，避免盖住直方图等浮动窗口
    ImDrawList* dlBg = ImGui::GetBackgroundDrawList();
    ImDrawList* dl = ImGui::GetForegroundDrawList();

    // 叠加标注限制在点云视区内，避免画进侧栏 / 2D 图像窗口
    const ImVec2 clipMin(view3dX_, view3dY_);
    const ImVec2 clipMax(view3dX_ + view3dW_, view3dY_ + view3dH_);
    dlBg->PushClipRect(clipMin, clipMax, true);
    dl->PushClipRect(clipMin, clipMax, true);

    // 框选拖拽中：屏幕矩形；完成后：3D 投影框（随视角动）
    const bool roiDraggingNow =
        measure_.roiDragging ||
        (measure_.roiX0 != measure_.roiX1 || measure_.roiY0 != measure_.roiY1);

    if (measure_.mode == ToolMode::Roi || measure_.mode == ToolMode::PlaneFit ||
        measure_.mode == ToolMode::SphereFit || measure_.mode == ToolMode::CircleFit ||
        measure_.mode == ToolMode::CylinderFit || measure_.mode == ToolMode::Flatness) {
        if (roiDraggingNow) {
            dlBg->AddRect(ImVec2(measure_.roiX0 * sx, measure_.roiY0 * sy),
                          ImVec2(measure_.roiX1 * sx, measure_.roiY1 * sy),
                          IM_COL32(64, 200, 180, 230), 0.f, 0, 2.f);
        } else if (!measure_.roiIndices.empty()) {
            const char* label = u8"框选区域";
            if (measure_.mode == ToolMode::PlaneFit) label = u8"平面拟合区域";
            if (measure_.mode == ToolMode::SphereFit) label = u8"球面拟合区域";
            if (measure_.mode == ToolMode::CircleFit) label = u8"圆拟合区域";
            if (measure_.mode == ToolMode::CylinderFit) label = u8"圆柱拟合区域";
            if (measure_.mode == ToolMode::Flatness) label = u8"平面度区域";
            if (measure_.mode == ToolMode::Roi) label = u8"ROI 区域";
            DrawRoiRegionOverlay(dlBg, winW, winH, measure_.roiIndices, label,
                                 IM_COL32(64, 200, 180, 240), IM_COL32(120, 230, 210, 255));
        }
        if (measure_.mode == ToolMode::Flatness && measure_.flatness.valid &&
            !measure_.flatness.indices.empty() && measure_.roiIndices.empty()) {
            DrawRoiRegionOverlay(dlBg, winW, winH, measure_.flatness.indices, u8"平面度区域",
                                 IM_COL32(64, 200, 180, 240), IM_COL32(120, 230, 210, 255));
        }
    }

    if (measure_.mode == ToolMode::StepGap) {
        if (roiDraggingNow) {
            dlBg->AddRect(ImVec2(measure_.roiX0 * sx, measure_.roiY0 * sy),
                          ImVec2(measure_.roiX1 * sx, measure_.roiY1 * sy),
                          IM_COL32(64, 200, 180, 230), 0.f, 0, 2.f);
        }
        DrawStepGapRegionOverlays(dlBg, winW, winH);
    }

    auto project = [&](const Vec3& p, ImVec2& out) -> bool {
        float px = 0.f, py = 0.f;
        if (!ProjectWorldToScreen(p, px, py)) return false;
        out.x = px * sx;
        out.y = py * sy;
        return true;
    };

    if (measure_.distA) {
        ImVec2 a;
        if (project(*measure_.distA, a)) {
            dl->AddCircleFilled(a, 7.f, IM_COL32(255, 230, 60, 255));
            dl->AddText(ImVec2(a.x + 12.f, a.y - 10.f), IM_COL32(255, 230, 60, 255), u8"点1");
        }
    }
    if (measure_.distB) {
        ImVec2 b;
        if (project(*measure_.distB, b)) {
            dl->AddCircleFilled(b, 7.f, IM_COL32(255, 90, 40, 255));
            dl->AddText(ImVec2(b.x + 12.f, b.y - 10.f), IM_COL32(255, 120, 60, 255), u8"点2");
        }
    }
    if (measure_.distA && measure_.distB) {
        ImVec2 a, b;
        if (project(*measure_.distA, a) && project(*measure_.distB, b)) {
            dl->AddLine(a, b, IM_COL32(255, 240, 80, 255), 2.5f);
            char label[64];
            std::snprintf(label, sizeof(label), u8"距离 %.6f", measure_.distance);
            dl->AddText(ImVec2((a.x + b.x) * 0.5f, (a.y + b.y) * 0.5f - 14.f),
                        IM_COL32(255, 250, 180, 255), label);
        }
    }

    // Step height annotation
    if (measure_.stepA) {
        ImVec2 a;
        if (project(*measure_.stepA, a)) {
            dl->AddCircleFilled(a, 7.f, IM_COL32(80, 200, 255, 255));
            dl->AddText(ImVec2(a.x + 10.f, a.y - 10.f), IM_COL32(120, 210, 255, 255), u8"A基准");
        }
    }
    if (measure_.stepB) {
        ImVec2 b;
        if (project(*measure_.stepB, b)) {
            dl->AddCircleFilled(b, 7.f, IM_COL32(255, 160, 60, 255));
            dl->AddText(ImVec2(b.x + 10.f, b.y - 10.f), IM_COL32(255, 180, 80, 255), u8"B测量");
        }
    }
    if (measure_.stepA && measure_.stepB) {
        ImVec2 a, b;
        if (project(*measure_.stepA, a) && project(*measure_.stepB, b)) {
            // Vertical height dimension at the right of the two points
            const float x = std::max(a.x, b.x) + 20.f;
            dl->AddLine(ImVec2(x, a.y), ImVec2(x, b.y), IM_COL32(255, 200, 80, 255), 2.f);
            dl->AddLine(ImVec2(a.x, a.y), ImVec2(x, a.y), IM_COL32(255, 200, 80, 160), 1.f);
            dl->AddLine(ImVec2(b.x, b.y), ImVec2(x, b.y), IM_COL32(255, 200, 80, 160), 1.f);
            char label[64];
            std::snprintf(label, sizeof(label), u8"ΔZ=%.4f mm", measure_.stepDeltaZ);
            dl->AddText(ImVec2(x + 6.f, (a.y + b.y) * 0.5f - 8.f), IM_COL32(255, 220, 120, 255),
                        label);
        }
    }

    // 截面 2D 轮廓选点 → 同步标在 3D 点云上
    {
        const auto& sec = measure_.section;
        ImVec2 sa, sb;
        bool hasA = false, hasB = false;
        if (sec.pickA && *sec.pickA < sec.points.size()) {
            hasA = project(sec.points[*sec.pickA].p3, sa);
            if (hasA) {
                dl->AddCircleFilled(sa, 8.f, IM_COL32(255, 230, 60, 255));
                dl->AddCircle(sa, 11.f, IM_COL32(255, 255, 255, 200), 0, 1.8f);
                dl->AddText(ImVec2(sa.x + 12.f, sa.y - 12.f), IM_COL32(255, 230, 60, 255), u8"A");
            }
        }
        if (sec.pickB && *sec.pickB < sec.points.size()) {
            hasB = project(sec.points[*sec.pickB].p3, sb);
            if (hasB) {
                dl->AddCircleFilled(sb, 8.f, IM_COL32(255, 100, 50, 255));
                dl->AddCircle(sb, 11.f, IM_COL32(255, 255, 255, 200), 0, 1.8f);
                dl->AddText(ImVec2(sb.x + 12.f, sb.y - 12.f), IM_COL32(255, 120, 60, 255), u8"B");
            }
        }
        if (hasA && hasB) {
            dl->AddLine(sa, sb, IM_COL32(255, 220, 100, 220), 2.2f);
            char label[80];
            std::snprintf(label, sizeof(label), u8"Z向 %.6f", sec.zDistance);
            dl->AddText(ImVec2((sa.x + sb.x) * 0.5f + 8.f, (sa.y + sb.y) * 0.5f - 10.f),
                        IM_COL32(140, 230, 255, 255), label);
        }
    }

    // Axis endpoint labels (mm) near origin axes
    if (showAxes_ && cloud_.bounds.Valid()) {
        const float L = axesLength_;
        ImVec2 o, xp, yp, zp;
        if (project({0, 0, 0}, o)) {
            dl->AddText(ImVec2(o.x + 4.f, o.y + 4.f), IM_COL32(200, 200, 200, 220), u8"O");
        }
        if (project({L, 0, 0}, xp)) {
            char buf[48];
            std::snprintf(buf, sizeof(buf), u8"X %.2fmm", L);
            dl->AddText(xp, IM_COL32(255, 90, 90, 255), buf);
        }
        if (project({0, L, 0}, yp)) {
            char buf[48];
            std::snprintf(buf, sizeof(buf), u8"Y %.2fmm", L);
            dl->AddText(yp, IM_COL32(90, 255, 120, 255), buf);
        }
        if (project({0, 0, L}, zp)) {
            char buf[48];
            std::snprintf(buf, sizeof(buf), u8"Z %.2fmm", L);
            dl->AddText(zp, IM_COL32(100, 160, 255, 255), buf);
        }
    }

    dl->PopClipRect();
    dlBg->PopClipRect();
}

void Application::DrawStepGapPanel() {
    auto& sg = measure_.stepGap;
    if (!sg.hasDistances || sg.signedDistB.empty()) return;

    const ImGuiViewport* vp = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(
        ImVec2(vp->Pos.x + vp->Size.x - 420.f - 12.f, vp->Pos.y + 120.f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(420.f, 360.f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowBgAlpha(0.98f);
    if (!ImGui::Begin(u8"段差 ΔZ 高度图", nullptr, ImGuiWindowFlags_NoCollapse)) {
        ImGui::End();
        return;
    }
    // 保证直方图窗口叠在其它普通窗口之上（区域框已改背景层，不会再盖住本窗）
    if (ImGui::IsWindowAppearing()) {
        ImGui::SetWindowFocus();
    }

    ImGui::Text(u8"段差 ΔZ = %.6f", sg.mean);
    ImGui::TextDisabled(u8"基准 A 平均 Z = %.6f", sg.zRefA);
    ImGui::Text(u8"中位数 %.6f　平均|ΔZ| %.6f", sg.median, sg.meanAbs);
    ImGui::Text(u8"范围 [%.6f , %.6f]　RMS %.6f", sg.minDist, sg.maxDist, sg.rms);
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Text(u8"ΔZ 直方图");
    ImGui::TextDisabled(u8"横轴 = ΔZ 数值 (mm)　纵轴 = 该区间点数");

    constexpr int bins = 32;
    float hist[bins] = {};
    const float lo = sg.minDist;
    const float hi = sg.maxDist;
    const float span = std::max(hi - lo, 1e-6f);
    for (float d : sg.signedDistB) {
        int b = static_cast<int>((d - lo) / span * bins);
        b = std::clamp(b, 0, bins - 1);
        hist[b] += 1.f;
    }
    float peak = 1.f;
    int peakBin = 0;
    for (int i = 0; i < bins; ++i) {
        if (hist[i] > peak) {
            peak = hist[i];
            peakBin = i;
        }
    }

    const float plotH = 160.f;
    const float leftPad = 46.f;
    const float bottomPad = 36.f;
    const float topPad = 8.f;
    const float rightPad = 10.f;
    const ImVec2 avail = ImGui::GetContentRegionAvail();
    const float plotW = std::max(avail.x, 180.f);
    const ImVec2 canvasPos = ImGui::GetCursorScreenPos();
    const ImVec2 canvasSize(plotW, plotH + bottomPad + topPad);
    ImGui::InvisibleButton(u8"##seghistcanvas", canvasSize);

    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImVec2 origin(canvasPos.x + leftPad, canvasPos.y + topPad + plotH);
    const float chartW = plotW - leftPad - rightPad;
    const float chartH = plotH;
    const float barW = chartW / static_cast<float>(bins);

    // Background
    dl->AddRectFilled(ImVec2(origin.x, origin.y - chartH), ImVec2(origin.x + chartW, origin.y),
                      IM_COL32(18, 22, 26, 255));
    dl->AddRect(ImVec2(origin.x, origin.y - chartH), ImVec2(origin.x + chartW, origin.y),
                IM_COL32(70, 90, 100, 200));

    // Bars
    for (int i = 0; i < bins; ++i) {
        const float t = (hist[i] / peak);
        const float h = t * (chartH - 2.f);
        const float x0 = origin.x + barW * static_cast<float>(i);
        const float x1 = x0 + barW - 1.f;
        const float colorT = (static_cast<float>(i) + 0.5f) / static_cast<float>(bins);
        const Vec3 c = DivergingColor(colorT);
        dl->AddRectFilled(ImVec2(x0, origin.y - h), ImVec2(x1, origin.y),
                          IM_COL32(static_cast<int>(c.x * 255), static_cast<int>(c.y * 255),
                                   static_cast<int>(c.z * 255), 230));
    }

    // Y axis ticks (点数)
    char yLabel[32];
    for (int i = 0; i <= 4; ++i) {
        const float t = static_cast<float>(i) / 4.f;
        const float y = origin.y - t * chartH;
        const float val = peak * t;
        dl->AddLine(ImVec2(origin.x - 4.f, y), ImVec2(origin.x, y), IM_COL32(160, 180, 190, 200));
        std::snprintf(yLabel, sizeof(yLabel), "%.0f", val);
        const ImVec2 ts = ImGui::CalcTextSize(yLabel);
        dl->AddText(ImVec2(origin.x - 6.f - ts.x, y - ts.y * 0.5f), IM_COL32(180, 200, 210, 255),
                    yLabel);
    }
    {
        const char* yl = u8"点数";
        const ImVec2 ts = ImGui::CalcTextSize(yl);
        dl->AddText(ImVec2(canvasPos.x + 2.f, canvasPos.y + topPad + (chartH - ts.y) * 0.5f),
                    IM_COL32(140, 200, 210, 255), yl);
    }

    // X axis ticks (ΔZ)
    char xLabel[48];
    for (int i = 0; i <= 4; ++i) {
        const float t = static_cast<float>(i) / 4.f;
        const float x = origin.x + t * chartW;
        const float val = lo + t * span;
        dl->AddLine(ImVec2(x, origin.y), ImVec2(x, origin.y + 4.f), IM_COL32(160, 180, 190, 200));
        std::snprintf(xLabel, sizeof(xLabel), "%.3f", val);
        const ImVec2 ts = ImGui::CalcTextSize(xLabel);
        dl->AddText(ImVec2(x - ts.x * 0.5f, origin.y + 6.f), IM_COL32(180, 200, 210, 255), xLabel);
    }
    {
        const char* xl = u8"ΔZ (mm)";
        const ImVec2 ts = ImGui::CalcTextSize(xl);
        dl->AddText(ImVec2(origin.x + chartW * 0.5f - ts.x * 0.5f, origin.y + 20.f),
                    IM_COL32(140, 200, 210, 255), xl);
    }

    // Hover: show bin ΔZ range + count
    if (ImGui::IsItemHovered()) {
        const ImVec2 mp = ImGui::GetMousePos();
        if (mp.x >= origin.x && mp.x <= origin.x + chartW && mp.y >= origin.y - chartH &&
            mp.y <= origin.y) {
            int b = static_cast<int>((mp.x - origin.x) / barW);
            b = std::clamp(b, 0, bins - 1);
            const float b0 = lo + span * (static_cast<float>(b) / bins);
            const float b1 = lo + span * (static_cast<float>(b + 1) / bins);
            ImGui::BeginTooltip();
            ImGui::Text(u8"ΔZ 区间: [%.4f , %.4f]", b0, b1);
            ImGui::Text(u8"点数: %.0f", hist[b]);
            ImGui::EndTooltip();
        }
    }

    ImGui::Spacing();
    const float peakLo = lo + span * (static_cast<float>(peakBin) / bins);
    const float peakHi = lo + span * (static_cast<float>(peakBin + 1) / bins);
    ImGui::Text(u8"峰值区间 ΔZ ∈ [%.4f , %.4f]，点数 %.0f", peakLo, peakHi, peak);
    ImGui::TextDisabled(u8"鼠标悬停柱条可查看该区间 ΔZ 与点数");

    // Color legend bar
    const ImVec2 p0 = ImGui::GetCursorScreenPos();
    const float w = ImGui::GetContentRegionAvail().x;
    const float h = 14.f;
    for (int i = 0; i < 64; ++i) {
        const float t = static_cast<float>(i) / 63.f;
        const Vec3 c = DivergingColor(t);
        const float x0 = p0.x + w * (static_cast<float>(i) / 64.f);
        const float x1 = p0.x + w * (static_cast<float>(i + 1) / 64.f);
        dl->AddRectFilled(ImVec2(x0, p0.y), ImVec2(x1, p0.y + h),
                          IM_COL32(static_cast<int>(c.x * 255), static_cast<int>(c.y * 255),
                                   static_cast<int>(c.z * 255), 255));
    }
    ImGui::Dummy(ImVec2(w, h + 4.f));
    ImGui::TextDisabled(u8"蓝 ← 负 ΔZ | 红 → 正 ΔZ");

    ImGui::End();
}

void Application::DrawSectionPanel() {
    auto& sec = measure_.section;
    const ImGuiViewport* vp = ImGui::GetMainViewport();
    const float panelW = 440.f;
    const float panelH = 400.f;
    ImGui::SetNextWindowPos(
        ImVec2(vp->WorkPos.x + vp->WorkSize.x - panelW - 12.f, vp->WorkPos.y + 12.f),
        ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(panelW, panelH), ImGuiCond_FirstUseEver);
    ImGui::Begin(u8"截面 2D 轮廓", nullptr);

    if (sec.points.empty()) {
        ImGui::TextDisabled(u8"尚未生成截面。可在 3D 中左键拖拽橙色切面，或点“生成截面”。");
        ImGui::End();
        return;
    }

    ImGui::Text(u8"轮廓点数: %zu", sec.points.size());
    ImGui::TextWrapped(
        u8"单击选点；按住点1/点2拖动可沿轮廓移动，垂线间距与 Z 向距离实时更新。");
    if (sec.pickA && sec.pickB) {
        ImGui::Text(u8"垂线间距 = %.6f", sec.lineDistance);
        ImGui::Text(u8"Z 向距离 = %.6f", sec.zDistance);
    }
    if (ImGui::Button(u8"清除截面测距")) {
        sec.pickA.reset();
        sec.pickB.reset();
        sec.lineDistance = 0.f;
        sec.zDistance = 0.f;
        sectionPlotDragTarget_ = 0;
    }

    ImVec2 canvasPos = ImGui::GetCursorScreenPos();
    ImVec2 canvasSize = ImGui::GetContentRegionAvail();
    if (canvasSize.x < 50.f) canvasSize.x = 50.f;
    if (canvasSize.y < 50.f) canvasSize.y = 50.f;

    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImVec2 canvasMax(canvasPos.x + canvasSize.x, canvasPos.y + canvasSize.y);
    dl->AddRectFilled(canvasPos, canvasMax, IM_COL32(18, 22, 26, 255), 6.f);
    dl->AddRect(canvasPos, canvasMax, IM_COL32(60, 90, 100, 255), 6.f);

    const float pad = 0.08f;
    const float du = (sec.uMax - sec.uMin) * (1.f + pad * 2.f);
    const float dv = (sec.vMax - sec.vMin) * (1.f + pad * 2.f);
    const float uStart = sec.uMin - (sec.uMax - sec.uMin) * pad;
    const float vStart = sec.vMin - (sec.vMax - sec.vMin) * pad;

    auto toScreen = [&](float u, float v) -> ImVec2 {
        const float nx = (u - uStart) / du;
        const float ny = 1.f - (v - vStart) / dv;
        return {canvasPos.x + nx * canvasSize.x, canvasPos.y + ny * canvasSize.y};
    };

    dl->AddText(ImVec2(canvasPos.x + 8.f, canvasPos.y + 8.f), IM_COL32(140, 180, 190, 255),
                sec.cutAlongX ? u8"横轴:Y  纵轴:Z" : u8"横轴:X  纵轴:Z");

    const int step = std::max(1, static_cast<int>(sec.points.size()) / 4000);
    for (std::size_t i = 0; i + static_cast<std::size_t>(step) < sec.points.size();
         i += static_cast<std::size_t>(step)) {
        ImVec2 a = toScreen(sec.points[i].u, sec.points[i].v);
        ImVec2 b = toScreen(sec.points[i + static_cast<std::size_t>(step)].u,
                            sec.points[i + static_cast<std::size_t>(step)].v);
        dl->AddLine(a, b, IM_COL32(80, 200, 220, 200), 1.2f);
    }
    for (std::size_t i = 0; i < sec.points.size(); i += static_cast<std::size_t>(step)) {
        ImVec2 p = toScreen(sec.points[i].u, sec.points[i].v);
        dl->AddCircleFilled(p, 1.6f, IM_COL32(120, 220, 230, 220));
    }

    auto drawVertLine = [&](float u, ImU32 col, const char* tag) {
        ImVec2 top = toScreen(u, sec.vMax + (sec.vMax - sec.vMin) * pad);
        ImVec2 bot = toScreen(u, sec.vMin - (sec.vMax - sec.vMin) * pad);
        // Clamp to canvas
        top.y = canvasPos.y + 4.f;
        bot.y = canvasMax.y - 4.f;
        top.x = bot.x = toScreen(u, (sec.vMin + sec.vMax) * 0.5f).x;
        dl->AddLine(top, bot, col, 2.0f);
        dl->AddText(ImVec2(top.x + 6.f, top.y + 4.f), col, tag);
    };

    if (sec.pickA) {
        const auto& a = sec.points[*sec.pickA];
        ImVec2 pa = toScreen(a.u, a.v);
        drawVertLine(a.u, IM_COL32(255, 230, 60, 230), u8"垂线A");
        dl->AddCircleFilled(pa, 6.f, IM_COL32(255, 230, 60, 255));
        dl->AddText(ImVec2(pa.x + 8.f, pa.y - 8.f), IM_COL32(255, 230, 60, 255), u8"A");
    }
    if (sec.pickB) {
        const auto& b = sec.points[*sec.pickB];
        ImVec2 pb = toScreen(b.u, b.v);
        drawVertLine(b.u, IM_COL32(255, 110, 50, 230), u8"垂线B");
        dl->AddCircleFilled(pb, 6.f, IM_COL32(255, 100, 50, 255));
        dl->AddText(ImVec2(pb.x + 8.f, pb.y - 8.f), IM_COL32(255, 120, 60, 255), u8"B");
    }
    if (sec.pickA && sec.pickB) {
        const auto& a = sec.points[*sec.pickA];
        const auto& b = sec.points[*sec.pickB];
        ImVec2 pa = toScreen(a.u, a.v);
        ImVec2 pb = toScreen(b.u, b.v);

        // Horizontal dimension between the two vertical lines (at mid height)
        const float midY = (canvasPos.y + canvasMax.y) * 0.5f;
        const float x1 = toScreen(a.u, a.v).x;
        const float x2 = toScreen(b.u, b.v).x;
        dl->AddLine(ImVec2(x1, midY), ImVec2(x2, midY), IM_COL32(255, 240, 120, 220), 1.8f);
        char lineLabel[80];
        std::snprintf(lineLabel, sizeof(lineLabel), u8"垂线间距 %.6f", sec.lineDistance);
        dl->AddText(ImVec2((x1 + x2) * 0.5f - 40.f, midY - 18.f), IM_COL32(255, 245, 160, 255),
                    lineLabel);

        // Z distance: vertical segment between the two points' Z, drawn beside them
        const float zx = std::max(x1, x2) + 18.f;
        dl->AddLine(ImVec2(zx, pa.y), ImVec2(zx, pb.y), IM_COL32(120, 220, 255, 230), 1.8f);
        char zLabel[80];
        std::snprintf(zLabel, sizeof(zLabel), u8"Z向 %.6f", sec.zDistance);
        dl->AddText(ImVec2(zx + 6.f, (pa.y + pb.y) * 0.5f - 8.f), IM_COL32(140, 230, 255, 255),
                    zLabel);
    }

    ImGui::InvisibleButton(u8"##sectioncanvas", canvasSize);
    const ImVec2 mp = ImGui::GetMousePos();
    const float localX = mp.x - canvasPos.x;
    const float localY = mp.y - canvasPos.y;

    if (ImGui::IsItemClicked(ImGuiMouseButton_Left)) {
        OnSectionPlotClick(localX, localY, canvasSize.x, canvasSize.y);
    }

    // Drag point1 / point2 along the contour with live measurement update.
    if (ImGui::IsItemActive() && sectionPlotDragTarget_ != 0 &&
        ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
        const auto nearest = FindNearestSectionPoint(localX, localY, canvasSize.x, canvasSize.y);
        if (nearest) {
            if (sectionPlotDragTarget_ == 1) {
                sec.pickA = *nearest;
            } else if (sectionPlotDragTarget_ == 2) {
                sec.pickB = *nearest;
            }
            if (sec.pickA && sec.pickB) {
                UpdateSectionDistances();
            } else if (sectionPlotDragTarget_ == 1) {
                measure_.status = u8"拖动点1中…再单击选择点2";
            }
        }
    }

    if (!ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
        sectionPlotDragTarget_ = 0;
    }

    // Change cursor when hovering markers
    if (ImGui::IsItemHovered()) {
        if (HitSectionPickMarker(localX, localY, canvasSize.x, canvasSize.y, true) ||
            HitSectionPickMarker(localX, localY, canvasSize.x, canvasSize.y, false) ||
            sectionPlotDragTarget_ != 0) {
            ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
        }
    }

    ImGui::End();
}

namespace {
const char* ToolModeLabel(ToolMode mode) {
    switch (mode) {
        case ToolMode::Navigate:
            return u8"漫游";
        case ToolMode::Pick:
            return u8"点选";
        case ToolMode::Distance:
            return u8"测距";
        case ToolMode::PlaneFit:
            return u8"平面拟合";
        case ToolMode::SphereFit:
            return u8"球面拟合";
        case ToolMode::CircleFit:
            return u8"圆拟合";
        case ToolMode::CylinderFit:
            return u8"圆柱拟合";
        case ToolMode::Roi:
            return u8"ROI框选";
        case ToolMode::ClipPlane:
            return u8"剖切平面";
        case ToolMode::Section:
            return u8"截面";
        case ToolMode::StepHeight:
            return u8"台阶高度";
        case ToolMode::Flatness:
            return u8"平面度";
        case ToolMode::StepGap:
            return u8"段差计算";
    }
    return u8"工具";
}
}  // namespace

void Application::ClearToolVisuals(bool resetStatus) {
    measure_.roiDragging = false;
    measure_.roiX0 = measure_.roiX1 = 0.f;
    measure_.roiY0 = measure_.roiY1 = 0.f;
    measure_.roiIndices.clear();
    measure_.picked.reset();
    measure_.distA.reset();
    measure_.distB.reset();
    measure_.distance = 0.f;
    measure_.plane.reset();
    measure_.sphere.reset();
    measure_.circle.reset();
    measure_.cylinder.reset();
    measure_.stepA.reset();
    measure_.stepB.reset();
    measure_.stepDeltaZ = 0.f;
    measure_.flatness = {};
    measure_.stepGap = {};
    measure_.stepGap.phase = StepGapPhase::SelectA;
    measure_.section.points.clear();
    measure_.section.pickA.reset();
    measure_.section.pickB.reset();
    measure_.section.lineDistance = 0.f;
    measure_.section.zDistance = 0.f;
    sectionDragging_ = false;
    sectionPlotDragTarget_ = 0;
    if (resetStatus) {
        measure_.status = u8"已清空当前工具显示";
    }
    needUpload_ = true;
    UpdateOverlays();
}

void Application::SetToolMode(ToolMode mode) {
    if (mode == measure_.mode) return;

    ClearToolVisuals(false);
    measure_.mode = mode;

    if (mode == ToolMode::Flatness) {
        measure_.status = u8"平面度：框选区域后点击计算";
    } else if (mode == ToolMode::StepGap) {
        measure_.stepGap.phase = StepGapPhase::SelectA;
        measure_.status = u8"段差：先框选基准区域 A";
    } else if (mode == ToolMode::Roi) {
        measure_.status = u8"ROI：左键拖拽框选可见表面（不含遮挡点）";
    } else if (mode == ToolMode::PlaneFit) {
        measure_.status = u8"平面拟合：框选可见表面后拟合";
    } else if (mode == ToolMode::SphereFit) {
        measure_.status = u8"球面拟合：框选可见表面后拟合";
    } else if (mode == ToolMode::CircleFit) {
        measure_.status = u8"圆拟合：框选可见表面后拟合";
    } else if (mode == ToolMode::CylinderFit) {
        measure_.status = u8"圆柱拟合：框选可见表面后拟合";
    } else if (mode == ToolMode::Section) {
        if (cloud_.bounds.Valid()) {
            measure_.section.position = measure_.section.cutAlongX ? cloud_.bounds.Center().x
                                                                  : cloud_.bounds.Center().y;
        }
        SyncSectionCutPlane();
        measure_.status = u8"截面：拖拽橙色切面或生成截面";
    } else if (mode == ToolMode::Navigate) {
        measure_.status = u8"漫游模式";
    } else if (mode == ToolMode::Pick) {
        measure_.status = u8"点选：单击读取坐标";
    } else if (mode == ToolMode::Distance) {
        measure_.status = u8"测距：依次点击两点";
    } else if (mode == ToolMode::StepHeight) {
        measure_.status = u8"台阶高度：依次点击 A、B";
    } else if (mode == ToolMode::ClipPlane) {
        measure_.status = u8"剖切：点击一点设置剖切面";
    } else {
        measure_.status = ToolModeLabel(mode);
    }

    needUpload_ = true;
    UpdateOverlays();
}

float Application::DrawMenuBar() {
    const ImGuiViewport* vp = ImGui::GetMainViewport();
    float menuBottom = vp->Pos.y;
    if (!ImGui::BeginMainMenuBar()) return menuBottom;

    if (ImGui::BeginMenu(u8"文件")) {
        if (ImGui::MenuItem(u8"打开点云…", "Ctrl+O")) {
            const std::string path = FileDialog::OpenPointCloudFile();
            if (!path.empty()) LoadPath(path);
        }
        if (ImGui::MenuItem(u8"打开深度图…")) {
            OpenDepthImage();
        }
        if (ImGui::MenuItem(u8"打开亮度图…")) {
            OpenBrightnessImage();
        }
        if (ImGui::MenuItem(u8"加载示例")) {
            LoadPath("assets/sample/sample.xyz");
        }
        ImGui::Separator();
        if (ImGui::MenuItem(u8"保存点云…", "Ctrl+S", false, !cloud_.points.empty())) {
            SaveCloud();
        }
        ImGui::Checkbox(u8"保存时仅可见点", &saveVisibleOnly_);
        ImGui::EndMenu();
    }

    if (ImGui::BeginMenu(u8"创建")) {
        if (ImGui::MenuItem(u8"球面点云…")) showCreateSphere_ = true;
        if (ImGui::MenuItem(u8"圆柱点云…")) showCreateCylinder_ = true;
        ImGui::EndMenu();
    }

    if (ImGui::BeginMenu(u8"编辑")) {
        const bool canUndo = history_.CanUndo();
        const bool canRedo = history_.CanRedo();
        if (ImGui::MenuItem(u8"撤销", "Ctrl+Z", false, canUndo)) Undo();
        if (ImGui::MenuItem(u8"重做", "Ctrl+Y", false, canRedo)) Redo();
        ImGui::Separator();
        if (ImGui::MenuItem(u8"清空显示")) ClearToolVisuals(true);
        ImGui::EndMenu();
    }

    if (ImGui::BeginMenu(u8"工具")) {
        auto toolItem = [&](ToolMode mode) {
            const bool selected = measure_.mode == mode;
            if (ImGui::MenuItem(ToolModeLabel(mode), nullptr, selected)) {
                SetToolMode(mode);
            }
        };
        toolItem(ToolMode::Navigate);
        toolItem(ToolMode::Pick);
        toolItem(ToolMode::Distance);
        toolItem(ToolMode::Roi);
        toolItem(ToolMode::ClipPlane);
        toolItem(ToolMode::Section);
        toolItem(ToolMode::StepHeight);
        ImGui::Separator();
        toolItem(ToolMode::Flatness);
        toolItem(ToolMode::StepGap);
        ImGui::EndMenu();
    }

    if (ImGui::BeginMenu(u8"拟合")) {
        auto toolItem = [&](ToolMode mode) {
            const bool selected = measure_.mode == mode;
            if (ImGui::MenuItem(ToolModeLabel(mode), nullptr, selected)) {
                SetToolMode(mode);
            }
        };
        toolItem(ToolMode::PlaneFit);
        toolItem(ToolMode::SphereFit);
        toolItem(ToolMode::CircleFit);
        toolItem(ToolMode::CylinderFit);
        ImGui::EndMenu();
    }

    if (ImGui::BeginMenu(u8"滤波")) {
        DrawFilterMenuItems();
        ImGui::EndMenu();
    }

    if (ImGui::BeginMenu(u8"显示")) {
        ImGui::TextDisabled(u8"点云外观");
        if (ImGui::SliderInt(u8"显示点数上限", &maxDisplayPoints_, 200000, 3000000, "%d")) {
            needUpload_ = true;
        }
        ImGui::SliderFloat(u8"点大小", &pointSize_, 1.f, 20.f, "%.1f");
        ImGui::SliderFloat(u8"透明度", &opacity_, 0.05f, 1.f, "%.2f");
        ImGui::Checkbox(u8"自动高度着色范围", &autoZRange_);
        if (!autoZRange_) {
            if (ImGui::DragFloat(u8"高度最小", &zMin_, 0.01f, 0.f, 0.f, "%.4f")) needUpload_ = true;
            if (ImGui::DragFloat(u8"高度最大", &zMax_, 0.01f, 0.f, 0.f, "%.4f")) needUpload_ = true;
        }
        if (!intensityColors_.empty() && intensityColors_.size() == cloud_.points.size()) {
            if (ImGui::Checkbox(u8"使用亮度着色", &useIntensityColors_)) needUpload_ = true;
        }
        if (ImGui::MenuItem(u8"重新上色")) needUpload_ = true;
        ImGui::Separator();
        if (ImGui::Checkbox(u8"显示坐标系 (mm)", &showAxes_)) {
            renderer_.SetAxes(showAxes_, axesLength_);
        }
        ImGui::EndMenu();
    }

    if (ImGui::BeginMenu(u8"窗口")) {
        if (ImGui::MenuItem(u8"算法编辑器", nullptr, algoEditor_.IsVisible())) {
            algoEditor_.ToggleVisible();
        }
        const bool canShowImage = depthImage_.valid() || brightnessImage_.valid();
        if (ImGui::MenuItem(u8"深度/亮度图窗口", nullptr, showImagePanel_, canShowImage)) {
            showImagePanel_ = !showImagePanel_;
        }
        ImGui::EndMenu();
    }

    if (ImGui::BeginMenu(u8"帮助")) {
        if (ImGui::MenuItem(u8"关于…")) {
            showAbout_ = true;
        }
        ImGui::EndMenu();
    }

    menuBottom = ImGui::GetWindowPos().y + ImGui::GetWindowSize().y;
    ImGui::EndMainMenuBar();
    return menuBottom;
}

void Application::DrawAboutPopup() {
    if (showAbout_) {
        ImGui::OpenPopup(u8"关于点云查看器");
        showAbout_ = false;
    }

    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(ImVec2(360.f, 0.f), ImGuiCond_Appearing);
    if (ImGui::BeginPopupModal(u8"关于点云查看器", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.40f, 0.85f, 0.90f, 1.f));
        ImGui::Text(u8"点云查看器");
        ImGui::PopStyleColor();
        ImGui::Spacing();
        ImGui::Text(u8"版本  %s", kAppVersion);
        ImGui::TextDisabled(u8"离线点云查看与测量");
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        ImGui::TextWrapped(
            u8"支持 PLY / PCD / XYZ / OBJ；可另开窗口查看深度图/亮度图（PNG/TIFF/BMP/JPEG）。"
            u8"含点选、测距、ROI、拟合、截面与台阶高度等工具。可保存 PLY/XYZ。");
        ImGui::Spacing();
        if (ImGui::Button(u8"确定", ImVec2(120.f, 0.f))) {
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
}

void Application::DrawCreatePopups() {
    if (showCreateSphere_) {
        ImGui::OpenPopup(u8"创建球面点云");
        showCreateSphere_ = false;
    }
    if (showCreateCylinder_) {
        ImGui::OpenPopup(u8"创建圆柱点云");
        showCreateCylinder_ = false;
    }

    ImGui::SetNextWindowSize(ImVec2(360.f, 0.f), ImGuiCond_Appearing);
    if (ImGui::BeginPopupModal(u8"创建球面点云", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::TextWrapped(u8"生成均匀分布的球面点云，可用于测试球面拟合。");
        ImGui::Spacing();
        ImGui::DragFloat(u8"半径", &genSphereRadius_, 0.1f, 0.01f, 1e6f, "%.3f");
        ImGui::DragInt(u8"点数", &genSpherePoints_, 100.f, 16, 2000000);
        ImGui::DragFloat(u8"径向噪声", &genSphereNoise_, 0.001f, 0.f, 100.f, "%.4f");
        ImGui::Spacing();
        if (ImGui::Button(u8"生成", ImVec2(120.f, 0))) {
            CreateSphereCloud();
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::Button(u8"取消", ImVec2(120.f, 0))) ImGui::CloseCurrentPopup();
        ImGui::EndPopup();
    }

    ImGui::SetNextWindowSize(ImVec2(360.f, 0.f), ImGuiCond_Appearing);
    if (ImGui::BeginPopupModal(u8"创建圆柱点云", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::TextWrapped(u8"生成沿 Z 轴的圆柱面点云，可用于测试圆柱拟合。");
        ImGui::Spacing();
        ImGui::DragFloat(u8"半径", &genCylRadius_, 0.1f, 0.01f, 1e6f, "%.3f");
        ImGui::DragFloat(u8"高度", &genCylHeight_, 0.1f, 0.01f, 1e6f, "%.3f");
        ImGui::DragInt(u8"点数", &genCylPoints_, 100.f, 32, 2000000);
        ImGui::DragFloat(u8"径向噪声", &genCylNoise_, 0.001f, 0.f, 100.f, "%.4f");
        ImGui::Spacing();
        if (ImGui::Button(u8"生成", ImVec2(120.f, 0))) {
            CreateCylinderCloud();
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::Button(u8"取消", ImVec2(120.f, 0))) ImGui::CloseCurrentPopup();
        ImGui::EndPopup();
    }
}

void Application::DrawImageWithSyncMarker(ImageView& view, const char* label) {
    if (!view.valid()) return;

    ImGui::TextDisabled("%s  %s  %dx%d", label, FileNameOf(view.path).c_str(), view.width,
                        view.height);
    if (!view.gray.empty()) {
        ImGui::SameLine();
        ImGui::TextDisabled(u8"  渲染 %.3f~%.3f", view.valueMin, view.valueMax);
    }
    ImGui::SameLine();
    ImGui::TextDisabled(u8"  缩放 %.0f%%", image2dZoom_ * 100.f);

    const ImVec2 avail = ImGui::GetContentRegionAvail();
    const float imgAspect =
        (view.height > 0) ? static_cast<float>(view.width) / static_cast<float>(view.height) : 1.f;
    float fitW = std::max(avail.x, 1.f);
    float fitH = fitW / imgAspect;
    if (fitH > avail.y && avail.y > 1.f) {
        fitH = avail.y;
        fitW = fitH * imgAspect;
    }
    const float drawW = std::max(fitW * image2dZoom_, 1.f);
    const float drawH = std::max(fitH * image2dZoom_, 1.f);

    const ImGuiID childId = ImGui::GetID(label);
    ImGui::BeginChild(childId, avail, ImGuiChildFlags_Borders,
                      ImGuiWindowFlags_HorizontalScrollbar);
    ImGuiIO& io = ImGui::GetIO();
    if (ImGui::IsWindowHovered() && io.KeyShift && io.MouseWheel != 0.f) {
        const float factor = 1.f + io.MouseWheel * 0.12f;
        image2dZoom_ = std::clamp(image2dZoom_ * factor, 0.1f, 32.f);
    }
    if (ImGui::IsWindowHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
        image2dZoom_ = 1.f;
    }

    const ImVec2 cursor = ImGui::GetCursorScreenPos();
    ImGui::Image((ImTextureID)(intptr_t)view.texId, ImVec2(drawW, drawH));
    const bool hovered = ImGui::IsItemHovered();
    const bool clicked = ImGui::IsItemClicked(ImGuiMouseButton_Left);

    if (imageSyncEnabled_ && syncHasPick_ && syncWidth_ == view.width &&
        syncHeight_ == view.height) {
        const float x = cursor.x + (static_cast<float>(syncCol_) + 0.5f) /
                                       static_cast<float>(view.width) * drawW;
        const float y = cursor.y + (static_cast<float>(syncRow_) + 0.5f) /
                                       static_cast<float>(view.height) * drawH;
        ImDrawList* dl = ImGui::GetWindowDrawList();
        const float arm = 14.f;
        dl->AddLine(ImVec2(x - arm, y), ImVec2(x + arm, y), IM_COL32(255, 230, 40, 255), 2.f);
        dl->AddLine(ImVec2(x, y - arm), ImVec2(x, y + arm), IM_COL32(255, 230, 40, 255), 2.f);
        dl->AddCircle(ImVec2(x, y), 7.f, IM_COL32(255, 80, 60, 255), 0, 2.2f);
        dl->AddCircleFilled(ImVec2(x, y), 2.5f, IM_COL32(255, 255, 255, 255));
    }

    if ((hovered || clicked) && view.width > 0 && view.height > 0) {
        const ImVec2 mp = ImGui::GetMousePos();
        const float u = (mp.x - cursor.x) / drawW;
        const float vv = (mp.y - cursor.y) / drawH;
        if (u >= 0.f && u < 1.f && vv >= 0.f && vv < 1.f) {
            const int col = std::clamp(static_cast<int>(u * view.width), 0, view.width - 1);
            const int row = std::clamp(static_cast<int>(vv * view.height), 0, view.height - 1);
            if (clicked && imageSyncEnabled_) {
                SetImageSyncPixel(col, row);
            }
            if (hovered) {
                ImGui::BeginTooltip();
                ImGui::Text(u8"像素  (%d, %d)", col, row);
                if (!view.gray.empty()) {
                    const std::size_t idx =
                        static_cast<std::size_t>(row) * static_cast<std::size_t>(view.width) +
                        static_cast<std::size_t>(col);
                    if (idx < view.gray.size()) ImGui::Text(u8"深度值  %.4f", view.gray[idx]);
                } else if (!view.rgb.empty()) {
                    const std::size_t bi =
                        (static_cast<std::size_t>(row) * static_cast<std::size_t>(view.width) +
                         static_cast<std::size_t>(col)) *
                        3u;
                    if (bi + 2 < view.rgb.size()) {
                        ImGui::Text(u8"RGB  %d, %d, %d", view.rgb[bi], view.rgb[bi + 1],
                                    view.rgb[bi + 2]);
                    }
                }
                if (imageSyncEnabled_) ImGui::TextDisabled(u8"单击同步到另一张图");
                ImGui::TextDisabled(u8"Shift+滚轮缩放，双击复位");
                ImGui::EndTooltip();
            }
        }
    }
    ImGui::EndChild();
}

void Application::DrawDepthRenderControls() {
    if (!depthImage_.valid() || depthImage_.gray.empty()) return;

    ImGui::Separator();
    ImGui::TextDisabled(u8"深度伪彩渲染范围");
    ImGui::TextDisabled(u8"数据 %.4f ~ %.4f（收窄范围可放大高度差）", depthDataMin_, depthDataMax_);

    bool changed = false;
    const float span = std::max(depthDataMax_ - depthDataMin_, 1e-6f);
    const float step = span * 0.001f;
    ImGui::SetNextItemWidth(-1);
    changed |= ImGui::DragFloat(u8"渲染最小", &depthDisplayMin_, step, depthDataMin_,
                                depthDisplayMax_ - 1e-6f, "%.6f");
    ImGui::SetNextItemWidth(-1);
    changed |= ImGui::DragFloat(u8"渲染最大", &depthDisplayMax_, step, depthDisplayMin_ + 1e-6f,
                                depthDataMax_, "%.6f");

    if (ImGui::Button(u8"复位全范围", ImVec2(-1.f, 0))) {
        depthDisplayMin_ = depthDataMin_;
        depthDisplayMax_ = depthDataMax_;
        changed = true;
    }
    // 一键收窄到中间 50%，让微小高度差更醒目
    if (ImGui::Button(u8"增强对比（中间 50%）", ImVec2(-1.f, 0))) {
        const float mid = 0.5f * (depthDataMin_ + depthDataMax_);
        const float half = 0.25f * span;
        depthDisplayMin_ = mid - half;
        depthDisplayMax_ = mid + half;
        changed = true;
    }
    if (ImGui::Checkbox(u8"忽略 0 值参与统计", &depthSkipZero_)) {
        ComputeGrayRange(depthImage_.gray, depthSkipZero_, depthDataMin_, depthDataMax_);
        depthDisplayMin_ = std::clamp(depthDisplayMin_, depthDataMin_, depthDataMax_);
        depthDisplayMax_ = std::clamp(depthDisplayMax_, depthDataMin_, depthDataMax_);
        if (!(depthDisplayMax_ > depthDisplayMin_)) {
            depthDisplayMin_ = depthDataMin_;
            depthDisplayMax_ = depthDataMax_;
        }
        changed = true;
    }
    if (changed) RebuildDepthDisplay();
}

void Application::DrawImagePanel() {
    if (!HasImagePanel()) return;

    const ImGuiViewport* vp = ImGui::GetMainViewport();
    const float maxW = std::max(vp->Size.x - 600.f, 240.f);
    imagePanelPreferredW_ = std::clamp(imagePanelPreferredW_, 240.f, maxW);
    const float imageW = imagePanelPreferredW_;
    const float panelX = vp->Pos.x + vp->Size.x - imageW;
    const float panelY = view3dY_;
    const float panelH = view3dH_;

    // 左边缘分割条：吸附右端，仅左右拖拽调宽
    constexpr float kSplitHit = 8.f;
    ImGui::SetNextWindowPos(ImVec2(panelX - kSplitHit * 0.5f, panelY));
    ImGui::SetNextWindowSize(ImVec2(kSplitHit, panelH));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.f, 0.f));
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.f, 0.f, 0.f, 0.f));
    ImGui::Begin(u8"##2d图像分割条", nullptr,
                 ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                     ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoSavedSettings |
                     ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoBackground |
                     ImGuiWindowFlags_NoBringToFrontOnFocus);
    ImGui::InvisibleButton(u8"##split", ImVec2(kSplitHit, panelH));
    const bool splitHover = ImGui::IsItemHovered();
    const bool splitDrag = ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Left);
    if (splitDrag) {
        const float mx = ImGui::GetIO().MousePos.x;
        imagePanelPreferredW_ = std::clamp(vp->Pos.x + vp->Size.x - mx, 240.f, maxW);
    }
    if (splitHover || splitDrag) {
        ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
    }
    ImGui::End();
    ImGui::PopStyleColor();
    ImGui::PopStyleVar();

    // 与左侧栏相同：贴边停靠，不可拖移，高度铺满内容区
    ImGui::SetNextWindowPos(ImVec2(panelX, panelY));
    ImGui::SetNextWindowSize(ImVec2(imageW, panelH));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(10.f, 8.f));
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.08f, 0.09f, 0.10f, 1.f));

    ImGui::Begin(u8"##2D图像停靠栏", nullptr,
                 ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                     ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoBringToFrontOnFocus |
                     ImGuiWindowFlags_NoScrollbar);

    // 顶栏：标题 + 关闭（替代系统标题栏，保持停靠外观）
    {
        ImGui::TextColored(ImVec4(0.45f, 0.85f, 0.90f, 1.f), u8"2D 图像");
        ImGui::SameLine(ImGui::GetContentRegionAvail().x - 28.f);
        if (ImGui::SmallButton(u8"×")) {
            showImagePanel_ = false;
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip(u8"关闭面板（可在「窗口」菜单再打开）");
    }

    // 左侧分割高亮线
    {
        ImDrawList* dl = ImGui::GetWindowDrawList();
        const ImU32 col = (splitHover || splitDrag) ? IM_COL32(90, 180, 200, 220)
                                                   : IM_COL32(50, 70, 80, 200);
        dl->AddLine(ImVec2(panelX, panelY), ImVec2(panelX, panelY + panelH), col,
                    (splitHover || splitDrag) ? 2.5f : 1.5f);
    }

    const bool canSync = depthImage_.valid() && brightnessImage_.valid();
    if (!canSync) ImGui::BeginDisabled();
    if (ImGui::Button(imageSyncEnabled_ ? u8"关闭深度/亮度联动" : u8"启用深度/亮度联动",
                      ImVec2(-1.f, 0))) {
        if (imageSyncEnabled_) {
            imageSyncEnabled_ = false;
            ClearImageSyncPick();
            measure_.status = u8"已关闭深度/亮度联动";
        } else {
            TryEnableImageSync();
        }
    }
    if (!canSync) ImGui::EndDisabled();
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
        ImGui::SetTooltip(u8"需先打开同尺寸的深度图与亮度图；在一图上点选，另一图同步十字线");
    }

    if (imageSyncEnabled_) {
        ImGui::TextColored(ImVec4(0.45f, 0.90f, 0.70f, 1.f), u8"网格 %d × %d", syncWidth_,
                           syncHeight_);
        if (syncHasPick_) {
            ImGui::Text(u8"选中像素 (%d, %d)", syncCol_, syncRow_);
            if (ImGui::SmallButton(u8"清除选点")) ClearImageSyncPick();
        } else {
            ImGui::TextDisabled(u8"在深度图或亮度图上单击即可联动");
        }
    }
    ImGui::Separator();

    const bool both = depthImage_.valid() && brightnessImage_.valid();
    if (both && imageSyncEnabled_) {
        const float halfH = std::max((ImGui::GetContentRegionAvail().y - 8.f) * 0.5f, 80.f);
        ImGui::BeginChild(u8"##depthImgPane", ImVec2(0.f, halfH), ImGuiChildFlags_None);
        DrawImageWithSyncMarker(depthImage_, u8"深度图");
        ImGui::EndChild();
        ImGui::Spacing();
        ImGui::BeginChild(u8"##brightImgPane", ImVec2(0.f, 0.f), ImGuiChildFlags_None);
        DrawImageWithSyncMarker(brightnessImage_, u8"亮度图");
        ImGui::EndChild();
        DrawDepthRenderControls();
    } else {
        if (ImGui::BeginTabBar(u8"##imageTabs")) {
            if (depthImage_.valid() && ImGui::BeginTabItem(u8"深度图")) {
                imagePanelTab_ = 0;
                ImGui::EndTabItem();
            }
            if (brightnessImage_.valid() && ImGui::BeginTabItem(u8"亮度图")) {
                imagePanelTab_ = 1;
                ImGui::EndTabItem();
            }
            ImGui::EndTabBar();
        }
        if (imagePanelTab_ == 0 && !depthImage_.valid() && brightnessImage_.valid())
            imagePanelTab_ = 1;
        if (imagePanelTab_ == 1 && !brightnessImage_.valid() && depthImage_.valid())
            imagePanelTab_ = 0;

        ImageView* view = (imagePanelTab_ == 1) ? &brightnessImage_ : &depthImage_;
        if (view->valid()) {
            DrawImageWithSyncMarker(*view, imagePanelTab_ == 0 ? u8"深度图" : u8"亮度图");
            if (imagePanelTab_ == 0) DrawDepthRenderControls();
        } else {
            ImGui::TextDisabled(u8"当前无图像");
        }
    }

    ImGui::End();
    ImGui::PopStyleColor();
    ImGui::PopStyleVar(3);
}

void Application::DrawToolbar(float y, float height) {
    const ImGuiViewport* vp = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(ImVec2(vp->Pos.x, y));
    ImGui::SetNextWindowSize(ImVec2(vp->Size.x, height));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(10.f, 6.f));
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.10f, 0.12f, 0.14f, 1.f));
    ImGui::Begin(u8"##工具栏", nullptr,
                 ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                     ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoCollapse |
                     ImGuiWindowFlags_NoBringToFrontOnFocus);

    const float btnH = height - 14.f;
    const bool canUndo = history_.CanUndo();
    const bool canRedo = history_.CanRedo();
    if (!canUndo) ImGui::BeginDisabled();
    if (ImGui::Button(u8"撤销", ImVec2(0.f, btnH))) Undo();
    if (!canUndo) ImGui::EndDisabled();
    ImGui::SameLine();
    if (!canRedo) ImGui::BeginDisabled();
    if (ImGui::Button(u8"重做", ImVec2(0.f, btnH))) Redo();
    if (!canRedo) ImGui::EndDisabled();

    ImGui::SameLine();
    ImGui::TextDisabled("|");
    ImGui::SameLine();
    ImGui::TextDisabled(u8"当前工具");
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(0.45f, 0.85f, 0.90f, 1.f), "%s", ToolModeLabel(measure_.mode));
    ImGui::SameLine();
    ImGui::TextDisabled(u8"（菜单「工具」中切换）");
    ImGui::SameLine(0.f, 16.f);
    ImGui::TextDisabled("|");
    ImGui::SameLine();
    if (ImGui::Button(u8"清空显示", ImVec2(0.f, btnH))) {
        ClearToolVisuals(true);
    }
    ImGui::SameLine(0.f, 16.f);
    ImGui::TextDisabled("|");
    ImGui::SameLine();
    const bool canSync = depthImage_.valid() && brightnessImage_.valid();
    if (!canSync) ImGui::BeginDisabled();
    if (ImGui::Button(imageSyncEnabled_ ? u8"关闭联动" : u8"深度亮度联动", ImVec2(0.f, btnH))) {
        if (imageSyncEnabled_) {
            imageSyncEnabled_ = false;
            ClearImageSyncPick();
            measure_.status = u8"已关闭深度/亮度联动";
        } else {
            TryEnableImageSync();
        }
    }
    if (!canSync) ImGui::EndDisabled();
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
        ImGui::SetTooltip(u8"深度图与亮度图同尺寸时，点选一图同步十字线到另一图");
    }

    if (filterCompareActive_) {
        ImGui::SameLine(0.f, 24.f);
        ImGui::TextDisabled("|");
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(0.95f, 0.55f, 0.35f, 1.f), u8"滤波对比中");
        ImGui::SameLine();
        if (ImGui::Button(u8"应用滤波", ImVec2(0.f, btnH))) ApplyFilterResult();
        ImGui::SameLine();
        if (ImGui::Button(u8"取消预览", ImVec2(0.f, btnH))) ClearFilterCompare();
    }

    ImGui::End();
    ImGui::PopStyleColor();
    ImGui::PopStyleVar(3);
}

void Application::DrawFilterMenuItems() {
    ImGui::TextDisabled(u8"体素下采样");
    ImGui::SetNextItemWidth(160.f);
    ImGui::DragFloat(u8"体素边长##v", &filterVoxelLeaf_, 0.01f, 1e-4f, 1e6f, "%.4f");
    if (ImGui::MenuItem(u8"预览体素滤波")) RunFilterPreview(0);
    ImGui::Separator();
    ImGui::TextDisabled(u8"半径离群点");
    ImGui::SetNextItemWidth(160.f);
    ImGui::DragFloat(u8"搜索半径##r", &filterRadius_, 0.01f, 1e-4f, 1e6f, "%.4f");
    ImGui::SetNextItemWidth(160.f);
    ImGui::DragInt(u8"最少邻居##r", &filterRadiusMinNeighbors_, 1, 1, 200);
    if (ImGui::MenuItem(u8"预览半径滤波")) RunFilterPreview(1);
    ImGui::Separator();
    ImGui::TextDisabled(u8"统计离群点");
    ImGui::SetNextItemWidth(160.f);
    ImGui::DragInt(u8"邻域点数 K##s", &filterStatMeanK_, 1, 2, 200);
    ImGui::SetNextItemWidth(160.f);
    ImGui::DragFloat(u8"标准差倍数##s", &filterStatStdMul_, 0.05f, 0.1f, 10.f, "%.2f");
    if (ImGui::MenuItem(u8"预览统计滤波")) RunFilterPreview(2);
    ImGui::Separator();
    if (filterCompareActive_) {
        if (ImGui::Checkbox(u8"仅显示滤波后（隐藏红色点）", &filterHideRemoved_)) {
            needUpload_ = true;
        }
        ImGui::Text(u8"保留 %d / 滤除 %d", filterLastKept_, filterLastRemoved_);
        if (ImGui::MenuItem(u8"应用滤波到点云")) ApplyFilterResult();
        if (ImGui::MenuItem(u8"取消预览")) ClearFilterCompare();
    } else {
        ImGui::TextDisabled(u8"先预览：青绿=保留，红=滤除");
    }
}

void Application::DrawViewAxisWidget(float contentTop, float contentBottom, float leftInset) {
    const float size = 118.f * 1.3f;
    const float margin = 12.f;
    const float x = view3dX_ + view3dW_ - size - margin;
    const float y = contentTop + margin;
    (void)contentBottom;
    (void)leftInset;

    ImGui::SetNextWindowPos(ImVec2(x, y));
    ImGui::SetNextWindowSize(ImVec2(size, size));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.f, 0.f));
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.f, 0.f, 0.f, 0.f));
    ImGui::Begin(u8"##视角坐标轴", nullptr,
                 ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                     ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoCollapse |
                     ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoBackground);

    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImVec2 origin(x + size * 0.42f, y + size * 0.58f);
    const float axisLen = 36.f * 1.3f;

    // Isometric-ish screen directions for a readable triad
    const ImVec2 dirX(axisLen * 0.95f, axisLen * 0.18f);    // +X → 侧视 X
    const ImVec2 dirY(-axisLen * 0.15f, -axisLen * 0.95f);  // +Y → 俯视
    const ImVec2 dirZ(-axisLen * 0.85f, axisLen * 0.35f);   // +Z → 侧视 Z
    const ImVec2 dirNegY(axisLen * 0.12f, axisLen * 0.55f); // −Y → 沿运动

    auto tip = [](ImVec2 o, ImVec2 d) { return ImVec2(o.x + d.x, o.y + d.y); };
    auto dist2 = [](ImVec2 a, ImVec2 b) {
        const float dx = a.x - b.x;
        const float dy = a.y - b.y;
        return dx * dx + dy * dy;
    };

    struct HitTarget {
        ImVec2 p;
        float r;
        int preset;
        const char* tip;
        ImU32 col;
        const char* label;
    };

    const float s = 1.3f;
    const HitTarget targets[] = {
        {tip(origin, dirX), 14.f * s, 1, u8"侧视 X（沿 +X 看）", IM_COL32(220, 80, 70, 255), "X"},
        {tip(origin, dirY), 14.f * s, 0, u8"俯视（沿 +Y 看）", IM_COL32(90, 200, 110, 255), "Y"},
        {tip(origin, dirZ), 14.f * s, 2, u8"侧视 Z（沿 +Z 看）", IM_COL32(70, 140, 230, 255), "Z"},
        {tip(origin, dirNegY), 12.f * s, 3, u8"沿运动方向 Y", IM_COL32(160, 210, 120, 255), "-Y"},
        {origin, 13.f * s, 4, u8"复位包围盒", IM_COL32(200, 210, 220, 255), ""},
    };

    // Axes lines (drawn under hit circles)
    dl->AddLine(origin, tip(origin, dirX), IM_COL32(220, 80, 70, 220), 2.4f * s);
    dl->AddLine(origin, tip(origin, dirY), IM_COL32(90, 200, 110, 220), 2.4f * s);
    dl->AddLine(origin, tip(origin, dirZ), IM_COL32(70, 140, 230, 220), 2.4f * s);
    dl->AddLine(origin, tip(origin, dirNegY), IM_COL32(120, 160, 100, 160), 1.6f * s);

    const ImVec2 mouse = ImGui::GetMousePos();
    int hover = -1;
    for (int i = 0; i < 5; ++i) {
        if (dist2(mouse, targets[i].p) <= targets[i].r * targets[i].r) hover = i;
    }

    for (int i = 0; i < 5; ++i) {
        const HitTarget& t = targets[i];
        const bool hot = (hover == i);
        const float r = hot ? t.r + 2.f : t.r;
        dl->AddCircleFilled(t.p, r, t.col);
        if (hot) dl->AddCircle(t.p, r + 1.5f, IM_COL32(255, 255, 255, 180), 0, 1.5f);
        if (t.label[0] != '\0') {
            const ImVec2 ts = ImGui::CalcTextSize(t.label);
            dl->AddText(ImVec2(t.p.x - ts.x * 0.5f, t.p.y - ts.y * 0.5f),
                        IM_COL32(20, 24, 28, 255), t.label);
        }
    }

    ImGui::InvisibleButton(u8"##viewaxis", ImVec2(size, size));
    if (hover >= 0) {
        ImGui::SetTooltip("%s", targets[hover].tip);
        ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
        if (ImGui::IsItemClicked(ImGuiMouseButton_Left)) {
            ApplyViewPreset(targets[hover].preset);
        }
    }

    ImGui::End();
    ImGui::PopStyleColor();
    ImGui::PopStyleVar(3);
}

void Application::DrawToolPanel() {
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.40f, 0.85f, 0.90f, 1.f));
    ImGui::TextUnformatted(ToolModeLabel(measure_.mode));
    ImGui::PopStyleColor();
    ImGui::TextDisabled(u8"当前工具参数与说明");
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.11f, 0.13f, 0.15f, 1.f));
    ImGui::BeginChild(u8"##toolhelp", ImVec2(0, 0), true);

    switch (measure_.mode) {
        case ToolMode::Navigate:
            ImGui::TextWrapped(
                u8"左键拖拽: 旋转\n"
                u8"中键拖拽: 平移（按住拖动）\n"
                u8"Shift+左键: 平移\n"
                u8"滚轮: 缩放 | 右键: 旋转\n"
                u8"双击点: 设为旋转中心");
            break;
        case ToolMode::Pick:
            ImGui::TextWrapped(u8"在点附近单击，读取原始世界坐标 XYZ。");
            if (measure_.picked) {
                const Vec3 w = cloud_.ToWorld(*measure_.picked);
                ImGui::Spacing();
                ImGui::Text(u8"X = %.6f", w.x);
                ImGui::Text(u8"Y = %.6f", w.y);
                ImGui::Text(u8"Z = %.6f", w.z);
            }
            break;
        case ToolMode::Distance:
            ImGui::TextWrapped(u8"依次点击两个点。黄色=点1，橙红=点2，并画出测距线段。");
            if (measure_.distA && measure_.distB) {
                ImGui::Spacing();
                ImGui::Text(u8"距离 = %.6f", measure_.distance);
            }
            if (ImGui::Button(u8"清除测距标记", ImVec2(-1, 0))) {
                measure_.distA.reset();
                measure_.distB.reset();
                measure_.distance = 0.f;
                measure_.status = u8"已清除测距标记";
                UpdateOverlays();
            }
            break;
        case ToolMode::PlaneFit:
            ImGui::TextWrapped(
                u8"① 左键拖拽框选可见表面（被挡住的点不选）\n"
                u8"② 点击下方按钮对框选点拟合\n"
                u8"③ 橙色半透明面为拟合结果");
            ImGui::Text(u8"当前框选: %zu 点", measure_.roiIndices.size());
            ImGui::Spacing();
            if (ImGui::Button(u8"对框选区域拟合平面", ImVec2(-1, 32.f))) {
                if (measure_.roiIndices.empty()) {
                    measure_.status = u8"请先框选区域，或改用“对全部可见点拟合”";
                } else {
                    std::string error;
                    PlaneModel plane;
                    if (MeasureTools::FitPlaneSVD(cloud_, measure_.roiIndices, plane, error)) {
                        measure_.plane = plane;
                        UpdateOverlays();
                        measure_.status = u8"框选拟合完成";
                    } else {
                        measure_.status = error;
                    }
                }
            }
            if (ImGui::Button(u8"对全部可见点拟合", ImVec2(-1, 0))) {
                std::string error;
                PlaneModel plane;
                std::vector<std::size_t> empty;
                if (MeasureTools::FitPlaneSVD(cloud_, empty, plane, error)) {
                    measure_.plane = plane;
                    UpdateOverlays();
                    measure_.status = u8"全点云拟合完成";
                } else {
                    measure_.status = error;
                }
            }
            if (measure_.plane && ImGui::Button(u8"清除拟合平面显示", ImVec2(-1, 0))) {
                measure_.plane.reset();
                UpdateOverlays();
            }
            break;
        case ToolMode::SphereFit:
            ImGui::TextWrapped(
                u8"① 左键拖拽框选可见表面\n"
                u8"② 点击下方按钮拟合球\n"
                u8"③ 橙色线框为拟合球面");
            ImGui::Text(u8"当前框选: %zu 点", measure_.roiIndices.size());
            ImGui::Spacing();
            if (ImGui::Button(u8"对框选区域拟合球", ImVec2(-1, 32.f))) {
                if (measure_.roiIndices.empty()) {
                    measure_.status = u8"请先框选区域，或改用“对全部可见点拟合”";
                } else {
                    std::string error;
                    SphereModel sphere;
                    if (MeasureTools::FitSphere(cloud_, measure_.roiIndices, sphere, error)) {
                        measure_.sphere = sphere;
                        measure_.circle.reset();
                        measure_.cylinder.reset();
                        UpdateOverlays();
                        char buf[160];
                        std::snprintf(buf, sizeof(buf),
                                      u8"球面拟合完成 R=%.6f RMS=%.6f（%d 点）", sphere.radius,
                                      sphere.rms, sphere.pointCount);
                        measure_.status = buf;
                    } else {
                        measure_.status = error;
                    }
                }
            }
            if (ImGui::Button(u8"对全部可见点拟合", ImVec2(-1, 0))) {
                std::string error;
                SphereModel sphere;
                std::vector<std::size_t> empty;
                if (MeasureTools::FitSphere(cloud_, empty, sphere, error)) {
                    measure_.sphere = sphere;
                    measure_.circle.reset();
                    measure_.cylinder.reset();
                    UpdateOverlays();
                    char buf[160];
                    std::snprintf(buf, sizeof(buf), u8"全点云球面拟合 R=%.6f RMS=%.6f",
                                  sphere.radius, sphere.rms);
                    measure_.status = buf;
                } else {
                    measure_.status = error;
                }
            }
            if (measure_.sphere) {
                ImGui::Spacing();
                ImGui::Text(u8"中心 = (%.4f, %.4f, %.4f)", measure_.sphere->center.x,
                            measure_.sphere->center.y, measure_.sphere->center.z);
                ImGui::Text(u8"半径 R = %.6f", measure_.sphere->radius);
                ImGui::Text(u8"RMS = %.6f", measure_.sphere->rms);
                ImGui::Text(u8"点数 = %d", measure_.sphere->pointCount);
            }
            if (measure_.sphere && ImGui::Button(u8"清除拟合显示", ImVec2(-1, 0))) {
                measure_.sphere.reset();
                UpdateOverlays();
            }
            break;
        case ToolMode::CircleFit:
            ImGui::TextWrapped(
                u8"① 左键拖拽框选可见表面\n"
                u8"② 先拟合支撑平面，再在平面内拟合圆\n"
                u8"③ 青色线框为拟合圆，黄线为法向");
            ImGui::Text(u8"当前框选: %zu 点", measure_.roiIndices.size());
            ImGui::Spacing();
            if (ImGui::Button(u8"对框选区域拟合圆", ImVec2(-1, 32.f))) {
                if (measure_.roiIndices.empty()) {
                    measure_.status = u8"请先框选区域，或改用“对全部可见点拟合”";
                } else {
                    std::string error;
                    CircleModel circle;
                    if (MeasureTools::FitCircle3D(cloud_, measure_.roiIndices, circle, error)) {
                        measure_.circle = circle;
                        measure_.sphere.reset();
                        measure_.cylinder.reset();
                        UpdateOverlays();
                        char buf[160];
                        std::snprintf(buf, sizeof(buf),
                                      u8"圆拟合完成 R=%.6f RMS=%.6f（%d 点）", circle.radius,
                                      circle.rms, circle.pointCount);
                        measure_.status = buf;
                    } else {
                        measure_.status = error;
                    }
                }
            }
            if (ImGui::Button(u8"对全部可见点拟合", ImVec2(-1, 0))) {
                std::string error;
                CircleModel circle;
                std::vector<std::size_t> empty;
                if (MeasureTools::FitCircle3D(cloud_, empty, circle, error)) {
                    measure_.circle = circle;
                    measure_.sphere.reset();
                    measure_.cylinder.reset();
                    UpdateOverlays();
                    char buf[160];
                    std::snprintf(buf, sizeof(buf), u8"全点云圆拟合 R=%.6f RMS=%.6f",
                                  circle.radius, circle.rms);
                    measure_.status = buf;
                } else {
                    measure_.status = error;
                }
            }
            if (measure_.circle) {
                ImGui::Spacing();
                ImGui::Text(u8"圆心 = (%.4f, %.4f, %.4f)", measure_.circle->center.x,
                            measure_.circle->center.y, measure_.circle->center.z);
                ImGui::Text(u8"法向 = (%.4f, %.4f, %.4f)", measure_.circle->normal.x,
                            measure_.circle->normal.y, measure_.circle->normal.z);
                ImGui::Text(u8"半径 R = %.6f", measure_.circle->radius);
                ImGui::Text(u8"RMS = %.6f", measure_.circle->rms);
                ImGui::Text(u8"点数 = %d", measure_.circle->pointCount);
            }
            if (measure_.circle && ImGui::Button(u8"清除拟合显示", ImVec2(-1, 0))) {
                measure_.circle.reset();
                UpdateOverlays();
            }
            break;
        case ToolMode::CylinderFit:
            ImGui::TextWrapped(
                u8"① 左键拖拽框选可见表面\n"
                u8"② PCA 候选轴 + 垂面圆拟合\n"
                u8"③ 品红线框为圆柱，黄线为轴线");
            ImGui::Text(u8"当前框选: %zu 点", measure_.roiIndices.size());
            ImGui::Spacing();
            if (ImGui::Button(u8"对框选区域拟合圆柱", ImVec2(-1, 32.f))) {
                if (measure_.roiIndices.empty()) {
                    measure_.status = u8"请先框选区域，或改用“对全部可见点拟合”";
                } else {
                    std::string error;
                    CylinderModel cyl;
                    if (MeasureTools::FitCylinder(cloud_, measure_.roiIndices, cyl, error)) {
                        measure_.cylinder = cyl;
                        measure_.sphere.reset();
                        measure_.circle.reset();
                        UpdateOverlays();
                        char buf[160];
                        std::snprintf(buf, sizeof(buf),
                                      u8"圆柱拟合完成 R=%.6f RMS=%.6f（%d 点）", cyl.radius,
                                      cyl.rms, cyl.pointCount);
                        measure_.status = buf;
                    } else {
                        measure_.status = error;
                    }
                }
            }
            if (ImGui::Button(u8"对全部可见点拟合", ImVec2(-1, 0))) {
                std::string error;
                CylinderModel cyl;
                std::vector<std::size_t> empty;
                if (MeasureTools::FitCylinder(cloud_, empty, cyl, error)) {
                    measure_.cylinder = cyl;
                    measure_.sphere.reset();
                    measure_.circle.reset();
                    UpdateOverlays();
                    char buf[160];
                    std::snprintf(buf, sizeof(buf), u8"全点云圆柱拟合 R=%.6f RMS=%.6f",
                                  cyl.radius, cyl.rms);
                    measure_.status = buf;
                } else {
                    measure_.status = error;
                }
            }
            if (measure_.cylinder) {
                ImGui::Spacing();
                ImGui::Text(u8"轴点 = (%.4f, %.4f, %.4f)", measure_.cylinder->axisPoint.x,
                            measure_.cylinder->axisPoint.y, measure_.cylinder->axisPoint.z);
                ImGui::Text(u8"轴向 = (%.4f, %.4f, %.4f)", measure_.cylinder->axisDir.x,
                            measure_.cylinder->axisDir.y, measure_.cylinder->axisDir.z);
                ImGui::Text(u8"半径 R = %.6f", measure_.cylinder->radius);
                ImGui::Text(u8"半高 = %.6f", measure_.cylinder->halfHeight);
                ImGui::Text(u8"RMS = %.6f", measure_.cylinder->rms);
                ImGui::Text(u8"点数 = %d", measure_.cylinder->pointCount);
            }
            if (measure_.cylinder && ImGui::Button(u8"清除拟合显示", ImVec2(-1, 0))) {
                measure_.cylinder.reset();
                UpdateOverlays();
            }
            break;
        case ToolMode::Flatness: {
            ImGui::TextWrapped(
                u8"① 左键拖拽框选可见表面\n"
                u8"② 点击计算平面度\n"
                u8"③ 颜色为相对拟合面的偏差图（蓝负/红正）\n"
                u8"平面度 PV = 最大偏差 − 最小偏差");
            ImGui::Text(u8"当前框选: %zu 点", measure_.roiIndices.size());
            ImGui::Spacing();
            if (ImGui::Button(u8"计算平面度", ImVec2(-1, 36.f))) {
                std::string error;
                FlatnessResult fr;
                if (MeasureTools::ComputeFlatness(cloud_, measure_.roiIndices, fr, error)) {
                    measure_.flatness = std::move(fr);
                    measure_.plane = measure_.flatness.plane;
                    needUpload_ = true;
                    UpdateOverlays();
                    char buf[192];
                    std::snprintf(buf, sizeof(buf),
                                  u8"平面度 PV=%.6f mm，RMS=%.6f（%d 点）",
                                  measure_.flatness.peakToValley, measure_.flatness.rms,
                                  measure_.flatness.plane.pointCount);
                    measure_.status = buf;
                } else {
                    measure_.status = error;
                }
            }
            if (measure_.flatness.valid) {
                ImGui::Spacing();
                ImGui::Text(u8"平面度 PV = %.6f", measure_.flatness.peakToValley);
                ImGui::Text(u8"最小偏差 = %.6f", measure_.flatness.minDev);
                ImGui::Text(u8"最大偏差 = %.6f", measure_.flatness.maxDev);
                ImGui::Text(u8"平均|偏差| = %.6f", measure_.flatness.meanAbs);
                ImGui::Text(u8"RMS = %.6f", measure_.flatness.rms);
                ImGui::Text(u8"拟合点数 = %d", measure_.flatness.plane.pointCount);
            }
            if (ImGui::Button(u8"清除结果", ImVec2(-1, 0))) {
                measure_.flatness = {};
                measure_.plane.reset();
                needUpload_ = true;
                UpdateOverlays();
            }
            break;
        }
        case ToolMode::StepGap: {
            auto& sg = measure_.stepGap;
            ImGui::TextWrapped(
                u8"① 框选基准区域 A（仅可见表面）→ 拟合平面\n"
                u8"② 框选测量区域 B（仅可见表面）→ 计算段差\n"
                u8"③ 段差 ΔZ = B.z − mean(A.z)（Z 轴高度差）\n"
                u8"黄=A，蓝/红=B 相对基准高度");
            ImGui::Spacing();
            const char* phaseName = u8"框选区域 A";
            if (sg.phase == StepGapPhase::SelectB ||
                (sg.hasPlane && !sg.hasDistances && sg.phase != StepGapPhase::SelectA))
                phaseName = u8"框选区域 B";
            if (sg.hasDistances) phaseName = u8"已完成";
            ImGui::TextColored(ImVec4(0.45f, 0.85f, 0.90f, 1.f), u8"步骤：%s", phaseName);
            ImGui::Text(u8"区域 A：%zu 点", sg.regionA.size());
            ImGui::Text(u8"区域 B：%zu 点", sg.regionB.size());
            ImGui::Spacing();

            if (ImGui::Button(u8"对区域 A 拟合平面", ImVec2(-1, 32.f))) {
                if (sg.regionA.empty()) {
                    measure_.status = u8"请先框选区域 A";
                } else {
                    std::string error;
                    PlaneModel plane;
                    if (MeasureTools::FitPlaneSVD(cloud_, sg.regionA, plane, error)) {
                        sg.planeA = plane;
                        sg.hasPlane = true;
                        sg.hasDistances = false;
                        sg.phase = StepGapPhase::SelectB;
                        measure_.plane = plane;
                        UpdateOverlays();
                        measure_.status = u8"区域 A 平面已拟合，请框选区域 B";
                        needUpload_ = true;
                    } else {
                        measure_.status = error;
                    }
                }
            }
            if (ImGui::Button(u8"计算段差 ΔZ", ImVec2(-1, 32.f))) {
                if (sg.regionA.empty()) {
                    measure_.status = u8"请先框选区域 A";
                } else if (sg.regionB.empty()) {
                    measure_.status = u8"请先框选区域 B";
                } else {
                    std::string error;
                    if (MeasureTools::ComputeStepGapZHeight(cloud_, sg.regionA, sg.regionB, sg,
                                                            error)) {
                        needUpload_ = true;
                        UpdateOverlays();
                        char buf[192];
                        std::snprintf(buf, sizeof(buf),
                                      u8"段差 ΔZ=%.6f（中位 %.6f，B=%zu 点）", sg.mean, sg.median,
                                      sg.regionB.size());
                        measure_.status = buf;
                    } else {
                        measure_.status = error;
                    }
                }
            }
            if (sg.hasDistances) {
                ImGui::Spacing();
                ImGui::Separator();
                ImGui::Text(u8"段差 ΔZ = %.6f", sg.mean);
                ImGui::TextDisabled(u8"A 平均 Z = %.6f", sg.zRefA);
                ImGui::Text(u8"中位数 = %.6f", sg.median);
                ImGui::Text(u8"平均|ΔZ| = %.6f", sg.meanAbs);
                ImGui::Text(u8"最小/最大 = %.6f / %.6f", sg.minDist, sg.maxDist);
                ImGui::Text(u8"RMS = %.6f", sg.rms);
            }
            if (ImGui::Button(u8"重新开始段差", ImVec2(-1, 0))) {
                sg = {};
                sg.phase = StepGapPhase::SelectA;
                measure_.roiIndices.clear();
                measure_.plane.reset();
                needUpload_ = true;
                UpdateOverlays();
                measure_.status = u8"段差：先框选基准区域 A";
            }
            break;
        }
        case ToolMode::Roi:
            ImGui::TextWrapped(u8"左键拖拽框选。框选后点绿色高亮，再选择操作：");
            ImGui::Text(u8"已选 %zu 点", measure_.roiIndices.size());
            ImGui::Spacing();
            if (ImGui::Button(u8"清除框选内的点", ImVec2(-1, 32.f))) {
                if (measure_.roiIndices.empty()) {
                    measure_.status = u8"请先框选区域";
                } else {
                    PushHistory(u8"清除框内点");
                    MeasureTools::ApplyRoiDelete(cloud_, measure_.roiIndices, true);
                    measure_.status = std::string(u8"已清除框内点，可见 ") +
                                      std::to_string(cloud_.VisibleCount());
                    measure_.roiIndices.clear();
                    needUpload_ = true;
                }
            }
            if (ImGui::Button(u8"清除框选外的点（只留框内）", ImVec2(-1, 32.f))) {
                if (measure_.roiIndices.empty()) {
                    measure_.status = u8"请先框选区域";
                } else {
                    PushHistory(u8"清除框外点");
                    MeasureTools::ApplyRoiDelete(cloud_, measure_.roiIndices, false);
                    measure_.status = std::string(u8"已清除框外点，可见 ") +
                                      std::to_string(cloud_.VisibleCount());
                    measure_.roiIndices.clear();
                    needUpload_ = true;
                }
            }
            if (ImGui::Button(u8"恢复全部点显示", ImVec2(-1, 0))) {
                PushHistory(u8"恢复全部点前");
                MeasureTools::RestoreAllPoints(cloud_);
                measure_.clipEnabled = false;
                measure_.status = u8"已恢复全部点";
                needUpload_ = true;
            }
            break;
        case ToolMode::ClipPlane:
            ImGui::TextWrapped(u8"点击一点设置剖切（优先用法向，否则 +Z）。");
            if (ImGui::Checkbox(u8"启用剖切", &measure_.clipEnabled)) {
                if (measure_.clipEnabled) PushHistory(u8"启用剖切");
                MeasureTools::ApplyClipMask(cloud_, measure_.clipNormal, measure_.clipD,
                                            measure_.clipEnabled);
                needUpload_ = true;
            }
            if (ImGui::Button(u8"清除剖切", ImVec2(-1, 0))) {
                PushHistory(u8"清除剖切前");
                measure_.clipEnabled = false;
                MeasureTools::RestoreAllPoints(cloud_);
                needUpload_ = true;
            }
            break;
        case ToolMode::StepHeight:
            ImGui::TextWrapped(
                u8"工业台阶/高度差测量：\n"
                u8"① 点击基准面点 A\n"
                u8"② 再点击测量面点 B\n"
                u8"③ 显示 ΔZ = B.Z − A.Z（单位 mm）");
            if (measure_.stepA && measure_.stepB) {
                ImGui::Spacing();
                ImGui::Text(u8"ΔZ = %.4f mm", measure_.stepDeltaZ);
                const Vec3 wa = cloud_.ToWorld(*measure_.stepA);
                const Vec3 wb = cloud_.ToWorld(*measure_.stepB);
                ImGui::Text(u8"A.Z = %.4f mm", wa.z);
                ImGui::Text(u8"B.Z = %.4f mm", wb.z);
            }
            if (ImGui::Button(u8"清除台阶测量", ImVec2(-1, 0))) {
                measure_.stepA.reset();
                measure_.stepB.reset();
                measure_.stepDeltaZ = 0.f;
                UpdateOverlays();
                measure_.status = u8"已清除台阶测量";
            }
            break;
        case ToolMode::Section: {
            ImGui::TextWrapped(
                u8"操作：\n"
                u8"· 3D 中左键拖拽橙色切面 → 移动切割位置\n"
                u8"· 松开后自动生成 2D 轮廓\n"
                u8"· 也可拖动下方滑条 / 点“生成截面”\n"
                u8"· 中键平移，右键旋转");
            const char* axes[] = {u8"沿 X 向切割 (剖面 YZ)", u8"沿 Y 向切割 (剖面 XZ)"};
            int axis = measure_.section.cutAlongX ? 0 : 1;
            ImGui::SetNextItemWidth(-1);
            if (ImGui::Combo(u8"切割方向", &axis, axes, 2)) {
                measure_.section.cutAlongX = (axis == 0);
                if (cloud_.bounds.Valid()) {
                    measure_.section.position = measure_.section.cutAlongX
                                                    ? cloud_.bounds.Center().x
                                                    : cloud_.bounds.Center().y;
                }
                SyncSectionCutPlane();
            }

            float amin = 0.f, amax = 1.f;
            if (cloud_.bounds.Valid()) {
                if (measure_.section.cutAlongX) {
                    amin = cloud_.bounds.min.x;
                    amax = cloud_.bounds.max.x;
                } else {
                    amin = cloud_.bounds.min.y;
                    amax = cloud_.bounds.max.y;
                }
            }
            ImGui::SetNextItemWidth(-1);
            if (ImGui::SliderFloat(u8"切割位置", &measure_.section.position, amin, amax, "%.4f")) {
                SyncSectionCutPlane();
            }
            ImGui::SetNextItemWidth(-1);
            ImGui::DragFloat(u8"截面厚度", &measure_.section.thickness, 0.001f, 1e-5f, 1e6f,
                             "%.5f");

            if (ImGui::Button(u8"生成截面", ImVec2(-1, 36.f))) {
                GenerateSection();
            }
            if (ImGui::Button(u8"清除截面结果", ImVec2(-1, 0))) {
                measure_.section.points.clear();
                measure_.section.pickA.reset();
                measure_.section.pickB.reset();
                measure_.section.lineDistance = 0.f;
                measure_.section.zDistance = 0.f;
                SyncSectionCutPlane();
                measure_.status = u8"已清除截面轮廓（切面仍可拖拽）";
            }
            if (!measure_.section.points.empty()) {
                ImGui::Text(u8"已生成 %zu 个轮廓点", measure_.section.points.size());
                if (measure_.section.pickA && measure_.section.pickB) {
                    ImGui::Text(u8"垂线间距 = %.6f", measure_.section.lineDistance);
                    ImGui::Text(u8"Z 向距离 = %.6f", measure_.section.zDistance);
                }
            }
            break;
        }
    }

    ImGui::EndChild();
    ImGui::PopStyleColor();
}

void Application::DrawUi() {
    const ImGuiViewport* vp = ImGui::GetMainViewport();
    const float sidebarW = 320.f;
    const float statusH = 42.f;
    const float toolbarH = 40.f;

    const float menuBottom = DrawMenuBar();

    // 算法编辑器：独占菜单下方区域，不显示点云侧栏/状态/叠加层
    if (algoEditor_.IsVisible()) {
        AlgoHost host;
        host.currentCloud = &cloud_;
        host.publishCloud = [this](PointCloud&& c, const char* status) {
            ApplyCloud(std::move(c), status);
        };
        algoEditor_.SetHost(host);
        DrawAboutPopup();
        DrawCreatePopups();
        algoEditor_.Draw(menuBottom);
        return;
    }

    DrawToolbar(menuBottom, toolbarH);
    const float contentTop = menuBottom + toolbarH;
    const float contentH = vp->Pos.y + vp->Size.y - contentTop - statusH;
    UpdateView3dLayout(contentTop, contentH, sidebarW);

    ImGui::SetNextWindowPos(ImVec2(vp->Pos.x, contentTop));
    ImGui::SetNextWindowSize(ImVec2(sidebarW, contentH));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.f);
    ImGui::Begin(u8"##侧栏", nullptr,
                 ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                     ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoBringToFrontOnFocus);

    ImGui::TextDisabled(u8"点云信息");
    ImGui::BeginChild(u8"##info", ImVec2(0, 118.f), true);
    ImGui::Text(u8"总点数　%d", static_cast<int>(cloud_.points.size()));
    ImGui::Text(u8"可见　　%d", static_cast<int>(cloud_.VisibleCount()));
    ImGui::Text(u8"GPU显示 %d", gpuPointCount_);
    if (cloud_.bounds.Valid()) {
        const Vec3 e = cloud_.bounds.Extent();
        ImGui::Text(u8"尺寸  %.2f × %.2f × %.2f mm", e.x, e.y, e.z);
        const Vec3 off = cloud_.originOffset;
        ImGui::TextDisabled(u8"原点偏移 %.3f, %.3f, %.3f", off.x, off.y, off.z);
    } else {
        ImGui::TextDisabled(u8"尚未加载点云");
    }
    ImGui::EndChild();

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    DrawToolPanel();

    ImGui::End();
    ImGui::PopStyleVar(2);

    ImGui::SetNextWindowPos(ImVec2(vp->Pos.x, vp->Pos.y + vp->Size.y - statusH));
    ImGui::SetNextWindowSize(ImVec2(vp->Size.x, statusH));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(16.f, 11.f));
    ImGui::Begin(u8"##状态栏", nullptr,
                 ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                     ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoCollapse);
    ImGui::TextColored(ImVec4(0.45f, 0.80f, 0.85f, 1.f), u8"状态");
    ImGui::SameLine(0.f, 12.f);
    ImGui::TextUnformatted(measure_.status.c_str());
    ImGui::End();
    ImGui::PopStyleVar(2);

    if (measure_.mode == ToolMode::Section) {
        DrawSectionPanel();
    }
    if (measure_.mode == ToolMode::StepGap && measure_.stepGap.hasDistances) {
        DrawStepGapPanel();
    }

    if (cloud_.points.empty()) {
        ImDrawList* tipDl = ImGui::GetBackgroundDrawList();
        const float cx = view3dX_ + view3dW_ * 0.5f;
        const float cy = contentTop + contentH * 0.45f;
        const char* tip = u8"打开点云文件开始查看";
        const ImVec2 sz = ImGui::CalcTextSize(tip);
        tipDl->AddText(ImVec2(cx - sz.x * 0.5f, cy), IM_COL32(130, 155, 165, 200), tip);
    }

    DrawViewAxisWidget(contentTop, contentTop + contentH, sidebarW);
    DrawImagePanel();
    // 左边缘拖拽可能改了宽度，同帧刷新点云视区，避免差一帧
    UpdateView3dLayout(contentTop, contentH, sidebarW);
    DrawAboutPopup();
    DrawCreatePopups();
    DrawOverlays();
}

void Application::Run() {
    while (!glfwWindowShouldClose(window_)) {
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        HandleInput();
        DrawUi();

        if (needUpload_ && !algoEditor_.IsVisible()) RefreshGpu();

        glfwGetFramebufferSize(window_, &fbW_, &fbH_);
        glDisable(GL_SCISSOR_TEST);
        glViewport(0, 0, fbW_, fbH_);
        glClearColor(0.07f, 0.08f, 0.10f, 1.f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        if (!algoEditor_.IsVisible()) {
            int vx = 0, vy = 0, vw = 0, vh = 0;
            GetView3dGlViewport(vx, vy, vw, vh);
            glViewport(vx, vy, vw, vh);
            glEnable(GL_SCISSOR_TEST);
            glScissor(vx, vy, vw, vh);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            renderer_.Draw(camera_, vw, vh, pointSize_, opacity_);
            glDisable(GL_SCISSOR_TEST);
            glViewport(0, 0, fbW_, fbH_);
        }

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window_);
    }
}

void Application::Shutdown() {
    DestroyImageView(depthImage_);
    DestroyImageView(brightnessImage_);
    renderer_.Shutdown();
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    if (window_) {
        glfwDestroyWindow(window_);
        window_ = nullptr;
    }
    glfwTerminate();
}
