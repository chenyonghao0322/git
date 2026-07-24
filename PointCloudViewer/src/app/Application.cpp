#include "app/Application.h"

#include "app/FileDialog.h"
#include "app/UiTheme.h"
#include "io/ImageIO.h"
#include "io/PointCloudGenerator.h"
#include "io/PointCloudIO.h"
#include "tools/FilterTools.h"
#include "tools/OpenCv2D.h"
#include "tools/PclTools.h"
#include "tools/PaddleOcrTools.h"

#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>
#include <functional>
#include <unordered_set>
#include <vector>

namespace {

void GlfwErrorCallback(int /*code*/, const char* /*desc*/) {}

void SetWindowIconFromAsset(GLFWwindow* window) {
    ImageIO::RgbImage img;
    std::string error;
    if (!ImageIO::LoadRgb("assets/pointcloud_icon.png", img, error)) return;

    std::vector<uint8_t> rgba(static_cast<std::size_t>(img.width * img.height * 4));
    for (int i = 0; i < img.width * img.height; ++i) {
        rgba[static_cast<std::size_t>(i) * 4 + 0] = img.rgb[static_cast<std::size_t>(i) * 3 + 0];
        rgba[static_cast<std::size_t>(i) * 4 + 1] = img.rgb[static_cast<std::size_t>(i) * 3 + 1];
        rgba[static_cast<std::size_t>(i) * 4 + 2] = img.rgb[static_cast<std::size_t>(i) * 3 + 2];
        rgba[static_cast<std::size_t>(i) * 4 + 3] = 255;
    }

    GLFWimage icon{};
    icon.width = img.width;
    icon.height = img.height;
    icon.pixels = rgba.data();
    glfwSetWindowIcon(window, 1, &icon);
}

ImFont* gMenuBarFont = nullptr;

constexpr float kMenuBarFontPx = 27.f;
constexpr float kMenuBarFramePadY = 10.f;
constexpr float kMenuBarFramePadXMul = 1.25f;

void PushMenuBarStyle() {
    ImGuiStyle& style = ImGui::GetStyle();
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding,
                        ImVec2(style.FramePadding.x * kMenuBarFramePadXMul, kMenuBarFramePadY));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing,
                        ImVec2(style.ItemSpacing.x * 1.2f, style.ItemSpacing.y * 1.1f));
    if (gMenuBarFont) ImGui::PushFont(gMenuBarFont);
}

void PopMenuBarStyle() {
    if (gMenuBarFont) ImGui::PopFont();
    ImGui::PopStyleVar(2);
}

float MenuBarFrameHeight() {
    PushMenuBarStyle();
    const float h = ImGui::GetFrameHeight();
    PopMenuBarStyle();
    return h;
}

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
            path, 18.0f, &cfg, io.Fonts->GetGlyphRangesChineseSimplifiedCommon());
        gMenuBarFont = io.Fonts->AddFontFromFileTTF(
            path, kMenuBarFontPx, &cfg, io.Fonts->GetGlyphRangesChineseSimplifiedCommon());
        if (font) return true;
    }
    return false;
}

void ProjectBulgePoint(float ax, float ay, float bx, float by, float mx, float my, float& p2x,
                       float& p2y) {
    const float mx0 = (ax + bx) * 0.5f;
    const float my0 = (ay + by) * 0.5f;
    const float dx = bx - ax;
    const float dy = by - ay;
    const float len = std::hypot(dx, dy);
    if (len < 1e-3f) {
        p2x = mx;
        p2y = my;
        return;
    }
    const float nx = -dy / len;
    const float ny = dx / len;
    const float t = (mx - mx0) * nx + (my - my0) * ny;
    p2x = mx0 + nx * t;
    p2y = my0 + ny * t;
}

float ArcChordBulgePx(float ax, float ay, float bx, float by, float p2x, float p2y) {
    const float mx0 = (ax + bx) * 0.5f;
    const float my0 = (ay + by) * 0.5f;
    const float dx = bx - ax;
    const float dy = by - ay;
    const float len = std::hypot(dx, dy);
    if (len < 1e-3f) return 0.f;
    return std::fabs((p2x - mx0) * dy - (p2y - my0) * dx) / len;
}

}  // namespace

#include <limits>

namespace {
PointCloud gEmptySceneCloud;
}  // namespace

bool Application::HasSceneClouds() const {
    return activeLayerIndex_ >= 0 && activeLayerIndex_ < static_cast<int>(sceneLayers_.size());
}

PointCloud& Application::MainCloud() {
    if (!HasSceneClouds()) return gEmptySceneCloud;
    return sceneLayers_[static_cast<std::size_t>(activeLayerIndex_)].cloud;
}

const PointCloud& Application::MainCloud() const {
    if (!HasSceneClouds()) return gEmptySceneCloud;
    return sceneLayers_[static_cast<std::size_t>(activeLayerIndex_)].cloud;
}

Application::SceneCloudLayer& Application::ActiveLayer() {
    static SceneCloudLayer s_empty;
    if (!HasSceneClouds()) return s_empty;
    return sceneLayers_[static_cast<std::size_t>(activeLayerIndex_)];
}

const Application::SceneCloudLayer& Application::ActiveLayer() const {
    static const SceneCloudLayer s_empty;
    if (!HasSceneClouds()) return s_empty;
    return sceneLayers_[static_cast<std::size_t>(activeLayerIndex_)];
}

PointCloudRenderer& Application::ActiveRenderer() { return ActiveLayer().renderer; }

void Application::InitLayerRenderer(SceneCloudLayer& layer) {
    if (layer.rendererReady) return;
    std::string err;
    if (!layer.renderer.Init(err)) {
        SetStatus(err);
        return;
    }
    layer.rendererReady = true;
}

void Application::ShutdownSceneLayers() {
    for (SceneCloudLayer& layer : sceneLayers_) {
        if (layer.rendererReady) {
            layer.renderer.Shutdown();
            layer.rendererReady = false;
        }
    }
}

void Application::SetActiveLayerById(int layerId) {
    for (int i = 0; i < static_cast<int>(sceneLayers_.size()); ++i) {
        if (sceneLayers_[static_cast<std::size_t>(i)].id != layerId) continue;
        if (activeLayerIndex_ != i) {
            activeLayerIndex_ = i;
            history_.Clear();
            measure_.roiIndices.clear();
            ClearFilterCompare();
            sceneGpuDirty_ = true;
        }
        dbSelectedLayerId_ = layerId;
        dbSelectedSpecial_ = DbEntityId::None;
        return;
    }
}

void Application::RemoveSceneLayerById(int layerId) {
    int removeIdx = -1;
    for (int i = 0; i < static_cast<int>(sceneLayers_.size()); ++i) {
        if (sceneLayers_[static_cast<std::size_t>(i)].id == layerId) {
            removeIdx = i;
            break;
        }
    }
    if (removeIdx < 0) return;

    CloseDualCloudView();
    const bool wasActive = (removeIdx == activeLayerIndex_);
    SceneCloudLayer& layer = sceneLayers_[static_cast<std::size_t>(removeIdx)];
    if (layer.rendererReady) {
        layer.renderer.Shutdown();
        layer.rendererReady = false;
    }
    sceneLayers_.erase(sceneLayers_.begin() + removeIdx);

    if (sceneLayers_.empty()) {
        activeLayerIndex_ = -1;
        dbSelectedLayerId_ = 0;
        dbSelectedSpecial_ = DbEntityId::None;
        history_.Clear();
        measure_ = {};
        ClearFilterCompare();
        sceneGpuDirty_ = true;
        SetStatus(u8"已删除点云，场景为空");
        return;
    }

    if (wasActive) {
        activeLayerIndex_ = std::min(removeIdx, static_cast<int>(sceneLayers_.size()) - 1);
    } else if (activeLayerIndex_ > removeIdx) {
        activeLayerIndex_--;
    }
    dbSelectedLayerId_ = sceneLayers_[static_cast<std::size_t>(activeLayerIndex_)].id;
    dbSelectedSpecial_ = DbEntityId::None;
    history_.Clear();
    measure_.roiIndices.clear();
    ClearFilterCompare();
    sceneGpuDirty_ = true;
    FitCameraToAllClouds();
    SetStatus(u8"已从 DB 树删除点云");
}

void Application::FitCameraToAllClouds() {
    Aabb merged;
    bool any = false;
    for (const SceneCloudLayer& layer : sceneLayers_) {
        if (!layer.visible || !layer.cloud.bounds.Valid()) continue;
        if (!any) {
            merged = layer.cloud.bounds;
            any = true;
        } else {
            merged.Expand(layer.cloud.bounds.min);
            merged.Expand(layer.cloud.bounds.max);
        }
    }
    if (!any) {
        FitCameraToCloud();
        return;
    }
    const Vec3 e = merged.Extent();
    const float maxExtent = std::max({e.x, e.y, e.z, 0.1f});
    camera_.SetTarget(merged.Center(), maxExtent * 2.f);
}

void Application::UpdatePlaneFitProfile(const PlaneModel& plane,
                                        const std::vector<std::size_t>& indices) {
    MeasureTools::ComputePlaneDistanceProfile(MainCloud(), indices, plane,
                                              measure_.planeDistProfile);
}

void Application::DrawPlaneDistanceHistogram(const PlaneDistanceProfile& profile) {
    if (!profile.valid || profile.signedDist.empty()) return;

    ImGui::Spacing();
    ImGui::TextDisabled(u8"点到拟合平面距离分布");
    char stat[160];
    std::snprintf(stat, sizeof(stat), u8"N=%d  min=%.4f  max=%.4f  RMS=%.4f", profile.pointCount,
                  profile.minD, profile.maxD, profile.rms);
    ImGui::TextUnformatted(stat);

    ImVec2 canvasPos = ImGui::GetCursorScreenPos();
    ImVec2 canvasSize = ImGui::GetContentRegionAvail();
    canvasSize.y = std::max(canvasSize.y, 110.f);
    if (canvasSize.x < 50.f) canvasSize.x = 50.f;

    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImVec2 canvasMax(canvasPos.x + canvasSize.x, canvasPos.y + canvasSize.y);
    dl->AddRectFilled(canvasPos, canvasMax, IM_COL32(18, 22, 26, 255), 4.f);
    dl->AddRect(canvasPos, canvasMax, IM_COL32(60, 90, 100, 255), 4.f);

    constexpr int kBins = 32;
    std::vector<int> bins(static_cast<std::size_t>(kBins), 0);
    const float range = std::max(profile.maxD - profile.minD, 1e-6f);
    int maxCount = 0;
    for (float d : profile.signedDist) {
        int bi = static_cast<int>((d - profile.minD) / range * static_cast<float>(kBins - 1));
        bi = std::clamp(bi, 0, kBins - 1);
        ++bins[static_cast<std::size_t>(bi)];
        maxCount = std::max(maxCount, bins[static_cast<std::size_t>(bi)]);
    }
    if (maxCount <= 0) maxCount = 1;

    const float barW = canvasSize.x / static_cast<float>(kBins);
    const float plotH = canvasSize.y - 22.f;
    for (int i = 0; i < kBins; ++i) {
        const float h = plotH * static_cast<float>(bins[static_cast<std::size_t>(i)]) /
                        static_cast<float>(maxCount);
        const float x0 = canvasPos.x + static_cast<float>(i) * barW + 1.f;
        const float y0 = canvasMax.y - 8.f - h;
        const float x1 = x0 + barW - 2.f;
        dl->AddRectFilled(ImVec2(x0, y0), ImVec2(x1, canvasMax.y - 8.f),
                          IM_COL32(70, 190, 210, 230), 2.f);
    }

    const float zeroX =
        canvasPos.x + (0.f - profile.minD) / range * canvasSize.x;
    if (zeroX >= canvasPos.x && zeroX <= canvasMax.x) {
        dl->AddLine(ImVec2(zeroX, canvasPos.y + 4.f), ImVec2(zeroX, canvasMax.y - 8.f),
                    IM_COL32(255, 200, 80, 200), 1.5f);
    }

    char axis[96];
    std::snprintf(axis, sizeof(axis), u8"%.3f", profile.minD);
    dl->AddText(ImVec2(canvasPos.x + 4.f, canvasMax.y - 18.f), IM_COL32(140, 170, 180, 255), axis);
    std::snprintf(axis, sizeof(axis), u8"%.3f", profile.maxD);
    const ImVec2 ts = ImGui::CalcTextSize(axis);
    dl->AddText(ImVec2(canvasMax.x - ts.x - 4.f, canvasMax.y - 18.f),
                IM_COL32(140, 170, 180, 255), axis);
    dl->AddText(ImVec2(canvasPos.x + 4.f, canvasPos.y + 4.f), IM_COL32(140, 170, 180, 255),
                u8"距离 mm");

    ImGui::Dummy(canvasSize);
}

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
    SetWindowIconFromAsset(window_);

    if (!gladLoadGL(reinterpret_cast<GLADloadfunc>(glfwGetProcAddress))) {
        return false;
    }

    std::string err;
    if (!filledRenderer_.Init(err)) {
        return false;
    }
    if (!icpRenderer_.Init(err)) {
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

    shapeTemplateWindow_.SetStatusCallback([this](const char* msg) { SetStatus(msg); });
    halconMatchWindow_.SetStatusCallback([this](const char* msg) { SetStatus(msg); });
    ocrWindow_.SetStatusCallback([this](const char* msg) { SetStatus(msg); });
    camera2DCalibWindow_.SetStatusCallback([this](const char* msg) { SetStatus(msg); });
    cameraIntrinsicsCalibWindow_.SetStatusCallback([this](const char* msg) { SetStatus(msg); });
    multiViewGeometryWindow_.SetStatusCallback([this](const char* msg) { SetStatus(msg); });
    multiViewGeometryWindow_.SetImportCloudCallback(
        [this](const std::vector<Vec3>& pts, const char* status) {
            PointCloud cloud;
            cloud.points = pts;
            cloud.ResetMask();
            cloud.RecomputeBounds();
            cloud.CenterToOrigin();
            ApplyCloud(std::move(cloud), status, false);
        });

    SetStatus(u8"请打开点云文件（PLY / PCD / XYZ / OBJ），或打开深度/亮度图对照查看");
    return true;
}

bool Application::ApplyCloud(PointCloud&& cloud, const char* statusMsg, bool append) {
    CloseDualCloudView();
    if (!append || !HasSceneClouds()) {
        ClearIcpState();
        ShutdownSceneLayers();
        sceneLayers_.clear();
        activeLayerIndex_ = -1;
        history_.Clear();
        measure_ = {};
        ClearFilterCompare();
        useIntensityColors_ = false;
        intensityColors_.clear();
        autoZRange_ = true;
        append = false;
    }

    SceneCloudLayer layer;
    layer.id = nextSceneLayerId_++;
    layer.cloud = std::move(cloud);
    layer.visible = true;
    layer.needUpload = true;
    InitLayerRenderer(layer);
    sceneLayers_.push_back(std::move(layer));
    activeLayerIndex_ = static_cast<int>(sceneLayers_.size()) - 1;
    dbSelectedLayerId_ = sceneLayers_[static_cast<std::size_t>(activeLayerIndex_)].id;
    dbSelectedSpecial_ = DbEntityId::None;

    measure_.section.cutAlongX = true;
    measure_.section.position = MainCloud().bounds.Center().x;
    measure_.section.thickness =
        std::max(MainCloud().bounds.Extent().x, MainCloud().bounds.Extent().y) * 0.002f;
    if (measure_.section.thickness < 1e-4f) measure_.section.thickness = 0.01f;
    UpdateAxesLength();

    if (MainCloud().points.size() > 5000000) {
        maxDisplayPoints_ = 800000;
    } else if (MainCloud().points.size() > 2000000) {
        maxDisplayPoints_ = 1200000;
    } else {
        maxDisplayPoints_ = 1500000;
    }

    if (autoZRange_ || !append) {
        zMin_ = MainCloud().bounds.min.z;
        zMax_ = MainCloud().bounds.max.z;
    }

    if (statusMsg && statusMsg[0]) {
        SetStatus(statusMsg);
    } else {
        char buf[256];
        std::snprintf(buf, sizeof(buf), u8"已加载 %zu 个点（显示上限约 %d）", MainCloud().points.size(),
                      maxDisplayPoints_);
        SetStatus(buf);
    }

    if (append) {
        FitCameraToAllClouds();
    } else {
        FitCameraToCloud();
    }
    sceneGpuDirty_ = true;
    return true;
}

bool Application::SaveCloud() {
    if (MainCloud().points.empty()) {
        SetStatus(u8"当前没有点云可保存");
        return false;
    }
    const std::string path = FileDialog::SavePointCloudFile();
    if (path.empty()) return false;
    std::string error;
    if (!PointCloudIO::Save(path, MainCloud(), error, saveVisibleOnly_)) {
        SetStatus(error);
        return false;
    }
    char buf[320];
    std::snprintf(buf, sizeof(buf), u8"已保存 %s（%s）", path.c_str(),
                  saveVisibleOnly_ ? u8"仅可见点" : u8"全部点");
    SetStatus(buf);
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

void RotateGray90CW(const std::vector<float>& src, int w, int h, std::vector<float>& dst, int& outW,
                    int& outH) {
    outW = h;
    outH = w;
    dst.resize(static_cast<std::size_t>(outW * outH));
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const int nx = h - 1 - y;
            const int ny = x;
            dst[static_cast<std::size_t>(ny) * static_cast<std::size_t>(outW) +
                static_cast<std::size_t>(nx)] =
                src[static_cast<std::size_t>(y) * static_cast<std::size_t>(w) +
                    static_cast<std::size_t>(x)];
        }
    }
}

void RotateGray90CCW(const std::vector<float>& src, int w, int h, std::vector<float>& dst, int& outW,
                     int& outH) {
    outW = h;
    outH = w;
    dst.resize(static_cast<std::size_t>(outW * outH));
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const int nx = y;
            const int ny = w - 1 - x;
            dst[static_cast<std::size_t>(ny) * static_cast<std::size_t>(outW) +
                static_cast<std::size_t>(nx)] =
                src[static_cast<std::size_t>(y) * static_cast<std::size_t>(w) +
                    static_cast<std::size_t>(x)];
        }
    }
}

void RotateRgb90CW(const std::vector<uint8_t>& src, int w, int h, std::vector<uint8_t>& dst,
                   int& outW, int& outH) {
    outW = h;
    outH = w;
    dst.resize(static_cast<std::size_t>(outW * outH * 3u));
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const int nx = h - 1 - y;
            const int ny = x;
            const std::size_t srcIdx =
                (static_cast<std::size_t>(y) * static_cast<std::size_t>(w) +
                 static_cast<std::size_t>(x)) *
                3u;
            const std::size_t dstIdx =
                (static_cast<std::size_t>(ny) * static_cast<std::size_t>(outW) +
                 static_cast<std::size_t>(nx)) *
                3u;
            dst[dstIdx] = src[srcIdx];
            dst[dstIdx + 1] = src[srcIdx + 1];
            dst[dstIdx + 2] = src[srcIdx + 2];
        }
    }
}

void RotateRgb90CCW(const std::vector<uint8_t>& src, int w, int h, std::vector<uint8_t>& dst,
                    int& outW, int& outH) {
    outW = h;
    outH = w;
    dst.resize(static_cast<std::size_t>(outW * outH * 3u));
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const int nx = y;
            const int ny = w - 1 - x;
            const std::size_t srcIdx =
                (static_cast<std::size_t>(y) * static_cast<std::size_t>(w) +
                 static_cast<std::size_t>(x)) *
                3u;
            const std::size_t dstIdx =
                (static_cast<std::size_t>(ny) * static_cast<std::size_t>(outW) +
                 static_cast<std::size_t>(nx)) *
                3u;
            dst[dstIdx] = src[srcIdx];
            dst[dstIdx + 1] = src[srcIdx + 1];
            dst[dstIdx + 2] = src[srcIdx + 2];
        }
    }
}

}  // namespace

bool Application::LoadDepthImageFromPath(const std::string& path) {
    if (path.empty()) return false;

    ImageIO::GrayImage gray;
    std::string error;
    if (!ImageIO::LoadGray(path, gray, error)) {
        SetStatus(error);
        return false;
    }

    DestroyImageView(depthImage_);
    measuredLines_.erase(
        std::remove_if(measuredLines_.begin(), measuredLines_.end(),
                       [](const MeasuredImageLine& l) { return l.imageSource == 0; }),
        measuredLines_.end());
    lineDistValid_ = false;
    depthImage_.path = path;
    depthImage_.width = gray.width;
    depthImage_.height = gray.height;
    depthImage_.gray = std::move(gray.pixels);
    ComputeGrayRange(depthImage_.gray, depthSkipZero_, depthDataMin_, depthDataMax_);
    PointCloudIO::SuggestDepthMapParams(depthImage_.gray, depthImage_.width, depthImage_.height,
                                        depthMapParams_);
    depthMapParams_.skipZero = depthSkipZero_;
    depthDisplayMin_ = depthDataMin_;
    depthDisplayMax_ = depthDataMax_;
    depthImage_.valueMin = depthDataMin_;
    depthImage_.valueMax = depthDataMax_;
    RebuildDepthDisplay();
    if (!depthImage_.valid()) {
        SetStatus(u8"深度图纹理上传失败");
        DestroyImageView(depthImage_);
        return false;
    }

    EnterView2DMode();
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
    SetStatus(buf);
    return true;
}

bool Application::OpenDepthImage() {
    const std::string path = FileDialog::OpenImageFile("打开深度图");
    if (path.empty()) return false;
    if (!LoadDepthImageFromPath(path)) return false;
    OpenDepthToCloudDialog();
    return true;
}

void Application::OpenDepthToCloudDialog() {
    if (depthImage_.gray.empty() || depthImage_.width <= 0 || depthImage_.height <= 0) {
        SetStatus(u8"请先打开深度图");
        return;
    }
    showDepthToCloudWindow_ = true;
    depthToCloudWindowFocus_ = true;
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
        SetStatus(error);
        return false;
    }

    DestroyImageView(brightnessImage_);
    measuredLines_.erase(
        std::remove_if(measuredLines_.begin(), measuredLines_.end(),
                       [](const MeasuredImageLine& l) { return l.imageSource == 1; }),
        measuredLines_.end());
    lineDistValid_ = false;
    brightnessImage_.path = path;
    brightnessImage_.width = rgb.width;
    brightnessImage_.height = rgb.height;
    brightnessImage_.rgb = std::move(rgb.rgb);
    brightnessImage_.gray.clear();
    brightnessImage_.valueMin = 0.f;
    brightnessImage_.valueMax = 255.f;
    if (!UploadImageTexture(brightnessImage_)) {
        SetStatus(u8"亮度图纹理上传失败");
        DestroyImageView(brightnessImage_);
        return false;
    }

    EnterView2DMode();
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
    SetStatus(buf);
    return true;
}

bool Application::HasImagePanel() const {
    return view2DMode_;
}

float Application::ImagePanelWidth() const {
    if (!view2DMode_) return 0.f;
    const ImGuiViewport* vp = ImGui::GetMainViewport();
    return std::max(vp->Size.x - SidebarWidth(), 1.f);
}

float Application::SidebarWidth() const {
    const ImGuiViewport* vp = ImGui::GetMainViewport();
    const float maxW = std::max(vp->Size.x - 480.f, 280.f);
    return std::clamp(sidebarPreferredW_, 280.f, maxW);
}

void Application::DrawSidebarSplitter(float contentTop, float contentH) {
    const ImGuiViewport* vp = ImGui::GetMainViewport();
    const float sidebarW = SidebarWidth();
    const float maxW = std::max(vp->Size.x - 480.f, 280.f);
    constexpr float kSplitHit = 8.f;
    ImGui::SetNextWindowPos(ImVec2(vp->Pos.x + sidebarW - kSplitHit * 0.5f, contentTop));
    ImGui::SetNextWindowSize(ImVec2(kSplitHit, contentH));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.f, 0.f));
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.f, 0.f, 0.f, 0.f));
    ImGui::Begin(u8"##侧栏分割条", nullptr,
                 ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                     ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoSavedSettings |
                     ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoBackground |
                     ImGuiWindowFlags_NoBringToFrontOnFocus);
    ImGui::InvisibleButton(u8"##sidebar_split", ImVec2(kSplitHit, contentH));
    const bool splitHover = ImGui::IsItemHovered();
    const bool splitDrag = ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Left);
    if (splitDrag) {
        sidebarPreferredW_ = std::clamp(ImGui::GetIO().MousePos.x - vp->Pos.x, 280.f, maxW);
    }
    if (splitHover || splitDrag) {
        ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
    }
    ImGui::End();
    ImGui::PopStyleColor();
    ImGui::PopStyleVar();
}

void Application::AppendConsoleLog(const std::string& msg) {
    if (msg.empty()) return;
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tmLocal{};
#if defined(_WIN32)
    localtime_s(&tmLocal, &t);
#else
    localtime_r(&t, &tmLocal);
#endif
    char timeBuf[16];
    std::snprintf(timeBuf, sizeof(timeBuf), "%02d:%02d:%02d", tmLocal.tm_hour, tmLocal.tm_min,
                  tmLocal.tm_sec);

    consoleLog_.push_back({timeBuf, msg});
    if (consoleLog_.size() > kConsoleMaxLines) {
        consoleLog_.erase(consoleLog_.begin(),
                          consoleLog_.begin() +
                              static_cast<std::ptrdiff_t>(consoleLog_.size() - kConsoleMaxLines));
    }
}

void Application::SetStatus(const std::string& msg, bool logConsole) {
    measure_.status = msg;
    if (logConsole) AppendConsoleLog(msg);
}

float Application::ConsoleHeight() const {
    const ImGuiViewport* vp = ImGui::GetMainViewport();
    const float maxH = std::max(vp->Size.y * 0.42f, 180.f);
    return std::clamp(consoleHeight_, 96.f, maxH);
}

void Application::DrawConsolePanel() {
    constexpr float kResizeHit = 10.f;
    const UiPalette& pal = GetUiPalette();
    const ImGuiViewport* vp = ImGui::GetMainViewport();
    const float h = ConsoleHeight();
    const float sidebarW = SidebarWidth();
    const float consoleW = std::max(vp->Size.x - sidebarW, 1.f);
    ImGui::SetNextWindowPos(ImVec2(vp->Pos.x + sidebarW, vp->Pos.y + vp->Size.y - h));
    ImGui::SetNextWindowSize(ImVec2(consoleW, h));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 6.f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 1.f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(12.f, 8.f));
    ImGui::PushStyleColor(ImGuiCol_WindowBg, pal.panel);
    ImGui::PushStyleColor(ImGuiCol_Border, pal.border);
    ImGui::Begin(u8"输出台", nullptr,
                 ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                     ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoScrollbar |
                     ImGuiWindowFlags_NoScrollWithMouse);

    const ImVec2 winPos = ImGui::GetWindowPos();
    const float winW = ImGui::GetWindowWidth();
    ImGui::SetCursorScreenPos(winPos);
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.f, 0.f, 0.f, 0.f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.28f, 0.48f, 0.58f, 0.30f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.30f, 0.58f, 0.68f, 0.45f));
    ImGui::InvisibleButton(u8"##console_resize", ImVec2(winW, kResizeHit));
    const bool resizeHover = ImGui::IsItemHovered();
    const bool resizeActive = ImGui::IsItemActive();
    if (resizeActive && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
        const float maxH = std::max(vp->Size.y * 0.42f, 180.f);
        consoleHeight_ =
            std::clamp(vp->Pos.y + vp->Size.y - ImGui::GetIO().MousePos.y, 96.f, maxH);
    }
    if (resizeHover || resizeActive) {
        ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);
    }
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const float gripY = winPos.y + kResizeHit * 0.5f;
    const ImU32 gripCol =
        ImGui::ColorConvertFloat4ToU32(resizeHover || resizeActive ? pal.accent : pal.borderLight);
    dl->AddLine(ImVec2(winPos.x + 48.f, gripY), ImVec2(winPos.x + winW - 48.f, gripY), gripCol,
                (resizeHover || resizeActive) ? 2.f : 1.f);
    ImGui::PopStyleColor(3);

    ImGui::SetCursorPos(ImVec2(ImGui::GetStyle().WindowPadding.x, kResizeHit + 2.f));

    UiSectionHeader(u8"输出台", nullptr, &pal.sectionTitle, true);
    ImGui::SameLine(0.f, 12.f);
    if (ImGui::SmallButton(u8"清空")) consoleLog_.clear();
    ImGui::SameLine();
    ImGui::Checkbox(u8"自动滚动", &consoleAutoScroll_);
    if (!consoleLog_.empty()) {
        ImGui::SameLine();
        char countBuf[24];
        std::snprintf(countBuf, sizeof(countBuf), u8"%zu 条", consoleLog_.size());
        ImGui::TextDisabled("%s", countBuf);
    }
    if (!measure_.status.empty()) {
        ImGui::SameLine(0.f, 16.f);
        ImGui::TextDisabled(u8"状态");
        ImGui::SameLine();
        ImGui::TextColored(pal.accent, "%s", measure_.status.c_str());
    }

    ImGui::Spacing();
    ImGui::PushStyleColor(ImGuiCol_Separator, pal.border);
    ImGui::Separator();
    ImGui::PopStyleColor();
    ImGui::Spacing();

    ImGui::PushStyleColor(ImGuiCol_ChildBg, pal.panelRaised);
    ImGui::PushStyleColor(ImGuiCol_Border, pal.borderLight);
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 4.f);
    ImGui::PushStyleVar(ImGuiStyleVar_ChildBorderSize, 1.f);
    ImGui::BeginChild(u8"##console_scroll", ImVec2(0.f, 0.f), true,
                      ImGuiWindowFlags_HorizontalScrollbar);
    for (const ConsoleLine& line : consoleLog_) {
        ImGui::TextColored(pal.consoleTime, "[%s]", line.time.c_str());
        ImGui::SameLine(0.f, 6.f);
        ImGui::TextUnformatted(line.text.c_str());
    }
    if (consoleAutoScroll_ && ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 2.f) {
        ImGui::SetScrollHereY(1.f);
    }
    ImGui::EndChild();
    ImGui::PopStyleVar(2);
    ImGui::PopStyleColor(2);

    ImGui::End();
    ImGui::PopStyleColor(2);
    ImGui::PopStyleVar(3);
}

void Application::ClearImageSyncPick() {
    syncHasPick_ = false;
    syncCol_ = -1;
    syncRow_ = -1;
}

bool Application::CanSyncRegionBlob() const {
    return imageSyncEnabled_ && depthImage_.valid() && brightnessImage_.valid() &&
           depthImage_.width > 0 && depthImage_.height > 0 &&
           depthImage_.width == brightnessImage_.width &&
           depthImage_.height == brightnessImage_.height;
}

bool Application::TryEnableImageSync() {
    if (!depthImage_.valid() || !brightnessImage_.valid()) {
        SetStatus(u8"请先同时打开深度图和亮度图");
        return false;
    }
    if (depthImage_.width != brightnessImage_.width ||
        depthImage_.height != brightnessImage_.height) {
        SetStatus(u8"深度图与亮度图尺寸不一致，无法联动");
        return false;
    }

    syncWidth_ = depthImage_.width;
    syncHeight_ = depthImage_.height;
    imageSyncEnabled_ = true;
    EnterView2DMode();
    ClearImageSyncPick();
    char buf[192];
    std::snprintf(buf, sizeof(buf),
                  u8"已启用深度/亮度联动 %dx%d：单击同步十字线；区域筛选将同步到两图", syncWidth_,
                  syncHeight_);
    SetStatus(buf);
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
    SetStatus(buf);
}

bool Application::LoadPath(const std::string& path, bool append) {
    std::string error;
    PointCloud cloud;
    if (!PointCloudIO::Load(path, cloud, error)) {
        SetStatus(error);
        return false;
    }
    char buf[256];
    std::snprintf(buf, sizeof(buf), u8"已打开 %zu 个点（显示上限约 %d，可在侧栏调节）",
                  cloud.points.size(),
                  cloud.points.size() > 5000000
                      ? 800000
                      : (cloud.points.size() > 2000000 ? 1200000 : 1500000));
    return ApplyCloud(std::move(cloud), buf, append);
}

void Application::CreateSphereCloud() {
    PointCloudGenerator::SphereParams p;
    p.radius = genSphereRadius_;
    p.pointCount = genSpherePoints_;
    p.noise = genSphereNoise_;
    std::string error;
    PointCloud cloud;
    if (!PointCloudGenerator::GenerateSphere(p, cloud, error)) {
        SetStatus(error);
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
        SetStatus(error);
        return;
    }
    char buf[192];
    std::snprintf(buf, sizeof(buf), u8"已创建圆柱点云 %zu 点（R=%.3f，H=%.3f，噪声=%.3f）",
                  cloud.points.size(), p.radius, p.height, p.noise);
    ApplyCloud(std::move(cloud), buf);
}

void Application::CreateDiskCloud() {
    PointCloudGenerator::DiskParams p;
    p.radius = genDiskRadius_;
    p.pointCount = genDiskPoints_;
    p.noise = genDiskNoise_;
    std::string error;
    PointCloud cloud;
    if (!PointCloudGenerator::GenerateDisk(p, cloud, error)) {
        SetStatus(error);
        return;
    }
    char buf[192];
    std::snprintf(buf, sizeof(buf), u8"已创建圆面点云 %zu 点（R=%.3f，Z噪声=%.3f）",
                  cloud.points.size(), p.radius, p.noise);
    ApplyCloud(std::move(cloud), buf);
}

void Application::CreatePlaneCloud() {
    PointCloudGenerator::PlaneParams p;
    p.extentX = genPlaneExtentX_;
    p.extentY = genPlaneExtentY_;
    p.pointCount = genPlanePoints_;
    p.noise = genPlaneNoise_;
    std::string error;
    PointCloud cloud;
    if (!PointCloudGenerator::GeneratePlane(p, cloud, error)) {
        SetStatus(error);
        return;
    }
    char buf[192];
    std::snprintf(buf, sizeof(buf), u8"已创建平面点云 %zu 点（%.1f×%.1f，Z噪声=%.3f）",
                  cloud.points.size(), p.extentX * 2.f, p.extentY * 2.f, p.noise);
    ApplyCloud(std::move(cloud), buf);
}

namespace {
constexpr const char* kFittedSphereSourceTag = u8"<拟合:球面>";
}  // namespace

void Application::RemoveFittedSphereLayers() {
    for (int i = static_cast<int>(sceneLayers_.size()) - 1; i >= 0; --i) {
        if (sceneLayers_[static_cast<std::size_t>(i)].cloud.sourcePath != kFittedSphereSourceTag) {
            continue;
        }
        SceneCloudLayer& layer = sceneLayers_[static_cast<std::size_t>(i)];
        if (layer.rendererReady) {
            layer.renderer.Shutdown();
            layer.rendererReady = false;
        }
        sceneLayers_.erase(sceneLayers_.begin() + i);
        if (activeLayerIndex_ == i) {
            activeLayerIndex_ = std::max(0, i - 1);
        } else if (activeLayerIndex_ > i) {
            activeLayerIndex_--;
        }
    }
    if (HasSceneClouds()) {
        dbSelectedLayerId_ = sceneLayers_[static_cast<std::size_t>(activeLayerIndex_)].id;
    }
    sceneGpuDirty_ = true;
}

void Application::SyncFittedSphereLayer(const SphereModel& sphere) {
    if (sphere.radius <= 1e-6f || !HasSceneClouds()) return;

    RemoveFittedSphereLayers();

    PointCloudGenerator::SphereParams p;
    p.radius = sphere.radius;
    p.pointCount = genSpherePoints_;
    p.centerX = sphere.center.x;
    p.centerY = sphere.center.y;
    p.centerZ = sphere.center.z;
    p.noise = 0.f;

    PointCloud cloud;
    std::string error;
    if (!PointCloudGenerator::GenerateSphere(p, cloud, error, false)) {
        SetStatus(error);
        return;
    }
    cloud.sourcePath = kFittedSphereSourceTag;
    cloud.originOffset = ActiveLayer().cloud.originOffset;
    cloud.colors.assign(cloud.points.size(), Vec3{1.f, 0.55f, 0.15f});

    SceneCloudLayer layer;
    layer.id = nextSceneLayerId_++;
    layer.cloud = std::move(cloud);
    layer.visible = true;
    layer.needUpload = true;
    InitLayerRenderer(layer);
    sceneLayers_.push_back(std::move(layer));
    sceneGpuDirty_ = true;
}

void Application::FitCameraToCloud() {
    if (!MainCloud().bounds.Valid()) return;
    const Vec3 e = MainCloud().bounds.Extent();
    const float maxExtent = std::max({e.x, e.y, e.z, 0.1f});
    const float dist = maxExtent * 2.0f;
    camera_.SetTarget(MainCloud().bounds.Center(), dist);
    camera_.Reset();
    UpdateAxesLength();
}

void Application::ApplyViewPreset(int preset) {
    if (!MainCloud().bounds.Valid()) return;
    const Vec3 e = MainCloud().bounds.Extent();
    const float maxExtent = std::max({e.x, e.y, e.z, 0.1f});
    const float dist = maxExtent * 2.0f;
    camera_.SetTarget(MainCloud().bounds.Center(), dist);

    constexpr float pi = 3.14159265f;
    switch (preset) {
        case 0:  // 俯视 +Y
            camera_.SetYawPitch(0.f, 1.52f);
            SetStatus(u8"视角: 俯视 (+Y → XZ 平面)");
            break;
        case 1:  // 侧视 从 +X 看 YZ
            camera_.SetYawPitch(0.f, 0.f);
            SetStatus(u8"视角: 侧视 (沿 +X 看 YZ)");
            break;
        case 2:  // 侧视 从 +Z 看 XY
            camera_.SetYawPitch(pi * 0.5f, 0.f);
            SetStatus(u8"视角: 侧视 (沿 +Z 看 XY)");
            break;
        case 3:  // 沿运动方向 Y
            camera_.SetYawPitch(0.f, -1.35f);
            SetStatus(u8"视角: 沿运动方向 (沿 Y)");
            break;
        default:
            camera_.Reset();
            SetStatus(u8"视角: 复位到包围盒");
            break;
    }
    UpdateAxesLength();
}

void Application::UpdateAxesLength() {
    if (!MainCloud().bounds.Valid()) {
        axesLength_ = 1.f;
    } else {
        const float d = MainCloud().bounds.Diagonal();
        axesLength_ = std::max(d * 0.15f, 0.01f);
    }
    ActiveRenderer().SetAxes(showAxes_, axesLength_);
}

void Application::PushHistory(const std::string& label, bool captureMainPoints) {
    PointCloud& target = EditableCloud();
    if (target.mask.size() != target.points.size()) target.ResetMask();
    const HistoryCloudTarget targetId =
        (&target == &filledCloud_) ? HistoryCloudTarget::Filled : HistoryCloudTarget::Main;
    history_.Push(CaptureCloudSnapshot(targetId, label, captureMainPoints));
}

void Application::PushMainCloudHistory(const std::string& label, bool captureMainPoints) {
    if (MainCloud().mask.size() != MainCloud().points.size()) MainCloud().ResetMask();
    history_.Push(CaptureCloudSnapshot(HistoryCloudTarget::Main, label, captureMainPoints));
}

CloudSnapshot Application::CaptureCloudSnapshot(HistoryCloudTarget target, const std::string& label,
                                                bool captureMainPoints) const {
    const PointCloud& cloud =
        (target == HistoryCloudTarget::Filled) ? filledCloud_ : MainCloud();
    CloudSnapshot snap;
    snap.target = target;
    snap.label = label;
    if (cloud.mask.size() == cloud.points.size()) {
        snap.mask = cloud.mask;
    }
    if (captureMainPoints && target == HistoryCloudTarget::Main) {
        snap.points = cloud.points;
    }
    return snap;
}

void Application::ApplyCloudSnapshot(const CloudSnapshot& snap, bool closeDualOnMain) {
    if (snap.target == HistoryCloudTarget::Main) {
        if (!snap.mask.empty()) {
            MainCloud().mask = snap.mask;
        } else if (MainCloud().mask.size() != MainCloud().points.size()) {
            MainCloud().ResetMask();
        }
        if (!snap.points.empty()) {
            MainCloud().points = snap.points;
            MainCloud().RecomputeBounds();
            MainCloud().colors.clear();
        }
        if (closeDualOnMain && DualCloudViewActive()) {
            CloseDualCloudView();
        }
        ClearFilterCompare();
        sceneGpuDirty_ = true;
        return;
    }

    if (!snap.mask.empty()) {
        filledCloud_.mask = snap.mask;
    } else if (filledCloud_.mask.size() != filledCloud_.points.size()) {
        filledCloud_.ResetMask();
    }
    needUploadFilled_ = true;
}

void Application::Undo() {
    CloudSnapshot restore;
    if (!history_.PopUndo(restore)) {
        SetStatus(u8"没有可撤销的操作");
        return;
    }

    if (restore.target == HistoryCloudTarget::Main) {
        if (MainCloud().mask.size() != MainCloud().points.size()) MainCloud().ResetMask();
    } else if (filledCloud_.mask.size() != filledCloud_.points.size()) {
        filledCloud_.ResetMask();
    }

    CloudSnapshot redoSnap = CaptureCloudSnapshot(
        restore.target, restore.label, !restore.points.empty());
    history_.PushRedo(std::move(redoSnap));
    ApplyCloudSnapshot(restore, true);
    SetStatus(std::string(u8"已撤销: ") + restore.label);
}

void Application::Redo() {
    CloudSnapshot restore;
    if (!history_.PopRedo(restore)) {
        SetStatus(u8"没有可重做的操作");
        return;
    }

    if (restore.target == HistoryCloudTarget::Main) {
        if (MainCloud().mask.size() != MainCloud().points.size()) MainCloud().ResetMask();
    } else if (filledCloud_.mask.size() != filledCloud_.points.size()) {
        filledCloud_.ResetMask();
    }

    CloudSnapshot undoSnap =
        CaptureCloudSnapshot(restore.target, restore.label, !restore.points.empty());
    history_.PushUndo(std::move(undoSnap));
    ApplyCloudSnapshot(restore, true);
    SetStatus(std::string(u8"已重做: ") + restore.label);
}

void Application::RefreshLayerGpu(SceneCloudLayer& layer, bool isActive) {
    PointCloud& cloud = layer.cloud;
    if (isActive && autoZRange_ && cloud.bounds.Valid()) {
        zMin_ = cloud.bounds.min.z;
        zMax_ = cloud.bounds.max.z;
    }

    if (isActive) RebuildAnalysisColors();

    UploadParams params;
    params.maxDisplayPoints = maxDisplayPoints_;
    if (cloud.bounds.Valid()) {
        params.zMin = isActive ? zMin_ : cloud.bounds.min.z;
        params.zMax = isActive ? zMax_ : cloud.bounds.max.z;
    }
    const bool sphereFitCompare =
        isActive && IsSphereFitMode(measure_.mode) && measure_.sphere &&
        !measure_.sphere->inlierIndices.empty();
    const bool circleFitCompare = isActive && measure_.mode == ToolMode::CircleFit && measure_.circle &&
                                  !measure_.circle->inlierIndices.empty();
    const bool showRoi =
        isActive && !measure_.roiIndices.empty() &&
        (measure_.mode == ToolMode::Roi || measure_.mode == ToolMode::PlaneFit ||
         measure_.mode == ToolMode::PlaneAlign ||
         measure_.mode == ToolMode::SphereFit || measure_.mode == ToolMode::SphereBodyFit ||
         measure_.mode == ToolMode::CircleFit ||
         measure_.mode == ToolMode::CylinderFit || measure_.mode == ToolMode::Flatness ||
         measure_.mode == ToolMode::StepGap) &&
        !measure_.flatness.valid && !measure_.stepGap.hasDistances && !sphereFitCompare &&
        !circleFitCompare;
    params.highlightRoi = showRoi ? &measure_.roiIndices : nullptr;
    params.usePointColors =
        (cloud.colors.size() == cloud.points.size()) &&
        (cloud.vertexColors ||
         (isActive && (filterCompareActive_ || measure_.flatness.valid ||
                       measure_.stepGap.hasDistances || useIntensityColors_ || sphereFitCompare ||
                       circleFitCompare)));
    params.ignoreMask = false;

    std::vector<uint8_t> maskBackup;
    if (isActive && filterCompareActive_ && filterKeepMask_.size() == cloud.points.size()) {
        maskBackup = cloud.mask;
        const auto wasVisibleBefore = [&](std::size_t i) {
            return filterBackupMask_.empty() || filterBackupMask_[i];
        };
        switch (filterCompareViewMode_) {
            case FilterCompareViewMode::Compare:
                params.ignoreMask = true;
                break;
            case FilterCompareViewMode::Original:
                if (!filterBackupMask_.empty()) cloud.mask = filterBackupMask_;
                break;
            case FilterCompareViewMode::FilteredOnly:
                cloud.mask = filterKeepMask_;
                break;
            case FilterCompareViewMode::RemovedOnly:
                cloud.mask.resize(cloud.points.size());
                for (std::size_t i = 0; i < cloud.points.size(); ++i) {
                    cloud.mask[i] =
                        wasVisibleBefore(i) && !filterKeepMask_[i] ? static_cast<uint8_t>(1) : 0;
                }
                break;
        }
    }

    layer.gpuPointCount = layer.renderer.Upload(cloud, params, &layer.displayIndices);
    if (!maskBackup.empty()) cloud.mask = std::move(maskBackup);
    layer.needUpload = false;
}

void Application::RefreshAllLayerGpu() {
    if (sceneLayers_.empty()) {
        sceneGpuDirty_ = false;
        return;
    }
    for (int i = 0; i < static_cast<int>(sceneLayers_.size()); ++i) {
        SceneCloudLayer& layer = sceneLayers_[static_cast<std::size_t>(i)];
        const bool isActive = (i == activeLayerIndex_);
        if (layer.needUpload || sceneGpuDirty_ || isActive) {
            RefreshLayerGpu(layer, isActive);
        }
    }
    UpdateOverlays();
    sceneGpuDirty_ = false;
}

void Application::RefreshGpu() { RefreshAllLayerGpu(); }

void Application::RebuildAnalysisColors() {
    if (MainCloud().points.empty()) {
        MainCloud().colors.clear();
        return;
    }

    if (filterCompareActive_ && filterKeepMask_.size() == MainCloud().points.size()) {
        MainCloud().colors.resize(MainCloud().points.size());
        const auto wasVisibleBefore = [&](std::size_t i) {
            return filterBackupMask_.empty() || filterBackupMask_[i];
        };
        if (filterCompareViewMode_ == FilterCompareViewMode::Original ||
            filterCompareViewMode_ == FilterCompareViewMode::FilteredOnly) {
            MainCloud().ApplyHeightColors(zMin_, zMax_);
            return;
        }
        if (filterCompareViewMode_ == FilterCompareViewMode::RemovedOnly) {
            MainCloud().ApplyHeightColors(zMin_, zMax_);
            for (std::size_t i = 0; i < MainCloud().points.size(); ++i) {
                if (wasVisibleBefore(i) && !filterKeepMask_[i]) {
                    MainCloud().colors[i] = {0.95f, 0.28f, 0.22f};
                }
            }
            return;
        }
        for (std::size_t i = 0; i < MainCloud().points.size(); ++i) {
            if (filterKeepMask_[i]) {
                MainCloud().colors[i] = {0.20f, 0.85f, 0.95f};
            } else {
                MainCloud().colors[i] = {0.95f, 0.28f, 0.22f};
            }
        }
        return;
    }

    if (measure_.mode == ToolMode::Flatness && measure_.flatness.valid) {
        MainCloud().colors.assign(MainCloud().points.size(), Vec3{0.22f, 0.24f, 0.26f});
        const float span = std::max(measure_.flatness.peakToValley, 1e-6f);
        const float mid = 0.5f * (measure_.flatness.minDev + measure_.flatness.maxDev);
        for (std::size_t k = 0; k < measure_.flatness.indices.size(); ++k) {
            const std::size_t idx = measure_.flatness.indices[k];
            if (idx >= MainCloud().points.size()) continue;
            const float t = 0.5f + (measure_.flatness.signedDist[k] - mid) / span;
            MainCloud().colors[idx] = DivergingColor(t);
        }
        return;
    }

    if (measure_.mode == ToolMode::StepGap && measure_.stepGap.hasDistances) {
        MainCloud().colors.assign(MainCloud().points.size(), Vec3{0.20f, 0.22f, 0.24f});
        for (std::size_t idx : measure_.stepGap.regionA) {
            if (idx < MainCloud().colors.size()) MainCloud().colors[idx] = {0.95f, 0.85f, 0.20f};  // A 黄
        }
        const float amax = std::max(std::fabs(measure_.stepGap.minDist),
                                    std::fabs(measure_.stepGap.maxDist));
        const float span = std::max(amax * 2.f, 1e-6f);
        for (std::size_t k = 0; k < measure_.stepGap.regionB.size(); ++k) {
            const std::size_t idx = measure_.stepGap.regionB[k];
            if (idx >= MainCloud().colors.size()) continue;
            const float t = 0.5f + measure_.stepGap.signedDistB[k] / span;
            MainCloud().colors[idx] = DivergingColor(t);
        }
        return;
    }

    if (IsSphereFitMode(measure_.mode) && measure_.sphere &&
        !measure_.sphere->inlierIndices.empty()) {
        MainCloud().colors.assign(MainCloud().points.size(), Vec3{0.52f, 0.55f, 0.60f});
        for (std::size_t idx : measure_.sphere->inlierIndices) {
            if (idx < MainCloud().colors.size()) MainCloud().colors[idx] = {0.20f, 0.82f, 0.98f};
        }
        return;
    }

    if (measure_.mode == ToolMode::CircleFit && measure_.circle &&
        !measure_.circle->inlierIndices.empty() && !DualCloudViewActive()) {
        MainCloud().colors.assign(MainCloud().points.size(), Vec3{0.52f, 0.55f, 0.60f});
        for (std::size_t idx : measure_.circle->inlierIndices) {
            if (idx < MainCloud().colors.size()) MainCloud().colors[idx] = {1.0f, 0.55f, 0.15f};
        }
        return;
    }

    if (useIntensityColors_ && intensityColors_.size() == MainCloud().points.size()) {
        MainCloud().colors = intensityColors_;
        return;
    }

    // 默认高度色
    MainCloud().ApplyHeightColors(zMin_, zMax_);
}

void Application::RunFilterPreview(int type, AlgorithmBackend backend) {
    (void)backend;
    const AlgorithmBackend active = EffectiveAlgoBackend();
    if (MainCloud().points.empty()) {
        SetStatus(u8"请先加载点云再滤波");
        return;
    }
    if (MainCloud().mask.size() != MainCloud().points.size()) MainCloud().ResetMask();

    filterBackupMask_ = MainCloud().mask;
    std::string error;
    int kept = 0;
    bool ok = false;
    const bool usePcl = active == AlgorithmBackend::PCL;
    if (type == 0) {
        ok = usePcl ? PclTools::VoxelDownsample(MainCloud(), filterVoxelLeaf_, filterKeepMask_, error,
                                                &kept)
                    : FilterTools::VoxelDownsample(MainCloud(), filterVoxelLeaf_, filterKeepMask_, error,
                                                   &kept);
    } else if (type == 1) {
        ok = usePcl ? PclTools::RadiusOutlier(MainCloud(), filterRadius_, filterRadiusMinNeighbors_,
                                              filterKeepMask_, error, &kept)
                    : FilterTools::RadiusOutlier(MainCloud(), filterRadius_, filterRadiusMinNeighbors_,
                                                 filterKeepMask_, error, &kept);
    } else {
        ok = usePcl ? PclTools::StatisticalOutlier(MainCloud(), filterStatMeanK_, filterStatStdMul_,
                                                   filterKeepMask_, error, &kept)
                    : FilterTools::StatisticalOutlier(MainCloud(), filterStatMeanK_, filterStatStdMul_,
                                                      filterKeepMask_, error, &kept);
    }
    if (!ok) {
        SetStatus(error);
        return;
    }

    filterLastKept_ = kept;
    filterLastRemoved_ = static_cast<int>(MainCloud().VisibleCount()) - kept;
    if (filterLastRemoved_ < 0) filterLastRemoved_ = 0;
    filterCompareActive_ = true;
    filterCompareViewMode_ = FilterCompareViewMode::Compare;
    sceneGpuDirty_ = true;

    char buf[192];
    std::snprintf(buf, sizeof(buf),
                  u8"%s 滤波预览：保留 %d，滤除 %d（可切换对比/原始/滤波后/差异）",
                  AlgorithmBackendLabel(active), filterLastKept_, filterLastRemoved_);
    SetStatus(buf);
}

bool Application::FitPlaneWithBackend(const std::vector<std::size_t>& indices, PlaneModel& plane,
                                      std::string& error, AlgorithmBackend backend) {
    (void)backend;
    const AlgorithmBackend active = EffectiveAlgoBackend();
    if (active == AlgorithmBackend::PCL) {
        return PclTools::FitPlaneRANSAC(EditableCloud(), indices, pclToolsPanel_.PlaneDistThresh(),
                                        pclToolsPanel_.PlaneMaxIter(), plane, error);
    }
    return MeasureTools::FitPlaneSVD(EditableCloud(), indices, plane, error);
}

AlgorithmBackend Application::EffectiveAlgoBackend() const {
    if (algoBackend_ == AlgorithmBackend::Native && !nativeAlgoUnlocked_) {
        return AlgorithmBackend::PCL;
    }
    return algoBackend_;
}

bool Application::FitSphereWithBackend(const std::vector<std::size_t>& indices,
                                       SphereModel& sphere, std::string& error,
                                       AlgorithmBackend backend) {
    (void)backend;
    const AlgorithmBackend active = EffectiveAlgoBackend();
    if (active == AlgorithmBackend::PCL) {
        return PclTools::FitSphereRANSAC(EditableCloud(), indices, pclToolsPanel_.PlaneDistThresh(),
                                         pclToolsPanel_.PlaneMaxIter(), sphere, error);
    }
    return MeasureTools::FitSphere(EditableCloud(), indices, sphere, error);
}

bool Application::FitCircleWithBackend(const std::vector<std::size_t>& indices, CircleModel& circle,
                                       std::string& error, AlgorithmBackend backend) {
    (void)backend;
    const AlgorithmBackend active = EffectiveAlgoBackend();
    if (active == AlgorithmBackend::PCL) {
        return PclTools::FitCircleRANSAC(EditableCloud(), indices, pclToolsPanel_.PlaneDistThresh(),
                                         pclToolsPanel_.PlaneMaxIter(), circle, error);
    }
    return MeasureTools::FitCircle3D(EditableCloud(), indices, circle, error);
}

bool Application::FitCylinderWithBackend(const std::vector<std::size_t>& indices,
                                         CylinderModel& cylinder, std::string& error,
                                         AlgorithmBackend backend) {
    (void)backend;
    const AlgorithmBackend active = EffectiveAlgoBackend();
    if (active == AlgorithmBackend::PCL) {
        return PclTools::FitCylinderRANSAC(EditableCloud(), indices, pclToolsPanel_.PlaneDistThresh(),
                                           pclToolsPanel_.PlaneMaxIter(), cylinder, error);
    }
    return MeasureTools::FitCylinder(EditableCloud(), indices, cylinder, error);
}

bool Application::ComputeFlatnessWithBackend(const std::vector<std::size_t>& indices,
                                           FlatnessResult& out, std::string& error,
                                           AlgorithmBackend backend) {
    (void)backend;
    const AlgorithmBackend active = EffectiveAlgoBackend();
    if (active == AlgorithmBackend::PCL) {
        return PclTools::ComputeFlatness(EditableCloud(), indices, pclToolsPanel_.PlaneDistThresh(),
                                         pclToolsPanel_.PlaneMaxIter(), out, error);
    }
    return MeasureTools::ComputeFlatness(EditableCloud(), indices, out, error);
}

bool Application::ComputeStepGapZHeightWithBackend(const std::vector<std::size_t>& regionA,
                                                   const std::vector<std::size_t>& regionB,
                                                   StepGapResult& out, std::string& error,
                                                   AlgorithmBackend backend) {
    (void)backend;
    const AlgorithmBackend active = EffectiveAlgoBackend();
    if (active == AlgorithmBackend::PCL) {
        return PclTools::ComputeStepGapZHeight(EditableCloud(), regionA, regionB, out, error);
    }
    return MeasureTools::ComputeStepGapZHeight(EditableCloud(), regionA, regionB, out, error);
}

bool Application::ExtractSectionWithBackend(bool cutAlongX, float position, float thickness,
                                          SectionData& out, std::string& error,
                                          AlgorithmBackend backend) {
    (void)backend;
    const AlgorithmBackend active = EffectiveAlgoBackend();
    if (active == AlgorithmBackend::PCL) {
        return PclTools::ExtractSection(EditableCloud(), cutAlongX, position, thickness, out, error);
    }
    return MeasureTools::ExtractSection(EditableCloud(), cutAlongX, position, thickness, out, error);
}

void Application::SelectRoiWithBackend(int fbW, int fbH, float x0, float y0, float x1, float y1,
                                       std::vector<std::size_t>& outIndices, RoiShape shape,
                                       bool useWorldSize, float worldRadius, float worldHalfW,
                                       float worldHalfH, const Vec3& worldCenter,
                                       const std::vector<float>* polyX,
                                       const std::vector<float>* polyY) {
    (void)EffectiveAlgoBackend();
    outIndices.clear();
    PointCloud& cloud = EditableCloud();
    if (shape == RoiShape::FreePolygon && polyX && polyY) {
        MeasureTools::SelectRoiPolygon(cloud, camera_, fbW, fbH, *polyX, *polyY, outIndices);
        return;
    }
    if (useWorldSize) {
        if (shape == RoiShape::Circle) {
            MeasureTools::SelectRoiWorldCircleXY(cloud, camera_, fbW, fbH, worldCenter,
                                                 worldRadius, outIndices);
        } else {
            MeasureTools::SelectRoiWorldRectXY(cloud, camera_, fbW, fbH, worldCenter, worldHalfW,
                                               worldHalfH, outIndices);
        }
        return;
    }
    if (shape == RoiShape::Circle) {
        const float cx = x0;
        const float cy = y0;
        const float r = std::hypot(x1 - x0, y1 - y0);
        MeasureTools::SelectRoiCircle(cloud, camera_, fbW, fbH, cx, cy, r, outIndices);
        return;
    }
    MeasureTools::SelectRoi(cloud, camera_, fbW, fbH, x0, y0, x1, y1, outIndices);
}

bool Application::MeasureHoleRadiusWithBackend(HoleMeasureResult& out, std::string& error) {
    if (measure_.roiIndices.empty()) {
        error = u8"请先用 ROI 框选孔缘环带区域，再测量孔径";
        return false;
    }
    if (EffectiveAlgoBackend() == AlgorithmBackend::PCL) {
        return PclTools::MeasureHoleRadius(MainCloud(), measure_.roiIndices, pclToolsPanel_.PlaneDistThresh(),
                                           pclToolsPanel_.PlaneMaxIter(), out, error);
    }
    return MeasureTools::MeasureHoleRadius(MainCloud(), measure_.roiIndices, out, error);
}

bool Application::DualCloudViewActive() const {
    return dualCloudView_ && !filledCloud_.points.empty();
}

PointCloud& Application::EditableCloud() {
    return (activeCloudPane_ == 1 && DualCloudViewActive()) ? filledCloud_ : MainCloud();
}

const PointCloud& Application::EditableCloud() const {
    return (activeCloudPane_ == 1 && DualCloudViewActive()) ? filledCloud_ : MainCloud();
}

void Application::CloseDualCloudView() {
    dualCloudView_ = false;
    activeCloudPane_ = 0;
    filledCloud_.Clear();
    displayFilledIndices_.clear();
    gpuFilledPointCount_ = 0;
    needUploadFilled_ = false;
    filledRenderer_.ClearFitWireOverlay();
}

int Application::CloudPaneAtMouse(float mouseX) const {
    if (!DualCloudViewActive()) return 0;
    if (mouseX >= view3dPane1X_ && mouseX < view3dPane1X_ + view3dPane1W_) return 1;
    return activeCloudPane_;
}

void Application::GetCloudPaneFbRect(int pane, int& x, int& y, int& w, int& h) const {
    int winW = 0, winH = 0;
    if (window_) glfwGetWindowSize(window_, &winW, &winH);
    const float sx = (winW > 0) ? static_cast<float>(fbW_) / static_cast<float>(winW) : 1.f;
    const float sy = (winH > 0) ? static_cast<float>(fbH_) / static_cast<float>(winH) : 1.f;
    const float px = (pane == 1 && DualCloudViewActive()) ? view3dPane1X_ : view3dPane0X_;
    const float pw = (pane == 1 && DualCloudViewActive()) ? view3dPane1W_ : view3dPane0W_;
    x = static_cast<int>(std::lround(px * sx));
    y = static_cast<int>(std::lround(view3dY_ * sy));
    w = std::max(1, static_cast<int>(std::lround(pw * sx)));
    h = std::max(1, static_cast<int>(std::lround(view3dH_ * sy)));
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x + w > fbW_) w = std::max(1, fbW_ - x);
    if (y + h > fbH_) h = std::max(1, fbH_ - y);
}

void Application::GetCloudPaneGlViewport(int pane, int& x, int& y, int& w, int& h) const {
    int fx = 0, fy = 0;
    GetCloudPaneFbRect(pane, fx, fy, w, h);
    x = fx;
    y = fbH_ - fy - h;
    if (y < 0) y = 0;
}

float Application::CloudPaneAspect(int pane) const {
    int x = 0, y = 0, w = 0, h = 0;
    GetCloudPaneFbRect(pane, x, y, w, h);
    return static_cast<float>(w) / static_cast<float>(std::max(h, 1));
}

void Application::ResetFitRoiSelection() {
    measure_.roiShape = RoiShape::Rect;
    measure_.roiUseWorldSize = false;
    measure_.roiHasWorldCenter = false;
    measure_.roiIndices.clear();
    measure_.roiPolyX.clear();
    measure_.roiPolyY.clear();
    measure_.roiPolyBuilding = false;
    measure_.roiX0 = measure_.roiX1 = 0.f;
    measure_.roiY0 = measure_.roiY1 = 0.f;
}

void Application::DrawFitRoiShapeControls() {
    ImGui::TextDisabled(u8"框选形状（仅作用于当前编辑点云）");
    int shape = static_cast<int>(measure_.roiShape);
    if (ImGui::RadioButton(u8"矩形", shape == 0)) measure_.roiShape = RoiShape::Rect;
    ImGui::SameLine();
    if (ImGui::RadioButton(u8"圆形", shape == 1)) measure_.roiShape = RoiShape::Circle;
    ImGui::SameLine();
    if (ImGui::RadioButton(u8"自由多边形", shape == 2)) {
        measure_.roiShape = RoiShape::FreePolygon;
    }
    if (measure_.roiShape == RoiShape::FreePolygon) {
        ImGui::Text(u8"顶点数: %zu", measure_.roiPolyX.size());
        if (ImGui::Button(u8"完成多边形", ImVec2(-1, 0))) FinishRoiPolygon();
        if (ImGui::Button(u8"清除多边形顶点", ImVec2(-1, 0))) {
            measure_.roiPolyX.clear();
            measure_.roiPolyY.clear();
            measure_.roiPolyBuilding = false;
        }
    }
    ImGui::Text(u8"当前框选: %zu 点", measure_.roiIndices.size());
}

void Application::BuildFilledPaneDisplayCloud(PointCloud& out) {
    out.Clear();
    if (!DualCloudViewActive()) return;

    std::unordered_set<std::size_t> filledInliers;
    if (measure_.mode == ToolMode::CircleFit && measure_.circle &&
        !measure_.circle->inlierIndices.empty()) {
        filledInliers.reserve(measure_.circle->inlierIndices.size() * 2 + 1);
        for (std::size_t idx : measure_.circle->inlierIndices) filledInliers.insert(idx);
    }

    out.points.reserve(MainCloud().points.size() + filledCloud_.points.size());
    out.colors.reserve(out.points.capacity());

    for (std::size_t i = 0; i < MainCloud().points.size(); ++i) {
        if (!MainCloud().mask.empty() && !MainCloud().mask[i]) continue;
        if (!dbFilledRefVisible_) continue;
        out.points.push_back(MainCloud().points[i]);
        out.colors.push_back({0.42f, 0.45f, 0.50f});
    }

    for (std::size_t i = 0; i < filledCloud_.points.size(); ++i) {
        if (!filledCloud_.mask.empty() && !filledCloud_.mask[i]) continue;
        if (!dbFilledResultVisible_) continue;
        out.points.push_back(filledCloud_.points[i]);
        if (filledInliers.count(i)) {
            out.colors.push_back({1.0f, 0.55f, 0.15f});
        } else {
            out.colors.push_back({0.15f, 0.88f, 0.82f});
        }
    }
    out.ResetMask();
    out.RecomputeBounds();
    out.sourcePath = u8"<填充视区显示>";
}

void Application::RefreshGpuFilled() {
    if (!DualCloudViewActive()) return;

    PointCloud display;
    BuildFilledPaneDisplayCloud(display);
    if (display.bounds.Valid()) {
        filledZMin_ = display.bounds.min.z;
        filledZMax_ = display.bounds.max.z;
    }

    UploadParams params;
    params.maxDisplayPoints = maxDisplayPoints_;
    params.zMin = filledZMin_;
    params.zMax = filledZMax_;
    params.usePointColors = true;
    params.highlightRoi = nullptr;
    displayFilledIndices_.clear();
    gpuFilledPointCount_ = filledRenderer_.Upload(display, params, nullptr);
    needUploadFilled_ = false;
}

void Application::ReparentCloudToTargetFrame(PointCloud& src) const {
    if (MainCloud().points.empty()) return;
    for (Vec3& p : src.points) {
        const Vec3 w = src.ToWorld(p);
        p = w - MainCloud().originOffset;
    }
    src.originOffset = MainCloud().originOffset;
    src.RecomputeBounds();
}

void Application::ClearIcpState() {
    icpSourceCloud_.Clear();
    icpAlignedSource_.Clear();
    icpSourceLoaded_ = false;
    icpOverlayActive_ = false;
    icpShowAligned_ = false;
    icpResult_ = {};
    needUploadIcp_ = false;
    icpRenderer_.ClearFitWireOverlay();
}

void Application::LoadIcpSourceCloud() {
    if (MainCloud().points.empty()) {
        SetStatus(u8"请先加载目标点云（主点云）");
        return;
    }
    const std::string path = FileDialog::OpenPointCloudFile();
    if (path.empty()) return;
    PointCloud loaded;
    std::string error;
    if (!PointCloudIO::Load(path, loaded, error)) {
        SetStatus(error);
        return;
    }
    ReparentCloudToTargetFrame(loaded);
    icpSourceCloud_ = std::move(loaded);
    icpAlignedSource_.Clear();
    icpSourceLoaded_ = true;
    icpOverlayActive_ = true;
    icpShowAligned_ = false;
    icpResult_ = {};
    if (MainCloud().bounds.Valid()) {
        icpParams_.maxCorrespondenceDist = MainCloud().bounds.Diagonal() * 0.05f;
        if (icpParams_.maxCorrespondenceDist < 0.1f) icpParams_.maxCorrespondenceDist = 0.1f;
    }
    needUploadIcp_ = true;
    char buf[256];
    std::snprintf(buf, sizeof(buf), u8"已加载 ICP 源点云 %zu 点：%s", icpSourceCloud_.points.size(),
                  path.c_str());
    SetStatus(buf);
}

void Application::RunIcpRegistration() {
    if (MainCloud().points.empty()) {
        SetStatus(u8"请先加载目标点云");
        return;
    }
    if (!icpSourceLoaded_) {
        SetStatus(u8"请先加载源点云");
        return;
    }
    std::string error;
    PointCloud aligned;
    if (!PclTools::RunIcp(MainCloud(), icpSourceCloud_, icpParams_, icpResult_, aligned, error)) {
        SetStatus(error);
        return;
    }
    icpAlignedSource_ = std::move(aligned);
    icpShowAligned_ = true;
    needUploadIcp_ = true;
    char buf[192];
    std::snprintf(buf, sizeof(buf), u8"ICP %s，fitness=%.6f",
                  icpResult_.converged ? u8"已收敛" : u8"未收敛", icpResult_.fitnessScore);
    SetStatus(buf);
}

void Application::RefreshIcpGpu() {
    if (!icpOverlayActive_ || !icpSourceLoaded_) return;
    const PointCloud& src =
        (icpShowAligned_ && !icpAlignedSource_.points.empty()) ? icpAlignedSource_ : icpSourceCloud_;
    PointCloud display = src;
    display.colors.assign(display.points.size(), Vec3{1.0f, 0.45f, 0.10f});
    if (display.bounds.Valid()) {
        icpZMin_ = display.bounds.min.z;
        icpZMax_ = display.bounds.max.z;
    }
    UploadParams params;
    params.maxDisplayPoints = maxDisplayPoints_;
    params.zMin = icpZMin_;
    params.zMax = icpZMax_;
    params.usePointColors = true;
    params.highlightRoi = nullptr;
    icpRenderer_.Upload(display, params, nullptr);
    needUploadIcp_ = false;
}

void Application::DrawIcpPanel() {
    ImGui::TextWrapped(
        u8"主点云=目标；加载源点云后以橙色叠加。配准后可复制变换矩阵或保存结果。");
    ImGui::Spacing();
    if (MainCloud().points.empty()) {
        ImGui::TextColored(ImVec4(0.95f, 0.75f, 0.35f, 1.f), u8"请先加载目标点云");
        return;
    }
    ImGui::Text(u8"目标: %zu 点", MainCloud().VisibleCount());
    if (ImGui::Button(u8"加载源点云…", ImVec2(-1, 32.f))) LoadIcpSourceCloud();
    if (icpSourceLoaded_) {
        ImGui::Text(u8"源: %zu 点", icpSourceCloud_.VisibleCount());
        if (icpResult_.success) {
            ImGui::Text(u8"收敛: %s", icpResult_.converged ? u8"是" : u8"否");
            ImGui::Text(u8"fitness: %.6f", icpResult_.fitnessScore);
        }
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::TextDisabled(u8"ICP 参数");

    const auto drawParamHint = [](const char* hint) {
        ImGui::Spacing();
        ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
        const float wrapX = ImGui::GetCursorPosX() + ImGui::GetContentRegionAvail().x;
        ImGui::PushTextWrapPos(wrapX);
        ImGui::TextUnformatted(hint);
        ImGui::PopTextWrapPos();
        ImGui::PopStyleColor();
    };

    const auto drawFloatParam = [&](const char* id, const char* label, const char* hint, float* v,
                                    float step, float vMin, float vMax, const char* fmt) {
        ImGui::TextUnformatted(label);
        ImGui::SetNextItemWidth(-1.f);
        ImGui::DragFloat(id, v, step, vMin, vMax, fmt);
        drawParamHint(hint);
    };

    const auto drawIntParam = [&](const char* id, const char* label, const char* hint, int* v,
                                  int step, int vMin, int vMax) {
        ImGui::TextUnformatted(label);
        ImGui::SetNextItemWidth(-1.f);
        ImGui::DragInt(id, v, step, vMin, vMax);
        drawParamHint(hint);
    };

    drawFloatParam("##icp_max_dist", u8"最大对应距离",
                   u8"最近邻搜索半径 (mm)。两云初始位姿接近时可设小；初始偏差大时适当加大。",
                   &icpParams_.maxCorrespondenceDist, 0.01f, 1e-4f, 1e6f, "%.4f");
    drawIntParam("##icp_max_iter", u8"最大迭代", u8"ICP 迭代次数上限，通常 30～100 即可。",
                 &icpParams_.maxIterations, 1, 1, 500);
    drawFloatParam("##icp_voxel", u8"体素下采样",
                   u8"配准前下采样以加速，0 表示不下采样。大点云建议 0.5～2 mm。",
                   &icpParams_.voxelLeaf, 0.01f, 0.f, 100.f, "%.3f");

    ImGui::Checkbox(u8"质心粗对齐", &icpParams_.useCentroidInit);
    drawParamHint(u8"用质心平移作为初始位姿，适合两次扫描大致同位、仅有平移偏差时。");

    ImGui::Spacing();
    if (!icpSourceLoaded_) ImGui::BeginDisabled();
    if (ImGui::Button(u8"执行 ICP 配准", ImVec2(-1, 36.f))) RunIcpRegistration();
    if (!icpSourceLoaded_) ImGui::EndDisabled();

    if (icpSourceLoaded_ && icpResult_.success) {
        ImGui::Checkbox(u8"显示配准后源点云", &icpShowAligned_);
        if (ImGui::IsItemEdited()) needUploadIcp_ = true;
        if (ImGui::Button(u8"复制 4×4 变换矩阵", ImVec2(-1, 0))) {
            char buf[512];
            std::snprintf(buf, sizeof(buf),
                          "%.8f,%.8f,%.8f,%.8f\n"
                          "%.8f,%.8f,%.8f,%.8f\n"
                          "%.8f,%.8f,%.8f,%.8f\n"
                          "%.8f,%.8f,%.8f,%.8f",
                          icpResult_.transform[0], icpResult_.transform[1], icpResult_.transform[2],
                          icpResult_.transform[3], icpResult_.transform[4], icpResult_.transform[5],
                          icpResult_.transform[6], icpResult_.transform[7], icpResult_.transform[8],
                          icpResult_.transform[9], icpResult_.transform[10], icpResult_.transform[11],
                          icpResult_.transform[12], icpResult_.transform[13], icpResult_.transform[14],
                          icpResult_.transform[15]);
            ImGui::SetClipboardText(buf);
            SetStatus(u8"已复制变换矩阵到剪贴板");
        }
        if (ImGui::Button(u8"保存配准后源点云…", ImVec2(-1, 0))) {
            const std::string path = FileDialog::SavePointCloudFile();
            if (!path.empty()) {
                std::string error;
                const PointCloud& toSave =
                    icpAlignedSource_.points.empty() ? icpSourceCloud_ : icpAlignedSource_;
                if (PointCloudIO::Save(path, toSave, error, true)) {
                    char sbuf[320];
                    std::snprintf(sbuf, sizeof(sbuf), u8"已保存配准后点云 %s", path.c_str());
                    SetStatus(sbuf);
                } else {
                    SetStatus(error);
                }
            }
        }
    }
    if (icpSourceLoaded_) {
        if (ImGui::Button(u8"清除 ICP", ImVec2(-1, 0))) {
            ClearIcpState();
            SetStatus(u8"已清除 ICP 配准");
        }
    }
}

bool Application::RunRoiProjectFill(std::string& error) {
    if (MainCloud().points.empty()) {
        error = u8"请先加载点云";
        return false;
    }
    const bool worldCircle =
        measure_.roiShape == RoiShape::Circle && measure_.roiUseWorldSize && measure_.roiHasWorldCenter;
    if (measure_.roiIndices.empty() && !worldCircle) {
        error = u8"请先用圆形 ROI 框选区域（推荐世界尺寸圆形）";
        return false;
    }

    bool clipCircle = measure_.roiShape == RoiShape::Circle;
    Vec3 clipCenter = measure_.roiWorldCenter;
    float clipRadius = measure_.roiWorldRadius;
    if (clipCircle && !measure_.roiUseWorldSize && !measure_.roiIndices.empty()) {
        clipCenter = Vec3{0, 0, 0};
        for (std::size_t idx : measure_.roiIndices) clipCenter += MainCloud().points[idx];
        clipCenter = clipCenter / static_cast<float>(measure_.roiIndices.size());
        clipRadius = 0.f;
        for (std::size_t idx : measure_.roiIndices) {
            const Vec3 d = MainCloud().points[idx] - clipCenter;
            clipRadius = std::max(clipRadius, std::hypot(d.x, d.y));
        }
    }

    PointCloud filled;
    PlaneModel plane;
    float gridStep = 0.f;
    PushMainCloudHistory(u8"执行填充");
    const bool ok = PclTools::RoiProjectFill(MainCloud(), measure_.roiIndices, 2, measure_.roiFillGridStep,
                                             clipCircle, clipCenter, clipRadius, filled, plane,
                                             gridStep, error);
    if (!ok) return false;

    filledCloud_ = std::move(filled);
    filledCloud_.colors.clear();
    if (filledCloud_.mask.size() != filledCloud_.points.size()) filledCloud_.ResetMask();
    dualCloudView_ = true;
    activeCloudPane_ = 1;
    dbSelectedLayerId_ = sceneLayers_[static_cast<std::size_t>(activeLayerIndex_)].id;
    dbSelectedSpecial_ = DbEntityId::FilledResult;
    measure_.circle.reset();
    measure_.plane = plane;
    measure_.holeMeasure = {};
    measure_.roiIndices.clear();
    measure_.roiHasWorldCenter = false;
    sceneGpuDirty_ = true;
    needUploadFilled_ = true;
    UpdateOverlays();

    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  u8"已生成填充视图：灰=原始参考，青=填充 (%zu 点，网格 %.4f mm)\n"
                  u8"原始点仅叠加显示，框选与拟合只作用于青色填充点",
                  filledCloud_.points.size(), gridStep);
    SetStatus(buf);
    return true;
}

void Application::ApplyProjectionToAxis(int axis) {
    if (MainCloud().points.empty()) {
        SetStatus(u8"请先加载点云");
        return;
    }
    Vec3 dir{0, 0, 1};
    const char* planeName = "XY";
    if (axis == 0) {
        dir = {1, 0, 0};
        planeName = "YZ";
    }
    if (axis == 1) {
        dir = {0, 1, 0};
        planeName = "XZ";
    }
    const Vec3 origin = MainCloud().bounds.Valid() ? MainCloud().bounds.Center() : Vec3{0, 0, 0};
    PushMainCloudHistory(u8"平面投影", true);
    std::string error;
    const bool ok = PclTools::ProjectOntoAxis(MainCloud(), origin, dir, error);
    if (!ok) {
        SetStatus(error);
        return;
    }
    MainCloud().colors.clear();
    FitCameraToCloud();
    sceneGpuDirty_ = true;
    char buf[96];
    std::snprintf(buf, sizeof(buf), u8"已投影到 %s 平面（法向 %s）", planeName,
                  axis == 0 ? "X" : (axis == 1 ? "Y" : "Z"));
    SetStatus(buf);
}

void Application::AlignCloudToReferencePlane(const std::vector<std::size_t>* roiIndices,
                                             const PlaneModel* existingPlane) {
    if (MainCloud().points.empty()) {
        SetStatus(u8"请先加载点云");
        return;
    }

    PlaneModel plane;
    if (existingPlane) {
        plane = *existingPlane;
    } else {
        if (!roiIndices || roiIndices->empty()) {
            SetStatus(u8"请先在 3D 视区框选基准平面区域");
            return;
        }
        std::string fitErr;
        if (!FitPlaneWithBackend(*roiIndices, plane, fitErr, EffectiveAlgoBackend())) {
            SetStatus(fitErr);
            return;
        }
    }

    Vec3 target{0.f, 0.f, 1.f};
    const char* targetName = u8"+Z（XY 水平面）";
    if (planeAlignTarget_ == 1) {
        target = {0.f, 1.f, 0.f};
        targetName = u8"+Y";
    } else if (planeAlignTarget_ == 2) {
        target = {1.f, 0.f, 0.f};
        targetName = u8"+X";
    }

    PushMainCloudHistory(u8"平面摆正", true);
    PlaneModel alignedPlane;
    std::string error;
    const bool ok = PclTools::AlignCloudToPlaneNormal(MainCloud(), plane, target, alignedPlane, error);
    if (!ok) {
        SetStatus(error);
        return;
    }

    if (DualCloudViewActive()) CloseDualCloudView();
    measure_.plane = alignedPlane;
    MainCloud().colors.clear();
    FitCameraToCloud();
    sceneGpuDirty_ = true;
    UpdateOverlays();

    const float tiltDeg =
        std::acos(std::clamp(plane.normal.Normalized().Dot(target), -1.f, 1.f)) * 57.2957795f;
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  u8"已以基准平面摆正点云：原倾斜约 %.2f°，法向现对齐 %s（RMS=%.4f mm）", tiltDeg,
                  targetName, plane.rms);
    SetStatus(buf);
}

std::optional<std::size_t> Application::PickNearestWithBackend(
    int fbW, int fbH, float mouseX, float mouseY, float maxPixelDist,
    const std::vector<std::size_t>* onlyIndices) {
    const PointCloud& cloud = EditableCloud();
    if (EffectiveAlgoBackend() == AlgorithmBackend::PCL) {
        return PclTools::PickNearest(cloud, camera_, fbW, fbH, mouseX, mouseY, maxPixelDist,
                                     onlyIndices);
    }
    return MeasureTools::PickNearest(cloud, camera_, fbW, fbH, mouseX, mouseY, maxPixelDist,
                                     onlyIndices);
}

void Application::ApplyRoiDeleteWithBackend(const std::vector<std::size_t>& roiIndices,
                                            bool deleteInside) {
    if (EffectiveAlgoBackend() == AlgorithmBackend::PCL) {
        PclTools::ApplyRoiDelete(EditableCloud(), roiIndices, deleteInside);
    } else {
        MeasureTools::ApplyRoiDelete(EditableCloud(), roiIndices, deleteInside);
    }
}

void Application::RestoreAllPointsWithBackend() {
    PointCloud& cloud = EditableCloud();
    if (EffectiveAlgoBackend() == AlgorithmBackend::PCL) {
        PclTools::RestoreAllPoints(cloud);
    } else {
        MeasureTools::RestoreAllPoints(cloud);
    }
    if (activeCloudPane_ == 1 && DualCloudViewActive()) {
        needUploadFilled_ = true;
    } else {
        sceneGpuDirty_ = true;
    }
}

void Application::ApplyFilterResult() {
    if (!filterCompareActive_ || filterKeepMask_.size() != MainCloud().points.size()) {
        SetStatus(u8"没有可应用的滤波结果");
        return;
    }
    PushMainCloudHistory(u8"滤波");
    MainCloud().mask = filterKeepMask_;
    filterCompareActive_ = false;
    filterCompareViewMode_ = FilterCompareViewMode::Compare;
    filterKeepMask_.clear();
    filterBackupMask_.clear();
    sceneGpuDirty_ = true;
    SetStatus(std::string(u8"已应用滤波，可见 ") + std::to_string(MainCloud().VisibleCount()));
}

void Application::ClearFilterCompare() {
    if (!filterBackupMask_.empty() && filterBackupMask_.size() == MainCloud().points.size()) {
        MainCloud().mask = filterBackupMask_;
    }
    filterCompareActive_ = false;
    filterCompareViewMode_ = FilterCompareViewMode::Compare;
    filterKeepMask_.clear();
    filterBackupMask_.clear();
    sceneGpuDirty_ = true;
    SetStatus(u8"已取消滤波预览");
}

void Application::DrawFilterCompareViewControls() {
    const FilterCompareViewMode prev = filterCompareViewMode_;
    int mode = static_cast<int>(filterCompareViewMode_);
    if (ImGui::RadioButton(u8"对比", &mode, 0)) {
    }
    ImGui::SameLine();
    if (ImGui::RadioButton(u8"原始", &mode, 1)) {
    }
    ImGui::SameLine();
    if (ImGui::RadioButton(u8"滤波后", &mode, 2)) {
    }
    ImGui::SameLine();
    if (ImGui::RadioButton(u8"差异", &mode, 3)) {
    }
    if (mode != static_cast<int>(prev)) {
        filterCompareViewMode_ = static_cast<FilterCompareViewMode>(mode);
        sceneGpuDirty_ = true;
    }
    ImGui::TextDisabled(u8"对比=青绿保留+红滤除 | 原始=滤波前 | 滤波后=保留点 | 差异=被滤除点");
}

DbTreeHost Application::BuildDbTreeHost() {
    DbTreeHost host;
    host.sceneLayers.reserve(sceneLayers_.size());
    for (std::size_t i = 0; i < sceneLayers_.size(); ++i) {
        SceneCloudLayer& layer = sceneLayers_[i];
        DbSceneLayerRef ref;
        ref.id = layer.id;
        ref.cloud = &layer.cloud;
        ref.visible = &layer.visible;
        ref.gpuCount = layer.gpuPointCount;
        ref.isActive = static_cast<int>(i) == activeLayerIndex_;
        host.sceneLayers.push_back(ref);
    }
    host.filledCloud = &filledCloud_;
    host.icpSourceCloud = &icpSourceCloud_;
    host.icpAlignedCloud = &icpAlignedSource_;
    host.dualCloudActive = DualCloudViewActive();
    host.icpSourceLoaded = icpSourceLoaded_;
    host.icpHasAligned = icpResult_.success && !icpAlignedSource_.points.empty();
    host.filledRefVisible = &dbFilledRefVisible_;
    host.filledResultVisible = &dbFilledResultVisible_;
    host.icpOverlayVisible = &icpOverlayActive_;
    host.icpShowAligned = &icpShowAligned_;
    host.selectedLayerId = &dbSelectedLayerId_;
    host.selectedSpecial = &dbSelectedSpecial_;
    host.filledGpuCount = gpuFilledPointCount_;
    host.onSelectLayer = [this](int layerId) { SetActiveLayerById(layerId); };
    host.onDeleteLayer = [this](int layerId) { RemoveSceneLayerById(layerId); };
    host.onSelectSpecial = [this](DbEntityId id) {
        dbSelectedSpecial_ = id;
        if (id == DbEntityId::IcpAligned) {
            icpShowAligned_ = true;
            needUploadIcp_ = true;
        } else if (id == DbEntityId::IcpSource) {
            icpShowAligned_ = false;
            needUploadIcp_ = true;
        }
    };
    host.onVisibilityChanged = [this]() {
        sceneGpuDirty_ = true;
        needUploadFilled_ = true;
        needUploadIcp_ = true;
        for (SceneCloudLayer& layer : sceneLayers_) layer.needUpload = true;
    };
    return host;
}

void Application::DrawDbTreePanel(float height) {
    const UiPalette& pal = GetUiPalette();
    UiSectionHeader(u8"DB 树", nullptr, &pal.sectionTitle);
    ImGui::PushStyleColor(ImGuiCol_ChildBg, pal.panelRaised);
    ImGui::PushStyleColor(ImGuiCol_Border, pal.border);
    ImGui::BeginChild(u8"##db_panel", ImVec2(0, height), true);
    DbTreePanel::Draw(BuildDbTreeHost());
    ImGui::EndChild();
    ImGui::PopStyleColor(2);
}

void Application::DrawDbTreeSplitter(float maxDbHeight) {
    constexpr float kSplitHit = 6.f;
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.f, 0.f, 0.f, 0.f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.5f, 0.6f, 0.35f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.3f, 0.6f, 0.7f, 0.5f));
    ImGui::InvisibleButton(u8"##db_tree_split", ImVec2(-1.f, kSplitHit));
    ImGui::PopStyleColor(3);
    if (ImGui::IsItemHovered() || ImGui::IsItemActive()) {
        ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);
    }
    if (ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
        dbTreePreferredH_ += ImGui::GetIO().MouseDelta.y;
        dbTreePreferredH_ = std::clamp(dbTreePreferredH_, 100.f, maxDbHeight);
    }
}

void Application::DrawEmptyCloudPrompt(float contentTop, float contentH) {
    if (!sceneLayers_.empty() || view2DMode_) return;

    const float cx = view3dX_ + view3dW_ * 0.5f;
    const float cy = contentTop + contentH * 0.45f;
    const char* title = u8"打开点云文件开始查看";
    const char* hint = u8"点击此处导入点云";
    const ImVec2 titleSz = ImGui::CalcTextSize(title);
    const ImVec2 hintSz = ImGui::CalcTextSize(hint);
    const float winW = std::max(titleSz.x, hintSz.x) + 48.f;
    const float winH = titleSz.y + hintSz.y + 44.f;

    ImGui::SetNextWindowPos(ImVec2(cx - winW * 0.5f, cy - winH * 0.5f));
    ImGui::SetNextWindowSize(ImVec2(winW, winH));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 8.f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(16.f, 12.f));
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.08f, 0.11f, 0.14f, 0.92f));
    ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.24f, 0.55f, 0.65f, 0.55f));
    ImGui::Begin(u8"##empty_cloud_prompt", nullptr,
                 ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                     ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoSavedSettings);

    const UiPalette& pal = GetUiPalette();
    ImGui::PushStyleColor(ImGuiCol_Button, pal.panelRaised);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.18f, 0.42f, 0.50f, 1.f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.22f, 0.52f, 0.60f, 1.f));
    if (ImGui::Button(title, ImVec2(-1.f, 0.f))) {
        const std::string path = FileDialog::OpenPointCloudFile();
        if (!path.empty()) LoadPath(path);
    }
    ImGui::PopStyleColor(3);
    ImGui::Spacing();
    ImGui::TextColored(pal.textMuted, "%s", hint);

    ImGui::End();
    ImGui::PopStyleColor(2);
    ImGui::PopStyleVar(2);
}

AlgoToolsHost Application::BuildAlgoToolsHost(AlgorithmBackend backend) {
    AlgoToolsHost host;
    host.backend = backend;
    host.cloud = &MainCloud();
    host.measure = &measure_;
    host.currentMode = measure_.mode;
    host.canUndo = history_.CanUndo();
    host.canRedo = history_.CanRedo();
    host.filterCompareActive = filterCompareActive_;
    host.filterLastKept = filterLastKept_;
    host.filterLastRemoved = filterLastRemoved_;
    host.filterCompareViewMode = &filterCompareViewMode_;
    host.filterVoxelLeaf = &filterVoxelLeaf_;
    host.filterRadius = &filterRadius_;
    host.filterRadiusMinNeighbors = &filterRadiusMinNeighbors_;
    host.filterStatMeanK = &filterStatMeanK_;
    host.filterStatStdMul = &filterStatStdMul_;
    host.showCreateSphere = &showCreateSphere_;
    host.showCreateCylinder = &showCreateCylinder_;
    host.genSphereRadius = &genSphereRadius_;
    host.genSpherePoints = &genSpherePoints_;
    host.genSphereNoise = &genSphereNoise_;
    host.genCylRadius = &genCylRadius_;
    host.genCylHeight = &genCylHeight_;
    host.genCylPoints = &genCylPoints_;
    host.genCylNoise = &genCylNoise_;

    host.setToolMode = [this](ToolMode mode) { SetToolMode(mode); };
    host.undo = [this]() { Undo(); };
    host.redo = [this]() { Redo(); };
    host.clearVisuals = [this]() { ClearToolVisuals(true); };
    host.runFilterPreview = [this, backend](int type) {
        algoBackend_ = backend;
        RunFilterPreview(type, backend);
    };
    host.applyFilter = [this]() { ApplyFilterResult(); };
    host.clearFilter = [this]() { ClearFilterCompare(); };
    host.fitPlane = [this, backend](const std::vector<std::size_t>& indices, PlaneModel& out,
                                    std::string& err) {
        algoBackend_ = backend;
        const bool ok = FitPlaneWithBackend(indices, out, err, backend);
        return ok;
    };
    host.fitSphere = [this, backend](const std::vector<std::size_t>& indices, SphereModel& out,
                                     std::string& err) {
        algoBackend_ = backend;
        if (backend == AlgorithmBackend::PCL) {
            const bool ok = PclTools::FitSphereRANSAC(
                MainCloud(), indices, pclToolsPanel_.PlaneDistThresh(), pclToolsPanel_.PlaneMaxIter(),
                out, err);
            if (ok) {
                measure_.sphere = out;
                measure_.circle.reset();
                measure_.cylinder.reset();
                SyncFittedSphereLayer(out);
                SetToolMode(ToolMode::SphereFit);
                UpdateOverlays();
            }
            return ok;
        }
        const bool ok = MeasureTools::FitSphere(MainCloud(), indices, out, err);
        if (ok) {
            measure_.sphere = out;
            measure_.circle.reset();
            measure_.cylinder.reset();
            SyncFittedSphereLayer(out);
            SetToolMode(ToolMode::SphereFit);
            UpdateOverlays();
        }
        return ok;
    };
    host.fitCircle = [this](const std::vector<std::size_t>& indices, CircleModel& out,
                            std::string& err) {
        const bool ok = FitCircleWithBackend(indices, out, err, algoBackend_);
        if (ok) {
            measure_.circle = out;
            measure_.sphere.reset();
            measure_.cylinder.reset();
            SetToolMode(ToolMode::CircleFit);
            UpdateOverlays();
        }
        return ok;
    };
    host.fitCylinder = [this](const std::vector<std::size_t>& indices, CylinderModel& out,
                              std::string& err) {
        const bool ok = FitCylinderWithBackend(indices, out, err, algoBackend_);
        if (ok) {
            measure_.cylinder = out;
            measure_.sphere.reset();
            measure_.circle.reset();
            SetToolMode(ToolMode::CylinderFit);
            UpdateOverlays();
        }
        return ok;
    };
    host.showPlane = [this](const PlaneModel& plane) {
        measure_.plane = plane;
        SetToolMode(ToolMode::PlaneFit);
        UpdateOverlays();
    };
    host.clearPlane = [this]() {
        measure_.plane.reset();
        UpdateOverlays();
        SetStatus(u8"已清除拟合平面");
    };
    host.setStatus = [this](const std::string& s) { SetStatus(s); };
    host.requestRefreshGpu = [this]() { sceneGpuDirty_ = true; };
    return host;
}

void Application::UpdateOverlays() {
    for (SceneCloudLayer& layer : sceneLayers_) {
        if (layer.rendererReady) layer.renderer.ClearFitWireOverlay();
    }
    filledRenderer_.ClearFitWireOverlay();
    PointCloudRenderer& activeR =
        (DualCloudViewActive() && activeCloudPane_ == 1) ? filledRenderer_ : ActiveRenderer();

    if (measure_.mode == ToolMode::StepHeight && (measure_.stepA || measure_.stepB)) {
        activeR.SetDistanceOverlay(measure_.stepA, measure_.stepB);
    } else if (measure_.distA || measure_.distB) {
        activeR.SetDistanceOverlay(measure_.distA, measure_.distB);
    } else {
        activeR.SetPickOverlay(measure_.picked);
    }
    if (measure_.mode == ToolMode::Section) {
        SyncSectionCutPlane();
    } else if (measure_.mode == ToolMode::Flatness && measure_.flatness.valid) {
        activeR.SetPlaneOverlay(measure_.flatness.plane);
    } else if (measure_.mode == ToolMode::StepGap && measure_.stepGap.hasPlane) {
        activeR.SetPlaneOverlay(measure_.stepGap.planeA);
    } else if (measure_.mode == ToolMode::PlaneFit || measure_.mode == ToolMode::PlaneAlign) {
        activeR.SetPlaneOverlay(measure_.plane);
    } else {
        activeR.SetPlaneOverlay(std::nullopt);
    }

    if (IsSphereFitMode(measure_.mode) && measure_.sphere) {
        activeR.SetSphereOverlay(measure_.sphere);
    } else {
        activeR.SetSphereOverlay(std::nullopt);
    }
    if (measure_.circle &&
               (measure_.mode == ToolMode::CircleFit || measure_.mode == ToolMode::Roi)) {
        activeR.SetCircleOverlay(measure_.circle);
    } else if (measure_.mode == ToolMode::CylinderFit && measure_.cylinder) {
        activeR.SetCylinderOverlay(measure_.cylinder);
    }
    ActiveRenderer().SetAxes(showAxes_, axesLength_);
    for (SceneCloudLayer& layer : sceneLayers_) {
        if (layer.rendererReady) layer.renderer.SetAxes(showAxes_, axesLength_);
    }
    filledRenderer_.SetAxes(showAxes_, axesLength_);
}

void Application::SyncSectionCutPlane() {
    if (!MainCloud().bounds.Valid()) {
        ActiveRenderer().SetPlaneOverlay(std::nullopt);
        return;
    }
    measure_.plane = MeasureTools::MakeSectionCutPlane(
        MainCloud(), measure_.section.cutAlongX, measure_.section.position);
    // Slightly more opaque / larger hint while dragging
    ActiveRenderer().SetPlaneOverlay(measure_.plane);
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
    SetStatus(u8"拖拽截面中…松开鼠标后自动生成 2D 轮廓", false);
}

void Application::UpdateSectionDrag(float mouseX, float mouseY) {
    if (!sectionDragging_ || !MainCloud().bounds.Valid()) return;

    const Vec3 axis = measure_.section.cutAlongX ? Vec3{1, 0, 0} : Vec3{0, 1, 0};
    PlaneModel plane = MeasureTools::MakeSectionCutPlane(
        MainCloud(), measure_.section.cutAlongX, measure_.section.position);
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

    float amin = measure_.section.cutAlongX ? MainCloud().bounds.min.x : MainCloud().bounds.min.y;
    float amax = measure_.section.cutAlongX ? MainCloud().bounds.max.x : MainCloud().bounds.max.y;
    measure_.section.position = std::clamp(measure_.section.position, amin, amax);

    lastSectionMouseX_ = mouseX;
    lastSectionMouseY_ = mouseY;
    SyncSectionCutPlane();

    char buf[96];
    std::snprintf(buf, sizeof(buf), u8"截面位置 = %.4f（拖拽中）", measure_.section.position);
    SetStatus(buf, false);
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
    if (measure_.roiUseWorldSize && measure_.mode == ToolMode::Roi) RefreshWorldRoiAt(mouseX, mouseY);
}

void Application::UpdateRoiDrag(float mouseX, float mouseY) {
    if (!measure_.roiDragging) return;
    measure_.roiX1 = mouseX;
    measure_.roiY1 = mouseY;
    if (measure_.roiUseWorldSize && measure_.mode == ToolMode::Roi) RefreshWorldRoiAt(mouseX, mouseY);
}

void Application::RefreshWorldRoiAt(float mouseX, float mouseY) {
    int vx = 0, vy = 0, vw = 0, vh = 0;
    GetView3dFbRect(vx, vy, vw, vh);
    const float lx = mouseX - static_cast<float>(vx);
    const float ly = mouseY - static_cast<float>(vy);

    if (const auto idx = PickNearestWithBackend(vw, vh, lx, ly, 24.f, nullptr)) {
        measure_.roiWorldCenter = EditableCloud().points[*idx];
        measure_.roiHasWorldCenter = true;
    }

    SelectRoiWithBackend(vw, vh, lx, ly, lx, ly, measure_.roiIndices, measure_.roiShape, true,
                         measure_.roiWorldRadius, measure_.roiWorldWidth * 0.5f,
                         measure_.roiWorldHeight * 0.5f, measure_.roiWorldCenter, nullptr,
                         nullptr);

    SetStatus(std::string(u8"世界尺寸框选预览 ") + std::to_string(measure_.roiIndices.size()) +
              u8" 点（松开确认）",
              false);
    if (activeCloudPane_ == 1 && DualCloudViewActive()) {
        needUploadFilled_ = true;
    } else {
        sceneGpuDirty_ = true;
    }
}

void Application::RunRoiSelection() {
    int vx = 0, vy = 0, vw = 0, vh = 0;
    GetView3dFbRect(vx, vy, vw, vh);
    const float lx0 = measure_.roiX0 - static_cast<float>(vx);
    const float ly0 = measure_.roiY0 - static_cast<float>(vy);
    const float lx1 = measure_.roiX1 - static_cast<float>(vx);
    const float ly1 = measure_.roiY1 - static_cast<float>(vy);

    if (measure_.roiUseWorldSize && measure_.mode == ToolMode::Roi) {
        const float pickX = lx1;
        const float pickY = ly1;
        if (const auto idx = PickNearestWithBackend(vw, vh, pickX, pickY, 24.f, nullptr)) {
            measure_.roiWorldCenter = EditableCloud().points[*idx];
            measure_.roiHasWorldCenter = true;
        }
        SelectRoiWithBackend(vw, vh, pickX, pickY, pickX, pickY, measure_.roiIndices,
                             measure_.roiShape, true, measure_.roiWorldRadius,
                             measure_.roiWorldWidth * 0.5f, measure_.roiWorldHeight * 0.5f,
                             measure_.roiWorldCenter, nullptr, nullptr);
    } else {
        SelectRoiWithBackend(vw, vh, lx0, ly0, lx1, ly1, measure_.roiIndices, measure_.roiShape,
                             false, measure_.roiWorldRadius, measure_.roiWorldWidth * 0.5f,
                             measure_.roiWorldHeight * 0.5f, measure_.roiWorldCenter, nullptr,
                             nullptr);
    }

    if (measure_.mode == ToolMode::StepGap) {
        auto& sg = measure_.stepGap;
        if (sg.hasPlane || sg.phase == StepGapPhase::SelectB || sg.phase == StepGapPhase::Done) {
            sg.regionB = measure_.roiIndices;
            sg.hasDistances = false;
            sg.signedDistB.clear();
            sg.phase = StepGapPhase::SelectB;
            SetStatus(std::string(u8"段差区域 B：") + std::to_string(sg.regionB.size()) + u8" 点，请计算段差");
        } else {
            sg.regionA = measure_.roiIndices;
            sg.hasPlane = false;
            sg.hasDistances = false;
            sg.regionB.clear();
            sg.signedDistB.clear();
            sg.phase = StepGapPhase::SelectA;
            SetStatus(std::string(u8"段差区域 A：") + std::to_string(sg.regionA.size()) + u8" 点，请拟合平面");
        }
    } else if (measure_.mode == ToolMode::Flatness) {
        measure_.flatness = {};
        SetStatus(std::string(u8"平面度框选 ") + std::to_string(measure_.roiIndices.size()) + u8" 个点");
    } else {
        SetStatus(std::string(u8"已框选 ") + std::to_string(measure_.roiIndices.size()) + u8" 个点");
    }

    measure_.roiX0 = measure_.roiX1 = 0.f;
    measure_.roiY0 = measure_.roiY1 = 0.f;
    if (activeCloudPane_ == 1 && DualCloudViewActive()) {
        needUploadFilled_ = true;
    } else {
        sceneGpuDirty_ = true;
    }
}

void Application::FinishRoiPolygon() {
    if (measure_.roiPolyX.size() < 3) {
        SetStatus(u8"自由多边形至少需要 3 个顶点");
        return;
    }
    int vx = 0, vy = 0, vw = 0, vh = 0;
    GetView3dFbRect(vx, vy, vw, vh);
    SelectRoiWithBackend(vw, vh, 0.f, 0.f, 0.f, 0.f, measure_.roiIndices, RoiShape::FreePolygon,
                         false, 0.f, 0.f, 0.f, measure_.roiWorldCenter, &measure_.roiPolyX,
                         &measure_.roiPolyY);
    measure_.roiPolyBuilding = false;
    SetStatus(std::string(u8"多边形框选 ") + std::to_string(measure_.roiIndices.size()) + u8" 个点");
    if (activeCloudPane_ == 1 && DualCloudViewActive()) {
        needUploadFilled_ = true;
    } else {
        sceneGpuDirty_ = true;
    }
}

void Application::EndRoiDrag() {
    if (!measure_.roiDragging) return;
    measure_.roiDragging = false;
    RunRoiSelection();
}

void Application::GenerateSection() {
    std::string error;
    if (!ExtractSectionWithBackend(measure_.section.cutAlongX, measure_.section.position,
                                   measure_.section.thickness, measure_.section, error,
                                   algoBackend_)) {
        SetStatus(error);
        SyncSectionCutPlane();
        return;
    }
    measure_.section.pickA.reset();
    measure_.section.pickB.reset();
    measure_.section.lineDistance = 0.f;
    measure_.section.zDistance = 0.f;
    SyncSectionCutPlane();
    showSectionPanel_ = true;
    char buf[160];
    std::snprintf(buf, sizeof(buf), u8"截面生成成功：%zu 个轮廓点（位置=%.4f）",
                  measure_.section.points.size(), measure_.section.position);
    SetStatus(buf);
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
    SetStatus(buf);
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
        SetStatus(u8"拖动点1中…", false);
        return;
    }
    if (HitSectionPickMarker(plotX, plotY, plotW, plotH, false)) {
        sectionPlotDragTarget_ = 2;
        SetStatus(u8"拖动点2中…", false);
        return;
    }

    float distPx = 0.f;
    const auto best = FindNearestSectionPoint(plotX, plotY, plotW, plotH, &distPx);
    if (!best || distPx > 12.f) {
        SetStatus(u8"截面图上未点到轮廓点");
        return;
    }

    if (!sec.pickA || sec.pickB) {
        sec.pickA = *best;
        sec.pickB.reset();
        sec.lineDistance = 0.f;
        sec.zDistance = 0.f;
        sectionPlotDragTarget_ = 1;
        SetStatus(u8"已选点1，可拖动调整；再点选点2");
    } else {
        sec.pickB = *best;
        sectionPlotDragTarget_ = 2;
        UpdateSectionDistances();
        SetStatus(std::string(measure_.status) + u8"（可拖动点1/点2实时调整）", false);
    }
}

void Application::OnLeftClick(float mouseX, float mouseY) {
    if (MainCloud().points.empty()) return;
    if (measure_.mode == ToolMode::Navigate) return;
    if (measure_.mode == ToolMode::Roi || measure_.mode == ToolMode::PlaneFit ||
        measure_.mode == ToolMode::PlaneAlign ||
        IsSphereFitMode(measure_.mode) || measure_.mode == ToolMode::CircleFit ||
        measure_.mode == ToolMode::CylinderFit || measure_.mode == ToolMode::Flatness ||
        measure_.mode == ToolMode::StepGap)
        return;
    if (measure_.mode == ToolMode::Section) return;

    int vx = 0, vy = 0, vw = 0, vh = 0;
    GetView3dFbRect(vx, vy, vw, vh);
    const std::vector<std::size_t>* disp =
        DualCloudViewActive()
            ? nullptr
            : (ActiveLayer().displayIndices.empty() ? nullptr : &ActiveLayer().displayIndices);
    const auto idx = PickNearestWithBackend(
        vw, vh, mouseX - static_cast<float>(vx), mouseY - static_cast<float>(vy), 12.f, disp);
    if (!idx) {
        SetStatus(u8"光标附近没有点");
        return;
    }
    const Vec3 pLocal = EditableCloud().points[*idx];
    const Vec3 p = EditableCloud().ToWorld(pLocal);

    if (measure_.mode == ToolMode::Pick) {
        measure_.picked = pLocal;
        measure_.distA.reset();
        measure_.distB.reset();
        char buf[160];
        std::snprintf(buf, sizeof(buf), u8"选中点 [#%zu]  X=%.4f  Y=%.4f  Z=%.4f", *idx, p.x, p.y,
                      p.z);
        SetStatus(buf);
        UpdateOverlays();
    } else if (measure_.mode == ToolMode::Distance) {
        measure_.picked.reset();
        if (!measure_.distA || measure_.distB) {
            measure_.distA = pLocal;
            measure_.distB.reset();
            SetStatus(u8"测距: 已标记第 1 点（黄），请再点第 2 点");
        } else {
            measure_.distB = pLocal;
            measure_.distance = (*measure_.distB - *measure_.distA).Length();
            char buf[128];
            std::snprintf(buf, sizeof(buf), u8"距离 = %.6f", measure_.distance);
            SetStatus(buf);
        }
        UpdateOverlays();
    } else if (measure_.mode == ToolMode::StepHeight) {
        measure_.picked.reset();
        measure_.distA.reset();
        measure_.distB.reset();
        if (!measure_.stepA || measure_.stepB) {
            measure_.stepA = pLocal;
            measure_.stepB.reset();
            measure_.stepDeltaZ = 0.f;
            SetStatus(u8"台阶: 已选基准点A，请再选测量点B");
        } else {
            measure_.stepB = pLocal;
            measure_.stepDeltaZ = measure_.stepB->z - measure_.stepA->z;
            const Vec3 wa = MainCloud().ToWorld(*measure_.stepA);
            const Vec3 wb = MainCloud().ToWorld(*measure_.stepB);
            char buf[192];
            std::snprintf(buf, sizeof(buf),
                          u8"台阶高度 ΔZ = %.4f mm  (A.Z=%.4f, B.Z=%.4f)", measure_.stepDeltaZ,
                          wa.z, wb.z);
            SetStatus(buf);
        }
        UpdateOverlays();
    }
}

void Application::UpdateView3dLayout(float contentTop, float contentH, float sidebarW) {
    const ImGuiViewport* vp = ImGui::GetMainViewport();
    view3dY_ = contentTop;
    view3dH_ = std::max(contentH, 1.f);
    if (view2DMode_) {
        view3dX_ = vp->Pos.x + sidebarW;
        view3dW_ = 0.f;
        view3dPane0X_ = view3dX_;
        view3dPane0W_ = 0.f;
        view3dPane1X_ = view3dX_;
        view3dPane1W_ = 0.f;
        return;
    }

    const float imageW = ImagePanelWidth();
    const float totalW = std::max(vp->Size.x - sidebarW - imageW, 1.f);
    view3dX_ = vp->Pos.x + sidebarW;
    view3dY_ = contentTop;
    view3dH_ = std::max(contentH, 1.f);
    if (DualCloudViewActive()) {
        view3dPane0X_ = view3dX_;
        view3dPane0W_ = 0.f;
        view3dPane1X_ = view3dX_;
        view3dPane1W_ = totalW;
        view3dW_ = totalW;
    } else {
        view3dPane0X_ = view3dX_;
        view3dPane0W_ = totalW;
        view3dPane1X_ = 0.f;
        view3dPane1W_ = 0.f;
        view3dW_ = totalW;
    }
}

bool Application::MouseInView3d(double mx, double my) const {
    if (my < static_cast<double>(view3dY_) || my >= static_cast<double>(view3dY_ + view3dH_)) {
        return false;
    }
    if (DualCloudViewActive()) {
        return mx >= static_cast<double>(view3dPane1X_) &&
               mx < static_cast<double>(view3dPane1X_ + view3dPane1W_);
    }
    return mx >= static_cast<double>(view3dX_) && mx < static_cast<double>(view3dX_ + view3dW_);
}

void Application::GetView3dFbRect(int& x, int& y, int& w, int& h) const {
    GetCloudPaneFbRect(activeCloudPane_, x, y, w, h);
}

void Application::GetView3dGlViewport(int& x, int& y, int& w, int& h) const {
    GetCloudPaneGlViewport(activeCloudPane_, x, y, w, h);
}

float Application::View3dAspect() const {
    return CloudPaneAspect(activeCloudPane_);
}

void Application::DrawDualCloudPaneLabels() {
    if (!DualCloudViewActive()) return;
    ImDrawList* dl = ImGui::GetBackgroundDrawList();
    const ImU32 bg = IM_COL32(12, 16, 20, 200);
    const ImU32 fg = IM_COL32(220, 235, 245, 255);
    const ImU32 hi = IM_COL32(255, 200, 80, 255);

    auto drawLabel = [&](float px, float pw, const char* text, bool active) {
        const ImVec2 ts = ImGui::CalcTextSize(text);
        const float tx = px + 10.f;
        const float ty = view3dY_ + 8.f;
        dl->AddRectFilled(ImVec2(tx - 4.f, ty - 2.f), ImVec2(tx + ts.x + 4.f, ty + ts.y + 2.f),
                          bg, 4.f);
        dl->AddText(ImVec2(tx, ty), active ? hi : fg, text);
    };
    drawLabel(view3dPane1X_, view3dPane1W_, u8"投影填充（灰=原始参考，青=填充）", true);
}

void Application::DrawIcpOverlayLabel() {
    if (!icpOverlayActive_ || !icpSourceLoaded_ || DualCloudViewActive()) return;
    ImDrawList* dl = ImGui::GetBackgroundDrawList();
    const ImU32 bg = IM_COL32(12, 16, 20, 200);
    const ImU32 fg = IM_COL32(255, 200, 80, 255);
    const char* text =
        icpShowAligned_ ? u8"ICP：主=目标，橙=配准后源" : u8"ICP：主=目标，橙=源（未配准）";
    const ImVec2 ts = ImGui::CalcTextSize(text);
    const float tx = view3dX_ + 10.f;
    const float ty = view3dY_ + 8.f;
    dl->AddRectFilled(ImVec2(tx - 4.f, ty - 2.f), ImVec2(tx + ts.x + 4.f, ty + ts.y + 2.f), bg,
                      4.f);
    dl->AddText(ImVec2(tx, ty), fg, text);
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
        const PointCloud& cloud = EditableCloud();
        if (idx >= cloud.points.size()) continue;
        if (!cloud.mask.empty() && !cloud.mask[idx]) continue;
        float sx = 0.f, sy = 0.f;
        if (!ProjectWorldToScreen(cloud.points[idx], sx, sy)) continue;
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

    // 算法编辑器 / 模板匹配 / OCR 打开时不处理点云视区交互
    if (algoEditor_.IsVisible() || halconMatchWindow_.IsVisible() || ocrWindow_.IsVisible() ||
        camera2DCalibWindow_.IsVisible() || cameraIntrinsicsCalibWindow_.IsVisible() ||
        multiViewGeometryWindow_.IsVisible())
        return;

    if (view2DMode_) {
        const bool ctrl = glfwGetKey(window_, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS ||
                          glfwGetKey(window_, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS;
        static bool zWas2d = false;
        const bool zDown = glfwGetKey(window_, GLFW_KEY_Z) == GLFW_PRESS;
        if (ctrl && zDown && !zWas2d) Undo2DOrCloud();
        zWas2d = zDown;
        return;
    }

    // HandleInput 在 DrawUi 之前执行，优先复用上一帧布局，避免框选坐标与视区错位
    if (cachedViewLayoutValid_) {
        UpdateView3dLayout(cachedContentTop_, cachedContentH_, cachedSidebarW_);
    } else {
        const ImGuiViewport* vp = ImGui::GetMainViewport();
        const float sidebarW = SidebarWidth();
        constexpr float toolbarH = 44.f;
        const float menuH = MenuBarFrameHeight();
        const float contentTop = vp->Pos.y + menuH + toolbarH;
        const float contentH = vp->Pos.y + vp->Size.y - contentTop;
        const float viewH = std::max(contentH - ConsoleHeight(), 1.f);
        UpdateView3dLayout(contentTop, viewH, sidebarW);
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
         measure_.mode == ToolMode::PlaneAlign ||
         measure_.mode == ToolMode::SphereFit || measure_.mode == ToolMode::SphereBodyFit ||
         measure_.mode == ToolMode::CircleFit ||
         measure_.mode == ToolMode::CylinderFit || measure_.mode == ToolMode::Flatness ||
         measure_.mode == ToolMode::StepGap);
    const bool sectionStyle = (measure_.mode == ToolMode::Section);

    const bool inView3d = MouseInView3d(mx, my);
    if (inView3d) {
        activeCloudPane_ = DualCloudViewActive() ? 1 : CloudPaneAtMouse(static_cast<float>(mx));
    }
    const bool uiCapture = io.WantCaptureMouse;
    // 点云视区内：中键 / Shift+左键 平移优先，不受 UI 抢鼠标影响
    const bool allowPan = inView3d && (!uiCapture || middle || (left && shift));
    const bool allow3d = inView3d && !uiCapture;

    // 双击点云：将旋转中心设为最近点
    if (allow3d && !EditableCloud().points.empty() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left) &&
        !shift && !alt) {
        int vx = 0, vy = 0, vw = 0, vh = 0;
        GetView3dFbRect(vx, vy, vw, vh);
        const std::vector<std::size_t>* disp =
            DualCloudViewActive()
                ? nullptr
                : (ActiveLayer().displayIndices.empty() ? nullptr : &ActiveLayer().displayIndices);
        const auto idx = PickNearestWithBackend(
            vw, vh, mouseX - static_cast<float>(vx),
            mouseY - static_cast<float>(vy), 14.f, disp);
        if (idx) {
            camera_.SetOrbitTarget(EditableCloud().points[*idx]);
            const Vec3 w = EditableCloud().ToWorld(EditableCloud().points[*idx]);
            char buf[160];
            std::snprintf(buf, sizeof(buf), u8"旋转中心已设为点 #%zu  (%.4f, %.4f, %.4f)", *idx, w.x,
                          w.y, w.z);
            SetStatus(buf);
            rotating_ = false;
            panning_ = false;
        } else {
            SetStatus(u8"双击附近没有点，无法设置旋转中心");
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
                if (measure_.mode == ToolMode::Roi &&
                    measure_.roiShape == RoiShape::FreePolygon) {
                    if (left && !leftWasDown && !shift) {
                        int vx = 0, vy = 0, vw = 0, vh = 0;
                        GetView3dFbRect(vx, vy, vw, vh);
                        measure_.roiPolyX.push_back(mouseX - static_cast<float>(vx));
                        measure_.roiPolyY.push_back(mouseY - static_cast<float>(vy));
                        measure_.roiPolyBuilding = true;
                        SetStatus(std::string(u8"多边形顶点 #") + std::to_string(measure_.roiPolyX.size()) + u8"，点击「完成多边形」闭合");
                    }
                } else {
                    if (left && !leftWasDown && !shift) BeginRoiDrag(mouseX, mouseY);
                    if (measure_.roiDragging && left) UpdateRoiDrag(mouseX, mouseY);
                    if (measure_.roiDragging && !left) EndRoiDrag();
                }

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
    if (ctrl && zDown && !zWas) Undo2DOrCloud();
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
    const float clipX = DualCloudViewActive() ? view3dPane1X_ : view3dX_;
    const float clipW = DualCloudViewActive() ? view3dPane1W_ : view3dW_;
    const ImVec2 clipMin(clipX, view3dY_);
    const ImVec2 clipMax(clipX + clipW, view3dY_ + view3dH_);
    dlBg->PushClipRect(clipMin, clipMax, true);
    dl->PushClipRect(clipMin, clipMax, true);

    // 框选拖拽中：屏幕矩形；完成后：3D 投影框（随视角动）
    const bool roiDraggingNow =
        measure_.roiDragging ||
        (measure_.roiX0 != measure_.roiX1 || measure_.roiY0 != measure_.roiY1);

    if (measure_.mode == ToolMode::Roi || measure_.mode == ToolMode::PlaneFit ||
        measure_.mode == ToolMode::PlaneAlign ||
        IsSphereFitMode(measure_.mode) || measure_.mode == ToolMode::CircleFit ||
        measure_.mode == ToolMode::CylinderFit || measure_.mode == ToolMode::Flatness) {
        if (roiDraggingNow) {
            if (measure_.roiUseWorldSize && measure_.roiHasWorldCenter) {
                const float aspect = View3dAspect();
                const int segs = 72;
                const Vec3 c = measure_.roiWorldCenter;
                if (measure_.roiShape == RoiShape::Circle) {
                    const float r = measure_.roiWorldRadius;
                    ImVec2 prev{};
                    bool hasPrev = false;
                    for (int si = 0; si <= segs; ++si) {
                        const float a =
                            6.2831853f * static_cast<float>(si) / static_cast<float>(segs);
                        const Vec3 p{c.x + r * std::cos(a), c.y + r * std::sin(a), c.z};
                        float px = 0.f, py = 0.f;
                        if (!ProjectWorldToScreen(p, px, py)) {
                            hasPrev = false;
                            continue;
                        }
                        const ImVec2 cur(px * sx, py * sy);
                        if (hasPrev) dlBg->AddLine(prev, cur, IM_COL32(255, 200, 60, 240), 2.f);
                        prev = cur;
                        hasPrev = true;
                    }
                    float ccx = 0.f, ccy = 0.f;
                    if (ProjectWorldToScreen(c, ccx, ccy)) {
                        dlBg->AddCircleFilled(ImVec2(ccx * sx, ccy * sy), 4.f,
                                              IM_COL32(255, 200, 60, 255));
                    }
                    (void)aspect;
                } else {
                    const float hw = measure_.roiWorldWidth * 0.5f;
                    const float hh = measure_.roiWorldHeight * 0.5f;
                    const Vec3 corners[4] = {
                        {c.x - hw, c.y - hh, c.z}, {c.x + hw, c.y - hh, c.z},
                        {c.x + hw, c.y + hh, c.z}, {c.x - hw, c.y + hh, c.z},
                    };
                    ImVec2 scr[4];
                    int valid = 0;
                    for (int ci = 0; ci < 4; ++ci) {
                        float px = 0.f, py = 0.f;
                        if (ProjectWorldToScreen(corners[ci], px, py)) {
                            scr[ci] = ImVec2(px * sx, py * sy);
                            ++valid;
                        }
                    }
                    if (valid == 4) {
                        for (int ci = 0; ci < 4; ++ci)
                            dlBg->AddLine(scr[ci], scr[(ci + 1) % 4], IM_COL32(255, 200, 60, 240),
                                          2.f);
                    }
                }
            } else if (measure_.roiShape == RoiShape::Circle && !measure_.roiUseWorldSize) {
                const float cx = measure_.roiX0 * sx;
                const float cy = measure_.roiY0 * sy;
                const float r = std::hypot((measure_.roiX1 - measure_.roiX0) * sx,
                                           (measure_.roiY1 - measure_.roiY0) * sy);
                dlBg->AddCircle(ImVec2(cx, cy), std::max(r, 1.f), IM_COL32(64, 200, 180, 230),
                                0, 2.f);
            } else {
                dlBg->AddRect(ImVec2(measure_.roiX0 * sx, measure_.roiY0 * sy),
                              ImVec2(measure_.roiX1 * sx, measure_.roiY1 * sy),
                              IM_COL32(64, 200, 180, 230), 0.f, 0, 2.f);
            }
        } else if (!measure_.roiIndices.empty()) {
            const char* label = u8"框选区域";
            if (measure_.mode == ToolMode::PlaneFit) label = u8"平面拟合区域";
            if (measure_.mode == ToolMode::PlaneAlign) label = u8"平面校准区域";
            if (measure_.mode == ToolMode::SphereFit) label = u8"球面拟合区域";
            if (measure_.mode == ToolMode::SphereBodyFit) label = u8"球体拟合区域";
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

    if (measure_.mode == ToolMode::Roi && measure_.roiShape == RoiShape::FreePolygon &&
        measure_.roiPolyX.size() >= 2) {
        for (std::size_t i = 1; i < measure_.roiPolyX.size(); ++i) {
            dlBg->AddLine(ImVec2(measure_.roiPolyX[i - 1] * sx, measure_.roiPolyY[i - 1] * sy),
                          ImVec2(measure_.roiPolyX[i] * sx, measure_.roiPolyY[i] * sy),
                          IM_COL32(64, 200, 180, 230), 2.f);
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
    if (showAxes_ && MainCloud().bounds.Valid()) {
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
    if (!ImGui::Begin(u8"截面 2D 轮廓", &showSectionPanel_, ImGuiWindowFlags_NoCollapse)) {
        ImGui::End();
        return;
    }

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
    ImGui::SameLine();
    if (ImGui::Button(u8"关闭")) {
        showSectionPanel_ = false;
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
                SetStatus(u8"拖动点1中…再单击选择点2", false);
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
        case ToolMode::PlaneAlign:
            return u8"平面校准";
        case ToolMode::SphereFit:
            return u8"球面拟合";
        case ToolMode::SphereBodyFit:
            return u8"球面拟合";
        case ToolMode::CircleFit:
            return u8"圆拟合";
        case ToolMode::CylinderFit:
            return u8"圆柱拟合";
        case ToolMode::Roi:
            return u8"ROI框选";
        case ToolMode::Section:
            return u8"截面";
        case ToolMode::StepHeight:
            return u8"台阶高度";
        case ToolMode::Flatness:
            return u8"平面度";
        case ToolMode::StepGap:
            return u8"段差计算";
        case ToolMode::Icp:
            return u8"ICP 配准";
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
    measure_.planeDistProfile = {};
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
    ClearAllMeasuredLines();
    if (resetStatus) {
        SetStatus(u8"已清空当前工具显示");
    }
    sceneGpuDirty_ = true;
    UpdateOverlays();
}

void Application::SetToolMode(ToolMode mode) {
    if (mode == ToolMode::SphereBodyFit) {
        mode = ToolMode::SphereFit;
    }
    if (mode == measure_.mode) return;

    ClearToolVisuals(false);
    measure_.mode = mode;

    if (mode == ToolMode::PlaneFit || mode == ToolMode::PlaneAlign || mode == ToolMode::SphereFit ||
        mode == ToolMode::SphereBodyFit || mode == ToolMode::CircleFit ||
        mode == ToolMode::CylinderFit || mode == ToolMode::Flatness || mode == ToolMode::StepGap) {
        ResetFitRoiSelection();
    }

    if (mode == ToolMode::Flatness) {
        SetStatus(u8"平面度：框选区域后点击计算");
    } else if (mode == ToolMode::StepGap) {
        measure_.stepGap.phase = StepGapPhase::SelectA;
        SetStatus(u8"段差：先框选基准区域 A");
    } else if (mode == ToolMode::Roi) {
        measure_.roiPolyX.clear();
        measure_.roiPolyY.clear();
        measure_.roiPolyBuilding = false;
        SetStatus(u8"ROI：矩形/圆形拖拽框选，或自由多边形逐点点击");
    } else if (mode == ToolMode::PlaneFit) {
        SetStatus(u8"平面拟合：框选可见表面后拟合");
    } else if (mode == ToolMode::PlaneAlign) {
        SetStatus(u8"平面校准：框选基准面后摆正点云（线扫倾斜校正）");
    } else if (mode == ToolMode::SphereFit) {
        SetStatus(u8"球面拟合：框选可见表面后拟合球体");
    } else if (mode == ToolMode::CircleFit) {
        SetStatus(u8"圆拟合：框选可见表面后拟合");
    } else if (mode == ToolMode::CylinderFit) {
        SetStatus(u8"圆柱拟合：框选可见表面后拟合");
    } else if (mode == ToolMode::Section) {
        showSectionPanel_ = true;
        if (MainCloud().bounds.Valid()) {
            measure_.section.position = measure_.section.cutAlongX ? MainCloud().bounds.Center().x
                                                                  : MainCloud().bounds.Center().y;
        }
        SyncSectionCutPlane();
        SetStatus(u8"截面：拖拽橙色切面或生成截面");
    } else if (mode == ToolMode::Navigate) {
        SetStatus(u8"漫游模式");
    } else if (mode == ToolMode::Pick) {
        SetStatus(u8"点选：单击读取坐标");
    } else if (mode == ToolMode::Distance) {
        SetStatus(u8"测距：依次点击两点");
    } else if (mode == ToolMode::StepHeight) {
        SetStatus(u8"台阶高度：依次点击 A、B");
    } else if (mode == ToolMode::Icp) {
        SetStatus(u8"ICP 配准：加载源点云后执行配准");
    } else {
        SetStatus(ToolModeLabel(mode));
    }

    sceneGpuDirty_ = true;
    if (DualCloudViewActive()) needUploadFilled_ = true;
    UpdateOverlays();
}

float Application::DrawMenuBar() {
    const UiPalette& pal = GetUiPalette();
    const ImGuiViewport* vp = ImGui::GetMainViewport();
    float menuBottom = vp->Pos.y;

    PushMenuBarStyle();
    if (!ImGui::BeginMainMenuBar()) {
        PopMenuBarStyle();
        return menuBottom;
    }

    ImGui::PushStyleColor(ImGuiCol_Text, pal.accent);
    ImGui::TextUnformatted(u8"  点云查看器");
    ImGui::PopStyleColor();
    ImGui::SameLine();
    ImGui::TextDisabled("v%s", kAppVersion);
    ImGui::SameLine(0.f, 18.f);
    ImGui::TextDisabled("|");
    ImGui::SameLine(0.f, 18.f);

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
        ImGui::Separator();
        if (ImGui::MenuItem(u8"加载示例")) {
            LoadPath("assets/sample/sample.xyz");
        }
        ImGui::Separator();
        if (ImGui::MenuItem(u8"保存点云…", "Ctrl+S", false, !MainCloud().points.empty())) {
            SaveCloud();
        }
        ImGui::Checkbox(u8"保存时仅可见点", &saveVisibleOnly_);
        ImGui::EndMenu();
    }

    if (ImGui::BeginMenu(u8"3D算子")) {
        if (ImGui::BeginMenu(u8"创建")) {
            if (ImGui::MenuItem(u8"球面点云…")) showCreateSphere_ = true;
            if (ImGui::MenuItem(u8"圆柱点云…")) showCreateCylinder_ = true;
            if (ImGui::MenuItem(u8"圆面点云…")) showCreateDisk_ = true;
            if (ImGui::MenuItem(u8"平面点云…")) showCreatePlane_ = true;
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
            toolItem(ToolMode::PlaneAlign);
            toolItem(ToolMode::Section);
            toolItem(ToolMode::StepHeight);
            ImGui::Separator();
            toolItem(ToolMode::Flatness);
            toolItem(ToolMode::StepGap);
            toolItem(ToolMode::Icp);
            ImGui::Separator();
            ImGui::TextDisabled(u8"PCL 正交投影");
            if (ImGui::MenuItem(u8"投影到 YZ 平面（法向 X）")) ApplyProjectionToAxis(0);
            if (ImGui::MenuItem(u8"投影到 XZ 平面（法向 Y）")) ApplyProjectionToAxis(1);
            if (ImGui::MenuItem(u8"投影到 XY 平面（法向 Z）")) ApplyProjectionToAxis(2);
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
            ImGui::TextDisabled(u8"后端: %s", AlgorithmBackendLabel(EffectiveAlgoBackend()));
            DrawFilterMenuItems();
            ImGui::EndMenu();
        }

        ImGui::EndMenu();
    }

    if (ImGui::BeginMenu(u8"2D算子")) {
        Draw2DOperatorMenuItems();
        ImGui::EndMenu();
    }

    if (ImGui::BeginMenu(u8"相机标定")) {
        if (ImGui::MenuItem(u8"九点标定…", nullptr, camera2DCalibWindow_.IsVisible())) {
            camera2DCalibWindow_.SetVisible(!camera2DCalibWindow_.IsVisible());
        }
        if (ImGui::MenuItem(u8"内参标定…", nullptr, cameraIntrinsicsCalibWindow_.IsVisible())) {
            cameraIntrinsicsCalibWindow_.SetVisible(!cameraIntrinsicsCalibWindow_.IsVisible());
        }
        ImGui::EndMenu();
    }

    if (ImGui::BeginMenu(u8"多视图几何")) {
        if (ImGui::MenuItem(u8"两视图对应…", nullptr, multiViewGeometryWindow_.IsVisible())) {
            multiViewGeometryWindow_.SetVisible(!multiViewGeometryWindow_.IsVisible());
        }
        ImGui::EndMenu();
    }

    if (ImGui::MenuItem(u8"模板匹配", nullptr, halconMatchWindow_.IsVisible())) {
        const bool next = !halconMatchWindow_.IsVisible();
        if (next) {
            algoEditor_.SetVisible(false);
            ocrWindow_.SetVisible(false);
        }
        halconMatchWindow_.SetVisible(next);
    }

    if (ImGui::MenuItem(u8"OCR 识别", nullptr, ocrWindow_.IsVisible())) {
        const bool next = !ocrWindow_.IsVisible();
        if (next) {
            algoEditor_.SetVisible(false);
            halconMatchWindow_.SetVisible(false);
        }
        ocrWindow_.SetVisible(next);
    }

    if (ImGui::BeginMenu(u8"设置")) {
        const bool nativeActive =
            algoBackend_ == AlgorithmBackend::Native && nativeAlgoUnlocked_;
        if (ImGui::MenuItem(u8"自研算法", nullptr, nativeActive)) {
            if (nativeAlgoUnlocked_) {
                algoBackend_ = AlgorithmBackend::Native;
                SetStatus(u8"已切换为自研算法");
            } else {
                showNativeAlgoPassword_ = true;
            }
        }
        if (ImGui::MenuItem(u8"PCL 算法", nullptr, algoBackend_ == AlgorithmBackend::PCL)) {
            algoBackend_ = AlgorithmBackend::PCL;
            SetStatus(u8"已切换为 PCL 算法");
        }
        ImGui::EndMenu();
    }

    if (ImGui::BeginMenu(u8"显示")) {
        ImGui::TextDisabled(u8"点云外观");
        if (ImGui::SliderInt(u8"显示点数上限", &maxDisplayPoints_, 200000, 3000000, "%d")) {
            sceneGpuDirty_ = true;
        }
        ImGui::SliderFloat(u8"点大小", &pointSize_, 1.f, 20.f, "%.1f");
        ImGui::SliderFloat(u8"透明度", &opacity_, 0.05f, 1.f, "%.2f");
        ImGui::Checkbox(u8"自动高度着色范围", &autoZRange_);
        if (!autoZRange_) {
            if (ImGui::DragFloat(u8"高度最小", &zMin_, 0.01f, 0.f, 0.f, "%.4f")) sceneGpuDirty_ = true;
            if (ImGui::DragFloat(u8"高度最大", &zMax_, 0.01f, 0.f, 0.f, "%.4f")) sceneGpuDirty_ = true;
        }
        if (!intensityColors_.empty() && intensityColors_.size() == MainCloud().points.size()) {
            if (ImGui::Checkbox(u8"使用亮度着色", &useIntensityColors_)) sceneGpuDirty_ = true;
        }
        if (ImGui::MenuItem(u8"重新上色")) sceneGpuDirty_ = true;
        ImGui::Separator();
        if (ImGui::Checkbox(u8"显示坐标系 (mm)", &showAxes_)) {
            ActiveRenderer().SetAxes(showAxes_, axesLength_);
        }
        ImGui::EndMenu();
    }

    if (ImGui::BeginMenu(u8"窗口")) {
        if (ImGui::MenuItem(u8"3D 模式", nullptr, !view2DMode_)) {
            EnterView3DMode();
            SetStatus(u8"已切换到 3D 模式");
        }
        if (ImGui::MenuItem(u8"2D 模式", nullptr, view2DMode_)) {
            EnterView2DMode();
            SetStatus(u8"已切换到 2D 模式");
        }
        ImGui::EndMenu();
    }

    if (ImGui::MenuItem(u8"算法编辑器", nullptr, algoEditor_.IsVisible())) {
        const bool next = !algoEditor_.IsVisible();
        if (next) {
            halconMatchWindow_.SetVisible(false);
            ocrWindow_.SetVisible(false);
        }
        algoEditor_.ToggleVisible();
    }

    if (ImGui::BeginMenu(u8"帮助")) {
        if (ImGui::MenuItem(u8"关于…")) {
            showAbout_ = true;
        }
        ImGui::EndMenu();
    }

    ImGui::SameLine();
    if (ImGui::MenuItem(u8"恢复")) {
        RestoreApplicationState();
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip(u8"将界面与工具状态恢复到软件启动时的默认状态");
    }

    menuBottom = ImGui::GetWindowPos().y + ImGui::GetWindowSize().y;
    ImGui::EndMainMenuBar();
    PopMenuBarStyle();

    ImDrawList* dl = ImGui::GetForegroundDrawList();
    const ImU32 lineCol = ImGui::ColorConvertFloat4ToU32(pal.border);
    dl->AddLine(ImVec2(vp->Pos.x, menuBottom), ImVec2(vp->Pos.x + vp->Size.x, menuBottom),
                lineCol, 1.0f);

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
        const UiPalette& pal = GetUiPalette();
        ImGui::PushStyleColor(ImGuiCol_Text, pal.sectionTitle);
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

void Application::DrawNativeAlgoPasswordPopup() {
    if (showNativeAlgoPassword_) {
        ImGui::OpenPopup(u8"启用自研算法");
        showNativeAlgoPassword_ = false;
        std::memset(nativeAlgoPasswordBuf_, 0, sizeof(nativeAlgoPasswordBuf_));
    }

    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(ImVec2(360.f, 0.f), ImGuiCond_Appearing);
    if (ImGui::BeginPopupModal(u8"启用自研算法", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::TextWrapped(u8"自研点云算法尚未完全替换，需输入密码后方可启用。");
        ImGui::Spacing();
        ImGui::SetNextItemWidth(-1.f);
        ImGui::InputText(u8"密码##native_algo", nativeAlgoPasswordBuf_, sizeof(nativeAlgoPasswordBuf_),
                         ImGuiInputTextFlags_Password);
        ImGui::Spacing();
        if (ImGui::Button(u8"确定", ImVec2(120.f, 0.f))) {
            if (std::strcmp(nativeAlgoPasswordBuf_, "111") == 0) {
                nativeAlgoUnlocked_ = true;
                algoBackend_ = AlgorithmBackend::Native;
                SetStatus(u8"自研算法已启用");
                std::memset(nativeAlgoPasswordBuf_, 0, sizeof(nativeAlgoPasswordBuf_));
                ImGui::CloseCurrentPopup();
            } else {
                SetStatus(u8"密码错误，仍使用 PCL 算法");
                std::memset(nativeAlgoPasswordBuf_, 0, sizeof(nativeAlgoPasswordBuf_));
            }
        }
        ImGui::SameLine();
        if (ImGui::Button(u8"取消", ImVec2(120.f, 0.f))) {
            std::memset(nativeAlgoPasswordBuf_, 0, sizeof(nativeAlgoPasswordBuf_));
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
    if (showCreateDisk_) {
        ImGui::OpenPopup(u8"创建圆面点云");
        showCreateDisk_ = false;
    }
    if (showCreatePlane_) {
        ImGui::OpenPopup(u8"创建平面点云");
        showCreatePlane_ = false;
    }

    ImGui::SetNextWindowSize(ImVec2(360.f, 0.f), ImGuiCond_Appearing);
    if (ImGui::BeginPopupModal(u8"创建球面点云", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::TextWrapped(u8"生成均匀分布的球面点云（黄金角螺旋），可用于测试球面拟合。");
        ImGui::Spacing();
        ImGui::DragFloat(u8"半径", &genSphereRadius_, 0.1f, 0.01f, 1e6f, "%.3f");
        ImGui::DragInt(u8"点数", &genSpherePoints_, 100.f, 16, 2000000);
        ImGui::DragFloat(u8"法向噪声", &genSphereNoise_, 0.001f, 0.f, 100.f, "%.4f");
        ImGui::TextDisabled(u8"噪声为 0 时球面光滑稳定；增大后可模拟测量抖动。");
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

    ImGui::SetNextWindowSize(ImVec2(360.f, 0.f), ImGuiCond_Appearing);
    if (ImGui::BeginPopupModal(u8"创建圆面点云", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::TextWrapped(
            u8"生成 XY 平面上的圆盘点云（黄金角螺旋排布），可用于测试圆拟合。");
        ImGui::Spacing();
        ImGui::DragFloat(u8"半径", &genDiskRadius_, 0.1f, 0.01f, 1e6f, "%.3f");
        ImGui::DragInt(u8"点数", &genDiskPoints_, 100.f, 16, 2000000);
        ImGui::DragFloat(u8"Z 向噪声", &genDiskNoise_, 0.001f, 0.f, 100.f, "%.4f");
        ImGui::TextDisabled(u8"噪声为 0 时圆面完全平整；有噪声时旋转视角可能出现深度闪烁。");
        ImGui::Spacing();
        if (ImGui::Button(u8"生成", ImVec2(120.f, 0))) {
            CreateDiskCloud();
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::Button(u8"取消", ImVec2(120.f, 0))) ImGui::CloseCurrentPopup();
        ImGui::EndPopup();
    }

    ImGui::SetNextWindowSize(ImVec2(360.f, 0.f), ImGuiCond_Appearing);
    if (ImGui::BeginPopupModal(u8"创建平面点云", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::TextWrapped(u8"生成 XY 平面上的矩形平面点云，可用于测试平面拟合与平面度。");
        ImGui::Spacing();
        ImGui::DragFloat(u8"X 半宽", &genPlaneExtentX_, 0.1f, 0.01f, 1e6f, "%.3f");
        ImGui::DragFloat(u8"Y 半宽", &genPlaneExtentY_, 0.1f, 0.01f, 1e6f, "%.3f");
        ImGui::DragInt(u8"点数", &genPlanePoints_, 100.f, 16, 2000000);
        ImGui::DragFloat(u8"Z 向噪声", &genPlaneNoise_, 0.001f, 0.f, 100.f, "%.4f");
        ImGui::TextDisabled(u8"噪声为 0 时平面完全平整。");
        ImGui::Spacing();
        if (ImGui::Button(u8"生成", ImVec2(120.f, 0))) {
            CreatePlaneCloud();
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
    ImGui::SameLine();
    if (ImGui::SmallButton(u8"复位视图")) {
        ResetImage2dView();
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip(u8"将图像缩放与平移恢复为初始状态（等同双击图像）");
    }

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
    const float basePanX = std::max((avail.x - drawW) * 0.5f, 0.f);
    const float basePanY = std::max((avail.y - drawH) * 0.5f, 0.f);

    const ImGuiID childId = ImGui::GetID(label);
    ImGui::BeginChild(childId, avail, ImGuiChildFlags_Borders,
                      ImGuiWindowFlags_HorizontalScrollbar);
    ImGuiIO& io = ImGui::GetIO();
    if (ImGui::IsWindowHovered() && io.KeyShift && io.MouseWheel != 0.f) {
        const float factor = 1.f + io.MouseWheel * 0.12f;
        image2dZoom_ = std::clamp(image2dZoom_ * factor, 0.1f, 32.f);
    }
    if (ImGui::IsWindowHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
        ResetImage2dView();
    }

    ImGui::SetCursorPos(ImVec2(basePanX + image2dPanX_, basePanY + image2dPanY_));
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

    if (showLineMeasureOverlay_) {
        DrawLineMeasureOverlay(ImGui::GetWindowDrawList(), view, cursor.x, cursor.y, drawW, drawH);
    }

    const bool lineCaliperActive = image2DTool_ == Image2DTool::CaliperLine;
    const bool arcCaliperActive = image2DTool_ == Image2DTool::CaliperArc;
    const bool circleFitActive = image2DTool_ == Image2DTool::CircleFit;
    const bool ellipseFitActive = image2DTool_ == Image2DTool::EllipseFit;
    const bool arcRoiActive = arcCaliperActive || circleFitActive || ellipseFitActive;
    const bool lineDistActive = image2DTool_ == Image2DTool::LineDistance;
    const bool arcDistActive = image2DTool_ == Image2DTool::ArcDistance;
    const bool pointDistActive = image2DTool_ == Image2DTool::PointDistance;
    const bool lineAngleActive = image2DTool_ == Image2DTool::LineAngle;
    const bool circleGapActive = image2DTool_ == Image2DTool::CircleGap;
    const bool pointLineActive = image2DTool_ == Image2DTool::PointLineDistance;
    const bool caliperPointActive = image2DTool_ == Image2DTool::CaliperPoint;
    const bool circleCaliperActive = image2DTool_ == Image2DTool::CaliperCircle;
    const bool arcLengthActive = image2DTool_ == Image2DTool::ArcLength;
    const bool threePointActive = image2DTool_ == Image2DTool::ThreePointCircle;
    const bool parallelDistActive = image2DTool_ == Image2DTool::ParallelLineDistance;
    const bool rectCaliperActive = image2DTool_ == Image2DTool::RectCaliper;
    const bool profileWidthActive = image2DTool_ == Image2DTool::ProfileWidth;
    const bool pointProjActive = image2DTool_ == Image2DTool::PointProjection;
    const bool concentricityActive = image2DTool_ == Image2DTool::Concentricity;
    const bool roundnessActive = image2DTool_ == Image2DTool::Roundness;
    const bool regionBlobActive = image2DTool_ == Image2DTool::RegionBlob;
    const bool depthHeightActive = image2DTool_ == Image2DTool::DepthHeightDiff;
    const bool depthProfileActive = image2DTool_ == Image2DTool::DepthProfile;
    const bool templateMatchActive = image2DTool_ == Image2DTool::TemplateMatch;
    const bool caliperActive = lineCaliperActive || arcRoiActive || caliperPointActive ||
                               circleCaliperActive || rectCaliperActive || profileWidthActive ||
                               regionBlobActive || depthProfileActive || templateMatchActive;
    if (view.width > 0 && view.height > 0) {
        const ImVec2 mp = ImGui::GetMousePos();
        const float u = (mp.x - cursor.x) / drawW;
        const float vv = (mp.y - cursor.y) / drawH;
        const bool inImage = u >= 0.f && u < 1.f && vv >= 0.f && vv < 1.f;
        const float px = u * static_cast<float>(view.width);
        const float py = vv * static_cast<float>(view.height);
        const int src = ImageSourceOf(view);

        if (hovered && io.KeyShift && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
            image2dPanX_ += io.MouseDelta.x;
            image2dPanY_ += io.MouseDelta.y;
        }

        if (lineCaliperActive && hovered && inImage && !io.KeyShift) {
            if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                lineMeasureDragging_ = true;
                lineMeasureDragSource_ = src;
                lineMeasureRoiX0_ = px;
                lineMeasureRoiY0_ = py;
                lineMeasureRoiX1_ = px;
                lineMeasureRoiY1_ = py;
            }
            if (lineMeasureDragging_ && lineMeasureDragSource_ == src &&
                ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
                lineMeasureRoiX1_ = px;
                lineMeasureRoiY1_ = py;
            }
            if (lineMeasureDragging_ && lineMeasureDragSource_ == src &&
                ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
                lineMeasureDragging_ = false;
                lineMeasureRoiX1_ = px;
                lineMeasureRoiY1_ = py;
                PreviewLineMeasure(view);
            }
        }

        if (arcRoiActive && hovered && inImage && !io.KeyShift) {
            bool skipRoiClick = false;
            if (circleFitActive && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                const int arcIdx = FindClosestMeasuredArc(src, px, py, 18.f);
                if (arcIdx >= 0) {
                    PreviewCircleFitFromMeasuredArc(arcIdx);
                    skipRoiClick = true;
                }
            }
            if (ellipseFitActive && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                const int arcIdx = FindClosestMeasuredArc(src, px, py, 18.f);
                if (arcIdx >= 0) {
                    PreviewEllipseFitFromMeasuredArc(arcIdx);
                    skipRoiClick = true;
                }
            }
            if (!skipRoiClick) {
            if (arcMeasurePhase_ == ArcMeasurePhase::PickA) {
                if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                    arcMeasureSource_ = src;
                    arcRoiP0X_ = px;
                    arcRoiP0Y_ = py;
                    arcMeasurePhase_ = ArcMeasurePhase::PickB;
                    SetStatus(u8"已设置 A 点，请点击 B 点");
                }
            } else if (arcMeasurePhase_ == ArcMeasurePhase::PickB && arcMeasureSource_ == src) {
                if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                    const float dx = px - arcRoiP0X_;
                    const float dy = py - arcRoiP0Y_;
                    if (dx * dx + dy * dy < 16.f) {
                        SetStatus(u8"B 点须与 A 点保持距离");
                    } else {
                        arcRoiP1X_ = px;
                        arcRoiP1Y_ = py;
                        ProjectBulgePoint(arcRoiP0X_, arcRoiP0Y_, arcRoiP1X_, arcRoiP1Y_, px, py,
                                          arcRoiP2X_, arcRoiP2Y_);
                        arcMeasurePhase_ = ArcMeasurePhase::DragBulge;
                        SetStatus(u8"请拖拽拱高线段调节弧线形状，松开后预览");
                    }
                }
            } else if (arcMeasurePhase_ == ArcMeasurePhase::DragBulge &&
                       arcMeasureSource_ == src) {
                if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                    arcBulgeDragging_ = true;
                    ProjectBulgePoint(arcRoiP0X_, arcRoiP0Y_, arcRoiP1X_, arcRoiP1Y_, px, py,
                                      arcRoiP2X_, arcRoiP2Y_);
                }
                if (arcBulgeDragging_ && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
                    ProjectBulgePoint(arcRoiP0X_, arcRoiP0Y_, arcRoiP1X_, arcRoiP1Y_, px, py,
                                      arcRoiP2X_, arcRoiP2Y_);
                } else if (!arcBulgeDragging_) {
                    ProjectBulgePoint(arcRoiP0X_, arcRoiP0Y_, arcRoiP1X_, arcRoiP1Y_, px, py,
                                      arcRoiP2X_, arcRoiP2Y_);
                }
                if (arcBulgeDragging_ && ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
                    arcBulgeDragging_ = false;
                    ProjectBulgePoint(arcRoiP0X_, arcRoiP0Y_, arcRoiP1X_, arcRoiP1Y_, px, py,
                                      arcRoiP2X_, arcRoiP2Y_);
                    if (ArcChordBulgePx(arcRoiP0X_, arcRoiP0Y_, arcRoiP1X_, arcRoiP1Y_,
                                        arcRoiP2X_, arcRoiP2Y_) >= 3.f) {
                        if (circleFitActive) {
                            PreviewCircleFitFromRoi(view);
                        } else if (ellipseFitActive) {
                            PreviewEllipseFitFromRoi(view);
                        } else {
                            PreviewArcMeasure(view);
                        }
                    } else {
                        SetStatus(u8"拱高过小，请继续拖拽调节线段");
                    }
                }
            }
            }
        }

        if (lineDistActive && hovered && inImage && !io.KeyShift &&
            ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            const int idx = FindClosestMeasuredLine(src, px, py, 18.f);
            if (idx >= 0) {
                PickLineForDistance(idx);
            } else {
                SetStatus(u8"未点中线段，请点击图像上已显示的线段");
            }
        }

        if (arcDistActive && hovered && inImage && !io.KeyShift &&
            ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            const int idx = FindClosestMeasuredArc(src, px, py, 18.f);
            if (idx >= 0) {
                PickArcForDistance(idx);
            } else {
                SetStatus(u8"未点中圆弧，请点击图像上已显示的圆弧");
            }
        }

        if (pointDistActive && hovered && inImage && !io.KeyShift &&
            ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            if (pointDistPhase_ == PointPickPhase::PickA) {
                pointDistSource_ = src;
                pointDistAx_ = px;
                pointDistAy_ = py;
                pointDistPhase_ = PointPickPhase::PickB;
                SetStatus(u8"已设置 A 点，请点击 B 点");
            } else if (pointDistSource_ == src) {
                AddPointDistance(pointDistAx_, pointDistAy_, px, py, src);
                pointDistPhase_ = PointPickPhase::PickA;
            }
        }

        if (lineAngleActive && hovered && inImage && !io.KeyShift &&
            ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            const int idx = FindClosestMeasuredLine(src, px, py, 18.f);
            if (idx >= 0) {
                PickLineForAngle(idx);
            } else {
                SetStatus(u8"未点中线段，请点击图像上已显示的线段");
            }
        }

        if (circleGapActive && hovered && inImage && !io.KeyShift &&
            ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            const int idx = FindClosestMeasuredCircleFit(src, px, py, 18.f);
            if (idx >= 0) {
                PickCircleForGap(idx);
            } else {
                SetStatus(u8"未点中圆，请点击图像上已显示的拟合圆");
            }
        }

        if (pointLineActive && hovered && inImage && !io.KeyShift &&
            ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            if (pointLinePhase_ == PointPickPhase::PickA) {
                PickPointForPointLine(px, py, src);
            } else {
                const int idx = FindClosestMeasuredLine(src, px, py, 18.f);
                if (idx >= 0) {
                    PickLineForPointLine(idx);
                } else {
                    SetStatus(u8"未点中线段，请点击图像上已显示的线段");
                }
            }
        }

        if (caliperPointActive && hovered && inImage && !io.KeyShift) {
            if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                caliperPointDragging_ = true;
                caliperPointDragSource_ = src;
                caliperPointRoiX0_ = px;
                caliperPointRoiY0_ = py;
                caliperPointRoiX1_ = px;
                caliperPointRoiY1_ = py;
            }
            if (caliperPointDragging_ && caliperPointDragSource_ == src &&
                ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
                caliperPointRoiX1_ = px;
                caliperPointRoiY1_ = py;
            }
            if (caliperPointDragging_ && caliperPointDragSource_ == src &&
                ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
                caliperPointDragging_ = false;
                caliperPointRoiX1_ = px;
                caliperPointRoiY1_ = py;
                PreviewCaliperPoint(view);
            }
        }

        if (circleCaliperActive && hovered && inImage && !io.KeyShift) {
            if (circleCaliperPhase_ == CircleCaliperPhase::PickCenter) {
                if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                    circleCaliperSource_ = src;
                    circleCaliperCx_ = px;
                    circleCaliperCy_ = py;
                    circleCaliperR_ = 0.f;
                    circleCaliperPhase_ = CircleCaliperPhase::DragRadius;
                    SetStatus(u8"已设置圆心，拖拽设置半径后松开");
                }
            } else if (circleCaliperSource_ == src) {
                if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                    circleCaliperDragging_ = true;
                }
                if (circleCaliperDragging_ && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
                    circleCaliperR_ = std::hypot(px - circleCaliperCx_, py - circleCaliperCy_);
                }
                if (circleCaliperDragging_ && ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
                    circleCaliperDragging_ = false;
                    circleCaliperR_ = std::hypot(px - circleCaliperCx_, py - circleCaliperCy_);
                    if (circleCaliperR_ >= 3.f) {
                        PreviewCircleCaliper(view);
                    } else {
                        SetStatus(u8"半径过小，请重新拖拽");
                    }
                } else if (!circleCaliperDragging_) {
                    circleCaliperR_ = std::hypot(px - circleCaliperCx_, py - circleCaliperCy_);
                }
            }
        }

        if (arcLengthActive && hovered && inImage && !io.KeyShift &&
            ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            const int idx = FindClosestMeasuredArc(src, px, py, 18.f);
            if (idx >= 0) {
                PickArcForLength(idx);
            } else {
                SetStatus(u8"未点中圆弧，请点击图像上已显示的圆弧");
            }
        }

        if (threePointActive && hovered && inImage && !io.KeyShift &&
            ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            const int idx = static_cast<int>(threePointPhase_);
            threePointSource_ = src;
            threePointX_[idx] = px;
            threePointY_[idx] = py;
            if (threePointPhase_ == ThreePointPhase::Pick0) {
                threePointPhase_ = ThreePointPhase::Pick1;
                SetStatus(u8"已设置点1，请点击点2");
            } else if (threePointPhase_ == ThreePointPhase::Pick1) {
                threePointPhase_ = ThreePointPhase::Pick2;
                SetStatus(u8"已设置点2，请点击点3");
            } else {
                threePointPhase_ = ThreePointPhase::Pick0;
                ConfirmThreePointCircle();
            }
        }

        if (parallelDistActive && hovered && inImage && !io.KeyShift &&
            ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            const int idx = FindClosestMeasuredLine(src, px, py, 18.f);
            if (idx >= 0)
                PickLineForParallelDist(idx);
            else
                SetStatus(u8"未点中线段");
        }

        if (pointProjActive && hovered && inImage && !io.KeyShift &&
            ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            if (pointProjPhase_ == PointPickPhase::PickA) {
                PickPointForProjection(px, py, src);
            } else {
                const int idx = FindClosestMeasuredLine(src, px, py, 18.f);
                if (idx >= 0)
                    PickLineForProjection(idx);
                else
                    SetStatus(u8"未点中线段");
            }
        }

        if (concentricityActive && hovered && inImage && !io.KeyShift &&
            ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            const int idx = FindClosestMeasuredCircleFit(src, px, py, 18.f);
            if (idx >= 0)
                PickCircleForConcentricity(idx);
            else
                SetStatus(u8"未点中拟合圆");
        }

        if (roundnessActive && hovered && inImage && !io.KeyShift &&
            ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            int idx = FindClosestMeasuredCircleFit(src, px, py, 18.f);
            if (idx >= 0) {
                roundnessCircleSource_ = 0;
                PickCircleForRoundness(idx);
            } else {
                idx = FindClosestMeasuredCircleCaliper(src, px, py, 18.f);
                if (idx >= 0) {
                    roundnessCircleSource_ = 1;
                    PickCircleForRoundness(idx);
                } else {
                    SetStatus(u8"未点中圆");
                }
            }
        }

        auto handleRectDrag = [&](bool& dragging, int& dragSource, float& x0, float& y0, float& x1,
                                  float& y1, auto&& onRelease) {
            if (!hovered || !inImage || io.KeyShift) return;
            if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                dragging = true;
                dragSource = src;
                x0 = x1 = px;
                y0 = y1 = py;
            }
            if (dragging && dragSource == src && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
                x1 = px;
                y1 = py;
            }
            if (dragging && dragSource == src && ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
                dragging = false;
                x1 = px;
                y1 = py;
                onRelease();
            }
        };

        if (rectCaliperActive) {
            handleRectDrag(rectCaliperDragging_, rectCaliperDragSource_, rectCaliperRoiX0_,
                           rectCaliperRoiY0_, rectCaliperRoiX1_, rectCaliperRoiY1_,
                           [&]() { PreviewRectCaliper(view); });
        }
        if (profileWidthActive) {
            handleRectDrag(profileWidthDragging_, profileWidthDragSource_, profileWidthRoiX0_,
                           profileWidthRoiY0_, profileWidthRoiX1_, profileWidthRoiY1_,
                           [&]() { PreviewProfileWidth(view); });
        }
        if (regionBlobActive) {
            handleRectDrag(regionBlobDragging_, regionBlobDragSource_, regionBlobRoiX0_,
                           regionBlobRoiY0_, regionBlobRoiX1_, regionBlobRoiY1_,
                           [&]() { PreviewRegionBlob(view); });
            if (regionBlobDragging_ && regionBlobDragSource_ == src && HasRegionBlobRoi()) {
                PreviewRegionBlob(view, true);
            }
        }

        if (templateMatchActive && src == 1) {
            if (templateMatchPhase_ == TemplateMatchPhase::DrawTemplate) {
                handleRectDrag(
                    templateMatchTplDragging_, templateMatchDragSource_, templateMatchTplX0_,
                    templateMatchTplY0_, templateMatchTplX1_, templateMatchTplY1_, [&]() {
                        templateMatchHasTemplate_ = true;
                        templateMatchPhase_ = TemplateMatchPhase::DrawSearch;
                        if (templateMatchSearchFull_) {
                            PreviewTemplateMatch(view);
                        } else {
                            SetStatus(u8"模板已框选，请拖拽搜索区域");
                        }
                    });
            } else if (!templateMatchSearchFull_) {
                handleRectDrag(templateMatchSearchDragging_, templateMatchDragSource_,
                               templateMatchSearchX0_, templateMatchSearchY0_,
                               templateMatchSearchX1_, templateMatchSearchY1_,
                               [&]() { PreviewTemplateMatch(view); });
            }
        }

        if (depthHeightActive && hovered && inImage && !io.KeyShift && src == 0 &&
            ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            AddDepthHeightSample(px, py, src);
        }

        if (depthProfileActive && src == 0 && !view.gray.empty()) {
            if (depthProfileMode_ == DepthProfileMode::FreeLine) {
                handleRectDrag(depthProfileDragging_, depthProfileDragSource_, depthProfileRoiX0_,
                               depthProfileRoiY0_, depthProfileRoiX1_, depthProfileRoiY1_,
                               [&]() { PreviewDepthProfile(view); });
            } else if (hovered && inImage && !io.KeyShift &&
                       ImGui::IsMouseClicked(ImGuiMouseButton_Left) && !depthProfileDragging_) {
                PreviewDepthProfileAt(px, py, view);
            }
        }

        if ((hovered || clicked) && inImage) {
            const int col = std::clamp(static_cast<int>(px), 0, view.width - 1);
            const int row = std::clamp(static_cast<int>(py), 0, view.height - 1);
            if (clicked && imageSyncEnabled_ && !caliperActive && !lineDistActive &&
                !arcDistActive && !pointDistActive && !lineAngleActive && !circleGapActive &&
                !pointLineActive && !arcLengthActive && !threePointActive && !parallelDistActive &&
                !pointProjActive && !concentricityActive && !roundnessActive &&
                !depthHeightActive && !templateMatchActive) {
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
                if (lineCaliperActive) {
                    ImGui::TextColored(ImVec4(0.5f, 0.95f, 0.7f, 1.f),
                                       u8"卡尺提线：拖拽绘制测量方向线");
                    ImGui::TextDisabled(u8"Shift+左键拖拽平移图像");
                } else if (arcCaliperActive) {
                    ImGui::TextColored(ImVec4(0.5f, 0.95f, 0.7f, 1.f), "%s",
                                       ArcMeasurePhaseHint(arcMeasurePhase_));
                    ImGui::TextDisabled(u8"Shift+左键拖拽平移图像");
                } else if (circleFitActive) {
                    ImGui::TextColored(ImVec4(0.5f, 0.95f, 0.7f, 1.f), "%s",
                                       ArcMeasurePhaseHint(arcMeasurePhase_));
                    ImGui::TextDisabled(u8"点击已有圆弧拟合，或设置 A/B 后拖拽拱高提取");
                    ImGui::TextDisabled(u8"Shift+左键拖拽平移图像");
                } else if (lineDistActive) {
                    ImGui::TextColored(ImVec4(0.5f, 0.95f, 0.7f, 1.f),
                                       u8"线线距离：点击图像上的线段选 A / B");
                    ImGui::TextDisabled(u8"Shift+左键拖拽平移图像");
                } else if (arcDistActive) {
                    ImGui::TextColored(ImVec4(0.5f, 0.95f, 0.7f, 1.f),
                                       u8"圆弧距离：点击图像上的圆弧选 A / B");
                    ImGui::TextDisabled(u8"Shift+左键拖拽平移图像");
                } else if (imageSyncEnabled_) {
                    ImGui::TextDisabled(u8"单击同步到另一张图");
                }
                ImGui::TextDisabled(u8"Shift+滚轮缩放，Shift+左键平移，双击或「复位视图」恢复");
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
        depthMapParams_.skipZero = depthSkipZero_;
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

void Application::ConvertDepthToPointCloud(bool append) {
    if (depthImage_.gray.empty() || depthImage_.width <= 0 || depthImage_.height <= 0) {
        SetStatus(u8"请先打开深度图");
        return;
    }

    const bool brightOk = depthMapUseBrightness_ && brightnessImage_.valid() &&
                          brightnessImage_.width == depthImage_.width &&
                          brightnessImage_.height == depthImage_.height &&
                          !brightnessImage_.rgb.empty();
    if (depthMapUseBrightness_ && brightnessImage_.valid() && !brightOk) {
        SetStatus(u8"亮度图尺寸与深度图不一致，已忽略亮度着色");
    }

    PointCloud cloud;
    std::string error;
    const std::vector<uint8_t>* rgb = brightOk ? &brightnessImage_.rgb : nullptr;
    if (!PointCloudIO::BuildFromDepthGray(depthImage_.gray, depthImage_.width, depthImage_.height,
                                          rgb, depthMapParams_, cloud, error, depthImage_.path)) {
        SetStatus(error);
        return;
    }

    char buf[320];
    std::snprintf(buf, sizeof(buf), u8"深度转点云完成：%zu 点（%dx%d）", cloud.points.size(),
                  depthImage_.width, depthImage_.height);
    ApplyCloud(std::move(cloud), buf, append);
    EnterView3DMode();
}

void Application::DrawDepthToCloudParamFields() {
    ImGui::PushID("DepthToCloudParams");

    const auto labeledInputFloat = [](const char* label, float* v, const char* fmt = "%.6f") {
        ImGui::TextUnformatted(label);
        ImGui::PushID(label);
        ImGui::SetNextItemWidth(-1.f);
        ImGui::InputFloat("##val", v, 0.f, 0.f, fmt);
        ImGui::PopID();
    };

    const PointCloudIO::DepthGrayStats stats =
        PointCloudIO::AnalyzeDepthGray(depthImage_.gray, depthImage_.width, depthImage_.height);
    if (stats.validCount > 0) {
        char statBuf[280];
        std::snprintf(statBuf, sizeof(statBuf),
                      u8"有效 %zu / %zu  原始 [%g, %g]  有效 [%g, %g]", stats.validCount,
                      stats.totalPixels, stats.rawMin, stats.rawMax, stats.validMin,
                      stats.validMax);
        ImGui::TextWrapped("%s", statBuf);
        if (stats.zeroCount > 0 || stats.maxUint16Count > 0) {
            ImGui::TextDisabled(u8"无效像素: 0×%zu  65535×%zu", stats.zeroCount,
                                stats.maxUint16Count);
        }
    } else {
        ImGui::TextColored(ImVec4(1.f, 0.55f, 0.45f, 1.f), u8"未检测到有效深度像素");
    }

    if (ImGui::Button(u8"应用推荐参数", ImVec2(-1.f, 0.f))) {
        PointCloudIO::SuggestDepthMapParams(depthImage_.gray, depthImage_.width, depthImage_.height,
                                            depthMapParams_);
        depthMapParams_.skipZero = depthSkipZero_;
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip(u8"推荐无效值过滤与 Z 量化；不修改 XY 间隔");
    }

    ImGui::Spacing();
    if (ImGui::CollapsingHeader(u8"XY 扫描间隔（必填）", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::TextDisabled(u8"激光线方向 → X；扫描/编码器方向 → Y");
        labeledInputFloat(u8"X 间隔 (mm/px)", &depthMapParams_.pixelSizeX);
        if (depthMapParams_.pixelSizeX < 0.f) depthMapParams_.pixelSizeX = 0.f;
        labeledInputFloat(u8"Y 间隔 (mm/px)", &depthMapParams_.pixelSizeY);
        if (depthMapParams_.pixelSizeY < 0.f) depthMapParams_.pixelSizeY = 0.f;
        if (depthMapParams_.pixelSizeX <= 0.f || depthMapParams_.pixelSizeY <= 0.f) {
            ImGui::TextColored(ImVec4(1.f, 0.45f, 0.35f, 1.f), u8"XY 间隔必须大于 0");
        }
    }

    ImGui::Spacing();
    if (ImGui::CollapsingHeader(u8"Z 高度换算（必填）", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox(u8"16bit 量化深度（线扫）", &depthMapParams_.useQuantizedZ);
        if (depthMapParams_.useQuantizedZ) {
            labeledInputFloat(u8"Z 全量程 (mm)", &depthMapParams_.zFullRangeMm, "%.3f");
            if (depthMapParams_.zFullRangeMm < 0.f) depthMapParams_.zFullRangeMm = 0.f;
            labeledInputFloat(u8"Z 量化除数", &depthMapParams_.zQuantDivisor, "%.0f");
            if (depthMapParams_.zQuantDivisor < 1.f) depthMapParams_.zQuantDivisor = 1.f;
            const float zScale = PointCloudIO::EffectiveDepthScale(depthMapParams_);
            ImGui::TextDisabled(u8"Z = raw × (%.8f) + 偏移", zScale);
            ImGui::TextDisabled(u8"即 全量程 %.3f ÷ 量化除数 %.0f", depthMapParams_.zFullRangeMm,
                                depthMapParams_.zQuantDivisor);
        } else {
            labeledInputFloat(u8"Z 缩放 (mm/DN)", &depthMapParams_.depthScale, "%.8f");
        }
        labeledInputFloat(u8"Z 偏移 (mm)", &depthMapParams_.zOffset, "%.4f");
    }

    ImGui::Spacing();
    if (ImGui::CollapsingHeader(u8"无效值与坐标")) {
        labeledInputFloat(u8"自定义无效值", &depthMapParams_.invalidValue, "%.4f");
        if (ImGui::Checkbox(u8"跳过 0", &depthMapParams_.skipZero)) {
            depthSkipZero_ = depthMapParams_.skipZero;
            ComputeGrayRange(depthImage_.gray, depthSkipZero_, depthDataMin_, depthDataMax_);
            RebuildDepthDisplay();
        }
        ImGui::Checkbox(u8"跳过 65535", &depthMapParams_.skipUint16Max);
        ImGui::Checkbox(u8"翻转 Y 轴", &depthMapParams_.flipY);
        ImGui::Checkbox(u8"交换 X/Y", &depthMapParams_.swapXY);
        ImGui::Checkbox(u8"居中到原点", &depthMapParams_.centerToOrigin);
    }

    if (ImGui::CollapsingHeader(u8"其它")) {
        ImGui::TextUnformatted(u8"降采样步长");
        ImGui::SetNextItemWidth(-1.f);
        ImGui::InputInt("##step", &depthMapParams_.step);
        depthMapParams_.step = std::clamp(depthMapParams_.step, 1, 32);
        const bool brightOk = brightnessImage_.valid() &&
                              brightnessImage_.width == depthImage_.width &&
                              brightnessImage_.height == depthImage_.height;
        if (brightnessImage_.valid() && !brightOk) {
            ImGui::TextColored(ImVec4(1.f, 0.55f, 0.45f, 1.f), u8"亮度图尺寸与深度图不一致");
        }
        if (brightOk) {
            ImGui::Checkbox(u8"使用亮度图着色", &depthMapUseBrightness_);
        }
        ImGui::Checkbox(u8"追加到场景", &depthMapAppend_);
    }

    ImGui::PopID();
}

void Application::DrawDepthToCloudWindow() {
    if (!showDepthToCloudWindow_) return;

    ImGui::SetNextWindowSize(ImVec2(400.f, 580.f), ImGuiCond_FirstUseEver);
    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_FirstUseEver, ImVec2(0.5f, 0.5f));
    if (depthToCloudWindowFocus_) {
        ImGui::SetNextWindowFocus();
        depthToCloudWindowFocus_ = false;
    }

    if (!ImGui::Begin(u8"深度图转点云", &showDepthToCloudWindow_, ImGuiWindowFlags_NoCollapse)) {
        return;
    }

    if (depthImage_.gray.empty()) {
        ImGui::TextDisabled(u8"请先打开深度图");
        ImGui::End();
        return;
    }

    char titleBuf[128];
    std::snprintf(titleBuf, sizeof(titleBuf), u8"图像 %d × %d", depthImage_.width,
                  depthImage_.height);
    ImGui::TextDisabled("%s", titleBuf);
    ImGui::Separator();

    const float footerH = 46.f;
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(12.f, 10.f));
    ImGui::BeginChild(u8"##depth2cloud_scroll", ImVec2(0.f, -footerH), ImGuiChildFlags_Borders,
                      ImGuiWindowFlags_AlwaysVerticalScrollbar);
    DrawDepthToCloudParamFields();
    ImGui::EndChild();
    ImGui::PopStyleVar();

    ImGui::Separator();
    if (ImGui::Button(u8"生成点云", ImVec2(130.f, 36.f))) {
        ConvertDepthToPointCloud(depthMapAppend_);
        showDepthToCloudWindow_ = false;
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip(u8"按当前参数生成点云并切换到 3D");
    }
    ImGui::SameLine();
    if (ImGui::Button(u8"仅 2D 查看", ImVec2(120.f, 36.f))) {
        EnterView2DMode();
        showDepthToCloudWindow_ = false;
    }
    ImGui::SameLine();
    if (ImGui::Button(u8"关闭", ImVec2(80.f, 36.f))) {
        showDepthToCloudWindow_ = false;
    }

    ImGui::End();
}

void Application::DrawDepthToCloudControls() {
    if (depthImage_.gray.empty() || depthImage_.width <= 0 || depthImage_.height <= 0) return;

    ImGui::Separator();
    ImGui::TextDisabled(u8"深度图转点云");
    ImGui::TextWrapped(u8"XY 间隔与 Z 换算请在「深度图转点云」窗口中设置。");
    if (ImGui::Button(u8"打开参数窗口…", ImVec2(-1.f, 32.f))) {
        OpenDepthToCloudDialog();
    }
}

void Application::DrawImageViewControls() {
    if (!view2DMode_) return;

    const UiPalette& pal = GetUiPalette();
    UiSectionHeader(u8"2D 图像", nullptr, &pal.tool2D, true);

    const bool canSync = depthImage_.valid() && brightnessImage_.valid();
    if (!canSync) ImGui::BeginDisabled();
    if (ImGui::Button(imageSyncEnabled_ ? u8"关闭深度/亮度联动" : u8"启用深度/亮度联动",
                      ImVec2(-1.f, 0))) {
        if (imageSyncEnabled_) {
            imageSyncEnabled_ = false;
            ClearImageSyncPick();
            SetStatus(u8"已关闭深度/亮度联动");
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

    const bool both = depthImage_.valid() && brightnessImage_.valid();
    if (both && !imageSyncEnabled_) {
        ImGui::Spacing();
        ImGui::TextDisabled(u8"显示图像");
        const float halfW = (ImGui::GetContentRegionAvail().x - 4.f) * 0.5f;
        {
            const bool sel = imagePanelTab_ == 0;
            if (sel) ImGui::PushStyleColor(ImGuiCol_Button, pal.tool2D);
            if (ImGui::Button(u8"深度图", ImVec2(halfW, 0.f))) imagePanelTab_ = 0;
            if (sel) ImGui::PopStyleColor();
        }
        ImGui::SameLine(0.f, 4.f);
        {
            const bool sel = imagePanelTab_ == 1;
            if (sel) ImGui::PushStyleColor(ImGuiCol_Button, pal.tool2D);
            if (ImGui::Button(u8"亮度图", ImVec2(halfW, 0.f))) imagePanelTab_ = 1;
            if (sel) ImGui::PopStyleColor();
        }
    }

    if (depthImage_.valid() || brightnessImage_.valid()) {
        ImGui::Spacing();
        const float halfW = (ImGui::GetContentRegionAvail().x - 4.f) * 0.5f;
        if (ImGui::Button(u8"左转 90°", ImVec2(halfW, 0.f))) {
            RotateImages2d90CCW();
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip(u8"深度图与亮度图同步逆时针旋转 90°，便于竖图横向查看");
        }
        ImGui::SameLine(0.f, 4.f);
        if (ImGui::Button(u8"右转 90°", ImVec2(halfW, 0.f))) {
            RotateImages2d90CW();
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip(u8"深度图与亮度图同步顺时针旋转 90°，便于竖图横向查看");
        }
    }

    ImGui::Separator();
}

void Application::DrawImagePanel() {
    if (!HasImagePanel()) return;

    const ImGuiViewport* vp = ImGui::GetMainViewport();
    const float sidebarW = SidebarWidth();
    const float panelY = view3dY_;
    const float panelH = view3dH_;
    const float panelX = vp->Pos.x + sidebarW;
    const float imageW = std::max(vp->Size.x - sidebarW, 1.f);

    ImGui::SetNextWindowPos(ImVec2(panelX, panelY));
    ImGui::SetNextWindowSize(ImVec2(imageW, panelH));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(6.f, 6.f));
    ImGui::PushStyleColor(ImGuiCol_WindowBg, GetUiPalette().bgDeep);

    ImGui::Begin(u8"##2D图像停靠栏", nullptr,
                 ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                     ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoBringToFrontOnFocus |
                     ImGuiWindowFlags_NoScrollbar);

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
        DrawDepthToCloudControls();
    } else {
        if (imagePanelTab_ == 0 && !depthImage_.valid() && brightnessImage_.valid())
            imagePanelTab_ = 1;
        if (imagePanelTab_ == 1 && !brightnessImage_.valid() && depthImage_.valid())
            imagePanelTab_ = 0;

        ImageView* view = (imagePanelTab_ == 1) ? &brightnessImage_ : &depthImage_;
        if (view->valid()) {
            DrawImageWithSyncMarker(*view, imagePanelTab_ == 0 ? u8"深度图" : u8"亮度图");
            if (imagePanelTab_ == 0) {
                DrawDepthRenderControls();
                DrawDepthToCloudControls();
            }
        } else {
            ImGui::TextDisabled(view2DMode_ ? u8"请从「文件」打开深度图或亮度图" : u8"当前无图像");
        }
    }

    ImGui::End();
    ImGui::PopStyleColor();
    ImGui::PopStyleVar(3);
}

void Application::DrawToolbar(float y, float height) {
    const UiPalette& pal = GetUiPalette();
    const ImGuiViewport* vp = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(ImVec2(vp->Pos.x, y));
    ImGui::SetNextWindowSize(ImVec2(vp->Size.x, height));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(12.f, 6.f));
    ImGui::PushStyleColor(ImGuiCol_WindowBg, pal.bgBar);
    ImGui::Begin(u8"##工具栏", nullptr,
                 ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                     ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoCollapse |
                     ImGuiWindowFlags_NoBringToFrontOnFocus);

    const float btnH = height - 12.f;
    const bool canUndo2D = CanUndoMeasuredLine();
    const bool canUndoCloud = history_.CanUndo();
    const bool canUndo = canUndo2D || canUndoCloud;
    const bool canRedo = history_.CanRedo();

    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 4.f);
    if (!canUndo) ImGui::BeginDisabled();
    if (ImGui::Button(u8"撤销", ImVec2(52.f, btnH))) Undo2DOrCloud();
    if (!canUndo) ImGui::EndDisabled();
    ImGui::SameLine();
    if (!canRedo) ImGui::BeginDisabled();
    if (ImGui::Button(u8"重做", ImVec2(52.f, btnH))) Redo();
    if (!canRedo) ImGui::EndDisabled();
    ImGui::PopStyleVar();

    ImGui::SameLine(0.f, 14.f);
    ImGui::TextDisabled("|");
    ImGui::SameLine(0.f, 14.f);
    ImGui::TextDisabled(u8"工具");
    ImGui::SameLine();
    if (image2DTool_ != Image2DTool::None) {
        UiStatusBadge(Image2DToolLabel(image2DTool_), pal.tool2D, btnH);
        ImGui::SameLine();
        ImGui::TextDisabled(u8"2D");
    } else if (view2DMode_) {
        UiStatusBadge(u8"2D 模式", pal.tool2D, btnH);
    } else {
        UiStatusBadge(ToolModeLabel(measure_.mode), pal.tool3D, btnH);
    }

    ImGui::SameLine(0.f, 18.f);
    ImGui::TextDisabled("|");
    ImGui::SameLine(0.f, 18.f);
    ImGui::TextDisabled(u8"算法");
    ImGui::SameLine();
    const bool isPcl = EffectiveAlgoBackend() == AlgorithmBackend::PCL;
    UiStatusBadge(AlgorithmBackendLabel(EffectiveAlgoBackend()), isPcl ? pal.success : pal.native,
                  btnH);

    ImGui::SameLine(0.f, 18.f);
    ImGui::TextDisabled("|");
    ImGui::SameLine(0.f, 18.f);
    if (ImGui::Button(u8"清空显示", ImVec2(0.f, btnH))) {
        ClearToolVisuals(true);
    }

    if (filterCompareActive_) {
        ImGui::SameLine(0.f, 20.f);
        ImGui::TextDisabled("|");
        ImGui::SameLine(0.f, 20.f);
        UiStatusBadge(u8"滤波对比中", pal.warning, btnH);
        ImGui::SameLine();
        UiStatusBadge(FilterCompareViewModeLabel(filterCompareViewMode_), pal.accent, btnH);
        ImGui::SameLine();
        if (ImGui::Button(u8"应用", ImVec2(52.f, btnH))) ApplyFilterResult();
        ImGui::SameLine();
        if (ImGui::Button(u8"取消", ImVec2(52.f, btnH))) ClearFilterCompare();
    }

    // 底部强调线
    {
        ImDrawList* dl = ImGui::GetWindowDrawList();
        const ImVec2 wmin = ImGui::GetWindowPos();
        const ImVec2 wmax = ImVec2(wmin.x + ImGui::GetWindowWidth(), wmin.y + ImGui::GetWindowHeight());
        dl->AddLine(ImVec2(wmin.x, wmax.y - 1.f), ImVec2(wmax.x, wmax.y - 1.f),
                    ImGui::ColorConvertFloat4ToU32(pal.border), 1.f);
        dl->AddLine(ImVec2(wmin.x, wmax.y), ImVec2(wmin.x + 120.f, wmax.y),
                    ImGui::ColorConvertFloat4ToU32(pal.accent), 2.f);
    }

    ImGui::End();
    ImGui::PopStyleColor();
    ImGui::PopStyleVar(3);
}

void Application::ClearLineMeasure() {
    lineMeasureDragging_ = false;
    lineMeasureDragSource_ = -1;
}

void Application::ClearArcMeasure() {
    arcMeasurePhase_ = ArcMeasurePhase::PickA;
    arcMeasureSource_ = -1;
    arcBulgeDragging_ = false;
    arcRoiP0X_ = 0.f;
    arcRoiP0Y_ = 0.f;
    arcRoiP1X_ = 0.f;
    arcRoiP1Y_ = 0.f;
    arcRoiP2X_ = 0.f;
    arcRoiP2Y_ = 0.f;
}

void Application::ResetArcMeasurePick() {
    arcMeasurePhase_ = ArcMeasurePhase::PickA;
    arcMeasureSource_ = -1;
    arcBulgeDragging_ = false;
    arcRoiP0X_ = 0.f;
    arcRoiP0Y_ = 0.f;
    arcRoiP1X_ = 0.f;
    arcRoiP1Y_ = 0.f;
    arcRoiP2X_ = 0.f;
    arcRoiP2Y_ = 0.f;
}

void Application::CancelCaliperPending() {
    lineMeasurePending_ = false;
    lineMeasurePendingSource_ = -1;
    lineMeasurePendingResult_ = {};
    arcMeasurePending_ = false;
    arcMeasurePendingSource_ = -1;
    arcMeasurePendingResult_ = {};
}

void Application::ClearMeasuredLinesOnly() {
    ClearLineMeasure();
    lineMeasurePending_ = false;
    lineMeasurePendingSource_ = -1;
    lineMeasurePendingResult_ = {};
    measuredLines_.clear();
    nextMeasuredLineId_ = 1;
    ClearLineDistance();
    ClearLineAngle();
    ClearParallelLineDistance();
    ClearPointLineDistance();
    ClearPointProjection();
}

void Application::ClearMeasuredArcsOnly() {
    ClearArcMeasure();
    arcMeasurePending_ = false;
    arcMeasurePendingSource_ = -1;
    arcMeasurePendingResult_ = {};
    measuredArcs_.clear();
    nextMeasuredArcId_ = 1;
    ClearArcDistance();
    ClearArcLength();
    measuredCircleFits_.clear();
    nextCircleFitId_ = 1;
    CancelCircleFitPending();
    measuredEllipseFits_.clear();
    nextEllipseFitId_ = 1;
    CancelEllipseFitPending();
    ClearCircleGap();
    ClearConcentricity();
    ClearRoundness();
}

void Application::ClearAllMeasuredLines() {
    ClearMeasuredLinesOnly();
    ClearMeasuredArcsOnly();
    CancelCaliperPending();
    measuredPointDists_.clear();
    nextPointDistId_ = 1;
    pointDistPhase_ = PointPickPhase::PickA;
    measuredCaliperPoints_.clear();
    nextCaliperPointId_ = 1;
    CancelCaliperPointPending();
    measuredCircleCalipers_.clear();
    nextCircleCaliperId_ = 1;
    CancelCircleCaliperPending();
    circleCaliperPhase_ = CircleCaliperPhase::PickCenter;
    measuredThreePointCircles_.clear();
    nextThreePointCircleId_ = 1;
    ClearThreePointCircle();
    measuredRectCalipers_.clear();
    nextRectCaliperId_ = 1;
    CancelRectCaliperPending();
    measuredProfileWidths_.clear();
    nextProfileWidthId_ = 1;
    CancelProfileWidthPending();
    measuredRegionBlobs_.clear();
    nextRegionBlobId_ = 1;
    CancelRegionBlobPending();
    ClearTemplateMatch();
    ClearDepthHeightDiff();
    ClearDepthProfile();
}

bool Application::CanUndoMeasuredLine() const {
    if (image2DTool_ == Image2DTool::None) return false;
    if (image2DTool_ == Image2DTool::LineDistance) return false;
    if (image2DTool_ == Image2DTool::ArcDistance) return false;
    if (image2DTool_ == Image2DTool::LineAngle) return false;
    if (image2DTool_ == Image2DTool::CircleGap) return false;
    if (image2DTool_ == Image2DTool::PointLineDistance) return false;
    if (image2DTool_ == Image2DTool::ArcLength) return false;
    if (image2DTool_ == Image2DTool::ParallelLineDistance) return false;
    if (image2DTool_ == Image2DTool::PointProjection) return false;
    if (image2DTool_ == Image2DTool::Concentricity) return false;
    if (image2DTool_ == Image2DTool::Roundness) return false;
    if (image2DTool_ == Image2DTool::DepthHeightDiff) return false;
    if (image2DTool_ == Image2DTool::DepthProfile) return false;
    if (image2DTool_ == Image2DTool::TemplateMatch) {
        return templateMatchPending_ || !measuredTemplateMatches_.empty();
    }
    if (image2DTool_ == Image2DTool::ThreePointCircle) return !measuredThreePointCircles_.empty();
    if (image2DTool_ == Image2DTool::RectCaliper) {
        return rectCaliperPending_ || !measuredRectCalipers_.empty();
    }
    if (image2DTool_ == Image2DTool::EllipseFit) {
        return ellipseFitPending_ || !measuredEllipseFits_.empty();
    }
    if (image2DTool_ == Image2DTool::ProfileWidth) {
        return profileWidthPending_ || !measuredProfileWidths_.empty();
    }
    if (image2DTool_ == Image2DTool::RegionBlob) {
        return regionBlobPending_ || !measuredRegionBlobs_.empty();
    }
    if (image2DTool_ == Image2DTool::PointDistance) return !measuredPointDists_.empty();
    if (image2DTool_ == Image2DTool::CaliperPoint) {
        return caliperPointPending_ || !measuredCaliperPoints_.empty();
    }
    if (image2DTool_ == Image2DTool::CaliperCircle) {
        return circleCaliperPending_ || !measuredCircleCalipers_.empty();
    }
    if (image2DTool_ == Image2DTool::CircleFit) {
        return circleFitPending_ || !measuredCircleFits_.empty();
    }
    if (image2DTool_ == Image2DTool::CaliperArc) {
        return arcMeasurePending_ || !measuredArcs_.empty();
    }
    return lineMeasurePending_ || !measuredLines_.empty();
}

void Application::UndoLastMeasuredLine() {
    if (image2DTool_ == Image2DTool::ThreePointCircle) {
        if (measuredThreePointCircles_.empty()) return;
        measuredThreePointCircles_.pop_back();
        SetStatus(u8"已撤回三点圆");
        return;
    }
    if (image2DTool_ == Image2DTool::RectCaliper) {
        if (rectCaliperPending_) {
            CancelRectCaliperPending();
            SetStatus(u8"已撤回矩形卡尺预览");
            return;
        }
        if (measuredRectCalipers_.empty()) return;
        measuredRectCalipers_.pop_back();
        SetStatus(u8"已撤回矩形卡尺");
        return;
    }
    if (image2DTool_ == Image2DTool::EllipseFit) {
        if (ellipseFitPending_) {
            CancelEllipseFitPending();
            arcMeasurePhase_ = ArcMeasurePhase::DragBulge;
            SetStatus(u8"已撤回椭圆拟合预览");
            return;
        }
        if (measuredEllipseFits_.empty()) return;
        measuredEllipseFits_.pop_back();
        SetStatus(u8"已撤回椭圆拟合");
        return;
    }
    if (image2DTool_ == Image2DTool::ProfileWidth) {
        if (profileWidthPending_) {
            CancelProfileWidthPending();
            SetStatus(u8"已撤回测宽预览");
            return;
        }
        if (measuredProfileWidths_.empty()) return;
        measuredProfileWidths_.pop_back();
        SetStatus(u8"已撤回测宽结果");
        return;
    }
    if (image2DTool_ == Image2DTool::RegionBlob) {
        if (regionBlobPending_) {
            CancelRegionBlobPending();
            SetStatus(u8"已撤回区域预览");
            return;
        }
        if (measuredRegionBlobs_.empty()) return;
        measuredRegionBlobs_.pop_back();
        SetStatus(u8"已撤回区域结果");
        return;
    }
    if (image2DTool_ == Image2DTool::TemplateMatch) {
        if (templateMatchPending_) {
            CancelTemplateMatchPending();
            SetStatus(u8"已撤回模板匹配预览");
            return;
        }
        if (measuredTemplateMatches_.empty()) return;
        const int removedId = measuredTemplateMatches_.back().id;
        measuredTemplateMatches_.pop_back();
        char buf[64];
        std::snprintf(buf, sizeof(buf), u8"已撤回模板匹配%d", removedId);
        SetStatus(buf);
        return;
    }
    if (image2DTool_ == Image2DTool::PointDistance) {
        if (measuredPointDists_.empty()) return;
        const int removedId = measuredPointDists_.back().id;
        measuredPointDists_.pop_back();
        char buf[64];
        std::snprintf(buf, sizeof(buf), u8"已撤回测距%d", removedId);
        SetStatus(buf);
        return;
    }
    if (image2DTool_ == Image2DTool::CaliperPoint) {
        if (caliperPointPending_) {
            CancelCaliperPointPending();
            SetStatus(u8"已撤回待确认边缘点");
            return;
        }
        if (measuredCaliperPoints_.empty()) return;
        const int removedId = measuredCaliperPoints_.back().id;
        measuredCaliperPoints_.pop_back();
        char buf[64];
        std::snprintf(buf, sizeof(buf), u8"已撤回边缘点%d", removedId);
        SetStatus(buf);
        return;
    }
    if (image2DTool_ == Image2DTool::CaliperCircle) {
        if (circleCaliperPending_) {
            CancelCircleCaliperPending();
            circleCaliperPhase_ = CircleCaliperPhase::DragRadius;
            SetStatus(u8"已撤回待确认圆卡尺");
            return;
        }
        if (measuredCircleCalipers_.empty()) return;
        const int removedId = measuredCircleCalipers_.back().id;
        measuredCircleCalipers_.pop_back();
        char buf[64];
        std::snprintf(buf, sizeof(buf), u8"已撤回圆卡尺%d", removedId);
        SetStatus(buf);
        return;
    }
    if (image2DTool_ == Image2DTool::CircleFit) {
        if (circleFitPending_) {
            CancelCircleFitPending();
            arcMeasurePhase_ = ArcMeasurePhase::DragBulge;
            SetStatus(u8"已撤回待确认圆拟合，可继续调节");
            return;
        }
        if (measuredCircleFits_.empty()) return;
        const int removedId = measuredCircleFits_.back().id;
        measuredCircleFits_.pop_back();
        char buf[96];
        std::snprintf(buf, sizeof(buf), u8"已撤回圆拟合%d", removedId);
        SetStatus(buf);
        return;
    }

    if (image2DTool_ == Image2DTool::CaliperArc) {
        if (arcMeasurePending_) {
            arcMeasurePending_ = false;
            arcMeasurePendingSource_ = -1;
            arcMeasurePendingResult_ = {};
            arcMeasurePhase_ = ArcMeasurePhase::DragBulge;
            SetStatus(u8"已撤回待确认圆弧卡尺，可继续调节拱高");
            return;
        }
        if (measuredArcs_.empty()) return;
        const int removedId = measuredArcs_.back().id;
        measuredArcs_.pop_back();
        char buf[96];
        std::snprintf(buf, sizeof(buf), u8"已撤回圆弧%d", removedId);
        SetStatus(buf);
        return;
    }

    if (lineMeasurePending_) {
        lineMeasurePending_ = false;
        lineMeasurePendingSource_ = -1;
        lineMeasurePendingResult_ = {};
        SetStatus(u8"已撤回待确认卡尺");
        return;
    }
    if (measuredLines_.empty()) return;

    const int removedId = measuredLines_.back().id;
    measuredLines_.pop_back();
    if (lineDistPickA_ >= static_cast<int>(measuredLines_.size())) lineDistPickA_ = -1;
    if (lineDistPickB_ >= static_cast<int>(measuredLines_.size())) lineDistPickB_ = -1;
    lineDistValid_ = false;
    lineDistSamples_.clear();

    char buf[96];
    std::snprintf(buf, sizeof(buf), u8"已撤回线段%d", removedId);
    SetStatus(buf);
}

void Application::Undo2DOrCloud() {
    if (CanUndoMeasuredLine()) {
        UndoLastMeasuredLine();
        return;
    }
    Undo();
}

int Application::ImageSourceOf(const ImageView& view) const {
    return (&view == &brightnessImage_) ? 1 : 0;
}

Application::ImageView* Application::ImageViewFromSource(int source) {
    return source == 1 ? &brightnessImage_ : &depthImage_;
}

const Application::ImageView* Application::ImageViewFromSource(int source) const {
    return source == 1 ? &brightnessImage_ : &depthImage_;
}

void Application::PreviewLineMeasure(ImageView& view) {
    if (!view.valid()) return;

    lineMeasureParams_.skipZero = (!view.gray.empty() && ImageSourceOf(view) == 0) ? depthSkipZero_
                                                                                    : false;
    OpenCv2D::CaliperLineResult result;
    std::string error;
    bool ok = false;
    if (!view.gray.empty()) {
        ok = OpenCv2D::MeasureLineWithCalipers(
            view.gray, view.width, view.height, lineMeasureRoiX0_, lineMeasureRoiY0_,
            lineMeasureRoiX1_, lineMeasureRoiY1_, lineMeasureParams_, result, error);
    } else if (!view.rgb.empty()) {
        ok = OpenCv2D::MeasureLineWithCalipersRgb(
            view.rgb, view.width, view.height, lineMeasureRoiX0_, lineMeasureRoiY0_,
            lineMeasureRoiX1_, lineMeasureRoiY1_, lineMeasureParams_, result, error);
    } else {
        SetStatus(u8"当前图像无可用灰度数据");
        return;
    }

    if (!ok) {
        lineMeasurePending_ = false;
        lineMeasurePendingSource_ = -1;
        lineMeasurePendingResult_ = {};
        SetStatus(error);
        return;
    }

    lineMeasurePending_ = true;
    lineMeasurePendingSource_ = ImageSourceOf(view);
    lineMeasurePendingResult_ = std::move(result);
    SetStatus(u8"卡尺预览完成，请在左侧点击「确认」添加线段");
}

void Application::ConfirmLineMeasure() {
    if (!lineMeasurePending_ || !lineMeasurePendingResult_.ok) {
        SetStatus(u8"当前没有待确认的卡尺结果");
        return;
    }

    MeasuredImageLine entry;
    entry.id = nextMeasuredLineId_++;
    entry.imageSource = lineMeasurePendingSource_;
    entry.result = std::move(lineMeasurePendingResult_);
    measuredLines_.push_back(std::move(entry));
    lineMeasurePending_ = false;
    lineMeasurePendingSource_ = -1;
    lineMeasurePendingResult_ = {};
    lineDistValid_ = false;
    lineDistSamples_.clear();

    char buf[192];
    std::snprintf(buf, sizeof(buf), u8"线段%d 已确认：%d 个有效点，RMS %.3f px",
                  measuredLines_.back().id, measuredLines_.back().result.validCount,
                  measuredLines_.back().result.fitRms);
    SetStatus(buf);
}

void Application::PreviewArcMeasure(ImageView& view) {
    if (!view.valid()) return;

    lineMeasureParams_.skipZero = (!view.gray.empty() && ImageSourceOf(view) == 0) ? depthSkipZero_
                                                                                    : false;
    OpenCv2D::CaliperArcResult result;
    std::string error;
    bool ok = false;
    if (!view.gray.empty()) {
        ok = OpenCv2D::MeasureArcWithCalipers(
            view.gray, view.width, view.height, arcRoiP0X_, arcRoiP0Y_, arcRoiP1X_, arcRoiP1Y_,
            arcRoiP2X_, arcRoiP2Y_, lineMeasureParams_, result, error);
    } else if (!view.rgb.empty()) {
        ok = OpenCv2D::MeasureArcWithCalipersRgb(
            view.rgb, view.width, view.height, arcRoiP0X_, arcRoiP0Y_, arcRoiP1X_, arcRoiP1Y_,
            arcRoiP2X_, arcRoiP2Y_, lineMeasureParams_, result, error);
    } else {
        SetStatus(u8"当前图像无可用灰度数据");
        return;
    }

    if (!ok) {
        arcMeasurePending_ = false;
        arcMeasurePendingSource_ = -1;
        arcMeasurePendingResult_ = {};
        SetStatus(error);
        return;
    }

    arcMeasurePending_ = true;
    arcMeasurePendingSource_ = ImageSourceOf(view);
    arcMeasurePendingResult_ = std::move(result);
    SetStatus(u8"圆弧卡尺预览完成，请在左侧点击「确认」添加圆弧");
}

void Application::ConfirmArcMeasure() {
    if (!arcMeasurePending_ || !arcMeasurePendingResult_.ok) {
        SetStatus(u8"当前没有待确认的圆弧卡尺结果");
        return;
    }

    MeasuredImageArc entry;
    entry.id = nextMeasuredArcId_++;
    entry.imageSource = arcMeasurePendingSource_;
    entry.result = std::move(arcMeasurePendingResult_);
    measuredArcs_.push_back(std::move(entry));
    arcMeasurePending_ = false;
    arcMeasurePendingSource_ = -1;
    arcMeasurePendingResult_ = {};
    ResetArcMeasurePick();

    char buf[224];
    std::snprintf(buf, sizeof(buf), u8"圆弧%d 已确认：半径 %.3f px，%d 个有效点，RMS %.3f px",
                  measuredArcs_.back().id, measuredArcs_.back().result.fitRadius,
                  measuredArcs_.back().result.validCount, measuredArcs_.back().result.fitRms);
    SetStatus(buf);
}

void Application::CancelCircleFitPending() {
    circleFitPending_ = false;
    circleFitPendingSource_ = -1;
    circleFitPendingFromArcId_ = -1;
    circleFitPendingResult_ = {};
    circleFitPendingEdgePoints_.clear();
}

void Application::PreviewCircleFitFromRoi(ImageView& view) {
    if (!view.valid()) return;

    lineMeasureParams_.skipZero = (!view.gray.empty() && ImageSourceOf(view) == 0) ? depthSkipZero_
                                                                                    : false;
    OpenCv2D::CaliperArcResult arcResult;
    std::string error;
    bool ok = false;
    if (!view.gray.empty()) {
        ok = OpenCv2D::MeasureArcWithCalipers(
            view.gray, view.width, view.height, arcRoiP0X_, arcRoiP0Y_, arcRoiP1X_, arcRoiP1Y_,
            arcRoiP2X_, arcRoiP2Y_, lineMeasureParams_, arcResult, error);
    } else if (!view.rgb.empty()) {
        ok = OpenCv2D::MeasureArcWithCalipersRgb(
            view.rgb, view.width, view.height, arcRoiP0X_, arcRoiP0Y_, arcRoiP1X_, arcRoiP1Y_,
            arcRoiP2X_, arcRoiP2Y_, lineMeasureParams_, arcResult, error);
    } else {
        SetStatus(u8"当前图像无可用灰度数据");
        return;
    }

    if (!ok || !arcResult.ok) {
        CancelCircleFitPending();
        SetStatus(error.empty() ? u8"圆弧边缘提取失败" : error);
        return;
    }

    OpenCv2D::CircleFitResult fit;
    if (!OpenCv2D::FitCircleFromEdgePoints(arcResult.edgePoints, fit)) {
        CancelCircleFitPending();
        SetStatus(u8"圆拟合失败，有效边缘点不足");
        return;
    }

    circleFitPending_ = true;
    circleFitPendingSource_ = ImageSourceOf(view);
    circleFitPendingFromArcId_ = -1;
    circleFitPendingResult_ = fit;
    circleFitPendingEdgePoints_ = arcResult.edgePoints;
    SetStatus(u8"圆拟合预览完成，请在左侧确认（半径已计算）");
}

void Application::PreviewCircleFitFromMeasuredArc(int arcIndex) {
    if (arcIndex < 0 || arcIndex >= static_cast<int>(measuredArcs_.size())) return;

    const MeasuredImageArc& arc = measuredArcs_[static_cast<std::size_t>(arcIndex)];
    if (!arc.result.ok) {
        SetStatus(u8"所选圆弧无效");
        return;
    }

    OpenCv2D::CircleFitResult fit;
    if (!OpenCv2D::FitCircleFromEdgePoints(arc.result.edgePoints, fit)) {
        SetStatus(u8"圆拟合失败，该圆弧有效边缘点不足");
        return;
    }

    circleFitPending_ = true;
    circleFitPendingSource_ = arc.imageSource;
    circleFitPendingFromArcId_ = arc.id;
    circleFitPendingResult_ = fit;
    circleFitPendingEdgePoints_ = arc.result.edgePoints;
    char buf[128];
    std::snprintf(buf, sizeof(buf), u8"圆弧%d 圆拟合预览：半径 %.3f px，%d 点，RMS %.3f px", arc.id,
                  fit.radius, fit.pointCount, fit.rms);
    SetStatus(buf);
}

void Application::ConfirmCircleFit() {
    if (!circleFitPending_ || !circleFitPendingResult_.ok) {
        SetStatus(u8"当前没有待确认的圆拟合结果");
        return;
    }

    MeasuredCircleFit entry;
    entry.id = nextCircleFitId_++;
    entry.imageSource = circleFitPendingSource_;
    entry.sourceArcId = circleFitPendingFromArcId_;
    entry.result = circleFitPendingResult_;
    entry.edgePoints = circleFitPendingEdgePoints_;
    measuredCircleFits_.push_back(std::move(entry));
    CancelCircleFitPending();
    ResetArcMeasurePick();

    char buf[192];
    std::snprintf(buf, sizeof(buf), u8"圆拟合%d 已确认：半径 = %.4f px（圆心 %.1f, %.1f）",
                  measuredCircleFits_.back().id, measuredCircleFits_.back().result.radius,
                  measuredCircleFits_.back().result.centerX, measuredCircleFits_.back().result.centerY);
    SetStatus(buf);
}

void Application::ComputeSelectedLineDistance() {
    lineDistValid_ = false;
    lineDistSamples_.clear();
    if (lineDistPickA_ < 0 || lineDistPickB_ < 0 ||
        lineDistPickA_ >= static_cast<int>(measuredLines_.size()) ||
        lineDistPickB_ >= static_cast<int>(measuredLines_.size()) ||
        lineDistPickA_ == lineDistPickB_) {
        SetStatus(u8"请选择两条不同的线段");
        return;
    }

    const MeasuredImageLine& la = measuredLines_[static_cast<std::size_t>(lineDistPickA_)];
    const MeasuredImageLine& lb = measuredLines_[static_cast<std::size_t>(lineDistPickB_)];
    if (la.imageSource != lb.imageSource) {
        SetStatus(u8"两条线段须在同一张图像上");
        return;
    }
    if (!la.result.ok || !lb.result.ok) {
        SetStatus(u8"所选线段无效");
        return;
    }

    OpenCv2D::AverageGapResult gap;
    if (!OpenCv2D::AverageGapDistance(
            la.result.fitX1, la.result.fitY1, la.result.fitX2, la.result.fitY2, lb.result.fitX1,
            lb.result.fitY1, lb.result.fitX2, lb.result.fitY2, lineDistSampleCount_, gap)) {
        SetStatus(u8"垂线与线段 B 无有效交点，请调整选线或采样数");
        return;
    }

    lineDistPx_ = gap.average;
    lineDistMinPx_ = gap.minDist;
    lineDistMaxPx_ = gap.maxDist;
    lineDistSamples_ = gap.samples;
    lineDistValid_ = true;

    char buf[224];
    std::snprintf(buf, sizeof(buf),
                  u8"线段%d(A) → 线段%d(B) 平均间隙 = %.3f px（%d/%d 点有效）", la.id, lb.id,
                  lineDistPx_, gap.validCount, gap.totalSamples);
    SetStatus(buf);
}

void Application::ClearLineDistance() {
    lineDistPickA_ = -1;
    lineDistPickB_ = -1;
    lineDistValid_ = false;
    lineDistSamples_.clear();
}

void Application::ResetImage2dView() {
    image2dZoom_ = 1.f;
    image2dPanX_ = 0.f;
    image2dPanY_ = 0.f;
}

void Application::EnterView2DMode() {
    view2DMode_ = true;
}

void Application::EnterView3DMode() {
    view2DMode_ = false;
    image2DTool_ = Image2DTool::None;
}

void Application::RestoreApplicationState() {
    if (algoEditor_.IsVisible()) {
        algoEditor_.SetVisible(false);
    }
    shapeTemplateWindow_.SetVisible(false);
    halconMatchWindow_.SetVisible(false);
    ocrWindow_.SetVisible(false);
    camera2DCalibWindow_.SetVisible(false);
    cameraIntrinsicsCalibWindow_.SetVisible(false);
    multiViewGeometryWindow_.SetVisible(false);

    showAbout_ = false;
    showCreateSphere_ = false;
    showCreateCylinder_ = false;
    showCreateDisk_ = false;
    showCreatePlane_ = false;
    showNativeAlgoPassword_ = false;

    CloseDualCloudView();
    ClearIcpState();
    ClearFilterCompare();

    ShutdownSceneLayers();
    sceneLayers_.clear();
    activeLayerIndex_ = -1;
    dbSelectedLayerId_ = 0;
    dbSelectedSpecial_ = DbEntityId::None;
    nextSceneLayerId_ = 1;
    history_.Clear();
    useIntensityColors_ = false;
    intensityColors_.clear();

    DestroyImageView(depthImage_);
    DestroyImageView(brightnessImage_);
    imagePanelTab_ = 0;

    ClearAllMeasuredLines();
    ClearDepthHeightDiff();
    ClearDepthProfile();

    EnterView3DMode();
    measure_ = {};
    measure_.mode = ToolMode::Navigate;
    ClearToolVisuals(false);
    ResetImage2dView();
    imageSyncEnabled_ = false;
    ClearImageSyncPick();
    showSectionPanel_ = false;
    showDepthProfilePanel_ = false;
    sidebarPreferredW_ = 360.f;
    dbTreePreferredH_ = 200.f;
    consoleHeight_ = 168.f;
    cachedViewLayoutValid_ = false;
    autoZRange_ = true;
    zMin_ = 0.f;
    zMax_ = 1.f;
    camera_.Reset();
    sceneGpuDirty_ = true;
    SetStatus(u8"请打开点云文件（PLY / PCD / XYZ / OBJ），或打开深度/亮度图对照查看");
}

bool Application::RotateImageView90CW(ImageView& view) {
    if (view.width <= 0 || view.height <= 0) return false;
    const int oldW = view.width;
    const int oldH = view.height;
    int newW = 0;
    int newH = 0;

    if (!view.gray.empty()) {
        std::vector<float> rotated;
        RotateGray90CW(view.gray, oldW, oldH, rotated, newW, newH);
        view.gray = std::move(rotated);
        view.width = newW;
        view.height = newH;
        BuildDepthRgb(view.gray, view.rgb, depthDisplayMin_, depthDisplayMax_, depthSkipZero_);
    } else if (!view.rgb.empty()) {
        std::vector<uint8_t> rotated;
        RotateRgb90CW(view.rgb, oldW, oldH, rotated, newW, newH);
        view.rgb = std::move(rotated);
        view.width = newW;
        view.height = newH;
    } else {
        return false;
    }

    if (!UploadImageTexture(view)) return false;
    return true;
}

bool Application::RotateImageView90CCW(ImageView& view) {
    if (view.width <= 0 || view.height <= 0) return false;
    const int oldW = view.width;
    const int oldH = view.height;
    int newW = 0;
    int newH = 0;

    if (!view.gray.empty()) {
        std::vector<float> rotated;
        RotateGray90CCW(view.gray, oldW, oldH, rotated, newW, newH);
        view.gray = std::move(rotated);
        view.width = newW;
        view.height = newH;
        BuildDepthRgb(view.gray, view.rgb, depthDisplayMin_, depthDisplayMax_, depthSkipZero_);
    } else if (!view.rgb.empty()) {
        std::vector<uint8_t> rotated;
        RotateRgb90CCW(view.rgb, oldW, oldH, rotated, newW, newH);
        view.rgb = std::move(rotated);
        view.width = newW;
        view.height = newH;
    } else {
        return false;
    }

    if (!UploadImageTexture(view)) return false;
    return true;
}

void Application::RotateImages2d90CW() {
    bool rotated = false;
    if (depthImage_.width > 0 && depthImage_.height > 0 && !depthImage_.gray.empty()) {
        rotated = RotateImageView90CW(depthImage_) || rotated;
    }
    if (brightnessImage_.width > 0 && brightnessImage_.height > 0 &&
        !brightnessImage_.rgb.empty()) {
        rotated = RotateImageView90CW(brightnessImage_) || rotated;
    }
    if (!rotated) {
        SetStatus(u8"当前没有可旋转的图像");
        return;
    }

    if (imageSyncEnabled_ && depthImage_.valid() && brightnessImage_.valid()) {
        if (depthImage_.width != brightnessImage_.width ||
            depthImage_.height != brightnessImage_.height) {
            imageSyncEnabled_ = false;
            ClearImageSyncPick();
            SetStatus(u8"旋转后深度图与亮度图尺寸不一致，已关闭联动");
        } else {
            syncWidth_ = depthImage_.width;
            syncHeight_ = depthImage_.height;
            ClearImageSyncPick();
        }
    } else {
        ClearImageSyncPick();
    }

    ClearAllMeasuredLines();
    ResetImage2dView();
    SetStatus(u8"图像已顺时针旋转 90°");
}

void Application::RotateImages2d90CCW() {
    bool rotated = false;
    if (depthImage_.width > 0 && depthImage_.height > 0 && !depthImage_.gray.empty()) {
        rotated = RotateImageView90CCW(depthImage_) || rotated;
    }
    if (brightnessImage_.width > 0 && brightnessImage_.height > 0 &&
        !brightnessImage_.rgb.empty()) {
        rotated = RotateImageView90CCW(brightnessImage_) || rotated;
    }
    if (!rotated) {
        SetStatus(u8"当前没有可旋转的图像");
        return;
    }

    if (imageSyncEnabled_ && depthImage_.valid() && brightnessImage_.valid()) {
        if (depthImage_.width != brightnessImage_.width ||
            depthImage_.height != brightnessImage_.height) {
            imageSyncEnabled_ = false;
            ClearImageSyncPick();
            SetStatus(u8"旋转后深度图与亮度图尺寸不一致，已关闭联动");
        } else {
            syncWidth_ = depthImage_.width;
            syncHeight_ = depthImage_.height;
            ClearImageSyncPick();
        }
    } else {
        ClearImageSyncPick();
    }

    ClearAllMeasuredLines();
    ResetImage2dView();
    SetStatus(u8"图像已逆时针旋转 90°");
}

void Application::ComputeSelectedArcDistance() {
    arcDistValid_ = false;
    arcDistSamples_.clear();
    if (arcDistPickA_ < 0 || arcDistPickB_ < 0 ||
        arcDistPickA_ >= static_cast<int>(measuredArcs_.size()) ||
        arcDistPickB_ >= static_cast<int>(measuredArcs_.size()) ||
        arcDistPickA_ == arcDistPickB_) {
        SetStatus(u8"请选择两条不同的圆弧");
        return;
    }

    const MeasuredImageArc& aa = measuredArcs_[static_cast<std::size_t>(arcDistPickA_)];
    const MeasuredImageArc& ab = measuredArcs_[static_cast<std::size_t>(arcDistPickB_)];
    if (aa.imageSource != ab.imageSource) {
        SetStatus(u8"两条圆弧须在同一张图像上");
        return;
    }
    if (!aa.result.ok || !ab.result.ok) {
        SetStatus(u8"所选圆弧无效");
        return;
    }

    OpenCv2D::AverageGapResult gap;
    if (!OpenCv2D::AverageArcGapDistance(
            aa.result.fitCenterX, aa.result.fitCenterY, aa.result.fitRadius,
            aa.result.fitStartAngle, aa.result.fitEndAngle, ab.result.fitCenterX,
            ab.result.fitCenterY, ab.result.fitRadius, ab.result.fitStartAngle,
            ab.result.fitEndAngle, arcDistSampleCount_, gap)) {
        SetStatus(u8"法向线与圆弧 B 无有效交点，请调整选弧或采样数");
        return;
    }

    arcDistPx_ = gap.average;
    arcDistMinPx_ = gap.minDist;
    arcDistMaxPx_ = gap.maxDist;
    arcDistSamples_ = gap.samples;
    arcDistValid_ = true;

    char buf[224];
    std::snprintf(buf, sizeof(buf),
                  u8"圆弧%d(A) → 圆弧%d(B) 平均间隙 = %.3f px（%d/%d 点有效）", aa.id, ab.id,
                  arcDistPx_, gap.validCount, gap.totalSamples);
    SetStatus(buf);
}

void Application::ClearArcDistance() {
    arcDistPickA_ = -1;
    arcDistPickB_ = -1;
    arcDistValid_ = false;
    arcDistSamples_.clear();
}

int Application::FindClosestMeasuredCircleFit(int imageSource, float px, float py,
                                              float maxDistPx) const {
    int bestIdx = -1;
    float bestDist = maxDistPx;
    for (std::size_t i = 0; i < measuredCircleFits_.size(); ++i) {
        const MeasuredCircleFit& fit = measuredCircleFits_[i];
        if (fit.imageSource != imageSource || !fit.result.ok) continue;
        const float d = std::fabs(
            std::hypot(px - fit.result.centerX, py - fit.result.centerY) - fit.result.radius);
        if (d < bestDist) {
            bestDist = d;
            bestIdx = static_cast<int>(i);
        }
    }
    return bestIdx;
}

void Application::AddPointDistance(float ax, float ay, float bx, float by, int imageSource) {
    MeasuredPointDist entry;
    entry.id = nextPointDistId_++;
    entry.imageSource = imageSource;
    entry.ax = ax;
    entry.ay = ay;
    entry.bx = bx;
    entry.by = by;
    entry.distance = OpenCv2D::PointPointDistance(ax, ay, bx, by, &entry.dx, &entry.dy);
    measuredPointDists_.push_back(entry);
    char buf[128];
    std::snprintf(buf, sizeof(buf), u8"测距%d：距离 %.3f px（ΔX %.3f  ΔY %.3f）", entry.id,
                  entry.distance, entry.dx, entry.dy);
    SetStatus(buf);
}

void Application::PickLineForAngle(int lineIndex) {
    if (lineIndex < 0 || lineIndex >= static_cast<int>(measuredLines_.size())) return;
    lineAngleValid_ = false;
    const int lineId = measuredLines_[static_cast<std::size_t>(lineIndex)].id;
    if (lineAnglePickA_ < 0) {
        lineAnglePickA_ = lineIndex;
        char buf[64];
        std::snprintf(buf, sizeof(buf), u8"线段%d 已选为 A", lineId);
        SetStatus(buf);
        return;
    }
    if (lineAnglePickB_ < 0 && lineIndex != lineAnglePickA_) {
        lineAnglePickB_ = lineIndex;
        char buf[64];
        std::snprintf(buf, sizeof(buf), u8"线段%d 已选为 B", lineId);
        SetStatus(buf);
        return;
    }
    lineAnglePickA_ = lineIndex;
    lineAnglePickB_ = -1;
    lineAngleValid_ = false;
    char buf[64];
    std::snprintf(buf, sizeof(buf), u8"线段%d 已重选为 A", lineId);
    SetStatus(buf);
}

void Application::ComputeSelectedLineAngle() {
    lineAngleValid_ = false;
    if (lineAnglePickA_ < 0 || lineAnglePickB_ < 0 ||
        lineAnglePickA_ >= static_cast<int>(measuredLines_.size()) ||
        lineAnglePickB_ >= static_cast<int>(measuredLines_.size()) ||
        lineAnglePickA_ == lineAnglePickB_) {
        SetStatus(u8"请选择两条不同的线段");
        return;
    }
    const MeasuredImageLine& la = measuredLines_[static_cast<std::size_t>(lineAnglePickA_)];
    const MeasuredImageLine& lb = measuredLines_[static_cast<std::size_t>(lineAnglePickB_)];
    if (la.imageSource != lb.imageSource || !la.result.ok || !lb.result.ok) {
        SetStatus(u8"所选线段无效或不在同一张图");
        return;
    }
    lineAngleDeg_ = OpenCv2D::AngleBetweenSegments(
        la.result.fitX1, la.result.fitY1, la.result.fitX2, la.result.fitY2, lb.result.fitX1,
        lb.result.fitY1, lb.result.fitX2, lb.result.fitY2, true);
    lineAngleValid_ = true;
    char buf[128];
    std::snprintf(buf, sizeof(buf), u8"线段%d 与 线段%d 夹角 = %.3f°", la.id, lb.id,
                  lineAngleDeg_);
    SetStatus(buf);
}

void Application::ClearLineAngle() {
    lineAnglePickA_ = -1;
    lineAnglePickB_ = -1;
    lineAngleValid_ = false;
    lineAngleDeg_ = 0.f;
}

void Application::PickCircleForGap(int circleIndex) {
    if (circleIndex < 0 || circleIndex >= static_cast<int>(measuredCircleFits_.size())) return;
    circleGapValid_ = false;
    const int circleId = measuredCircleFits_[static_cast<std::size_t>(circleIndex)].id;
    if (circleGapPickA_ < 0) {
        circleGapPickA_ = circleIndex;
        char buf[64];
        std::snprintf(buf, sizeof(buf), u8"圆%d 已选为 A", circleId);
        SetStatus(buf);
        return;
    }
    if (circleGapPickB_ < 0 && circleIndex != circleGapPickA_) {
        circleGapPickB_ = circleIndex;
        char buf[64];
        std::snprintf(buf, sizeof(buf), u8"圆%d 已选为 B", circleId);
        SetStatus(buf);
        return;
    }
    circleGapPickA_ = circleIndex;
    circleGapPickB_ = -1;
    circleGapValid_ = false;
    char buf[64];
    std::snprintf(buf, sizeof(buf), u8"圆%d 已重选为 A", circleId);
    SetStatus(buf);
}

void Application::ComputeSelectedCircleGap() {
    circleGapValid_ = false;
    if (circleGapPickA_ < 0 || circleGapPickB_ < 0 ||
        circleGapPickA_ >= static_cast<int>(measuredCircleFits_.size()) ||
        circleGapPickB_ >= static_cast<int>(measuredCircleFits_.size()) ||
        circleGapPickA_ == circleGapPickB_) {
        SetStatus(u8"请选择两个不同的圆");
        return;
    }
    const MeasuredCircleFit& ca = measuredCircleFits_[static_cast<std::size_t>(circleGapPickA_)];
    const MeasuredCircleFit& cb = measuredCircleFits_[static_cast<std::size_t>(circleGapPickB_)];
    if (ca.imageSource != cb.imageSource || !ca.result.ok || !cb.result.ok) {
        SetStatus(u8"所选圆无效或不在同一张图");
        return;
    }
    OpenCv2D::CircleGapResult gap;
    if (!OpenCv2D::ComputeCircleGap(ca.result.centerX, ca.result.centerY, ca.result.radius,
                                    cb.result.centerX, cb.result.centerY, cb.result.radius,
                                    gap)) {
        SetStatus(u8"圆间隙计算失败");
        return;
    }
    circleGapCenterDist_ = gap.centerDistance;
    circleGapSurfaceGap_ = gap.surfaceGap;
    circleGapValid_ = true;
    char buf[160];
    std::snprintf(buf, sizeof(buf), u8"圆%d 与 圆%d：圆心距 %.3f px，表面间隙 %.3f px", ca.id,
                  cb.id, circleGapCenterDist_, circleGapSurfaceGap_);
    SetStatus(buf);
}

void Application::ClearCircleGap() {
    circleGapPickA_ = -1;
    circleGapPickB_ = -1;
    circleGapValid_ = false;
    circleGapCenterDist_ = 0.f;
    circleGapSurfaceGap_ = 0.f;
}

void Application::PickPointForPointLine(float px, float py, int imageSource) {
    pointLinePx_ = px;
    pointLinePy_ = py;
    pointLineSource_ = imageSource;
    pointLinePhase_ = PointPickPhase::PickB;
    pointLineValid_ = false;
    SetStatus(u8"已选点，请点击一条线段");
}

void Application::PickLineForPointLine(int lineIndex) {
    if (lineIndex < 0 || lineIndex >= static_cast<int>(measuredLines_.size())) return;
    pointLinePick_ = lineIndex;
    pointLineValid_ = false;
    const int lineId = measuredLines_[static_cast<std::size_t>(lineIndex)].id;
    char buf[64];
    std::snprintf(buf, sizeof(buf), u8"线段%d 已选中", lineId);
    SetStatus(buf);
}

void Application::ComputePointLineDistance() {
    pointLineValid_ = false;
    if (pointLinePhase_ != PointPickPhase::PickB || pointLinePick_ < 0 ||
        pointLinePick_ >= static_cast<int>(measuredLines_.size())) {
        SetStatus(u8"请先选点，再选线段");
        return;
    }
    const MeasuredImageLine& line = measuredLines_[static_cast<std::size_t>(pointLinePick_)];
    if (line.imageSource != pointLineSource_ || !line.result.ok) {
        SetStatus(u8"点与线段须在同一张图");
        return;
    }
    if (!OpenCv2D::PointToSegmentDistance(pointLinePx_, pointLinePy_, line.result.fitX1,
                                            line.result.fitY1, line.result.fitX2,
                                            line.result.fitY2, pointLineDistPx_, pointLineFootX_,
                                            pointLineFootY_)) {
        SetStatus(u8"点线距离计算失败");
        return;
    }
    pointLineValid_ = true;
    char buf[128];
    std::snprintf(buf, sizeof(buf), u8"点到线段%d 垂直距离 = %.3f px", line.id,
                  pointLineDistPx_);
    SetStatus(buf);
}

void Application::ClearPointLineDistance() {
    pointLinePhase_ = PointPickPhase::PickA;
    pointLineSource_ = -1;
    pointLinePick_ = -1;
    pointLineValid_ = false;
    pointLineDistPx_ = 0.f;
}

void Application::PreviewCaliperPoint(ImageView& view) {
    if (!view.valid()) return;
    OpenCv2D::CaliperLineParams params = lineMeasureParams_;
    params.numCalipers = 1;
    params.skipZero = (!view.gray.empty() && ImageSourceOf(view) == 0) ? depthSkipZero_ : false;
    OpenCv2D::CaliperLineResult result;
    std::string error;
    bool ok = false;
    if (!view.gray.empty()) {
        ok = OpenCv2D::MeasureLineWithCalipers(
            view.gray, view.width, view.height, caliperPointRoiX0_, caliperPointRoiY0_,
            caliperPointRoiX1_, caliperPointRoiY1_, params, result, error);
    } else if (!view.rgb.empty()) {
        ok = OpenCv2D::MeasureLineWithCalipersRgb(
            view.rgb, view.width, view.height, caliperPointRoiX0_, caliperPointRoiY0_,
            caliperPointRoiX1_, caliperPointRoiY1_, params, result, error);
    } else {
        SetStatus(u8"当前图像无可用灰度数据");
        return;
    }
    if (!ok || result.validCount < 1) {
        caliperPointPending_ = false;
        SetStatus(error.empty() ? u8"未检测到边缘" : error);
        return;
    }
    for (const OpenCv2D::CaliperEdgePoint& ep : result.edgePoints) {
        if (ep.valid) {
            caliperPointPendingEdge_ = ep;
            caliperPointPending_ = true;
            caliperPointPendingSource_ = ImageSourceOf(view);
            SetStatus(u8"单点卡尺预览完成，请在左侧确认");
            return;
        }
    }
    caliperPointPending_ = false;
    SetStatus(u8"未检测到有效边缘点");
}

void Application::ConfirmCaliperPoint() {
    if (!caliperPointPending_ || !caliperPointPendingEdge_.valid) {
        SetStatus(u8"当前没有待确认的单点卡尺");
        return;
    }
    MeasuredCaliperPoint entry;
    entry.id = nextCaliperPointId_++;
    entry.imageSource = caliperPointPendingSource_;
    entry.x = caliperPointPendingEdge_.x;
    entry.y = caliperPointPendingEdge_.y;
    entry.roiX0 = caliperPointRoiX0_;
    entry.roiY0 = caliperPointRoiY0_;
    entry.roiX1 = caliperPointRoiX1_;
    entry.roiY1 = caliperPointRoiY1_;
    measuredCaliperPoints_.push_back(entry);
    caliperPointPending_ = false;
    char buf[96];
    std::snprintf(buf, sizeof(buf), u8"边缘点%d 已确认 (%.2f, %.2f)", entry.id, entry.x, entry.y);
    SetStatus(buf);
}

void Application::CancelCaliperPointPending() {
    caliperPointPending_ = false;
    caliperPointPendingSource_ = -1;
    caliperPointPendingEdge_ = {};
}

void Application::PreviewCircleCaliper(ImageView& view) {
    if (!view.valid()) return;
    OpenCv2D::CaliperCircleParams params = lineMeasureParams_;
    params.skipZero = (!view.gray.empty() && ImageSourceOf(view) == 0) ? depthSkipZero_ : false;
    OpenCv2D::CaliperCircleResult result;
    std::string error;
    bool ok = false;
    if (!view.gray.empty()) {
        ok = OpenCv2D::MeasureCircleWithCalipers(view.gray, view.width, view.height, circleCaliperCx_,
                                                 circleCaliperCy_, circleCaliperR_, params, result,
                                                 error);
    } else if (!view.rgb.empty()) {
        ok = OpenCv2D::MeasureCircleWithCalipersRgb(
            view.rgb, view.width, view.height, circleCaliperCx_, circleCaliperCy_, circleCaliperR_,
            params, result, error);
    } else {
        SetStatus(u8"当前图像无可用灰度数据");
        return;
    }
    if (!ok) {
        circleCaliperPending_ = false;
        SetStatus(error);
        return;
    }
    circleCaliperPending_ = true;
    circleCaliperPendingSource_ = ImageSourceOf(view);
    circleCaliperPendingResult_ = std::move(result);
    SetStatus(u8"圆卡尺预览完成，请在左侧点击「确认」");
}

void Application::ConfirmCircleCaliper() {
    if (!circleCaliperPending_ || !circleCaliperPendingResult_.ok) {
        SetStatus(u8"当前没有待确认的圆卡尺");
        return;
    }
    MeasuredCircleCaliper entry;
    entry.id = nextCircleCaliperId_++;
    entry.imageSource = circleCaliperPendingSource_;
    entry.result = std::move(circleCaliperPendingResult_);
    measuredCircleCalipers_.push_back(std::move(entry));
    circleCaliperPending_ = false;
    circleCaliperPhase_ = CircleCaliperPhase::PickCenter;
    char buf[128];
    std::snprintf(buf, sizeof(buf), u8"圆卡尺%d 已确认：半径 %.3f px，RMS %.3f",
                  measuredCircleCalipers_.back().id, measuredCircleCalipers_.back().result.fitRadius,
                  measuredCircleCalipers_.back().result.fitRms);
    SetStatus(buf);
}

void Application::CancelCircleCaliperPending() {
    circleCaliperPending_ = false;
    circleCaliperPendingSource_ = -1;
    circleCaliperPendingResult_ = {};
}

void Application::PickArcForLength(int arcIndex) {
    if (arcIndex < 0 || arcIndex >= static_cast<int>(measuredArcs_.size())) return;
    arcLengthPick_ = arcIndex;
    arcLengthValid_ = false;
    const int arcId = measuredArcs_[static_cast<std::size_t>(arcIndex)].id;
    char buf[64];
    std::snprintf(buf, sizeof(buf), u8"圆弧%d 已选中", arcId);
    SetStatus(buf);
}

void Application::ComputeSelectedArcLength() {
    arcLengthValid_ = false;
    if (arcLengthPick_ < 0 || arcLengthPick_ >= static_cast<int>(measuredArcs_.size())) {
        SetStatus(u8"请先选择一条圆弧");
        return;
    }
    const MeasuredImageArc& arc = measuredArcs_[static_cast<std::size_t>(arcLengthPick_)];
    if (!arc.result.ok) {
        SetStatus(u8"所选圆弧无效");
        return;
    }
    if (!OpenCv2D::ComputeArcMetrics(arc.result.fitCenterX, arc.result.fitCenterY,
                                     arc.result.fitRadius, arc.result.fitStartAngle,
                                     arc.result.fitEndAngle, arcLengthMetrics_)) {
        SetStatus(u8"弧长计算失败");
        return;
    }
    arcLengthValid_ = true;
    char buf[192];
    std::snprintf(buf, sizeof(buf),
                  u8"圆弧%d：弧长 %.3f px，弦长 %.3f px，弓高 %.3f px", arc.id,
                  arcLengthMetrics_.arcLength, arcLengthMetrics_.chordLength,
                  arcLengthMetrics_.sagitta);
    SetStatus(buf);
}

void Application::ClearArcLength() {
    arcLengthPick_ = -1;
    arcLengthValid_ = false;
    arcLengthMetrics_ = {};
}

void Application::ConfirmThreePointCircle() {
    float cx = 0.f;
    float cy = 0.f;
    float r = 0.f;
    if (!OpenCv2D::CircleFromThreePoints(threePointX_[0], threePointY_[0], threePointX_[1],
                                         threePointY_[1], threePointX_[2], threePointY_[2], cx, cy,
                                         r)) {
        SetStatus(u8"三点近乎共线，无法定圆");
        return;
    }
    MeasuredThreePointCircle entry;
    entry.id = nextThreePointCircleId_++;
    entry.imageSource = threePointSource_;
    entry.centerX = cx;
    entry.centerY = cy;
    entry.radius = r;
    measuredThreePointCircles_.push_back(entry);
    threePointPhase_ = ThreePointPhase::Pick0;
    char buf[128];
    std::snprintf(buf, sizeof(buf), u8"三点圆%d：半径 %.3f px（圆心 %.1f, %.1f）", entry.id, r, cx,
                  cy);
    SetStatus(buf);
}

void Application::ClearThreePointCircle() {
    threePointPhase_ = ThreePointPhase::Pick0;
    threePointSource_ = -1;
}

void Application::PickLineForParallelDist(int lineIndex) {
    if (lineIndex < 0 || lineIndex >= static_cast<int>(measuredLines_.size())) return;
    parallelDistValid_ = false;
    const int lineId = measuredLines_[static_cast<std::size_t>(lineIndex)].id;
    if (parallelDistPickA_ < 0) {
        parallelDistPickA_ = lineIndex;
        SetStatus((std::string(u8"线段") + std::to_string(lineId) + u8" 已选为 A").c_str());
        return;
    }
    if (parallelDistPickB_ < 0 && lineIndex != parallelDistPickA_) {
        parallelDistPickB_ = lineIndex;
        SetStatus((std::string(u8"线段") + std::to_string(lineId) + u8" 已选为 B").c_str());
        return;
    }
    parallelDistPickA_ = lineIndex;
    parallelDistPickB_ = -1;
    parallelDistValid_ = false;
}

void Application::ComputeParallelLineDistance() {
    parallelDistValid_ = false;
    if (parallelDistPickA_ < 0 || parallelDistPickB_ < 0) {
        SetStatus(u8"请选择两条线段");
        return;
    }
    const MeasuredImageLine& la = measuredLines_[static_cast<std::size_t>(parallelDistPickA_)];
    const MeasuredImageLine& lb = measuredLines_[static_cast<std::size_t>(parallelDistPickB_)];
    if (la.imageSource != lb.imageSource || !la.result.ok || !lb.result.ok) {
        SetStatus(u8"线段无效或不在同一张图");
        return;
    }
    parallelDistPx_ = OpenCv2D::ParallelLineDistance(
        la.result.fitX1, la.result.fitY1, la.result.fitX2, la.result.fitY2, lb.result.fitX1,
        lb.result.fitY1, lb.result.fitX2, lb.result.fitY2);
    parallelDistValid_ = true;
    char buf[96];
    std::snprintf(buf, sizeof(buf), u8"平行线间距 = %.4f px", parallelDistPx_);
    SetStatus(buf);
}

void Application::ClearParallelLineDistance() {
    parallelDistPickA_ = -1;
    parallelDistPickB_ = -1;
    parallelDistValid_ = false;
}

void Application::PreviewRectCaliper(ImageView& view) {
    if (!view.valid()) return;
    OpenCv2D::CaliperLineParams params = lineMeasureParams_;
    params.skipZero = (!view.gray.empty() && ImageSourceOf(view) == 0) ? depthSkipZero_ : false;
    OpenCv2D::CaliperRectResult result;
    std::string error;
    bool ok = false;
    if (!view.gray.empty()) {
        ok = OpenCv2D::MeasureRectWithCalipers(view.gray, view.width, view.height, rectCaliperRoiX0_,
                                               rectCaliperRoiY0_, rectCaliperRoiX1_,
                                               rectCaliperRoiY1_, params, result, error);
    } else if (!view.rgb.empty()) {
        ok = OpenCv2D::MeasureRectWithCalipersRgb(
            view.rgb, view.width, view.height, rectCaliperRoiX0_, rectCaliperRoiY0_,
            rectCaliperRoiX1_, rectCaliperRoiY1_, params, result, error);
    } else {
        SetStatus(u8"当前图像无可用灰度数据");
        return;
    }
    if (!ok) {
        rectCaliperPending_ = false;
        SetStatus(error);
        return;
    }
    rectCaliperPending_ = true;
    rectCaliperPendingSource_ = ImageSourceOf(view);
    rectCaliperPendingResult_ = std::move(result);
    SetStatus(u8"矩形卡尺预览完成，请在左侧确认");
}

void Application::ConfirmRectCaliper() {
    if (!rectCaliperPending_ || !rectCaliperPendingResult_.ok) {
        SetStatus(u8"当前没有待确认的矩形卡尺");
        return;
    }
    MeasuredRectCaliper entry;
    entry.id = nextRectCaliperId_++;
    entry.imageSource = rectCaliperPendingSource_;
    entry.result = std::move(rectCaliperPendingResult_);
    measuredRectCalipers_.push_back(std::move(entry));
    rectCaliperPending_ = false;
    char buf[128];
    std::snprintf(buf, sizeof(buf), u8"矩形%d：%.1f×%.1f px，角度 %.1f°",
                  measuredRectCalipers_.back().id, measuredRectCalipers_.back().result.width,
                  measuredRectCalipers_.back().result.height,
                  measuredRectCalipers_.back().result.angleDeg);
    SetStatus(buf);
}

void Application::CancelRectCaliperPending() {
    rectCaliperPending_ = false;
    rectCaliperPendingSource_ = -1;
    rectCaliperPendingResult_ = {};
}

void Application::PreviewEllipseFitFromRoi(ImageView& view) {
    if (!view.valid()) return;
    lineMeasureParams_.skipZero = (!view.gray.empty() && ImageSourceOf(view) == 0) ? depthSkipZero_
                                                                                    : false;
    OpenCv2D::CaliperArcResult arcResult;
    std::string error;
    bool ok = false;
    if (!view.gray.empty()) {
        ok = OpenCv2D::MeasureArcWithCalipers(
            view.gray, view.width, view.height, arcRoiP0X_, arcRoiP0Y_, arcRoiP1X_, arcRoiP1Y_,
            arcRoiP2X_, arcRoiP2Y_, lineMeasureParams_, arcResult, error);
    } else if (!view.rgb.empty()) {
        ok = OpenCv2D::MeasureArcWithCalipersRgb(
            view.rgb, view.width, view.height, arcRoiP0X_, arcRoiP0Y_, arcRoiP1X_, arcRoiP1Y_,
            arcRoiP2X_, arcRoiP2Y_, lineMeasureParams_, arcResult, error);
    }
    if (!ok) {
        CancelEllipseFitPending();
        SetStatus(error);
        return;
    }
    OpenCv2D::EllipseFitResult fit;
    if (!OpenCv2D::FitEllipseFromEdgePoints(arcResult.edgePoints, fit)) {
        CancelEllipseFitPending();
        SetStatus(u8"椭圆拟合需要至少 5 个有效边缘点");
        return;
    }
    ellipseFitPending_ = true;
    ellipseFitPendingSource_ = ImageSourceOf(view);
    ellipseFitPendingFromArcId_ = -1;
    ellipseFitPendingResult_ = fit;
    ellipseFitPendingEdgePoints_ = arcResult.edgePoints;
    SetStatus(u8"椭圆拟合预览完成，请在左侧确认");
}

void Application::PreviewEllipseFitFromMeasuredArc(int arcIndex) {
    if (arcIndex < 0 || arcIndex >= static_cast<int>(measuredArcs_.size())) return;
    const MeasuredImageArc& arc = measuredArcs_[static_cast<std::size_t>(arcIndex)];
    OpenCv2D::EllipseFitResult fit;
    if (!OpenCv2D::FitEllipseFromEdgePoints(arc.result.edgePoints, fit)) {
        SetStatus(u8"椭圆拟合失败（边缘点不足）");
        return;
    }
    ellipseFitPending_ = true;
    ellipseFitPendingSource_ = arc.imageSource;
    ellipseFitPendingFromArcId_ = arc.id;
    ellipseFitPendingResult_ = fit;
    ellipseFitPendingEdgePoints_ = arc.result.edgePoints;
    SetStatus(u8"椭圆拟合预览完成，请在左侧确认");
}

void Application::ConfirmEllipseFit() {
    if (!ellipseFitPending_ || !ellipseFitPendingResult_.ok) {
        SetStatus(u8"当前没有待确认的椭圆拟合");
        return;
    }
    MeasuredEllipseFit entry;
    entry.id = nextEllipseFitId_++;
    entry.imageSource = ellipseFitPendingSource_;
    entry.sourceArcId = ellipseFitPendingFromArcId_;
    entry.result = ellipseFitPendingResult_;
    entry.edgePoints = ellipseFitPendingEdgePoints_;
    measuredEllipseFits_.push_back(std::move(entry));
    CancelEllipseFitPending();
    ResetArcMeasurePick();
    char buf[128];
    std::snprintf(buf, sizeof(buf), u8"椭圆%d：长轴 %.2f  短轴 %.2f px",
                  measuredEllipseFits_.back().id, measuredEllipseFits_.back().result.axisA * 2.f,
                  measuredEllipseFits_.back().result.axisB * 2.f);
    SetStatus(buf);
}

void Application::CancelEllipseFitPending() {
    ellipseFitPending_ = false;
    ellipseFitPendingSource_ = -1;
    ellipseFitPendingFromArcId_ = -1;
    ellipseFitPendingResult_ = {};
    ellipseFitPendingEdgePoints_.clear();
}

void Application::PreviewProfileWidth(ImageView& view) {
    if (!view.valid()) return;
    OpenCv2D::CaliperLineParams params = lineMeasureParams_;
    params.skipZero = (!view.gray.empty() && ImageSourceOf(view) == 0) ? depthSkipZero_ : false;
    OpenCv2D::ProfileWidthResult result;
    std::string error;
    bool ok = false;
    if (!view.gray.empty()) {
        ok = OpenCv2D::MeasureProfileWidth(view.gray, view.width, view.height, profileWidthRoiX0_,
                                           profileWidthRoiY0_, profileWidthRoiX1_,
                                           profileWidthRoiY1_, params, result, error);
    } else if (!view.rgb.empty()) {
        ok = OpenCv2D::MeasureProfileWidthRgb(
            view.rgb, view.width, view.height, profileWidthRoiX0_, profileWidthRoiY0_,
            profileWidthRoiX1_, profileWidthRoiY1_, params, result, error);
    } else {
        SetStatus(u8"当前图像无可用灰度数据");
        return;
    }
    if (!ok) {
        profileWidthPending_ = false;
        SetStatus(error);
        return;
    }
    profileWidthPending_ = true;
    profileWidthPendingSource_ = ImageSourceOf(view);
    profileWidthPendingResult_ = std::move(result);
    SetStatus(u8"剖面测宽预览完成，请在左侧确认");
}

void Application::ConfirmProfileWidth() {
    if (!profileWidthPending_ || !profileWidthPendingResult_.ok) {
        SetStatus(u8"当前没有待确认的测宽结果");
        return;
    }
    MeasuredProfileWidth entry;
    entry.id = nextProfileWidthId_++;
    entry.imageSource = profileWidthPendingSource_;
    entry.result = std::move(profileWidthPendingResult_);
    measuredProfileWidths_.push_back(std::move(entry));
    profileWidthPending_ = false;
    char buf[64];
    std::snprintf(buf, sizeof(buf), u8"宽度%d = %.4f px", measuredProfileWidths_.back().id,
                  measuredProfileWidths_.back().result.width);
    SetStatus(buf);
}

void Application::CancelProfileWidthPending() {
    profileWidthPending_ = false;
    profileWidthPendingSource_ = -1;
    profileWidthPendingResult_ = {};
}

void Application::PickPointForProjection(float px, float py, int imageSource) {
    pointProjPx_ = px;
    pointProjPy_ = py;
    pointProjSource_ = imageSource;
    pointProjPhase_ = PointPickPhase::PickB;
    pointProjValid_ = false;
    SetStatus(u8"已选点，请点击线段");
}

void Application::PickLineForProjection(int lineIndex) {
    pointProjLinePick_ = lineIndex;
    pointProjValid_ = false;
}

void Application::ComputePointProjection() {
    pointProjValid_ = false;
    if (pointProjLinePick_ < 0 || pointProjLinePick_ >= static_cast<int>(measuredLines_.size())) {
        SetStatus(u8"请先选点再选线段");
        return;
    }
    const MeasuredImageLine& line = measuredLines_[static_cast<std::size_t>(pointProjLinePick_)];
    if (line.imageSource != pointProjSource_ || !line.result.ok) {
        SetStatus(u8"点与线段须在同一张图");
        return;
    }
    if (!OpenCv2D::ProjectPointOntoSegment(pointProjPx_, pointProjPy_, line.result.fitX1,
                                             line.result.fitY1, line.result.fitX2,
                                             line.result.fitY2, pointProjResult_)) {
        SetStatus(u8"投影计算失败");
        return;
    }
    pointProjValid_ = true;
    char buf[160];
    std::snprintf(buf, sizeof(buf), u8"垂足 (%.2f, %.2f)  垂直距 %.3f  参数 t=%.3f",
                  pointProjResult_.footX, pointProjResult_.footY, pointProjResult_.perpDist,
                  pointProjResult_.alongT);
    SetStatus(buf);
}

void Application::ClearPointProjection() {
    pointProjPhase_ = PointPickPhase::PickA;
    pointProjSource_ = -1;
    pointProjLinePick_ = -1;
    pointProjValid_ = false;
}

void Application::PickCircleForConcentricity(int circleIndex) {
    if (circleIndex < 0 || circleIndex >= static_cast<int>(measuredCircleFits_.size())) return;
    concentricityValid_ = false;
    if (concentricityPickA_ < 0) {
        concentricityPickA_ = circleIndex;
        return;
    }
    if (concentricityPickB_ < 0 && circleIndex != concentricityPickA_) {
        concentricityPickB_ = circleIndex;
        return;
    }
    concentricityPickA_ = circleIndex;
    concentricityPickB_ = -1;
}

void Application::ComputeConcentricity() {
    concentricityValid_ = false;
    if (concentricityPickA_ < 0 || concentricityPickB_ < 0) {
        SetStatus(u8"请选择两个圆");
        return;
    }
    const MeasuredCircleFit& ca = measuredCircleFits_[static_cast<std::size_t>(concentricityPickA_)];
    const MeasuredCircleFit& cb = measuredCircleFits_[static_cast<std::size_t>(concentricityPickB_)];
    if (!OpenCv2D::ComputeConcentricity(ca.result.centerX, ca.result.centerY, cb.result.centerX,
                                        cb.result.centerY, concentricityResult_)) {
        SetStatus(u8"同心度计算失败");
        return;
    }
    concentricityValid_ = true;
    char buf[128];
    std::snprintf(buf, sizeof(buf), u8"同心度偏移 ΔX=%.3f ΔY=%.3f  距离=%.3f px",
                  concentricityResult_.offsetX, concentricityResult_.offsetY,
                  concentricityResult_.offsetDist);
    SetStatus(buf);
}

void Application::ClearConcentricity() {
    concentricityPickA_ = -1;
    concentricityPickB_ = -1;
    concentricityValid_ = false;
}

int Application::FindClosestMeasuredCircleCaliper(int imageSource, float px, float py,
                                                float maxDistPx) const {
    int bestIdx = -1;
    float bestDist = maxDistPx;
    for (std::size_t i = 0; i < measuredCircleCalipers_.size(); ++i) {
        const MeasuredCircleCaliper& cc = measuredCircleCalipers_[i];
        if (cc.imageSource != imageSource || !cc.result.ok) continue;
        const float d = std::fabs(std::hypot(px - cc.result.fitCenterX, py - cc.result.fitCenterY) -
                                  cc.result.fitRadius);
        if (d < bestDist) {
            bestDist = d;
            bestIdx = static_cast<int>(i);
        }
    }
    return bestIdx;
}

void Application::PickCircleForRoundness(int circleIndex) {
    roundnessPick_ = circleIndex;
    roundnessCircleSource_ = 0;
    roundnessValid_ = false;
}

void Application::ComputeRoundness() {
    roundnessValid_ = false;
    if (roundnessPick_ < 0) {
        SetStatus(u8"请先选择圆");
        return;
    }
    if (roundnessCircleSource_ == 0) {
        if (roundnessPick_ >= static_cast<int>(measuredCircleFits_.size())) return;
        const MeasuredCircleFit& fit = measuredCircleFits_[static_cast<std::size_t>(roundnessPick_)];
        if (!OpenCv2D::ComputeRoundness(fit.result.centerX, fit.result.centerY, fit.result.radius,
                                        fit.edgePoints, roundnessResult_)) {
            SetStatus(u8"圆度计算失败");
            return;
        }
    } else {
        if (roundnessPick_ >= static_cast<int>(measuredCircleCalipers_.size())) return;
        const MeasuredCircleCaliper& cc =
            measuredCircleCalipers_[static_cast<std::size_t>(roundnessPick_)];
        if (!OpenCv2D::ComputeRoundness(cc.result.fitCenterX, cc.result.fitCenterY,
                                        cc.result.fitRadius, cc.result.edgePoints,
                                        roundnessResult_)) {
            SetStatus(u8"圆度计算失败");
            return;
        }
    }
    roundnessValid_ = true;
    char buf[128];
    std::snprintf(buf, sizeof(buf), u8"圆度 RMS=%.4f  最大偏差=%.4f  最小偏差=%.4f",
                  roundnessResult_.rms, roundnessResult_.maxDev, roundnessResult_.minDev);
    SetStatus(buf);
}

void Application::ClearRoundness() {
    roundnessPick_ = -1;
    roundnessValid_ = false;
}

void Application::PreviewRegionBlob(ImageView& view, bool quiet) {
    if (!view.valid() && !CanSyncRegionBlob()) return;

    ImageView* computeView = &view;
    if (CanSyncRegionBlob() && brightnessImage_.valid()) {
        computeView = &brightnessImage_;
    } else if (!view.valid()) {
        return;
    }

    OpenCv2D::RegionBlobResult result;
    std::string error;
    bool ok = false;
    if (!computeView->gray.empty()) {
        ok = OpenCv2D::ComputeRegionBlob(computeView->gray, computeView->width, computeView->height,
                                         regionBlobRoiX0_, regionBlobRoiY0_, regionBlobRoiX1_,
                                         regionBlobRoiY1_, regionBlobThreshold_,
                                         regionBlobGreaterThan_, result, error);
    } else if (!computeView->rgb.empty()) {
        ok = OpenCv2D::ComputeRegionBlobRgb(
            computeView->rgb, computeView->width, computeView->height, regionBlobRoiX0_,
            regionBlobRoiY0_, regionBlobRoiX1_, regionBlobRoiY1_, regionBlobThreshold_,
            regionBlobGreaterThan_, result, error);
    } else {
        if (!quiet) SetStatus(u8"当前图像无可用灰度数据");
        return;
    }
    if (!ok) {
        regionBlobPending_ = false;
        regionBlobPendingDepthStats_ = {};
        if (!quiet) SetStatus(error);
        return;
    }
    regionBlobPending_ = true;
    regionBlobPendingSource_ = ImageSourceOf(*computeView);
    regionBlobPendingResult_ = std::move(result);
    UpdateRegionBlobDepthStats();
    if (!quiet) {
        if (regionBlobPendingResult_.pixelCount > 0) {
            if (CanSyncRegionBlob() && regionBlobPendingDepthStats_.valid) {
                SetStatus(u8"亮度区域已筛选，深度图已同步高亮");
            } else {
                SetStatus(u8"区域分析预览完成，请在左侧确认");
            }
        } else {
            SetStatus(u8"当前阈值下无满足条件的像素，请调整阈值");
        }
    }
}

void Application::UpdateRegionBlobDepthStats() {
    regionBlobPendingDepthStats_ = {};
    if (!regionBlobPending_ || !regionBlobPendingResult_.ok ||
        regionBlobPendingResult_.pixelCount <= 0 || !depthImage_.valid() ||
        depthImage_.gray.empty()) {
        return;
    }
    const OpenCv2D::RegionBlobResult& blob = regionBlobPendingResult_;
    const int roiW = blob.roiWidth;
    const int roiH = blob.roiHeight;
    const int x0 = static_cast<int>(blob.roiX0);
    const int y0 = static_cast<int>(blob.roiY0);
    double sum = 0.0;
    int count = 0;
    float vmin = 0.f;
    float vmax = 0.f;
    for (int ry = 0; ry < roiH; ++ry) {
        for (int rx = 0; rx < roiW; ++rx) {
            if (blob.hitMask[static_cast<std::size_t>(ry * roiW + rx)] == 0) continue;
            const int gx = x0 + rx;
            const int gy = y0 + ry;
            const std::size_t di =
                static_cast<std::size_t>(gy) * static_cast<std::size_t>(depthImage_.width) +
                static_cast<std::size_t>(gx);
            if (di >= depthImage_.gray.size()) continue;
            const float z = depthImage_.gray[di];
            if (!std::isfinite(z)) continue;
            if (depthSkipZero_ && std::fabs(z) <= 1e-6f) continue;
            if (count == 0) {
                vmin = vmax = z;
            } else {
                vmin = std::min(vmin, z);
                vmax = std::max(vmax, z);
            }
            sum += static_cast<double>(z);
            ++count;
        }
    }
    if (count <= 0) return;
    regionBlobPendingDepthStats_.count = count;
    regionBlobPendingDepthStats_.min = vmin;
    regionBlobPendingDepthStats_.max = vmax;
    regionBlobPendingDepthStats_.mean = static_cast<float>(sum / static_cast<double>(count));
    regionBlobPendingDepthStats_.valid = true;
}

bool Application::HasRegionBlobRoi() const {
    return std::abs(regionBlobRoiX1_ - regionBlobRoiX0_) >= 1.f &&
           std::abs(regionBlobRoiY1_ - regionBlobRoiY0_) >= 1.f;
}

void Application::RefreshRegionBlobPreview() {
    if (image2DTool_ != Image2DTool::RegionBlob || !HasRegionBlobRoi()) return;
    if (CanSyncRegionBlob() && brightnessImage_.valid()) {
        PreviewRegionBlob(brightnessImage_, true);
        return;
    }
    int src = regionBlobPendingSource_;
    if (src < 0) src = regionBlobDragSource_;
    if (src < 0) src = 0;

    ImageView* view = ImageViewFromSource(src);
    if (!view || !view->valid()) {
        view = ImageViewFromSource(src == 0 ? 1 : 0);
        if (view && view->valid()) src = src == 0 ? 1 : 0;
    }
    if (!view || !view->valid()) return;
    PreviewRegionBlob(*view, true);
}

void Application::ConfirmRegionBlob() {
    if (!regionBlobPending_ || !regionBlobPendingResult_.ok ||
        regionBlobPendingResult_.pixelCount <= 0) {
        SetStatus(u8"当前没有待确认的有效区域结果");
        return;
    }
    MeasuredRegionBlob entry;
    entry.id = nextRegionBlobId_++;
    entry.imageSource = regionBlobPendingSource_;
    entry.syncedPair = CanSyncRegionBlob();
    entry.result = std::move(regionBlobPendingResult_);
    measuredRegionBlobs_.push_back(std::move(entry));
    regionBlobPending_ = false;
    char buf[128];
    std::snprintf(buf, sizeof(buf), u8"区域%d：面积 %d px²  质心 (%.1f, %.1f)",
                  measuredRegionBlobs_.back().id, measuredRegionBlobs_.back().result.pixelCount,
                  measuredRegionBlobs_.back().result.centroidX,
                  measuredRegionBlobs_.back().result.centroidY);
    SetStatus(buf);
}

void Application::CancelRegionBlobPending() {
    regionBlobPending_ = false;
    regionBlobPendingSource_ = -1;
    regionBlobPendingResult_ = {};
    regionBlobPendingDepthStats_ = {};
}

void Application::AddDepthHeightSample(float px, float py, int imageSource) {
    if (imageSource != 0) {
        SetStatus(u8"高度差测量仅支持深度图");
        return;
    }
    const ImageView* view = ImageViewFromSource(0);
    if (!view || view->gray.empty()) {
        SetStatus(u8"请先打开深度图");
        return;
    }
    const int col = std::clamp(static_cast<int>(px), 0, view->width - 1);
    const int row = std::clamp(static_cast<int>(py), 0, view->height - 1);
    const float z = view->gray[static_cast<std::size_t>(row) * static_cast<std::size_t>(view->width) +
                               static_cast<std::size_t>(col)];
    if (depthHeightPhase_ == PointPickPhase::PickA) {
        depthHeightSource_ = 0;
        depthHeightAx_ = px;
        depthHeightAy_ = py;
        depthHeightAz_ = z;
        depthHeightPhase_ = PointPickPhase::PickB;
        depthHeightValid_ = false;
        SetStatus(u8"已选 A 点，请点击 B 点");
    } else {
        depthHeightBz_ = z;
        depthHeightDelta_ = depthHeightBz_ - depthHeightAz_;
        depthHeightValid_ = true;
        depthHeightPhase_ = PointPickPhase::PickA;
        char buf[128];
        std::snprintf(buf, sizeof(buf), u8"ΔZ = %.6f（A=%.6f  B=%.6f）", depthHeightDelta_,
                      depthHeightAz_, depthHeightBz_);
        SetStatus(buf);
    }
}

void Application::ClearDepthHeightDiff() {
    depthHeightPhase_ = PointPickPhase::PickA;
    depthHeightValid_ = false;
}

void Application::PreviewDepthProfile(ImageView& view) {
    if (ImageSourceOf(view) != 0 || view.gray.empty()) {
        SetStatus(u8"剖面高度仅支持深度图");
        return;
    }
    depthProfileSamples_.clear();
    depthProfileMeasureValid_ = false;
    depthProfilePickA_ = -1;
    depthProfilePickB_ = -1;
    depthProfileHoverIdx_ = -1;

    bool ok = false;
    if (depthProfileMode_ == DepthProfileMode::FreeLine) {
        ok = OpenCv2D::SampleLineProfile(view.gray, view.width, view.height, depthProfileRoiX0_,
                                         depthProfileRoiY0_, depthProfileRoiX1_, depthProfileRoiY1_,
                                         depthProfileSampleCount_, depthProfileSamples_,
                                         depthSkipZero_);
        char buf[96];
        std::snprintf(buf, sizeof(buf), u8"任意剖面 (%.0f,%.0f)→(%.0f,%.0f)", depthProfileRoiX0_,
                      depthProfileRoiY0_, depthProfileRoiX1_, depthProfileRoiY1_);
        depthProfileLabel_ = buf;
    } else if (depthProfileMode_ == DepthProfileMode::ScanRow) {
        const int row = std::clamp(static_cast<int>(std::lround(depthProfileRoiY0_)), 0,
                                   view.height - 1);
        depthProfileRoiX0_ = 0.f;
        depthProfileRoiY0_ = static_cast<float>(row);
        depthProfileRoiX1_ = static_cast<float>(view.width - 1);
        depthProfileRoiY1_ = static_cast<float>(row);
        ok = OpenCv2D::SampleRowProfile(view.gray, view.width, view.height, row,
                                        depthProfileSamples_, depthSkipZero_);
        char buf[64];
        std::snprintf(buf, sizeof(buf), u8"扫描行 Y=%d（X-Z Profile）", row);
        depthProfileLabel_ = buf;
    } else {
        const int col = std::clamp(static_cast<int>(std::lround(depthProfileRoiX0_)), 0,
                                   view.width - 1);
        depthProfileRoiX0_ = static_cast<float>(col);
        depthProfileRoiY0_ = 0.f;
        depthProfileRoiX1_ = static_cast<float>(col);
        depthProfileRoiY1_ = static_cast<float>(view.height - 1);
        ok = OpenCv2D::SampleColumnProfile(view.gray, view.width, view.height, col,
                                           depthProfileSamples_, depthSkipZero_);
        char buf[64];
        std::snprintf(buf, sizeof(buf), u8"扫描列 X=%d（Y-Z Profile）", col);
        depthProfileLabel_ = buf;
    }

    if (!ok || depthProfileSamples_.empty()) {
        depthProfileValid_ = false;
        SetStatus(u8"剖面采样失败");
        return;
    }
    depthProfileValid_ = true;
    depthProfileDragSource_ = 0;
    showDepthProfilePanel_ = true;
    SetStatus(u8"Profile 曲线已生成，见曲线面板");
}

void Application::PreviewDepthProfileAt(float px, float py, ImageView& view) {
    depthProfileRoiX0_ = px;
    depthProfileRoiY0_ = py;
    depthProfileRoiX1_ = px;
    depthProfileRoiY1_ = py;
    PreviewDepthProfile(view);
}

void Application::ClearDepthProfile() {
    depthProfileValid_ = false;
    depthProfileSamples_.clear();
    depthProfileLabel_.clear();
    depthProfileHoverIdx_ = -1;
    depthProfilePickA_ = -1;
    depthProfilePickB_ = -1;
    depthProfileMeasureValid_ = false;
}

void Application::DrawDepthProfilePanel() {
    if (!depthProfileValid_ || depthProfileSamples_.empty()) return;

    const ImGuiViewport* vp = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(ImVec2(vp->Pos.x + vp->Size.x - 480.f - 12.f, vp->Pos.y + 100.f),
                            ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(480.f, 420.f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowBgAlpha(0.98f);
    if (!ImGui::Begin(u8"深度 Profile 曲线", &showDepthProfilePanel_, ImGuiWindowFlags_NoCollapse)) {
        ImGui::End();
        return;
    }

    ImGui::TextWrapped("%s", depthProfileLabel_.c_str());
    ImGui::Spacing();

    float zMin = std::numeric_limits<float>::max();
    float zMax = std::numeric_limits<float>::lowest();
    double zSum = 0.0;
    int validCount = 0;
    float distMin = std::numeric_limits<float>::max();
    float distMax = std::numeric_limits<float>::lowest();
    for (const OpenCv2D::LineProfileSample& s : depthProfileSamples_) {
        if (std::isnan(s.value)) continue;
        zMin = std::min(zMin, s.value);
        zMax = std::max(zMax, s.value);
        zSum += static_cast<double>(s.value);
        ++validCount;
        distMin = std::min(distMin, s.distance);
        distMax = std::max(distMax, s.distance);
    }
    if (validCount == 0) {
        ImGui::TextColored(ImVec4(1.f, 0.5f, 0.4f, 1.f), u8"无有效高度点（检查 0 值过滤）");
        ImGui::End();
        return;
    }
    const float zMean = static_cast<float>(zSum / static_cast<double>(validCount));
    const float zSpan = std::max(zMax - zMin, 1e-6f);
    const float distSpan = std::max(distMax - distMin, 1e-6f);

    ImGui::Text(u8"Z: 最小 %.4f　最大 %.4f　均值 %.4f　范围 %.4f", zMin, zMax, zMean, zSpan);
    ImGui::TextDisabled(u8"有效点 %d / %zu", validCount,
                        depthProfileSamples_.size());
    if (depthProfileMeasureValid_ && depthProfilePickA_ >= 0 && depthProfilePickB_ >= 0) {
        ImGui::TextColored(ImVec4(0.55f, 0.95f, 0.75f, 1.f), u8"ΔZ = %.6f", depthProfileMeasureDelta_);
    }
    ImGui::Spacing();
    ImGui::Separator();

    const char* xAxisLabel = (depthProfileMode_ == DepthProfileMode::ScanRow)   ? u8"X (px)"
                             : (depthProfileMode_ == DepthProfileMode::ScanColumn) ? u8"Y (px)"
                                                                                   : u8"距离 (px)";
    ImGui::TextDisabled(u8"横轴 = %s　纵轴 = Z 高度", xAxisLabel);
    ImGui::TextDisabled(u8"在曲线上点击选 A/B 测高度差；悬停查看数值");

    constexpr float leftPad = 52.f;
    constexpr float bottomPad = 34.f;
    constexpr float topPad = 8.f;
    constexpr float rightPad = 12.f;
    const float plotH = 200.f;
    const ImVec2 avail = ImGui::GetContentRegionAvail();
    const float plotW = std::max(avail.x, 200.f);
    const ImVec2 canvasPos = ImGui::GetCursorScreenPos();
    const ImVec2 canvasSize(plotW, plotH + bottomPad + topPad);
    ImGui::InvisibleButton(u8"##depthprofcanvas", canvasSize);
    const bool chartHovered = ImGui::IsItemHovered();

    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImVec2 origin(canvasPos.x + leftPad, canvasPos.y + topPad + plotH);
    const float chartW = plotW - leftPad - rightPad;
    const float chartH = plotH;

    dl->AddRectFilled(ImVec2(origin.x, origin.y - chartH), ImVec2(origin.x + chartW, origin.y),
                      IM_COL32(18, 22, 28, 255));
    dl->AddRect(ImVec2(origin.x, origin.y - chartH), ImVec2(origin.x + chartW, origin.y),
                IM_COL32(70, 90, 110, 220));

    auto mapX = [&](float dist) -> float {
        return origin.x + (dist - distMin) / distSpan * chartW;
    };
    auto mapY = [&](float z) -> float { return origin.y - (z - zMin) / zSpan * chartH; };

    for (int g = 0; g <= 4; ++g) {
        const float t = static_cast<float>(g) / 4.f;
        const float y = origin.y - t * chartH;
        dl->AddLine(ImVec2(origin.x, y), ImVec2(origin.x + chartW, y), IM_COL32(45, 55, 65, 180));
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.3f", zMin + t * zSpan);
        const ImVec2 ts = ImGui::CalcTextSize(buf);
        dl->AddText(ImVec2(origin.x - 6.f - ts.x, y - ts.y * 0.5f), IM_COL32(160, 185, 200, 255),
                    buf);
    }
    for (int g = 0; g <= 4; ++g) {
        const float t = static_cast<float>(g) / 4.f;
        const float x = origin.x + t * chartW;
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.0f", distMin + t * distSpan);
        const ImVec2 ts = ImGui::CalcTextSize(buf);
        dl->AddText(ImVec2(x - ts.x * 0.5f, origin.y + 4.f), IM_COL32(160, 185, 200, 255), buf);
    }
    dl->AddText(ImVec2(origin.x + chartW * 0.5f - ImGui::CalcTextSize(xAxisLabel).x * 0.5f,
                       origin.y + 18.f),
                IM_COL32(120, 175, 210, 255), xAxisLabel);
    {
        const char* yl = u8"Z";
        const ImVec2 ts = ImGui::CalcTextSize(yl);
        dl->AddText(ImVec2(canvasPos.x + 4.f, canvasPos.y + topPad + (chartH - ts.y) * 0.5f),
                    IM_COL32(120, 175, 210, 255), yl);
    }

    ImVec2 prev;
    bool hasPrev = false;
    for (const OpenCv2D::LineProfileSample& s : depthProfileSamples_) {
        if (std::isnan(s.value)) {
            hasPrev = false;
            continue;
        }
        const ImVec2 pt(mapX(s.distance), mapY(s.value));
        if (hasPrev) dl->AddLine(prev, pt, IM_COL32(80, 210, 255, 255), 1.8f);
        prev = pt;
        hasPrev = true;
    }

  int hoverIdx = -1;
    if (chartHovered) {
        const ImVec2 mp = ImGui::GetMousePos();
        const float relX = mp.x - origin.x;
        if (relX >= 0.f && relX <= chartW) {
            const float targetDist = distMin + (relX / chartW) * distSpan;
            float bestD = std::numeric_limits<float>::max();
            for (std::size_t i = 0; i < depthProfileSamples_.size(); ++i) {
                const OpenCv2D::LineProfileSample& s = depthProfileSamples_[i];
                if (std::isnan(s.value)) continue;
                const float d = std::fabs(s.distance - targetDist);
                if (d < bestD) {
                    bestD = d;
                    hoverIdx = static_cast<int>(i);
                }
            }
        }
    }
    depthProfileHoverIdx_ = hoverIdx;

    auto drawPickMarker = [&](int idx, ImU32 col, const char* tag) {
        if (idx < 0 || idx >= static_cast<int>(depthProfileSamples_.size())) return;
        const OpenCv2D::LineProfileSample& s = depthProfileSamples_[static_cast<std::size_t>(idx)];
        if (std::isnan(s.value)) return;
        const ImVec2 pt(mapX(s.distance), mapY(s.value));
        dl->AddCircleFilled(pt, 5.f, col);
        dl->AddCircle(pt, 7.f, IM_COL32(255, 255, 255, 200), 0, 1.5f);
        dl->AddText(ImVec2(pt.x + 8.f, pt.y - 16.f), col, tag);
    };
    drawPickMarker(depthProfilePickA_, IM_COL32(255, 230, 60, 255), u8"A");
    drawPickMarker(depthProfilePickB_, IM_COL32(80, 220, 255, 255), u8"B");

    if (hoverIdx >= 0) {
        const OpenCv2D::LineProfileSample& hs =
            depthProfileSamples_[static_cast<std::size_t>(hoverIdx)];
        if (!std::isnan(hs.value)) {
            const ImVec2 hp(mapX(hs.distance), mapY(hs.value));
            dl->AddLine(ImVec2(hp.x, origin.y - chartH), ImVec2(hp.x, origin.y),
                        IM_COL32(255, 200, 80, 120));
            dl->AddCircleFilled(hp, 4.f, IM_COL32(255, 200, 80, 255));
            ImGui::BeginTooltip();
            ImGui::Text(u8"距离 %.1f px　Z = %.6f", hs.distance, hs.value);
            ImGui::Text(u8"图像坐标 (%.1f, %.1f)", hs.x, hs.y);
            ImGui::EndTooltip();
        }
    }

    if (chartHovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left) && hoverIdx >= 0) {
        if (depthProfilePickA_ < 0 || depthProfileMeasureValid_) {
            depthProfilePickA_ = hoverIdx;
            depthProfilePickB_ = -1;
            depthProfileMeasureValid_ = false;
        } else {
            depthProfilePickB_ = hoverIdx;
            const float za =
                depthProfileSamples_[static_cast<std::size_t>(depthProfilePickA_)].value;
            const float zb = depthProfileSamples_[static_cast<std::size_t>(hoverIdx)].value;
            depthProfileMeasureDelta_ = zb - za;
            depthProfileMeasureValid_ = !std::isnan(za) && !std::isnan(zb);
        }
    }

    ImGui::Spacing();
    if (ImGui::Button(u8"清除测点 A/B", ImVec2(120.f, 0))) {
        depthProfilePickA_ = -1;
        depthProfilePickB_ = -1;
        depthProfileMeasureValid_ = false;
    }
    ImGui::SameLine();
    if (ImGui::Button(u8"清除剖面", ImVec2(120.f, 0))) ClearDepthProfile();
    ImGui::SameLine();
    if (depthProfileMeasureValid_ &&
        ImGui::Button(u8"复制 ΔZ", ImVec2(100.f, 0))) {
        char clip[64];
        std::snprintf(clip, sizeof(clip), "%.6f", depthProfileMeasureDelta_);
        ImGui::SetClipboardText(clip);
        SetStatus(u8"已复制 ΔZ");
    }

    ImGui::End();
}

void Application::PreviewTemplateMatch(ImageView& view) {
    if (ImageSourceOf(view) != 1 || view.rgb.empty()) {
        SetStatus(u8"模板匹配仅支持亮度图");
        return;
    }
    if (!templateMatchHasTemplate_) {
        SetStatus(u8"请先框选模板区域");
        return;
    }
    OpenCv2D::TemplateMatchResult result;
    std::string error;
    OpenCv2D::TemplateMatchParams params = templateMatchParams_;
    params.searchFullImage = templateMatchSearchFull_;
    const bool ok = OpenCv2D::MatchTemplateRgb(
        view.rgb, view.width, view.height, templateMatchTplX0_, templateMatchTplY0_,
        templateMatchTplX1_, templateMatchTplY1_, templateMatchSearchX0_, templateMatchSearchY0_,
        templateMatchSearchX1_, templateMatchSearchY1_, params, result, error);
    if (!ok) {
        templateMatchPending_ = false;
        SetStatus(error);
        return;
    }
    templateMatchPending_ = true;
    templateMatchPendingSource_ = 1;
    templateMatchPendingResult_ = std::move(result);
  char buf[128];
    std::snprintf(buf, sizeof(buf), u8"模板匹配预览：找到 %zu 个结果（最高得分 %.3f）",
                  templateMatchPendingResult_.hits.size(),
                  templateMatchPendingResult_.hits.empty()
                      ? 0.f
                      : templateMatchPendingResult_.hits.front().score);
    SetStatus(buf);
}

void Application::ConfirmTemplateMatch() {
    if (!templateMatchPending_ || !templateMatchPendingResult_.ok) {
        SetStatus(u8"当前没有待确认的模板匹配结果");
        return;
    }
    MeasuredTemplateMatch entry;
    entry.id = nextTemplateMatchId_++;
    entry.imageSource = templateMatchPendingSource_;
    entry.result = std::move(templateMatchPendingResult_);
    measuredTemplateMatches_.push_back(std::move(entry));
    templateMatchPending_ = false;
    const MeasuredTemplateMatch& back = measuredTemplateMatches_.back();
    char buf[160];
    if (!back.result.hits.empty()) {
        const OpenCv2D::TemplateMatchHit& h = back.result.hits.front();
        std::snprintf(buf, sizeof(buf),
                      u8"模板匹配%d 已确认：中心 (%.1f, %.1f)  得分 %.3f  角度 %.1f°  缩放 %.2f",
                      back.id, h.centerX, h.centerY, h.score, h.angleDeg, h.scale);
    } else {
        std::snprintf(buf, sizeof(buf), u8"模板匹配%d 已确认", back.id);
    }
    SetStatus(buf);
}

void Application::CancelTemplateMatchPending() {
    templateMatchPending_ = false;
    templateMatchPendingSource_ = -1;
    templateMatchPendingResult_ = {};
}

void Application::ResetTemplateMatchPick() {
    templateMatchPhase_ = TemplateMatchPhase::DrawTemplate;
    templateMatchTplDragging_ = false;
    templateMatchSearchDragging_ = false;
    templateMatchDragSource_ = -1;
    templateMatchHasTemplate_ = false;
    CancelTemplateMatchPending();
}

void Application::ClearTemplateMatch() {
    ResetTemplateMatchPick();
    measuredTemplateMatches_.clear();
    nextTemplateMatchId_ = 1;
}

namespace {

float DistancePointToSegment(float px, float py, float x1, float y1, float x2, float y2) {
    const float dx = x2 - x1;
    const float dy = y2 - y1;
    const float len2 = dx * dx + dy * dy;
    if (len2 < 1e-6f) return std::hypot(px - x1, py - y1);
    float t = ((px - x1) * dx + (py - y1) * dy) / len2;
    t = std::clamp(t, 0.f, 1.f);
    const float cx = x1 + t * dx;
    const float cy = y1 + t * dy;
    return std::hypot(px - cx, py - cy);
}

float ArcNormAnglePos(float a) {
    constexpr float kTwoPi = 6.283185307179586f;
    while (a < 0.f) a += kTwoPi;
    while (a >= kTwoPi) a -= kTwoPi;
    return a;
}

bool IsAngleOnArcSpan(float angle, float startAngle, float endAngle) {
    const float d0 = ArcNormAnglePos(angle - startAngle);
    const float d01 = ArcNormAnglePos(endAngle - startAngle);
    return d0 <= d01 + 1e-4f;
}

float DistancePointToArc(float px, float py, float cx, float cy, float radius, float startAngle,
                         float endAngle) {
    const float dx = px - cx;
    const float dy = py - cy;
    const float dist = std::hypot(dx, dy);
    const float ang = std::atan2(dy, dx);
    if (IsAngleOnArcSpan(ang, startAngle, endAngle)) {
        return std::fabs(dist - radius);
    }
    const float x0 = cx + radius * std::cos(startAngle);
    const float y0 = cy + radius * std::sin(startAngle);
    const float x1 = cx + radius * std::cos(endAngle);
    const float y1 = cy + radius * std::sin(endAngle);
    return std::min(DistancePointToSegment(px, py, x0, y0, x1, y1),
                    DistancePointToSegment(px, py, x1, y1, x0, y0));
}

}  // namespace

int Application::FindClosestMeasuredLine(int imageSource, float px, float py,
                                         float maxDistPx) const {
    int bestIdx = -1;
    float bestDist = maxDistPx;
    for (std::size_t i = 0; i < measuredLines_.size(); ++i) {
        const MeasuredImageLine& line = measuredLines_[i];
        if (line.imageSource != imageSource || !line.result.ok) continue;
        const float d = DistancePointToSegment(px, py, line.result.fitX1, line.result.fitY1,
                                               line.result.fitX2, line.result.fitY2);
        if (d < bestDist) {
            bestDist = d;
            bestIdx = static_cast<int>(i);
        }
    }
    return bestIdx;
}

void Application::PickLineForDistance(int lineIndex) {
    if (lineIndex < 0 || lineIndex >= static_cast<int>(measuredLines_.size())) return;

    lineDistValid_ = false;
    lineDistSamples_.clear();
    const int lineId = measuredLines_[static_cast<std::size_t>(lineIndex)].id;

    if (lineDistPickA_ < 0) {
        lineDistPickA_ = lineIndex;
        char buf[64];
        std::snprintf(buf, sizeof(buf), u8"线段%d 已选为 A", lineId);
        SetStatus(buf);
        return;
    }
    if (lineDistPickB_ < 0 && lineIndex != lineDistPickA_) {
        lineDistPickB_ = lineIndex;
        char buf[64];
        std::snprintf(buf, sizeof(buf), u8"线段%d 已选为 B", lineId);
        SetStatus(buf);
        return;
    }
    if (lineIndex == lineDistPickA_) {
        lineDistPickA_ = -1;
        SetStatus(u8"已取消 A 选线");
        return;
    }
    if (lineIndex == lineDistPickB_) {
        lineDistPickB_ = -1;
        SetStatus(u8"已取消 B 选线");
        return;
    }

    lineDistPickB_ = lineIndex;
    char buf[64];
    std::snprintf(buf, sizeof(buf), u8"线段%d 已替换为 B", lineId);
    SetStatus(buf);
}

int Application::FindClosestMeasuredArc(int imageSource, float px, float py,
                                        float maxDistPx) const {
    int bestIdx = -1;
    float bestDist = maxDistPx;
    for (std::size_t i = 0; i < measuredArcs_.size(); ++i) {
        const MeasuredImageArc& arc = measuredArcs_[i];
        if (arc.imageSource != imageSource || !arc.result.ok) continue;
        const float d = DistancePointToArc(px, py, arc.result.fitCenterX, arc.result.fitCenterY,
                                         arc.result.fitRadius, arc.result.fitStartAngle,
                                         arc.result.fitEndAngle);
        if (d < bestDist) {
            bestDist = d;
            bestIdx = static_cast<int>(i);
        }
    }
    return bestIdx;
}

void Application::PickArcForDistance(int arcIndex) {
    if (arcIndex < 0 || arcIndex >= static_cast<int>(measuredArcs_.size())) return;

    arcDistValid_ = false;
    arcDistSamples_.clear();
    const int arcId = measuredArcs_[static_cast<std::size_t>(arcIndex)].id;

    if (arcDistPickA_ < 0) {
        arcDistPickA_ = arcIndex;
        char buf[64];
        std::snprintf(buf, sizeof(buf), u8"圆弧%d 已选为 A", arcId);
        SetStatus(buf);
        return;
    }
    if (arcDistPickB_ < 0 && arcIndex != arcDistPickA_) {
        arcDistPickB_ = arcIndex;
        char buf[64];
        std::snprintf(buf, sizeof(buf), u8"圆弧%d 已选为 B", arcId);
        SetStatus(buf);
        return;
    }
    if (arcIndex == arcDistPickA_) {
        arcDistPickA_ = -1;
        SetStatus(u8"已取消 A 选弧");
        return;
    }
    if (arcIndex == arcDistPickB_) {
        arcDistPickB_ = -1;
        SetStatus(u8"已取消 B 选弧");
        return;
    }

    arcDistPickB_ = arcIndex;
    char buf[64];
    std::snprintf(buf, sizeof(buf), u8"圆弧%d 已替换为 B", arcId);
    SetStatus(buf);
}

namespace {

void DrawCaliperLineParamsPanel(OpenCv2D::CaliperLineParams& params, bool arcMode) {
    ImGui::TextDisabled(u8"卡尺参数");
    ImGui::SetNextItemWidth(-1.f);
    ImGui::DragInt(arcMode ? u8"卡尺数量" : u8"测量线数量", &params.numCalipers, 1,
                   arcMode ? 3 : 2, 200);
    if (arcMode) {
        ImGui::TextDisabled(u8"沿圆弧 ROI 均匀布置的径向卡尺条数；越多采样越密（建议 10~40）");
    } else {
        ImGui::TextDisabled(
            u8"沿 ROI 测量方向均匀布置的垂直卡尺条数；越多边缘采样越密（建议 10~40）");
    }
    ImGui::SetNextItemWidth(-1.f);
    ImGui::DragFloat(arcMode ? u8"卡尺半长(px)" : u8"测量半长(px)", &params.caliperHalfLength, 1.f,
                     4.f, 500.f, "%.0f");
    ImGui::TextDisabled(
        u8"每条卡尺沿搜索方向的半长（像素）；应覆盖边缘到 ROI 的最大偏移，过小会漏检");
    ImGui::SetNextItemWidth(-1.f);
    ImGui::DragInt(u8"平均宽度(px)", &params.caliperWidth, 1, 1, 31);
    ImGui::TextDisabled(u8"沿卡尺方向的灰度平均窗口宽度（奇数）；用于平滑噪声，常用 3~7");
    ImGui::SetNextItemWidth(-1.f);
    ImGui::DragFloat(u8"最小对比度", &params.minContrast, 0.1f, 0.f, 100.f, "%.2f");
    ImGui::TextDisabled(u8"边缘梯度强度阈值；低于此值的候选边缘将被忽略，噪声大时可适当提高");
    const char* polarityItems[] = {u8"全部（最大梯度）", u8"由暗到亮", u8"由亮到暗"};
    int polarity = static_cast<int>(params.polarity);
    ImGui::SetNextItemWidth(-1.f);
    if (ImGui::Combo(u8"边缘极性", &polarity, polarityItems, 3)) {
        params.polarity = static_cast<OpenCv2D::EdgePolarity>(polarity);
    }
    ImGui::TextDisabled(u8"限定边缘过渡方向：由暗到亮 / 由亮到暗；选「全部」则取最强梯度");
}

}  // namespace

namespace {

ImU32 LineColorForId(int id) {
    static const ImU32 palette[] = {
        IM_COL32(50, 255, 100, 240),  IM_COL32(80, 200, 255, 240), IM_COL32(255, 200, 60, 240),
        IM_COL32(255, 120, 120, 240), IM_COL32(200, 120, 255, 240)};
    return palette[(id - 1) % 5];
}

void DrawLabelTag(ImDrawList* dl, ImVec2 pos, const char* text, ImU32 col) {
    const ImVec2 ts = ImGui::CalcTextSize(text);
    const ImVec2 p0(pos.x - 4.f, pos.y - 2.f);
    const ImVec2 p1(pos.x + ts.x + 4.f, pos.y + ts.y + 2.f);
    dl->AddRectFilled(p0, p1, IM_COL32(10, 12, 14, 210), 3.f);
    dl->AddRect(p0, p1, col, 3.f);
    dl->AddText(pos, IM_COL32(255, 255, 255, 255), text);
}

void DrawCaliperResultDetail(ImDrawList* dl, const OpenCv2D::CaliperLineResult& result,
                             const std::function<ImVec2(float, float)>& toScreen, ImU32 fitCol,
                             float fitThickness, bool pending) {
    if (!result.ok) return;
    const ImU32 calCol =
        pending ? IM_COL32(120, 220, 255, 200) : IM_COL32(80, 200, 255, 180);
    for (const OpenCv2D::LineSegment& c : result.calipers) {
        dl->AddLine(toScreen(c.x1, c.y1), toScreen(c.x2, c.y2), calCol, 1.f);
    }
    for (const OpenCv2D::CaliperEdgePoint& ep : result.edgePoints) {
        if (!ep.valid) continue;
        const ImVec2 p = toScreen(ep.x, ep.y);
        dl->AddCircleFilled(p, 3.5f, IM_COL32(255, 70, 70, 240));
        dl->AddCircle(p, 4.5f, IM_COL32(255, 255, 255, 200), 0, 1.2f);
    }
    dl->AddLine(toScreen(result.fitX1, result.fitY1), toScreen(result.fitX2, result.fitY2),
                fitCol, fitThickness);
}

void DrawArcPolyline(ImDrawList* dl, float cx, float cy, float radius, float startAngle,
                     float endAngle, const std::function<ImVec2(float, float)>& toScreen,
                     ImU32 col, float thickness) {
    std::vector<float> ax;
    std::vector<float> ay;
    OpenCv2D::SampleArcPolyline(cx, cy, radius, startAngle, endAngle, 48, ax, ay);
    for (std::size_t i = 1; i < ax.size(); ++i) {
        dl->AddLine(toScreen(ax[i - 1], ay[i - 1]), toScreen(ax[i], ay[i]), col, thickness);
    }
}

void DrawThreePointRoiArc(ImDrawList* dl, float p0x, float p0y, float p1x, float p1y, float p2x,
                          float p2y, const std::function<ImVec2(float, float)>& toScreen,
                          ImU32 col, float thickness) {
    float cx = 0.f;
    float cy = 0.f;
    float r = 0.f;
    float a0 = 0.f;
    float a1 = 0.f;
    if (!OpenCv2D::CircleFromThreePoints(p0x, p0y, p1x, p1y, p2x, p2y, cx, cy, r)) return;
    if (!OpenCv2D::ArcSpanThroughMiddle(cx, cy, p0x, p0y, p1x, p1y, p2x, p2y, a0, a1)) return;
    DrawArcPolyline(dl, cx, cy, r, a0, a1, toScreen, col, thickness);
}

void DrawArcPickMarker(ImDrawList* dl, ImVec2 p, const char* label, ImU32 col) {
    dl->AddCircleFilled(p, 5.f, col);
    dl->AddCircle(p, 7.f, IM_COL32(255, 255, 255, 220), 0, 1.5f);
    DrawLabelTag(dl, ImVec2(p.x + 8.f, p.y - 16.f), label, col);
}

void DrawCaliperArcResultDetail(ImDrawList* dl, const OpenCv2D::CaliperArcResult& result,
                                const std::function<ImVec2(float, float)>& toScreen, ImU32 fitCol,
                                float fitThickness, bool pending) {
    if (!result.ok) return;
    const ImU32 calCol =
        pending ? IM_COL32(120, 220, 255, 200) : IM_COL32(80, 200, 255, 180);
    for (const OpenCv2D::LineSegment& c : result.calipers) {
        dl->AddLine(toScreen(c.x1, c.y1), toScreen(c.x2, c.y2), calCol, 1.f);
    }
    for (const OpenCv2D::CaliperEdgePoint& ep : result.edgePoints) {
        if (!ep.valid) continue;
        const ImVec2 p = toScreen(ep.x, ep.y);
        dl->AddCircleFilled(p, 3.5f, IM_COL32(255, 70, 70, 240));
        dl->AddCircle(p, 4.5f, IM_COL32(255, 255, 255, 200), 0, 1.2f);
    }
    DrawArcPolyline(dl, result.roiCenterX, result.roiCenterY, result.roiRadius,
                    result.roiStartAngle, result.roiEndAngle, toScreen,
                    pending ? IM_COL32(255, 220, 60, 160) : IM_COL32(255, 220, 60, 120), 1.5f);
    DrawArcPolyline(dl, result.fitCenterX, result.fitCenterY, result.fitRadius,
                    result.fitStartAngle, result.fitEndAngle, toScreen, fitCol, fitThickness);
    dl->AddCircle(toScreen(result.fitCenterX, result.fitCenterY), 3.f, fitCol, 0, 1.5f);
}

void DrawCircleFitDetail(ImDrawList* dl, const OpenCv2D::CircleFitResult& result,
                         const std::vector<OpenCv2D::CaliperEdgePoint>& edgePoints,
                         const std::function<ImVec2(float, float)>& toScreen, float screenScale,
                         ImU32 col, bool pending) {
    if (!result.ok) return;
    for (const OpenCv2D::CaliperEdgePoint& ep : edgePoints) {
        if (!ep.valid) continue;
        const ImVec2 p = toScreen(ep.x, ep.y);
        dl->AddCircleFilled(p, 3.f, IM_COL32(255, 70, 70, 240));
        dl->AddCircle(p, 4.f, IM_COL32(255, 255, 255, 200), 0, 1.f);
    }
    const ImVec2 center = toScreen(result.centerX, result.centerY);
    const float rScreen = result.radius * screenScale;
    dl->AddCircle(center, rScreen, col, 72, pending ? 2.8f : 2.2f);
    dl->AddCircleFilled(center, 3.5f, col);
    char label[48];
    std::snprintf(label, sizeof(label), u8"R=%.3f px", result.radius);
    DrawLabelTag(dl, ImVec2(center.x + 8.f, center.y - rScreen - 8.f), label, col);
}

void DrawRegionBlobMask(ImDrawList* dl, const OpenCv2D::RegionBlobResult& blob,
                        const std::function<ImVec2(float, float)>& toScreen, ImU32 fillCol) {
    if (!dl || !blob.ok || blob.hitMask.empty() || blob.roiWidth <= 0 || blob.roiHeight <= 0) {
        return;
    }
    const int roiW = blob.roiWidth;
    const int roiH = blob.roiHeight;
    const int x0 = static_cast<int>(blob.roiX0);
    const int y0 = static_cast<int>(blob.roiY0);
    for (int ry = 0; ry < roiH; ++ry) {
        int rx = 0;
        while (rx < roiW) {
            while (rx < roiW && blob.hitMask[static_cast<std::size_t>(ry * roiW + rx)] == 0) {
                ++rx;
            }
            if (rx >= roiW) break;
            int rxEnd = rx;
            while (rxEnd < roiW && blob.hitMask[static_cast<std::size_t>(ry * roiW + rxEnd)] != 0) {
                ++rxEnd;
            }
            const ImVec2 p0 = toScreen(static_cast<float>(x0 + rx), static_cast<float>(y0 + ry));
            const ImVec2 p1 =
                toScreen(static_cast<float>(x0 + rxEnd), static_cast<float>(y0 + ry + 1));
            dl->AddRectFilled(ImVec2(std::min(p0.x, p1.x), std::min(p0.y, p1.y)),
                              ImVec2(std::max(p0.x, p1.x), std::max(p0.y, p1.y)), fillCol);
            rx = rxEnd;
        }
    }
}

}  // namespace

void Application::DrawLineMeasureOverlay(ImDrawList* dl, const ImageView& view, float cursorX,
                                         float cursorY, float drawW, float drawH) {
    if (!dl || view.width <= 0 || view.height <= 0) return;
    const float sx = drawW / static_cast<float>(view.width);
    const float sy = drawH / static_cast<float>(view.height);
    const int src = ImageSourceOf(view);
    auto toScreen = [&](float x, float y) -> ImVec2 {
        return ImVec2(cursorX + x * sx, cursorY + y * sy);
    };

    if (lineMeasureDragging_ && lineMeasureDragSource_ == src) {
        dl->AddLine(toScreen(lineMeasureRoiX0_, lineMeasureRoiY0_),
                    toScreen(lineMeasureRoiX1_, lineMeasureRoiY1_), IM_COL32(255, 220, 60, 230),
                    2.f);
    }

    if (lineMeasurePending_ && lineMeasurePendingSource_ == src &&
        lineMeasurePendingResult_.ok) {
        DrawCaliperResultDetail(dl, lineMeasurePendingResult_, toScreen,
                                IM_COL32(255, 200, 60, 240), 2.5f, true);
        const ImVec2 mid(toScreen((lineMeasurePendingResult_.fitX1 +
                                   lineMeasurePendingResult_.fitX2) *
                                      0.5f,
                                  (lineMeasurePendingResult_.fitY1 +
                                   lineMeasurePendingResult_.fitY2) *
                                      0.5f));
        DrawLabelTag(dl, ImVec2(mid.x + 6.f, mid.y - 18.f), u8"待确认",
                     IM_COL32(255, 200, 60, 255));
    }

    for (std::size_t i = 0; i < measuredLines_.size(); ++i) {
        const MeasuredImageLine& line = measuredLines_[i];
        if (line.imageSource != src || !line.result.ok) continue;

        const bool lineDistActive = image2DTool_ == Image2DTool::LineDistance;
        const bool isA = lineDistActive && static_cast<int>(i) == lineDistPickA_;
        const bool isB = lineDistActive && static_cast<int>(i) == lineDistPickB_;
        const bool lineAngleOn = image2DTool_ == Image2DTool::LineAngle;
        const bool angleA = lineAngleOn && static_cast<int>(i) == lineAnglePickA_;
        const bool angleB = lineAngleOn && static_cast<int>(i) == lineAnglePickB_;
        const bool pointLineOn = image2DTool_ == Image2DTool::PointLineDistance;
        const bool plLine = pointLineOn && static_cast<int>(i) == pointLinePick_;
        ImU32 col = LineColorForId(line.id);
        float thickness = 2.5f;
        if (isA || angleA) {
            col = IM_COL32(255, 230, 60, 255);
            thickness = 3.5f;
        } else if (isB || angleB) {
            col = IM_COL32(80, 220, 255, 255);
            thickness = 3.5f;
        } else if (plLine) {
            col = IM_COL32(180, 120, 255, 255);
            thickness = 3.5f;
        }

        DrawCaliperResultDetail(dl, line.result, toScreen, col, thickness, false);

        char label[24];
        std::snprintf(label, sizeof(label), u8"线段%d", line.id);
        const ImVec2 mid(toScreen((line.result.fitX1 + line.result.fitX2) * 0.5f,
                                  (line.result.fitY1 + line.result.fitY2) * 0.5f));
        DrawLabelTag(dl, ImVec2(mid.x + 6.f, mid.y - 18.f), label, col);
    }

    if ((image2DTool_ == Image2DTool::CaliperArc || image2DTool_ == Image2DTool::CircleFit ||
         image2DTool_ == Image2DTool::EllipseFit) &&
        arcMeasureSource_ == src && !arcMeasurePending_ && !circleFitPending_ &&
        !ellipseFitPending_) {
        if (arcMeasurePhase_ == ArcMeasurePhase::PickB ||
            arcMeasurePhase_ == ArcMeasurePhase::DragBulge) {
            DrawArcPickMarker(dl, toScreen(arcRoiP0X_, arcRoiP0Y_), u8"A",
                              IM_COL32(255, 230, 60, 255));
        }
        if (arcMeasurePhase_ == ArcMeasurePhase::DragBulge) {
            DrawArcPickMarker(dl, toScreen(arcRoiP1X_, arcRoiP1Y_), u8"B",
                              IM_COL32(80, 220, 255, 255));
            dl->AddLine(toScreen(arcRoiP0X_, arcRoiP0Y_), toScreen(arcRoiP1X_, arcRoiP1Y_),
                        IM_COL32(180, 180, 180, 180), 1.5f);
            const float mx0 = (arcRoiP0X_ + arcRoiP1X_) * 0.5f;
            const float my0 = (arcRoiP0Y_ + arcRoiP1Y_) * 0.5f;
            dl->AddLine(toScreen(mx0, my0), toScreen(arcRoiP2X_, arcRoiP2Y_),
                        IM_COL32(255, 220, 60, 230), 2.f);
            DrawArcPickMarker(dl, toScreen(arcRoiP2X_, arcRoiP2Y_), u8"拱高",
                              IM_COL32(255, 200, 60, 255));
            DrawThreePointRoiArc(dl, arcRoiP0X_, arcRoiP0Y_, arcRoiP1X_, arcRoiP1Y_, arcRoiP2X_,
                                 arcRoiP2Y_, toScreen, IM_COL32(255, 220, 60, 200), 2.f);
        }
    }

    if (arcMeasurePending_ && arcMeasurePendingSource_ == src && arcMeasurePendingResult_.ok) {
        DrawCaliperArcResultDetail(dl, arcMeasurePendingResult_, toScreen,
                                   IM_COL32(255, 200, 60, 240), 2.5f, true);
        const float midAng =
            (arcMeasurePendingResult_.fitStartAngle + arcMeasurePendingResult_.fitEndAngle) * 0.5f;
        const ImVec2 mid(toScreen(
            arcMeasurePendingResult_.fitCenterX +
                arcMeasurePendingResult_.fitRadius * std::cos(midAng),
            arcMeasurePendingResult_.fitCenterY +
                arcMeasurePendingResult_.fitRadius * std::sin(midAng)));
        DrawLabelTag(dl, ImVec2(mid.x + 6.f, mid.y - 18.f), u8"待确认",
                     IM_COL32(255, 200, 60, 255));
    }

    for (std::size_t i = 0; i < measuredArcs_.size(); ++i) {
        const MeasuredImageArc& arc = measuredArcs_[i];
        if (arc.imageSource != src || !arc.result.ok) continue;
        const bool arcDistActive = image2DTool_ == Image2DTool::ArcDistance;
        const bool isA = arcDistActive && static_cast<int>(i) == arcDistPickA_;
        const bool isB = arcDistActive && static_cast<int>(i) == arcDistPickB_;
        const bool arcLenOn = image2DTool_ == Image2DTool::ArcLength;
        const bool arcLenPick = arcLenOn && static_cast<int>(i) == arcLengthPick_;
        ImU32 col = LineColorForId(arc.id);
        float thickness = 2.5f;
        if (isA) {
            col = IM_COL32(255, 230, 60, 255);
            thickness = 3.5f;
        } else if (isB) {
            col = IM_COL32(80, 220, 255, 255);
            thickness = 3.5f;
        } else if (arcLenPick) {
            col = IM_COL32(200, 120, 255, 255);
            thickness = 3.5f;
        }
        DrawCaliperArcResultDetail(dl, arc.result, toScreen, col, thickness, false);
        char label[24];
        std::snprintf(label, sizeof(label), u8"圆弧%d", arc.id);
        const float midAng = (arc.result.fitStartAngle + arc.result.fitEndAngle) * 0.5f;
        const ImVec2 mid(toScreen(arc.result.fitCenterX + arc.result.fitRadius * std::cos(midAng),
                                  arc.result.fitCenterY + arc.result.fitRadius * std::sin(midAng)));
        DrawLabelTag(dl, ImVec2(mid.x + 6.f, mid.y - 18.f), label, col);
    }

    if (circleFitPending_ && circleFitPendingSource_ == src && circleFitPendingResult_.ok) {
        DrawCircleFitDetail(dl, circleFitPendingResult_, circleFitPendingEdgePoints_, toScreen, sx,
                            IM_COL32(255, 200, 60, 255), true);
        DrawLabelTag(dl,
                     toScreen(circleFitPendingResult_.centerX, circleFitPendingResult_.centerY),
                     u8"待确认", IM_COL32(255, 200, 60, 255));
    }

    for (std::size_t fi = 0; fi < measuredCircleFits_.size(); ++fi) {
        const MeasuredCircleFit& fit = measuredCircleFits_[fi];
        if (fit.imageSource != src || !fit.result.ok) continue;
        const bool circleGapOn = image2DTool_ == Image2DTool::CircleGap;
        const bool gapA = circleGapOn && static_cast<int>(fi) == circleGapPickA_;
        const bool gapB = circleGapOn && static_cast<int>(fi) == circleGapPickB_;
        ImU32 col = LineColorForId(fit.id);
        if (gapA) col = IM_COL32(255, 230, 60, 255);
        else if (gapB) col = IM_COL32(80, 220, 255, 255);
        DrawCircleFitDetail(dl, fit.result, fit.edgePoints, toScreen, sx, col, false);
        char label[32];
        std::snprintf(label, sizeof(label), u8"圆%d", fit.id);
        DrawLabelTag(dl, toScreen(fit.result.centerX, fit.result.centerY), label, col);
    }

    if (image2DTool_ == Image2DTool::LineDistance && lineDistValid_ && lineDistPickA_ >= 0 &&
        lineDistPickB_ >= 0) {
        const MeasuredImageLine& la = measuredLines_[static_cast<std::size_t>(lineDistPickA_)];
        const MeasuredImageLine& lb = measuredLines_[static_cast<std::size_t>(lineDistPickB_)];
        if (la.imageSource == src && lb.imageSource == src) {
            for (const OpenCv2D::GapSample& gs : lineDistSamples_) {
                const ImVec2 pa = toScreen(gs.ax, gs.ay);
                const ImVec2 pb = toScreen(gs.bx, gs.by);
                dl->AddLine(pa, pb, IM_COL32(255, 120, 220, 140), 1.f);
                dl->AddCircleFilled(pa, 2.f, IM_COL32(255, 120, 220, 200));
            }

            char distLabel[64];
            std::snprintf(distLabel, sizeof(distLabel), u8"平均 %.2f px", lineDistPx_);
            if (!lineDistSamples_.empty()) {
                const OpenCv2D::GapSample& midS =
                    lineDistSamples_[lineDistSamples_.size() / 2];
                const ImVec2 pa = toScreen(midS.ax, midS.ay);
                const ImVec2 pb = toScreen(midS.bx, midS.by);
                const ImVec2 mid((pa.x + pb.x) * 0.5f, (pa.y + pb.y) * 0.5f);
                DrawLabelTag(dl, ImVec2(mid.x + 4.f, mid.y - 20.f), distLabel,
                             IM_COL32(255, 120, 220, 255));
            }
        }
    }

    if (image2DTool_ == Image2DTool::ArcDistance && arcDistValid_ && arcDistPickA_ >= 0 &&
        arcDistPickB_ >= 0) {
        const MeasuredImageArc& aa = measuredArcs_[static_cast<std::size_t>(arcDistPickA_)];
        const MeasuredImageArc& ab = measuredArcs_[static_cast<std::size_t>(arcDistPickB_)];
        if (aa.imageSource == src && ab.imageSource == src) {
            for (const OpenCv2D::GapSample& gs : arcDistSamples_) {
                const ImVec2 pa = toScreen(gs.ax, gs.ay);
                const ImVec2 pb = toScreen(gs.bx, gs.by);
                dl->AddLine(pa, pb, IM_COL32(120, 220, 255, 140), 1.f);
                dl->AddCircleFilled(pa, 2.f, IM_COL32(120, 220, 255, 200));
            }

            char distLabel[64];
            std::snprintf(distLabel, sizeof(distLabel), u8"平均 %.2f px", arcDistPx_);
            if (!arcDistSamples_.empty()) {
                const OpenCv2D::GapSample& midS = arcDistSamples_[arcDistSamples_.size() / 2];
                const ImVec2 pa = toScreen(midS.ax, midS.ay);
                const ImVec2 pb = toScreen(midS.bx, midS.by);
                const ImVec2 mid((pa.x + pb.x) * 0.5f, (pa.y + pb.y) * 0.5f);
                DrawLabelTag(dl, ImVec2(mid.x + 4.f, mid.y - 20.f), distLabel,
                             IM_COL32(120, 220, 255, 255));
            }
        }
    }

    for (const MeasuredPointDist& pd : measuredPointDists_) {
        if (pd.imageSource != src) continue;
        const ImVec2 pa = toScreen(pd.ax, pd.ay);
        const ImVec2 pb = toScreen(pd.bx, pd.by);
        dl->AddLine(pa, pb, IM_COL32(255, 180, 80, 230), 2.f);
        dl->AddCircleFilled(pa, 5.f, IM_COL32(255, 230, 60, 255));
        dl->AddCircleFilled(pb, 5.f, IM_COL32(80, 220, 255, 255));
        char label[48];
        std::snprintf(label, sizeof(label), u8"%.2f px", pd.distance);
        DrawLabelTag(dl, ImVec2((pa.x + pb.x) * 0.5f, (pa.y + pb.y) * 0.5f - 18.f), label,
                     IM_COL32(255, 200, 100, 255));
    }
    if (image2DTool_ == Image2DTool::PointDistance && pointDistPhase_ == PointPickPhase::PickB &&
        pointDistSource_ == src) {
        const ImVec2 pa = toScreen(pointDistAx_, pointDistAy_);
        dl->AddCircleFilled(pa, 5.f, IM_COL32(255, 230, 60, 255));
        DrawLabelTag(dl, ImVec2(pa.x + 8.f, pa.y - 16.f), u8"A", IM_COL32(255, 230, 60, 255));
    }

    if (image2DTool_ == Image2DTool::PointLineDistance && pointLineSource_ == src) {
        if (pointLinePhase_ == PointPickPhase::PickB || pointLineValid_) {
            const ImVec2 pp = toScreen(pointLinePx_, pointLinePy_);
            dl->AddCircleFilled(pp, 5.f, IM_COL32(200, 120, 255, 255));
            DrawLabelTag(dl, ImVec2(pp.x + 8.f, pp.y - 16.f), u8"P", IM_COL32(200, 120, 255, 255));
        }
        if (pointLineValid_ && pointLinePick_ >= 0 &&
            pointLinePick_ < static_cast<int>(measuredLines_.size())) {
            const ImVec2 pf = toScreen(pointLineFootX_, pointLineFootY_);
            const ImVec2 pp = toScreen(pointLinePx_, pointLinePy_);
            dl->AddLine(pp, pf, IM_COL32(200, 120, 255, 220), 2.f);
            dl->AddCircleFilled(pf, 4.f, IM_COL32(255, 255, 255, 220));
            char label[48];
            std::snprintf(label, sizeof(label), u8"%.2f px", pointLineDistPx_);
            DrawLabelTag(dl, ImVec2((pp.x + pf.x) * 0.5f, (pp.y + pf.y) * 0.5f - 16.f), label,
                         IM_COL32(200, 120, 255, 255));
        }
    }

    if (caliperPointDragging_ && caliperPointDragSource_ == src) {
        dl->AddLine(toScreen(caliperPointRoiX0_, caliperPointRoiY0_),
                    toScreen(caliperPointRoiX1_, caliperPointRoiY1_), IM_COL32(255, 220, 60, 230),
                    2.f);
    }
    if (caliperPointPending_ && caliperPointPendingSource_ == src &&
        caliperPointPendingEdge_.valid) {
        const ImVec2 p = toScreen(caliperPointPendingEdge_.x, caliperPointPendingEdge_.y);
        dl->AddCircleFilled(p, 6.f, IM_COL32(255, 200, 60, 255));
        DrawLabelTag(dl, ImVec2(p.x + 8.f, p.y - 16.f), u8"待确认", IM_COL32(255, 200, 60, 255));
    }
    for (const MeasuredCaliperPoint& cp : measuredCaliperPoints_) {
        if (cp.imageSource != src) continue;
        const ImVec2 p = toScreen(cp.x, cp.y);
        dl->AddLine(toScreen(cp.roiX0, cp.roiY0), toScreen(cp.roiX1, cp.roiY1),
                    IM_COL32(120, 200, 255, 160), 1.5f);
        dl->AddCircleFilled(p, 5.f, IM_COL32(255, 100, 100, 255));
        char label[24];
        std::snprintf(label, sizeof(label), u8"E%d", cp.id);
        DrawLabelTag(dl, ImVec2(p.x + 8.f, p.y - 16.f), label, IM_COL32(255, 100, 100, 255));
    }

    if (image2DTool_ == Image2DTool::CaliperCircle && circleCaliperSource_ == src &&
        circleCaliperPhase_ == CircleCaliperPhase::DragRadius && circleCaliperR_ > 1.f) {
        const ImVec2 center = toScreen(circleCaliperCx_, circleCaliperCy_);
        dl->AddCircle(center, circleCaliperR_ * sx, IM_COL32(255, 220, 60, 200), 64, 2.f);
        dl->AddCircleFilled(center, 4.f, IM_COL32(255, 230, 60, 255));
    }
    if (circleCaliperPending_ && circleCaliperPendingSource_ == src &&
        circleCaliperPendingResult_.ok) {
        const auto& r = circleCaliperPendingResult_;
        OpenCv2D::CircleFitResult fit;
        fit.centerX = r.fitCenterX;
        fit.centerY = r.fitCenterY;
        fit.radius = r.fitRadius;
        fit.ok = true;
        DrawCircleFitDetail(dl, fit, r.edgePoints, toScreen, sx, IM_COL32(255, 200, 60, 255), true);
        DrawLabelTag(dl, toScreen(r.fitCenterX, r.fitCenterY), u8"待确认",
                     IM_COL32(255, 200, 60, 255));
    }
    for (const MeasuredCircleCaliper& cc : measuredCircleCalipers_) {
        if (cc.imageSource != src || !cc.result.ok) continue;
        OpenCv2D::CircleFitResult fit;
        fit.centerX = cc.result.fitCenterX;
        fit.centerY = cc.result.fitCenterY;
        fit.radius = cc.result.fitRadius;
        fit.ok = true;
        DrawCircleFitDetail(dl, fit, cc.result.edgePoints, toScreen, sx, LineColorForId(cc.id),
                            false);
        char label[32];
        std::snprintf(label, sizeof(label), u8"圆卡%d R=%.2f", cc.id, cc.result.fitRadius);
        DrawLabelTag(dl, toScreen(cc.result.fitCenterX, cc.result.fitCenterY), label,
                     LineColorForId(cc.id));
    }

    if (image2DTool_ == Image2DTool::CircleGap && circleGapValid_ && circleGapPickA_ >= 0 &&
        circleGapPickB_ >= 0) {
        const MeasuredCircleFit& ca = measuredCircleFits_[static_cast<std::size_t>(circleGapPickA_)];
        const MeasuredCircleFit& cb = measuredCircleFits_[static_cast<std::size_t>(circleGapPickB_)];
        if (ca.imageSource == src && cb.imageSource == src) {
            const ImVec2 c1 = toScreen(ca.result.centerX, ca.result.centerY);
            const ImVec2 c2 = toScreen(cb.result.centerX, cb.result.centerY);
            dl->AddLine(c1, c2, IM_COL32(255, 200, 80, 220), 2.f);
            char label[64];
            std::snprintf(label, sizeof(label), u8"圆心距 %.2f", circleGapCenterDist_);
            DrawLabelTag(dl, ImVec2((c1.x + c2.x) * 0.5f, (c1.y + c2.y) * 0.5f - 16.f), label,
                         IM_COL32(255, 200, 80, 255));
        }
    }

    if (image2DTool_ == Image2DTool::ArcLength && arcLengthValid_ && arcLengthPick_ >= 0 &&
        arcLengthPick_ < static_cast<int>(measuredArcs_.size())) {
        const MeasuredImageArc& arc = measuredArcs_[static_cast<std::size_t>(arcLengthPick_)];
        if (arc.imageSource == src && arc.result.ok) {
            const float x0 = arc.result.fitCenterX +
                             arc.result.fitRadius * std::cos(arc.result.fitStartAngle);
            const float y0 = arc.result.fitCenterY +
                             arc.result.fitRadius * std::sin(arc.result.fitStartAngle);
            const float x1 = arc.result.fitCenterX +
                             arc.result.fitRadius * std::cos(arc.result.fitEndAngle);
            const float y1 = arc.result.fitCenterY +
                             arc.result.fitRadius * std::sin(arc.result.fitEndAngle);
            dl->AddLine(toScreen(x0, y0), toScreen(x1, y1), IM_COL32(200, 120, 255, 220), 2.f);
            char label[80];
            std::snprintf(label, sizeof(label), u8"弧长%.1f 弦长%.1f", arcLengthMetrics_.arcLength,
                          arcLengthMetrics_.chordLength);
            const float midAng =
                (arc.result.fitStartAngle + arc.result.fitEndAngle) * 0.5f;
            const ImVec2 mid(toScreen(
                arc.result.fitCenterX + arc.result.fitRadius * std::cos(midAng),
                arc.result.fitCenterY + arc.result.fitRadius * std::sin(midAng)));
            DrawLabelTag(dl, ImVec2(mid.x + 6.f, mid.y - 20.f), label, IM_COL32(200, 120, 255, 255));
        }
    }

    for (const MeasuredThreePointCircle& tc : measuredThreePointCircles_) {
        if (tc.imageSource != src) continue;
        const ImVec2 c = toScreen(tc.centerX, tc.centerY);
        dl->AddCircle(c, tc.radius * sx, IM_COL32(100, 220, 180, 220), 64, 2.f);
        dl->AddCircleFilled(c, 3.f, IM_COL32(100, 220, 180, 255));
        char label[32];
        std::snprintf(label, sizeof(label), u8"三点圆%d", tc.id);
        DrawLabelTag(dl, ImVec2(c.x + 8.f, c.y - tc.radius * sx - 8.f), label,
                     IM_COL32(100, 220, 180, 255));
    }
    if (image2DTool_ == Image2DTool::ThreePointCircle && threePointSource_ == src) {
        const int n = static_cast<int>(threePointPhase_);
        for (int i = 0; i < n; ++i) {
            dl->AddCircleFilled(toScreen(threePointX_[i], threePointY_[i]), 5.f,
                                IM_COL32(255, 230, 60, 255));
        }
    }

    auto drawDragRect = [&](bool dragging, int dragSource, float x0, float y0, float x1, float y1,
                            ImU32 col) {
        if (!dragging || dragSource != src) return;
        const float l = std::min(x0, x1);
        const float r = std::max(x0, x1);
        const float t = std::min(y0, y1);
        const float b = std::max(y0, y1);
        dl->AddRect(toScreen(l, t), toScreen(r, b), col, 0.f, 0, 2.f);
    };
    drawDragRect(rectCaliperDragging_, rectCaliperDragSource_, rectCaliperRoiX0_,
                 rectCaliperRoiY0_, rectCaliperRoiX1_, rectCaliperRoiY1_,
                 IM_COL32(80, 200, 255, 220));
    drawDragRect(regionBlobDragging_ && !CanSyncRegionBlob(), regionBlobDragSource_,
                 regionBlobRoiX0_, regionBlobRoiY0_, regionBlobRoiX1_, regionBlobRoiY1_,
                 IM_COL32(255, 70, 70, 240));
    if (regionBlobDragging_ && CanSyncRegionBlob()) {
        const float l = std::min(regionBlobRoiX0_, regionBlobRoiX1_);
        const float r = std::max(regionBlobRoiX0_, regionBlobRoiX1_);
        const float t = std::min(regionBlobRoiY0_, regionBlobRoiY1_);
        const float b = std::max(regionBlobRoiY0_, regionBlobRoiY1_);
        const ImU32 col = (src == 0) ? IM_COL32(60, 200, 255, 240) : IM_COL32(255, 70, 70, 240);
        dl->AddRect(toScreen(l, t), toScreen(r, b), col, 0.f, 0, 2.f);
    }

    auto drawRegionBlobOverlay = [&](const OpenCv2D::RegionBlobResult& b, bool onDepthView) {
        const ImU32 boxCol =
            onDepthView ? IM_COL32(60, 200, 255, 255) : IM_COL32(255, 45, 45, 255);
        const ImU32 fillCol =
            onDepthView ? IM_COL32(40, 160, 255, 140) : IM_COL32(255, 30, 30, 175);
        dl->AddRect(toScreen(b.roiX0, b.roiY0), toScreen(b.roiX1, b.roiY1), boxCol, 0.f, 0, 2.5f);
        DrawRegionBlobMask(dl, b, toScreen, fillCol);
        if (b.pixelCount > 0) {
            const ImVec2 c = toScreen(b.centroidX, b.centroidY);
            dl->AddCircleFilled(c, 5.f, boxCol);
            dl->AddCircle(c, 8.f, IM_COL32(255, 255, 255, 230), 0, 2.f);
        }
    };

    if (regionBlobPending_ && regionBlobPendingResult_.ok) {
        const bool showHere = regionBlobPendingSource_ == src ||
                              (CanSyncRegionBlob() && view.width == syncWidth_ &&
                               view.height == syncHeight_);
        if (showHere) {
            drawRegionBlobOverlay(regionBlobPendingResult_, src == 0);
        }
    }
    for (const MeasuredRegionBlob& rb : measuredRegionBlobs_) {
        if (!rb.result.ok) continue;
        const bool showHere = rb.imageSource == src ||
                              (rb.syncedPair && CanSyncRegionBlob() && view.width == syncWidth_ &&
                               view.height == syncHeight_);
        if (!showHere) continue;
        drawRegionBlobOverlay(rb.result, src == 0);
    }

    auto drawDragLine = [&](bool dragging, int dragSource, float x0, float y0, float x1, float y1,
                            ImU32 col) {
        if (!dragging || dragSource != src) return;
        dl->AddLine(toScreen(x0, y0), toScreen(x1, y1), col, 2.f);
    };
    drawDragLine(profileWidthDragging_, profileWidthDragSource_, profileWidthRoiX0_,
                 profileWidthRoiY0_, profileWidthRoiX1_, profileWidthRoiY1_,
                 IM_COL32(255, 180, 80, 230));
    drawDragLine(depthProfileDragging_, depthProfileDragSource_, depthProfileRoiX0_,
                 depthProfileRoiY0_, depthProfileRoiX1_, depthProfileRoiY1_,
                 IM_COL32(80, 220, 255, 230));

    if (depthProfileValid_ && depthProfileDragSource_ == src) {
        const ImU32 profCol = IM_COL32(80, 220, 255, 220);
        if (depthProfileMode_ == DepthProfileMode::ScanRow ||
            depthProfileMode_ == DepthProfileMode::ScanColumn) {
            dl->AddLine(toScreen(depthProfileRoiX0_, depthProfileRoiY0_),
                        toScreen(depthProfileRoiX1_, depthProfileRoiY1_), profCol, 2.f);
        } else if (!depthProfileDragging_) {
            dl->AddLine(toScreen(depthProfileRoiX0_, depthProfileRoiY0_),
                        toScreen(depthProfileRoiX1_, depthProfileRoiY1_), profCol, 2.f);
        }
        if (depthProfileHoverIdx_ >= 0 &&
            depthProfileHoverIdx_ < static_cast<int>(depthProfileSamples_.size())) {
            const OpenCv2D::LineProfileSample& hs =
                depthProfileSamples_[static_cast<std::size_t>(depthProfileHoverIdx_)];
            if (!std::isnan(hs.value)) {
                const ImVec2 hp = toScreen(hs.x, hs.y);
                dl->AddCircleFilled(hp, 5.f, IM_COL32(255, 200, 80, 255));
                dl->AddCircle(hp, 8.f, IM_COL32(255, 255, 255, 180), 0, 1.5f);
            }
        }
        auto drawProfPick = [&](int idx, ImU32 col) {
            if (idx < 0 || idx >= static_cast<int>(depthProfileSamples_.size())) return;
            const OpenCv2D::LineProfileSample& ps =
                depthProfileSamples_[static_cast<std::size_t>(idx)];
            if (std::isnan(ps.value)) return;
            dl->AddCircleFilled(toScreen(ps.x, ps.y), 6.f, col);
        };
        drawProfPick(depthProfilePickA_, IM_COL32(255, 230, 60, 255));
        drawProfPick(depthProfilePickB_, IM_COL32(80, 220, 255, 255));
    }

    if (rectCaliperPending_ && rectCaliperPendingSource_ == src && rectCaliperPendingResult_.ok) {
        const auto& r = rectCaliperPendingResult_;
        const float hw = r.width * 0.5f;
        const float hh = r.height * 0.5f;
        const float rad = r.angleDeg * 3.14159265f / 180.f;
        const float cosA = std::cos(rad);
        const float sinA = std::sin(rad);
        ImVec2 corners[4];
        const float lx[4] = {-hw, hw, hw, -hw};
        const float ly[4] = {-hh, -hh, hh, hh};
        for (int i = 0; i < 4; ++i) {
            corners[i] = toScreen(r.centerX + lx[i] * cosA - ly[i] * sinA,
                                  r.centerY + lx[i] * sinA + ly[i] * cosA);
        }
        for (int i = 0; i < 4; ++i) {
            dl->AddLine(corners[i], corners[(i + 1) % 4], IM_COL32(255, 200, 60, 230), 2.f);
        }
    }
    for (const MeasuredRectCaliper& rc : measuredRectCalipers_) {
        if (rc.imageSource != src || !rc.result.ok) continue;
        const auto& r = rc.result;
        dl->AddRect(toScreen(r.roiX0, r.roiY0), toScreen(r.roiX1, r.roiY1),
                    LineColorForId(rc.id), 0.f, 0, 1.5f);
    }

    if (profileWidthPending_ && profileWidthPendingSource_ == src &&
        profileWidthPendingResult_.ok) {
        const auto& w = profileWidthPendingResult_;
        dl->AddLine(toScreen(w.edge1X, w.edge1Y), toScreen(w.edge2X, w.edge2Y),
                    IM_COL32(255, 200, 60, 230), 2.5f);
    }
    for (const MeasuredProfileWidth& pw : measuredProfileWidths_) {
        if (pw.imageSource != src || !pw.result.ok) continue;
        dl->AddLine(toScreen(pw.result.edge1X, pw.result.edge1Y),
                    toScreen(pw.result.edge2X, pw.result.edge2Y), LineColorForId(pw.id), 2.f);
    }

    if (ellipseFitPending_ && ellipseFitPendingSource_ == src && ellipseFitPendingResult_.ok) {
        const auto& e = ellipseFitPendingResult_;
        const ImVec2 c = toScreen(e.centerX, e.centerY);
        dl->AddEllipse(c, ImVec2(e.axisA * sx, e.axisB * sy), e.angleDeg, IM_COL32(255, 200, 60, 220),
                       0, 2.f);
    }

    if (pointProjValid_ && pointProjSource_ == src && pointProjLinePick_ >= 0) {
        const ImVec2 pp = toScreen(pointProjPx_, pointProjPy_);
        const ImVec2 pf = toScreen(pointProjResult_.footX, pointProjResult_.footY);
        dl->AddLine(pp, pf, IM_COL32(180, 140, 255, 230), 2.f);
        dl->AddCircleFilled(pf, 4.f, IM_COL32(255, 255, 255, 230));
    }

    if (concentricityValid_ && concentricityPickA_ >= 0 && concentricityPickB_ >= 0) {
        const auto& ca = measuredCircleFits_[static_cast<std::size_t>(concentricityPickA_)];
        const auto& cb = measuredCircleFits_[static_cast<std::size_t>(concentricityPickB_)];
        if (ca.imageSource == src) {
            dl->AddLine(toScreen(ca.result.centerX, ca.result.centerY),
                        toScreen(cb.result.centerX, cb.result.centerY), IM_COL32(255, 160, 80, 220),
                        2.f);
        }
    }

    if (depthHeightValid_ && depthHeightSource_ == src) {
        const ImVec2 pa = toScreen(depthHeightAx_, depthHeightAy_);
        dl->AddCircleFilled(pa, 5.f, IM_COL32(255, 230, 60, 255));
        char label[48];
        std::snprintf(label, sizeof(label), u8"ΔZ=%.4f", depthHeightDelta_);
        DrawLabelTag(dl, ImVec2(pa.x + 8.f, pa.y - 16.f), label, IM_COL32(80, 220, 255, 255));
    }

    auto drawMatchHits = [&](const OpenCv2D::TemplateMatchResult& res, ImU32 boxCol, ImU32 labelCol,
                             const char* tag) {
        if (!res.ok) return;
        for (std::size_t hi = 0; hi < res.hits.size(); ++hi) {
            const OpenCv2D::TemplateMatchHit& h = res.hits[hi];
            const float hw = h.templateWidth * 0.5f;
            const float hh = h.templateHeight * 0.5f;
            const float rad = h.angleDeg * 3.14159265f / 180.f;
            const float cosA = std::cos(rad);
            const float sinA = std::sin(rad);
            ImVec2 corners[4];
            const float lx[4] = {-hw, hw, hw, -hw};
            const float ly[4] = {-hh, -hh, hh, hh};
            for (int i = 0; i < 4; ++i) {
                corners[i] = toScreen(h.centerX + lx[i] * cosA - ly[i] * sinA,
                                      h.centerY + lx[i] * sinA + ly[i] * cosA);
            }
            for (int i = 0; i < 4; ++i) {
                dl->AddLine(corners[i], corners[(i + 1) % 4], boxCol, 2.5f);
            }
            dl->AddCircleFilled(toScreen(h.centerX, h.centerY), 3.f, boxCol);
            char label[72];
            std::snprintf(label, sizeof(label), u8"%s%.3f ∠%.0f° ×%.2f", tag, h.score, h.angleDeg,
                          h.scale);
            DrawLabelTag(dl, ImVec2(corners[0].x, corners[0].y - 18.f), label, labelCol);
        }
    };

    if (templateMatchHasTemplate_) {
        const float l = std::min(templateMatchTplX0_, templateMatchTplX1_);
        const float r = std::max(templateMatchTplX0_, templateMatchTplX1_);
        const float t = std::min(templateMatchTplY0_, templateMatchTplY1_);
        const float b = std::max(templateMatchTplY0_, templateMatchTplY1_);
        dl->AddRect(toScreen(l, t), toScreen(r, b), IM_COL32(80, 220, 255, 230), 0.f, 0, 2.f);
        DrawLabelTag(dl, toScreen(l, t - 4.f), u8"模板", IM_COL32(80, 220, 255, 255));
    }
    drawDragRect(templateMatchTplDragging_, templateMatchDragSource_, templateMatchTplX0_,
                 templateMatchTplY0_, templateMatchTplX1_, templateMatchTplY1_,
                 IM_COL32(80, 220, 255, 220));
    if (!templateMatchSearchFull_ && templateMatchHasTemplate_) {
        const float l = std::min(templateMatchSearchX0_, templateMatchSearchX1_);
        const float r = std::max(templateMatchSearchX0_, templateMatchSearchX1_);
        const float t = std::min(templateMatchSearchY0_, templateMatchSearchY1_);
        const float b = std::max(templateMatchSearchY0_, templateMatchSearchY1_);
        dl->AddRect(toScreen(l, t), toScreen(r, b), IM_COL32(255, 220, 80, 180), 0.f, 0, 1.5f);
    }
    drawDragRect(templateMatchSearchDragging_, templateMatchDragSource_, templateMatchSearchX0_,
                 templateMatchSearchY0_, templateMatchSearchX1_, templateMatchSearchY1_,
                 IM_COL32(255, 220, 80, 220));
    if (templateMatchPending_ && templateMatchPendingSource_ == src) {
        drawMatchHits(templateMatchPendingResult_, IM_COL32(255, 200, 60, 240),
                      IM_COL32(255, 200, 60, 255), u8"待确认 ");
    }
    for (const MeasuredTemplateMatch& tm : measuredTemplateMatches_) {
        if (tm.imageSource != src) continue;
        drawMatchHits(tm.result, LineColorForId(tm.id), LineColorForId(tm.id), "");
    }
}

void Application::DrawImage2DToolPanel() {
    if (image2DTool_ == Image2DTool::CircleFit) {
        ImGui::TextColored(ImVec4(0.55f, 0.95f, 0.75f, 1.f), "%s",
                           ArcMeasurePhaseHint(arcMeasurePhase_));
        ImGui::TextWrapped(
            u8"对圆弧线段进行最小二乘圆拟合，输出半径与圆心。\n"
            u8"方式一：点击图像上已有圆弧（使用其边缘点拟合）\n"
            u8"方式二：设置 A/B 点并拖拽拱高，沿弧提取边缘后拟合");
        if (ImGui::Button(u8"重新选点", ImVec2(-1.f, 0))) {
            ResetArcMeasurePick();
            CancelCircleFitPending();
            SetStatus(u8"已重置，请重新设置 A 点或点击圆弧");
        }
        ImGui::Spacing();
        DrawCaliperLineParamsPanel(lineMeasureParams_, true);
        ImGui::Checkbox(u8"显示测量叠加", &showLineMeasureOverlay_);

        if (circleFitPending_ && circleFitPendingResult_.ok) {
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::TextColored(ImVec4(1.f, 0.85f, 0.45f, 1.f), u8"待确认圆拟合");
            if (circleFitPendingFromArcId_ >= 0) {
                ImGui::Text(u8"来源圆弧 %d", circleFitPendingFromArcId_);
            } else {
                ImGui::TextDisabled(u8"来源：沿弧新提取边缘");
            }
            ImGui::TextColored(ImVec4(0.55f, 0.95f, 0.75f, 1.f), u8"拟合半径 = %.4f px",
                               circleFitPendingResult_.radius);
            ImGui::Text(u8"圆心 (%.2f, %.2f)  有效点 %d  RMS %.4f px",
                        circleFitPendingResult_.centerX, circleFitPendingResult_.centerY,
                        circleFitPendingResult_.pointCount, circleFitPendingResult_.rms);
            if (ImGui::Button(u8"确认圆拟合", ImVec2(-1.f, 36.f))) {
                ConfirmCircleFit();
            }
            if (ImGui::Button(u8"取消预览", ImVec2(-1.f, 0))) {
                CancelCircleFitPending();
                SetStatus(u8"已取消圆拟合预览");
            }
        }

        ImGui::Spacing();
        ImGui::Separator();
        if (!CanUndoMeasuredLine()) ImGui::BeginDisabled();
        if (ImGui::Button(u8"撤回圆拟合 / 预览", ImVec2(-1.f, 0))) {
            UndoLastMeasuredLine();
        }
        if (!CanUndoMeasuredLine()) ImGui::EndDisabled();

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::TextDisabled(u8"已确认圆拟合 (%zu)", measuredCircleFits_.size());
        if (measuredCircleFits_.empty()) {
            ImGui::TextDisabled(u8"暂无结果，请点击圆弧或沿弧提取后确认");
        } else {
            for (std::size_t i = 0; i < measuredCircleFits_.size(); ++i) {
                const MeasuredCircleFit& fit = measuredCircleFits_[i];
                ImGui::PushID(static_cast<int>(i) + 40000);
                ImGui::Text(u8"圆%d", fit.id);
                if (fit.sourceArcId >= 0) {
                    ImGui::SameLine();
                    ImGui::TextDisabled(u8"(圆弧%d)", fit.sourceArcId);
                }
                ImGui::TextColored(ImVec4(0.55f, 0.95f, 0.75f, 1.f), u8"半径 = %.4f px",
                                   fit.result.radius);
                ImGui::Text(u8"圆心 (%.2f, %.2f)  RMS %.4f px", fit.result.centerX,
                            fit.result.centerY, fit.result.rms);
                if (ImGui::Button(u8"删除")) {
                    measuredCircleFits_.erase(measuredCircleFits_.begin() +
                                              static_cast<std::ptrdiff_t>(i));
                    ImGui::PopID();
                    break;
                }
                ImGui::Separator();
                ImGui::PopID();
            }
        }
        if (ImGui::Button(u8"清除全部圆拟合", ImVec2(-1.f, 0))) {
            measuredCircleFits_.clear();
            nextCircleFitId_ = 1;
            CancelCircleFitPending();
            SetStatus(u8"已清除全部圆拟合");
        }
        return;
    }

    if (image2DTool_ == Image2DTool::CaliperArc) {
        ImGui::TextColored(ImVec4(0.55f, 0.95f, 0.75f, 1.f), "%s",
                           ArcMeasurePhaseHint(arcMeasurePhase_));
        ImGui::TextWrapped(
            u8"① 点击设置圆弧端点 A\n"
            u8"② 点击设置圆弧端点 B\n"
            u8"③ 拖拽 AB 中垂线上的拱高线段调节弧线，松开后卡尺预览\n"
            u8"④ 确认后添加圆弧");
        if (ImGui::Button(u8"重新选点", ImVec2(-1.f, 0))) {
            ResetArcMeasurePick();
            arcMeasurePending_ = false;
            arcMeasurePendingSource_ = -1;
            arcMeasurePendingResult_ = {};
            SetStatus(u8"已重置，请重新设置 A 点");
        }
        ImGui::Spacing();
        DrawCaliperLineParamsPanel(lineMeasureParams_, true);
        ImGui::Checkbox(u8"显示测量叠加", &showLineMeasureOverlay_);

        if (arcMeasurePending_ && arcMeasurePendingResult_.ok) {
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::TextColored(ImVec4(1.f, 0.85f, 0.45f, 1.f), u8"待确认圆弧卡尺");
            ImGui::Text(u8"半径 %.3f px  有效点 %d  RMS %.3f px",
                        arcMeasurePendingResult_.fitRadius, arcMeasurePendingResult_.validCount,
                        arcMeasurePendingResult_.fitRms);
            if (ImGui::Button(u8"确认添加圆弧", ImVec2(-1.f, 36.f))) {
                ConfirmArcMeasure();
            }
            if (ImGui::Button(u8"取消预览", ImVec2(-1.f, 0))) {
                arcMeasurePending_ = false;
                arcMeasurePendingSource_ = -1;
                arcMeasurePendingResult_ = {};
                arcMeasurePhase_ = ArcMeasurePhase::DragBulge;
                SetStatus(u8"已取消圆弧卡尺预览，可继续调节拱高");
            }
        }

        ImGui::Spacing();
        ImGui::Separator();
        if (!CanUndoMeasuredLine()) ImGui::BeginDisabled();
        if (ImGui::Button(u8"撤回圆弧 / 预览", ImVec2(-1.f, 0))) {
            UndoLastMeasuredLine();
        }
        if (!CanUndoMeasuredLine()) ImGui::EndDisabled();

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::TextDisabled(u8"已提取圆弧 (%zu)", measuredArcs_.size());
        if (measuredArcs_.empty()) {
            ImGui::TextDisabled(u8"暂无圆弧，请先设置 A/B 点并调节拱高");
        } else {
            for (std::size_t i = 0; i < measuredArcs_.size(); ++i) {
                const MeasuredImageArc& arc = measuredArcs_[i];
                const char* imgName = arc.imageSource == 1 ? u8"亮度" : u8"深度";
                ImGui::PushID(static_cast<int>(i) + 10000);
                ImGui::Text(u8"圆弧%d", arc.id);
                ImGui::SameLine();
                ImGui::TextDisabled(u8"(%s)", imgName);
                ImGui::Text(u8"半径 %.3f px  有效点 %d  RMS %.3f px", arc.result.fitRadius,
                            arc.result.validCount, arc.result.fitRms);
                if (ImGui::Button(u8"删除")) {
                    measuredArcs_.erase(measuredArcs_.begin() + static_cast<std::ptrdiff_t>(i));
                    ImGui::PopID();
                    break;
                }
                ImGui::Separator();
                ImGui::PopID();
            }
        }
        if (ImGui::Button(u8"清除全部圆弧", ImVec2(-1.f, 0))) {
            ClearArcMeasure();
            arcMeasurePending_ = false;
            arcMeasurePendingSource_ = -1;
            arcMeasurePendingResult_ = {};
            measuredArcs_.clear();
            nextMeasuredArcId_ = 1;
            arcDistPickA_ = -1;
            arcDistPickB_ = -1;
            arcDistValid_ = false;
            arcDistSamples_.clear();
            SetStatus(u8"已清除全部圆弧");
        }
        return;
    }

    if (image2DTool_ == Image2DTool::LineDistance) {
        ImGui::TextWrapped(
            u8"在图像上直接点击线段选 A、B（或在下方的列表中选），再计算平均间隙。\n"
            u8"可选线段为当前已提取并显示在图像上的线段（不限于当前正在使用的算子）。");
        ImGui::Spacing();
        ImGui::TextDisabled(u8"算法说明");
        ImGui::TextWrapped(
            u8"选为 A 的线段上均匀采样，过每点作垂直于 A 的直线，与线段 B 求交；"
            u8"无交点不计入，有效距离取平均。");
        ImGui::SetNextItemWidth(-1.f);
        ImGui::DragInt(u8"采样点数", &lineDistSampleCount_, 1, 4, 200);
        ImGui::TextDisabled(u8"A 线段上的均匀采样数量；越多统计越稳定，但计算稍慢");
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::TextDisabled(u8"已提取线段 (%zu)", measuredLines_.size());
        if (measuredLines_.empty()) {
            ImGui::TextColored(ImVec4(1.f, 0.65f, 0.45f, 1.f),
                               u8"当前无已显示线段，请先用卡尺提线提取线段");
        } else {
            if (measuredLines_.size() < 2) {
                ImGui::TextColored(ImVec4(1.f, 0.65f, 0.45f, 1.f),
                                   u8"至少需要 2 条线段才能测量距离");
            }
            for (std::size_t i = 0; i < measuredLines_.size(); ++i) {
                const MeasuredImageLine& line = measuredLines_[i];
                const char* imgName = line.imageSource == 1 ? u8"亮度" : u8"深度";
                ImGui::PushID(static_cast<int>(i) + 20000);
                ImGui::Text(u8"线段%d", line.id);
                ImGui::SameLine();
                ImGui::TextDisabled(u8"(%s)", imgName);
                const bool pickedA = static_cast<int>(i) == lineDistPickA_;
                const bool pickedB = static_cast<int>(i) == lineDistPickB_;
                if (pickedA) ImGui::TextColored(ImVec4(1.f, 0.9f, 0.4f, 1.f), u8"  [A]");
                if (pickedB) ImGui::TextColored(ImVec4(0.4f, 0.85f, 1.f, 1.f), u8"  [B]");
                if (ImGui::Button(u8"选为 A")) {
                    lineDistPickA_ = static_cast<int>(i);
                    lineDistValid_ = false;
                    lineDistSamples_.clear();
                }
                ImGui::SameLine();
                if (ImGui::Button(u8"选为 B")) {
                    lineDistPickB_ = static_cast<int>(i);
                    lineDistValid_ = false;
                    lineDistSamples_.clear();
                }
                ImGui::Separator();
                ImGui::PopID();
            }
        }
        if (lineDistPickA_ >= 0 && lineDistPickA_ < static_cast<int>(measuredLines_.size())) {
            ImGui::Text(u8"A: 线段%d", measuredLines_[static_cast<std::size_t>(lineDistPickA_)].id);
        } else {
            ImGui::TextDisabled(u8"A: 未选择");
        }
        if (lineDistPickB_ >= 0 && lineDistPickB_ < static_cast<int>(measuredLines_.size())) {
            ImGui::Text(u8"B: 线段%d", measuredLines_[static_cast<std::size_t>(lineDistPickB_)].id);
        } else {
            ImGui::TextDisabled(u8"B: 未选择");
        }
        if (ImGui::Button(u8"计算平均间隙", ImVec2(-1.f, 36.f))) {
            ComputeSelectedLineDistance();
        }
        if (lineDistValid_) {
            ImGui::TextColored(ImVec4(0.55f, 0.95f, 0.75f, 1.f), u8"平均间隙 = %.4f px",
                               lineDistPx_);
            ImGui::TextDisabled(u8"有效 %zu 点  最小 %.4f  最大 %.4f", lineDistSamples_.size(),
                                lineDistMinPx_, lineDistMaxPx_);
        }
        if (ImGui::Button(u8"清除选线与结果", ImVec2(-1.f, 0))) {
            ClearLineDistance();
            SetStatus(u8"已清除线线距离选线与结果");
        }
        ImGui::Checkbox(u8"显示测量叠加", &showLineMeasureOverlay_);
        return;
    }

    if (image2DTool_ == Image2DTool::ArcDistance) {
        ImGui::TextWrapped(
            u8"在图像上直接点击圆弧选 A、B（或在下方的列表中选），再计算平均间隙。\n"
            u8"可选圆弧为当前已提取并显示在图像上的圆弧。");
        ImGui::Spacing();
        ImGui::TextDisabled(u8"算法说明");
        ImGui::TextWrapped(
            u8"选为 A 的圆弧上均匀采样，过每点作法向（径向）直线，与圆弧 B 求交；"
            u8"无交点不计入，有效距离取平均。");
        ImGui::SetNextItemWidth(-1.f);
        ImGui::DragInt(u8"采样点数", &arcDistSampleCount_, 1, 4, 200);
        ImGui::TextDisabled(u8"A 圆弧上的均匀采样数量；越多统计越稳定，但计算稍慢");
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::TextDisabled(u8"已提取圆弧 (%zu)", measuredArcs_.size());
        if (measuredArcs_.empty()) {
            ImGui::TextColored(ImVec4(1.f, 0.65f, 0.45f, 1.f),
                               u8"当前无已显示圆弧，请先用卡尺提弧提取圆弧");
        } else {
            if (measuredArcs_.size() < 2) {
                ImGui::TextColored(ImVec4(1.f, 0.65f, 0.45f, 1.f),
                                   u8"至少需要 2 条圆弧才能测量距离");
            }
            for (std::size_t i = 0; i < measuredArcs_.size(); ++i) {
                const MeasuredImageArc& arc = measuredArcs_[i];
                const char* imgName = arc.imageSource == 1 ? u8"亮度" : u8"深度";
                ImGui::PushID(static_cast<int>(i) + 30000);
                ImGui::Text(u8"圆弧%d", arc.id);
                ImGui::SameLine();
                ImGui::TextDisabled(u8"(%s)", imgName);
                const bool pickedA = static_cast<int>(i) == arcDistPickA_;
                const bool pickedB = static_cast<int>(i) == arcDistPickB_;
                if (pickedA) ImGui::TextColored(ImVec4(1.f, 0.9f, 0.4f, 1.f), u8"  [A]");
                if (pickedB) ImGui::TextColored(ImVec4(0.4f, 0.85f, 1.f, 1.f), u8"  [B]");
                if (ImGui::Button(u8"选为 A")) {
                    arcDistPickA_ = static_cast<int>(i);
                    arcDistValid_ = false;
                    arcDistSamples_.clear();
                }
                ImGui::SameLine();
                if (ImGui::Button(u8"选为 B")) {
                    arcDistPickB_ = static_cast<int>(i);
                    arcDistValid_ = false;
                    arcDistSamples_.clear();
                }
                ImGui::Separator();
                ImGui::PopID();
            }
        }
        if (arcDistPickA_ >= 0 && arcDistPickA_ < static_cast<int>(measuredArcs_.size())) {
            ImGui::Text(u8"A: 圆弧%d", measuredArcs_[static_cast<std::size_t>(arcDistPickA_)].id);
        } else {
            ImGui::TextDisabled(u8"A: 未选择");
        }
        if (arcDistPickB_ >= 0 && arcDistPickB_ < static_cast<int>(measuredArcs_.size())) {
            ImGui::Text(u8"B: 圆弧%d", measuredArcs_[static_cast<std::size_t>(arcDistPickB_)].id);
        } else {
            ImGui::TextDisabled(u8"B: 未选择");
        }
        if (ImGui::Button(u8"计算平均间隙", ImVec2(-1.f, 36.f))) {
            ComputeSelectedArcDistance();
        }
        if (arcDistValid_) {
            ImGui::TextColored(ImVec4(0.55f, 0.95f, 0.75f, 1.f), u8"平均间隙 = %.4f px",
                               arcDistPx_);
            ImGui::TextDisabled(u8"有效 %zu 点  最小 %.4f  最大 %.4f", arcDistSamples_.size(),
                                arcDistMinPx_, arcDistMaxPx_);
        }
        if (ImGui::Button(u8"清除选弧与结果", ImVec2(-1.f, 0))) {
            ClearArcDistance();
            SetStatus(u8"已清除圆弧距离选弧与结果");
        }
        ImGui::Checkbox(u8"显示测量叠加", &showLineMeasureOverlay_);
        return;
    }

    if (image2DTool_ == Image2DTool::PointDistance) {
        ImGui::TextWrapped(u8"在图像上依次点击 A、B 两点，自动记录距离。");
        ImGui::TextColored(ImVec4(0.55f, 0.95f, 0.75f, 1.f), "%s",
                           pointDistPhase_ == PointPickPhase::PickA ? u8"① 点击设置 A 点"
                                                                    : u8"② 点击设置 B 点");
        ImGui::Spacing();
        ImGui::TextDisabled(u8"已测距离 (%zu)", measuredPointDists_.size());
        for (std::size_t i = 0; i < measuredPointDists_.size(); ++i) {
            const MeasuredPointDist& pd = measuredPointDists_[i];
            ImGui::PushID(static_cast<int>(i) + 50000);
            ImGui::Text(u8"测距%d", pd.id);
            ImGui::TextColored(ImVec4(0.55f, 0.95f, 0.75f, 1.f), u8"距离 = %.4f px", pd.distance);
            ImGui::TextDisabled(u8"ΔX %.3f  ΔY %.3f", pd.dx, pd.dy);
            if (ImGui::Button(u8"删除")) {
                measuredPointDists_.erase(measuredPointDists_.begin() +
                                          static_cast<std::ptrdiff_t>(i));
                ImGui::PopID();
                break;
            }
            ImGui::Separator();
            ImGui::PopID();
        }
        if (ImGui::Button(u8"清除全部测距", ImVec2(-1, 0))) {
            measuredPointDists_.clear();
            nextPointDistId_ = 1;
            pointDistPhase_ = PointPickPhase::PickA;
            SetStatus(u8"已清除全部点点距离");
        }
        ImGui::Checkbox(u8"显示测量叠加", &showLineMeasureOverlay_);
        return;
    }

    if (image2DTool_ == Image2DTool::LineAngle) {
        ImGui::TextWrapped(u8"在图像上点击两条线段（或下方列表选），计算夹角。");
        ImGui::Spacing();
        ImGui::TextDisabled(u8"已提取线段 (%zu)", measuredLines_.size());
        for (std::size_t i = 0; i < measuredLines_.size(); ++i) {
            const MeasuredImageLine& line = measuredLines_[i];
            ImGui::PushID(static_cast<int>(i) + 51000);
            ImGui::Text(u8"线段%d", line.id);
            if (static_cast<int>(i) == lineAnglePickA_)
                ImGui::TextColored(ImVec4(1.f, 0.9f, 0.4f, 1.f), u8"  [A]");
            if (static_cast<int>(i) == lineAnglePickB_)
                ImGui::TextColored(ImVec4(0.4f, 0.85f, 1.f, 1.f), u8"  [B]");
            if (ImGui::Button(u8"选为 A")) {
                lineAnglePickA_ = static_cast<int>(i);
                lineAngleValid_ = false;
            }
            ImGui::SameLine();
            if (ImGui::Button(u8"选为 B")) {
                lineAnglePickB_ = static_cast<int>(i);
                lineAngleValid_ = false;
            }
            ImGui::Separator();
            ImGui::PopID();
        }
        if (ImGui::Button(u8"计算夹角", ImVec2(-1, 36.f))) ComputeSelectedLineAngle();
        if (lineAngleValid_) {
            ImGui::TextColored(ImVec4(0.55f, 0.95f, 0.75f, 1.f), u8"夹角 = %.3f°", lineAngleDeg_);
        }
        if (ImGui::Button(u8"清除选线", ImVec2(-1, 0))) ClearLineAngle();
        ImGui::Checkbox(u8"显示测量叠加", &showLineMeasureOverlay_);
        return;
    }

    if (image2DTool_ == Image2DTool::CircleGap) {
        ImGui::TextWrapped(u8"在图像上点击两个拟合圆（或下方列表选），计算圆心距与表面间隙。");
        ImGui::Spacing();
        ImGui::TextDisabled(u8"已拟合圆 (%zu)", measuredCircleFits_.size());
        for (std::size_t i = 0; i < measuredCircleFits_.size(); ++i) {
            const MeasuredCircleFit& fit = measuredCircleFits_[i];
            ImGui::PushID(static_cast<int>(i) + 52000);
            ImGui::Text(u8"圆%d  R=%.3f", fit.id, fit.result.radius);
            if (static_cast<int>(i) == circleGapPickA_)
                ImGui::TextColored(ImVec4(1.f, 0.9f, 0.4f, 1.f), u8"  [A]");
            if (static_cast<int>(i) == circleGapPickB_)
                ImGui::TextColored(ImVec4(0.4f, 0.85f, 1.f, 1.f), u8"  [B]");
            if (ImGui::Button(u8"选为 A")) {
                circleGapPickA_ = static_cast<int>(i);
                circleGapValid_ = false;
            }
            ImGui::SameLine();
            if (ImGui::Button(u8"选为 B")) {
                circleGapPickB_ = static_cast<int>(i);
                circleGapValid_ = false;
            }
            ImGui::Separator();
            ImGui::PopID();
        }
        if (ImGui::Button(u8"计算圆间隙", ImVec2(-1, 36.f))) ComputeSelectedCircleGap();
        if (circleGapValid_) {
            ImGui::TextColored(ImVec4(0.55f, 0.95f, 0.75f, 1.f), u8"圆心距 = %.4f px",
                               circleGapCenterDist_);
            ImGui::Text(u8"表面间隙 = %.4f px", circleGapSurfaceGap_);
        }
        if (ImGui::Button(u8"清除选圆", ImVec2(-1, 0))) ClearCircleGap();
        ImGui::Checkbox(u8"显示测量叠加", &showLineMeasureOverlay_);
        return;
    }

    if (image2DTool_ == Image2DTool::PointLineDistance) {
        ImGui::TextWrapped(u8"先在图像上点击一点，再点击一条线段，计算垂直距离。");
        ImGui::TextColored(ImVec4(0.55f, 0.95f, 0.75f, 1.f), "%s",
                           pointLinePhase_ == PointPickPhase::PickA ? u8"① 点击选点"
                                                                    : u8"② 点击选线段");
        if (ImGui::Button(u8"重新选点", ImVec2(-1, 0))) ClearPointLineDistance();
        if (ImGui::Button(u8"计算距离", ImVec2(-1, 36.f))) ComputePointLineDistance();
        if (pointLineValid_) {
            ImGui::TextColored(ImVec4(0.55f, 0.95f, 0.75f, 1.f), u8"垂直距离 = %.4f px",
                               pointLineDistPx_);
        }
        ImGui::Checkbox(u8"显示测量叠加", &showLineMeasureOverlay_);
        return;
    }

    if (image2DTool_ == Image2DTool::CaliperPoint) {
        ImGui::TextWrapped(u8"在图像上拖拽一条短测量线（搜索方向），松开后提取单个边缘点。");
        DrawCaliperLineParamsPanel(lineMeasureParams_, false);
        if (caliperPointPending_ && caliperPointPendingEdge_.valid) {
            ImGui::TextColored(ImVec4(1.f, 0.85f, 0.45f, 1.f), u8"待确认边缘点");
            ImGui::Text(u8"位置 (%.2f, %.2f)", caliperPointPendingEdge_.x,
                        caliperPointPendingEdge_.y);
            if (ImGui::Button(u8"确认边缘点", ImVec2(-1, 36.f))) ConfirmCaliperPoint();
            if (ImGui::Button(u8"取消预览", ImVec2(-1, 0))) CancelCaliperPointPending();
        }
        ImGui::TextDisabled(u8"已确认边缘点 (%zu)", measuredCaliperPoints_.size());
        for (std::size_t i = 0; i < measuredCaliperPoints_.size(); ++i) {
            const MeasuredCaliperPoint& cp = measuredCaliperPoints_[i];
            ImGui::PushID(static_cast<int>(i) + 53000);
            ImGui::Text(u8"E%d  (%.2f, %.2f)", cp.id, cp.x, cp.y);
            if (ImGui::Button(u8"删除")) {
                measuredCaliperPoints_.erase(measuredCaliperPoints_.begin() +
                                             static_cast<std::ptrdiff_t>(i));
                ImGui::PopID();
                break;
            }
            ImGui::PopID();
        }
        ImGui::Checkbox(u8"显示测量叠加", &showLineMeasureOverlay_);
        return;
    }

    if (image2DTool_ == Image2DTool::CaliperCircle) {
        ImGui::TextWrapped(u8"① 点击设置圆心  ② 拖拽设置半径  ③ 松开后圆卡尺预览");
        ImGui::TextColored(ImVec4(0.55f, 0.95f, 0.75f, 1.f), "%s",
                           circleCaliperPhase_ == CircleCaliperPhase::PickCenter
                               ? u8"① 点击设置圆心"
                               : u8"② 拖拽设置半径");
        if (ImGui::Button(u8"重新选圆心", ImVec2(-1, 0))) {
            circleCaliperPhase_ = CircleCaliperPhase::PickCenter;
            CancelCircleCaliperPending();
        }
        DrawCaliperLineParamsPanel(lineMeasureParams_, true);
        if (circleCaliperPending_ && circleCaliperPendingResult_.ok) {
            ImGui::TextColored(ImVec4(1.f, 0.85f, 0.45f, 1.f), u8"待确认圆卡尺");
            ImGui::Text(u8"半径 %.3f px  RMS %.3f", circleCaliperPendingResult_.fitRadius,
                        circleCaliperPendingResult_.fitRms);
            if (ImGui::Button(u8"确认圆卡尺", ImVec2(-1, 36.f))) ConfirmCircleCaliper();
            if (ImGui::Button(u8"取消预览", ImVec2(-1, 0))) CancelCircleCaliperPending();
        }
        ImGui::TextDisabled(u8"已确认圆卡尺 (%zu)", measuredCircleCalipers_.size());
        for (std::size_t i = 0; i < measuredCircleCalipers_.size(); ++i) {
            const MeasuredCircleCaliper& cc = measuredCircleCalipers_[i];
            ImGui::PushID(static_cast<int>(i) + 54000);
            ImGui::Text(u8"圆卡%d  R=%.3f", cc.id, cc.result.fitRadius);
            if (ImGui::Button(u8"删除")) {
                measuredCircleCalipers_.erase(measuredCircleCalipers_.begin() +
                                              static_cast<std::ptrdiff_t>(i));
                ImGui::PopID();
                break;
            }
            ImGui::PopID();
        }
        ImGui::Checkbox(u8"显示测量叠加", &showLineMeasureOverlay_);
        return;
    }

    if (image2DTool_ == Image2DTool::ArcLength) {
        ImGui::TextWrapped(u8"在图像上点击一条已提取圆弧，计算弧长、弦长与弓高。");
        ImGui::Spacing();
        ImGui::TextDisabled(u8"已提取圆弧 (%zu)", measuredArcs_.size());
        for (std::size_t i = 0; i < measuredArcs_.size(); ++i) {
            const MeasuredImageArc& arc = measuredArcs_[i];
            ImGui::PushID(static_cast<int>(i) + 55000);
            ImGui::Text(u8"圆弧%d  R=%.3f", arc.id, arc.result.fitRadius);
            if (static_cast<int>(i) == arcLengthPick_)
                ImGui::TextColored(ImVec4(0.8f, 0.6f, 1.f, 1.f), u8"  [已选]");
            if (ImGui::Button(u8"选择")) PickArcForLength(static_cast<int>(i));
            ImGui::Separator();
            ImGui::PopID();
        }
        if (ImGui::Button(u8"计算弧长", ImVec2(-1, 36.f))) ComputeSelectedArcLength();
        if (arcLengthValid_) {
            ImGui::TextColored(ImVec4(0.55f, 0.95f, 0.75f, 1.f), u8"弧长 = %.4f px",
                               arcLengthMetrics_.arcLength);
            ImGui::Text(u8"弦长 = %.4f px", arcLengthMetrics_.chordLength);
            ImGui::Text(u8"弓高 = %.4f px", arcLengthMetrics_.sagitta);
        }
        if (ImGui::Button(u8"清除选择", ImVec2(-1, 0))) ClearArcLength();
        ImGui::Checkbox(u8"显示测量叠加", &showLineMeasureOverlay_);
        return;
    }

    if (image2DTool_ == Image2DTool::EllipseFit) {
        ImGui::TextColored(ImVec4(0.55f, 0.95f, 0.75f, 1.f), "%s",
                           ArcMeasurePhaseHint(arcMeasurePhase_));
        ImGui::TextWrapped(u8"点击已有圆弧，或设置 A/B/拱高提取边缘后拟合椭圆（至少 5 点）。");
        if (ImGui::Button(u8"重新选点", ImVec2(-1, 0))) {
            ResetArcMeasurePick();
            CancelEllipseFitPending();
        }
        DrawCaliperLineParamsPanel(lineMeasureParams_, true);
        if (ellipseFitPending_ && ellipseFitPendingResult_.ok) {
            ImGui::Text(u8"长轴 %.2f  短轴 %.2f px", ellipseFitPendingResult_.axisA * 2.f,
                        ellipseFitPendingResult_.axisB * 2.f);
            if (ImGui::Button(u8"确认椭圆", ImVec2(-1, 36.f))) ConfirmEllipseFit();
            if (ImGui::Button(u8"取消预览", ImVec2(-1, 0))) CancelEllipseFitPending();
        }
        ImGui::Checkbox(u8"显示测量叠加", &showLineMeasureOverlay_);
        return;
    }

    if (image2DTool_ == Image2DTool::ThreePointCircle) {
        ImGui::TextWrapped(u8"依次点击三个点定圆。");
        ImGui::TextDisabled(u8"已测 (%zu)", measuredThreePointCircles_.size());
        if (ImGui::Button(u8"清除", ImVec2(-1, 0))) {
            measuredThreePointCircles_.clear();
            ClearThreePointCircle();
        }
        ImGui::Checkbox(u8"显示测量叠加", &showLineMeasureOverlay_);
        return;
    }

    if (image2DTool_ == Image2DTool::ParallelLineDistance) {
        ImGui::TextWrapped(u8"选两条线段，计算平行线间距。");
        if (ImGui::Button(u8"计算间距", ImVec2(-1, 36.f))) ComputeParallelLineDistance();
        if (parallelDistValid_)
            ImGui::TextColored(ImVec4(0.55f, 0.95f, 0.75f, 1.f), u8"间距 = %.4f px", parallelDistPx_);
        if (ImGui::Button(u8"清除", ImVec2(-1, 0))) ClearParallelLineDistance();
        ImGui::Checkbox(u8"显示测量叠加", &showLineMeasureOverlay_);
        return;
    }

    if (image2DTool_ == Image2DTool::RectCaliper) {
        ImGui::TextWrapped(u8"拖拽矩形 ROI，四边卡尺提边并拟合最小外接矩形。");
        DrawCaliperLineParamsPanel(lineMeasureParams_, false);
        if (rectCaliperPending_ && rectCaliperPendingResult_.ok) {
            ImGui::Text(u8"%.1f×%.1f px  角度 %.1f°", rectCaliperPendingResult_.width,
                        rectCaliperPendingResult_.height, rectCaliperPendingResult_.angleDeg);
            if (ImGui::Button(u8"确认矩形", ImVec2(-1, 36.f))) ConfirmRectCaliper();
            if (ImGui::Button(u8"取消预览", ImVec2(-1, 0))) CancelRectCaliperPending();
        }
        ImGui::Checkbox(u8"显示测量叠加", &showLineMeasureOverlay_);
        return;
    }

    if (image2DTool_ == Image2DTool::ProfileWidth) {
        ImGui::TextWrapped(u8"拖拽测量线（过槽中心），自动检测两侧边缘宽度。");
        DrawCaliperLineParamsPanel(lineMeasureParams_, false);
        if (profileWidthPending_ && profileWidthPendingResult_.ok) {
            ImGui::Text(u8"宽度 = %.4f px", profileWidthPendingResult_.width);
            if (ImGui::Button(u8"确认宽度", ImVec2(-1, 36.f))) ConfirmProfileWidth();
            if (ImGui::Button(u8"取消预览", ImVec2(-1, 0))) CancelProfileWidthPending();
        }
        ImGui::Checkbox(u8"显示测量叠加", &showLineMeasureOverlay_);
        return;
    }

    if (image2DTool_ == Image2DTool::PointProjection) {
        ImGui::TextWrapped(u8"先点击一点，再点击线段，显示垂足与投影参数。");
        if (ImGui::Button(u8"计算投影", ImVec2(-1, 36.f))) ComputePointProjection();
        if (pointProjValid_) {
            ImGui::Text(u8"垂足 (%.2f, %.2f)", pointProjResult_.footX, pointProjResult_.footY);
            ImGui::Text(u8"垂直距 %.4f  t=%.3f", pointProjResult_.perpDist, pointProjResult_.alongT);
        }
        if (ImGui::Button(u8"清除", ImVec2(-1, 0))) ClearPointProjection();
        ImGui::Checkbox(u8"显示测量叠加", &showLineMeasureOverlay_);
        return;
    }

    if (image2DTool_ == Image2DTool::Concentricity) {
        ImGui::TextWrapped(u8"选两个拟合圆，计算圆心偏移量。");
        if (ImGui::Button(u8"计算同心度", ImVec2(-1, 36.f))) ComputeConcentricity();
        if (concentricityValid_) {
            ImGui::Text(u8"ΔX=%.3f  ΔY=%.3f", concentricityResult_.offsetX,
                        concentricityResult_.offsetY);
            ImGui::Text(u8"偏移距离 %.4f px", concentricityResult_.offsetDist);
        }
        if (ImGui::Button(u8"清除", ImVec2(-1, 0))) ClearConcentricity();
        ImGui::Checkbox(u8"显示测量叠加", &showLineMeasureOverlay_);
        return;
    }

    if (image2DTool_ == Image2DTool::Roundness) {
        ImGui::TextWrapped(u8"点击拟合圆或圆卡尺，计算圆度（RMS/最大最小偏差）。");
        if (ImGui::Button(u8"计算圆度", ImVec2(-1, 36.f))) ComputeRoundness();
        if (roundnessValid_) {
            ImGui::Text(u8"RMS=%.4f  最大=%.4f  最小=%.4f", roundnessResult_.rms,
                        roundnessResult_.maxDev, roundnessResult_.minDev);
        }
        if (ImGui::Button(u8"清除", ImVec2(-1, 0))) ClearRoundness();
        ImGui::Checkbox(u8"显示测量叠加", &showLineMeasureOverlay_);
        return;
    }

    if (image2DTool_ == Image2DTool::RegionBlob) {
        if (CanSyncRegionBlob()) {
            ImGui::TextColored(ImVec4(0.45f, 0.85f, 1.f, 1.f),
                               u8"联动模式：亮度阈值筛选，深度图同步高亮（蓝）");
            ImGui::TextDisabled(u8"在亮度图或深度图上拖拽矩形均可；阈值始终按亮度计算");
        } else {
            ImGui::TextWrapped(u8"拖拽矩形区域，按阈值统计面积与质心；满足阈值的像素以红色高亮。");
            ImGui::TextDisabled(u8"启用深度/亮度联动后，可同步显示到深度图");
        }
        regionBlobThreshold_ = std::clamp(regionBlobThreshold_, 0.f, 255.f);
        ImGui::SetNextItemWidth(-1.f);
        if (ImGui::SliderFloat(u8"亮度阈值", &regionBlobThreshold_, 0.f, 255.f, "%.0f")) {
            RefreshRegionBlobPreview();
        }
        if (ImGui::Checkbox(u8"大于阈值", &regionBlobGreaterThan_)) {
            RefreshRegionBlobPreview();
        }
        if (regionBlobPending_ && regionBlobPendingResult_.ok) {
            if (regionBlobPendingResult_.pixelCount > 0) {
                ImGui::Text(u8"亮度面积 %d px²  质心 (%.1f, %.1f)",
                            regionBlobPendingResult_.pixelCount, regionBlobPendingResult_.centroidX,
                            regionBlobPendingResult_.centroidY);
                if (regionBlobPendingDepthStats_.valid) {
                    ImGui::Text(u8"对应深度  min=%.4f  max=%.4f  mean=%.4f（%d 点）",
                                regionBlobPendingDepthStats_.min, regionBlobPendingDepthStats_.max,
                                regionBlobPendingDepthStats_.mean,
                                regionBlobPendingDepthStats_.count);
                }
                if (ImGui::Button(u8"确认区域", ImVec2(-1, 36.f))) ConfirmRegionBlob();
            } else {
                ImGui::TextColored(ImVec4(1.f, 0.72f, 0.35f, 1.f), u8"当前阈值下无满足条件的像素");
            }
            if (ImGui::Button(u8"取消预览", ImVec2(-1, 0))) CancelRegionBlobPending();
        }
        ImGui::Checkbox(u8"显示测量叠加", &showLineMeasureOverlay_);
        return;
    }

    if (image2DTool_ == Image2DTool::DepthHeightDiff) {
        ImGui::TextWrapped(u8"仅在深度图上：依次点击 A、B 两点，读取高度差 ΔZ。");
        if (depthHeightValid_)
            ImGui::TextColored(ImVec4(0.55f, 0.95f, 0.75f, 1.f), u8"ΔZ = %.6f", depthHeightDelta_);
        if (ImGui::Button(u8"清除", ImVec2(-1, 0))) ClearDepthHeightDiff();
        ImGui::Checkbox(u8"显示测量叠加", &showLineMeasureOverlay_);
        return;
    }

    if (image2DTool_ == Image2DTool::DepthProfile) {
        ImGui::TextWrapped(u8"在深度图上查看 Z-Profile 曲线，支持线扫扫描行剖面。");
        if (!depthImage_.valid()) {
            ImGui::TextColored(ImVec4(1.f, 0.55f, 0.45f, 1.f), u8"请先打开深度图");
        }
        int mode = static_cast<int>(depthProfileMode_);
        ImGui::TextDisabled(u8"剖面类型");
        if (ImGui::RadioButton(u8"扫描行 (X-Z)", mode == 0)) depthProfileMode_ = DepthProfileMode::ScanRow;
        ImGui::SameLine();
        if (ImGui::RadioButton(u8"扫描列 (Y-Z)", mode == 1)) {
            depthProfileMode_ = DepthProfileMode::ScanColumn;
        }
        if (ImGui::RadioButton(u8"任意线段", mode == 2)) depthProfileMode_ = DepthProfileMode::FreeLine;
        if (depthProfileMode_ == DepthProfileMode::FreeLine) {
            ImGui::SetNextItemWidth(-1.f);
            ImGui::DragInt(u8"采样点数", &depthProfileSampleCount_, 1, 8, 4096);
            ImGui::TextDisabled(u8"在深度图上拖拽绘制剖面线");
        } else if (depthProfileMode_ == DepthProfileMode::ScanRow) {
            ImGui::TextDisabled(u8"在深度图上单击选择扫描行（Y）");
        } else {
            ImGui::TextDisabled(u8"在深度图上单击选择扫描列（X）");
        }
        ImGui::Checkbox(u8"显示曲线面板", &showDepthProfilePanel_);
        if (depthProfileValid_) {
            ImGui::TextColored(ImVec4(0.55f, 0.9f, 0.75f, 1.f), u8"已生成：%s", depthProfileLabel_.c_str());
            if (!showDepthProfilePanel_ && ImGui::Button(u8"打开曲线面板", ImVec2(-1, 0))) {
                showDepthProfilePanel_ = true;
            }
        }
        if (ImGui::Button(u8"清除剖面", ImVec2(-1, 0))) ClearDepthProfile();
        ImGui::Checkbox(u8"显示测量叠加", &showLineMeasureOverlay_);
        return;
    }

    if (image2DTool_ == Image2DTool::TemplateMatch) {
        ImGui::TextWrapped(
            u8"① 在亮度图上拖拽框选模板\n"
            u8"② 全图搜索或框选搜索区域\n"
            u8"③ 设置旋转/缩放搜索范围后点「重新匹配」");
        if (!brightnessImage_.valid()) {
            ImGui::TextColored(ImVec4(1.f, 0.55f, 0.45f, 1.f), u8"请先打开亮度图");
        }
        ImGui::SetNextItemWidth(-1.f);
        if (ImGui::DragFloat(u8"最小得分", &templateMatchParams_.minScore, 0.01f, 0.f, 1.f,
                             "%.3f")) {
            if (templateMatchHasTemplate_ && brightnessImage_.valid()) {
                PreviewTemplateMatch(brightnessImage_);
            }
        }
        ImGui::SetNextItemWidth(-1.f);
        if (ImGui::DragInt(u8"最多匹配数", &templateMatchParams_.maxMatches, 1, 1, 16)) {
            if (templateMatchHasTemplate_ && brightnessImage_.valid()) {
                PreviewTemplateMatch(brightnessImage_);
            }
        }
        ImGui::Separator();
        ImGui::TextDisabled(u8"旋转搜索（度）");
        ImGui::SetNextItemWidth(-1.f);
        ImGui::DragFloat(u8"角度最小##tm", &templateMatchParams_.angleMinDeg, 0.5f, -180.f, 180.f,
                          "%.1f");
        ImGui::SetNextItemWidth(-1.f);
        ImGui::DragFloat(u8"角度最大##tm", &templateMatchParams_.angleMaxDeg, 0.5f, -180.f, 180.f,
                          "%.1f");
        ImGui::SetNextItemWidth(-1.f);
        ImGui::DragFloat(u8"角度步长##tm", &templateMatchParams_.angleStepDeg, 0.5f, 0.5f, 45.f,
                          "%.1f");
        ImGui::TextDisabled(u8"缩放搜索（相对模板）");
        ImGui::SetNextItemWidth(-1.f);
        ImGui::DragFloat(u8"缩放最小##tm", &templateMatchParams_.scaleMin, 0.01f, 0.1f, 3.f, "%.2f");
        ImGui::SetNextItemWidth(-1.f);
        ImGui::DragFloat(u8"缩放最大##tm", &templateMatchParams_.scaleMax, 0.01f, 0.1f, 3.f, "%.2f");
        ImGui::SetNextItemWidth(-1.f);
        ImGui::DragFloat(u8"缩放步长##tm", &templateMatchParams_.scaleStep, 0.01f, 0.01f, 0.5f,
                          "%.2f");
        ImGui::TextDisabled(u8"范围越大搜索越慢，调好后点「重新匹配」");
        ImGui::Separator();
        ImGui::TextDisabled(u8"加速与精度");
        if (ImGui::Checkbox(u8"图像金字塔粗搜", &templateMatchParams_.usePyramid)) {
            if (templateMatchHasTemplate_ && brightnessImage_.valid()) {
                PreviewTemplateMatch(brightnessImage_);
            }
        }
        ImGui::SameLine();
        ImGui::TextDisabled(u8"(?)");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip(u8"顶层低分辨率粗搜，逐层细化位置/角度/缩放，大幅提速");
        }
        if (ImGui::Checkbox(u8"亚像素精修", &templateMatchParams_.subPixelRefine)) {
            if (templateMatchHasTemplate_ && brightnessImage_.valid()) {
                PreviewTemplateMatch(brightnessImage_);
            }
        }
        ImGui::SameLine();
        ImGui::TextDisabled(u8"(?)");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip(u8"对得分图峰值做抛物线插值，提高定位精度（全分辨率层）");
        }
        ImGui::SetNextItemWidth(-1.f);
        if (ImGui::DragInt(u8"金字塔层数（0=自动）", &templateMatchParams_.pyramidLevels, 1, 0, 5)) {
            templateMatchParams_.pyramidLevels = std::clamp(templateMatchParams_.pyramidLevels, 0, 5);
        }
        ImGui::Separator();
        if (ImGui::Checkbox(u8"全图搜索", &templateMatchSearchFull_)) {
            if (templateMatchHasTemplate_ && brightnessImage_.valid()) {
                if (templateMatchSearchFull_) {
                    PreviewTemplateMatch(brightnessImage_);
                } else {
                    CancelTemplateMatchPending();
                    SetStatus(u8"请拖拽搜索区域");
                }
            }
        }
        if (templateMatchHasTemplate_ && brightnessImage_.valid()) {
            if (ImGui::Button(u8"重新匹配", ImVec2(-1, 32.f))) {
                PreviewTemplateMatch(brightnessImage_);
            }
        }
        if (ImGui::Button(u8"重新框模板", ImVec2(-1, 0))) {
            ResetTemplateMatchPick();
            SetStatus(u8"请重新框选模板区域");
        }
        if (templateMatchPending_ && templateMatchPendingResult_.ok) {
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::TextColored(ImVec4(1.f, 0.85f, 0.45f, 1.f), u8"待确认匹配");
            for (std::size_t i = 0; i < templateMatchPendingResult_.hits.size(); ++i) {
                const OpenCv2D::TemplateMatchHit& h = templateMatchPendingResult_.hits[i];
                ImGui::Text(u8"#%zu  中心(%.2f, %.2f)  得分 %.3f  ∠%.1f°  ×%.2f", i + 1,
                            h.centerX, h.centerY, h.score, h.angleDeg, h.scale);
            }
            if (ImGui::Button(u8"确认匹配", ImVec2(-1, 36.f))) ConfirmTemplateMatch();
            if (ImGui::Button(u8"取消预览", ImVec2(-1, 0))) CancelTemplateMatchPending();
            if (!templateMatchPendingResult_.hits.empty()) {
                const OpenCv2D::TemplateMatchHit& h = templateMatchPendingResult_.hits.front();
                if (ImGui::Button(u8"复制最佳结果", ImVec2(-1, 0))) {
                    char clip[160];
                    std::snprintf(clip, sizeof(clip), "%.3f,%.3f,%.3f,%.1f,%.3f", h.centerX,
                                  h.centerY, h.score, h.angleDeg, h.scale);
                    ImGui::SetClipboardText(clip);
                    SetStatus(u8"已复制：centerX,centerY,score,angle,scale");
                }
            }
        }
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::TextDisabled(u8"已确认匹配 (%zu)", measuredTemplateMatches_.size());
        for (const MeasuredTemplateMatch& tm : measuredTemplateMatches_) {
            if (tm.result.hits.empty()) continue;
            const OpenCv2D::TemplateMatchHit& h = tm.result.hits.front();
            ImGui::Text(u8"匹配%d  (%.2f, %.2f)  %.3f  ∠%.0f° ×%.2f", tm.id, h.centerX, h.centerY,
                        h.score, h.angleDeg, h.scale);
        }
        if (!CanUndoMeasuredLine()) ImGui::BeginDisabled();
        if (ImGui::Button(u8"撤回匹配 / 预览", ImVec2(-1, 0))) UndoLastMeasuredLine();
        if (!CanUndoMeasuredLine()) ImGui::EndDisabled();
        ImGui::Checkbox(u8"显示测量叠加", &showLineMeasureOverlay_);
        return;
    }

    ImGui::TextWrapped(
        u8"① 在右侧图像上拖拽绘制测量方向线（ROI）\n"
        u8"② 松开后预览卡尺结果，点击「确认」添加线段");
    ImGui::Spacing();
    DrawCaliperLineParamsPanel(lineMeasureParams_, false);
    ImGui::Checkbox(u8"显示测量叠加", &showLineMeasureOverlay_);

    if (lineMeasurePending_ && lineMeasurePendingResult_.ok) {
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::TextColored(ImVec4(1.f, 0.85f, 0.45f, 1.f), u8"待确认卡尺");
        ImGui::Text(u8"有效点 %d  RMS %.3f px", lineMeasurePendingResult_.validCount,
                    lineMeasurePendingResult_.fitRms);
        if (ImGui::Button(u8"确认添加线段", ImVec2(-1.f, 36.f))) {
            ConfirmLineMeasure();
        }
        if (ImGui::Button(u8"取消预览", ImVec2(-1.f, 0))) {
            lineMeasurePending_ = false;
            lineMeasurePendingSource_ = -1;
            lineMeasurePendingResult_ = {};
            SetStatus(u8"已取消卡尺预览");
        }
    }

    ImGui::Spacing();
    ImGui::Separator();
    if (!CanUndoMeasuredLine()) ImGui::BeginDisabled();
    if (ImGui::Button(u8"撤回线段 / 预览", ImVec2(-1.f, 0))) {
        UndoLastMeasuredLine();
    }
    if (!CanUndoMeasuredLine()) ImGui::EndDisabled();

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::TextDisabled(u8"已提取线段 (%zu)", measuredLines_.size());
    if (measuredLines_.empty()) {
        ImGui::TextDisabled(u8"暂无线段，请在图像上拖拽提取");
    } else {
        for (std::size_t i = 0; i < measuredLines_.size(); ++i) {
            const MeasuredImageLine& line = measuredLines_[i];
            const char* imgName = line.imageSource == 1 ? u8"亮度" : u8"深度";
            ImGui::PushID(static_cast<int>(i));
            ImGui::Text(u8"线段%d", line.id);
            ImGui::SameLine();
            ImGui::TextDisabled(u8"(%s)", imgName);
            ImGui::Text(u8"有效点 %d  RMS %.3f px", line.result.validCount, line.result.fitRms);
            if (ImGui::Button(u8"删除")) {
                measuredLines_.erase(measuredLines_.begin() + static_cast<std::ptrdiff_t>(i));
                if (lineDistPickA_ == static_cast<int>(i))
                    lineDistPickA_ = -1;
                else if (lineDistPickA_ > static_cast<int>(i))
                    --lineDistPickA_;
                if (lineDistPickB_ == static_cast<int>(i))
                    lineDistPickB_ = -1;
                else if (lineDistPickB_ > static_cast<int>(i))
                    --lineDistPickB_;
                lineDistValid_ = false;
                ImGui::PopID();
                break;
            }
            ImGui::Separator();
            ImGui::PopID();
        }
    }

    if (ImGui::Button(u8"清除全部线段", ImVec2(-1.f, 0))) {
        ClearAllMeasuredLines();
        SetStatus(u8"已清除全部线段");
    }
}

void Application::Draw2DOperatorMenuItems() {
    auto toggleTool = [&](const char* label, Image2DTool tool, const char* onMsg, const char* offMsg) {
        const bool on = image2DTool_ == tool;
        if (ImGui::MenuItem(label, nullptr, on)) {
            image2DTool_ = on ? Image2DTool::None : tool;
            EnterView2DMode();
            SetStatus(on ? offMsg : onMsg);
        }
    };

    if (ImGui::BeginMenu(u8"卡尺提取")) {
        const bool caliperLineOn = image2DTool_ == Image2DTool::CaliperLine;
        if (ImGui::MenuItem(u8"提取线段", nullptr, caliperLineOn)) {
            if (caliperLineOn) {
                image2DTool_ = Image2DTool::None;
                ClearLineMeasure();
                SetStatus(u8"已退出卡尺提线");
            } else {
                image2DTool_ = Image2DTool::CaliperLine;
                ClearArcMeasure();
                arcMeasurePending_ = false;
                arcMeasurePendingSource_ = -1;
                arcMeasurePendingResult_ = {};
                EnterView2DMode();
                SetStatus(u8"卡尺提线：在图像上拖拽绘制测量方向线");
            }
        }
        const bool caliperArcOn = image2DTool_ == Image2DTool::CaliperArc;
        if (ImGui::MenuItem(u8"提取圆弧", nullptr, caliperArcOn)) {
            if (caliperArcOn) {
                image2DTool_ = Image2DTool::None;
                ClearArcMeasure();
                SetStatus(u8"已退出卡尺提弧");
            } else {
                image2DTool_ = Image2DTool::CaliperArc;
                ClearLineMeasure();
                ClearArcMeasure();
                lineMeasurePending_ = false;
                lineMeasurePendingSource_ = -1;
                lineMeasurePendingResult_ = {};
                EnterView2DMode();
                SetStatus(u8"卡尺提弧：先点击 A、B 点，再拖拽拱高线段");
            }
        }
        const bool caliperPointOn = image2DTool_ == Image2DTool::CaliperPoint;
        if (ImGui::MenuItem(u8"单点卡尺", nullptr, caliperPointOn)) {
            image2DTool_ = caliperPointOn ? Image2DTool::None : Image2DTool::CaliperPoint;
            CancelCaliperPointPending();
            EnterView2DMode();
            SetStatus(caliperPointOn ? u8"已退出单点卡尺" : u8"单点卡尺：拖拽测量方向线提取边缘");
        }
        const bool caliperCircleOn = image2DTool_ == Image2DTool::CaliperCircle;
        if (ImGui::MenuItem(u8"圆卡尺（整圆）", nullptr, caliperCircleOn)) {
            image2DTool_ = caliperCircleOn ? Image2DTool::None : Image2DTool::CaliperCircle;
            circleCaliperPhase_ = CircleCaliperPhase::PickCenter;
            CancelCircleCaliperPending();
            EnterView2DMode();
            SetStatus(caliperCircleOn ? u8"已退出圆卡尺" : u8"圆卡尺：点击圆心后拖拽半径");
        }
        toggleTool(u8"矩形卡尺", Image2DTool::RectCaliper, u8"矩形卡尺：拖拽矩形 ROI",
                   u8"已退出矩形卡尺");
        toggleTool(u8"剖面测宽", Image2DTool::ProfileWidth, u8"剖面测宽：拖拽测量线",
                   u8"已退出剖面测宽");
        ImGui::EndMenu();
    }

    if (ImGui::BeginMenu(u8"几何拟合")) {
        const bool circleFitOn = image2DTool_ == Image2DTool::CircleFit;
        if (ImGui::MenuItem(u8"拟合圆", nullptr, circleFitOn)) {
            if (circleFitOn) {
                image2DTool_ = Image2DTool::None;
                SetStatus(u8"已退出拟合圆");
            } else {
                image2DTool_ = Image2DTool::CircleFit;
                ClearArcMeasure();
                CancelCircleFitPending();
                EnterView2DMode();
                SetStatus(u8"拟合圆：点击已有圆弧，或设置 A/B 后拖拽拱高提取边缘");
            }
        }
        toggleTool(u8"拟合椭圆", Image2DTool::EllipseFit, u8"拟合椭圆：点击圆弧或沿弧提取",
                   u8"已退出拟合椭圆");
        toggleTool(u8"三点定圆", Image2DTool::ThreePointCircle, u8"三点定圆：依次点击三个点",
                   u8"已退出三点定圆");
        ImGui::EndMenu();
    }

    if (ImGui::BeginMenu(u8"距离与角度")) {
        const bool pointDistOn = image2DTool_ == Image2DTool::PointDistance;
        if (ImGui::MenuItem(u8"点点距离", nullptr, pointDistOn)) {
            image2DTool_ = pointDistOn ? Image2DTool::None : Image2DTool::PointDistance;
            pointDistPhase_ = PointPickPhase::PickA;
            EnterView2DMode();
            SetStatus(pointDistOn ? u8"已退出点点距离" : u8"点点距离：依次点击 A、B 两点");
        }
        const bool pointLineOn = image2DTool_ == Image2DTool::PointLineDistance;
        if (ImGui::MenuItem(u8"点线距离", nullptr, pointLineOn)) {
            image2DTool_ = pointLineOn ? Image2DTool::None : Image2DTool::PointLineDistance;
            ClearPointLineDistance();
            EnterView2DMode();
            SetStatus(pointLineOn ? u8"已退出点线距离" : u8"点线距离：先点击一点，再点击线段");
        }
        const bool lineDistOn = image2DTool_ == Image2DTool::LineDistance;
        if (ImGui::MenuItem(u8"线线距离（平均间隙）", nullptr, lineDistOn)) {
            if (lineDistOn) {
                image2DTool_ = Image2DTool::None;
                SetStatus(u8"已退出线线距离测量");
            } else {
                image2DTool_ = Image2DTool::LineDistance;
                EnterView2DMode();
                if (measuredLines_.empty()) {
                    SetStatus(u8"线线距离：请先在图像上点击已提取的线段（或左侧列表选线）");
                } else {
                    SetStatus(u8"线线距离：在图像上点击线段选 A、B，再计算平均间隙");
                }
            }
        }
        const bool arcDistOn = image2DTool_ == Image2DTool::ArcDistance;
        if (ImGui::MenuItem(u8"圆弧线线距离（平均间隙）", nullptr, arcDistOn)) {
            if (arcDistOn) {
                image2DTool_ = Image2DTool::None;
                SetStatus(u8"已退出圆弧距离测量");
            } else {
                image2DTool_ = Image2DTool::ArcDistance;
                EnterView2DMode();
                if (measuredArcs_.empty()) {
                    SetStatus(u8"圆弧距离：请先在图像上点击已提取的圆弧（或左侧列表选弧）");
                } else {
                    SetStatus(u8"圆弧距离：在图像上点击圆弧选 A、B，再计算平均间隙");
                }
            }
        }
        toggleTool(u8"平行线距离", Image2DTool::ParallelLineDistance, u8"平行线距离：点击两条线段",
                   u8"已退出平行线距离");
        const bool lineAngleOn = image2DTool_ == Image2DTool::LineAngle;
        if (ImGui::MenuItem(u8"两线夹角", nullptr, lineAngleOn)) {
            image2DTool_ = lineAngleOn ? Image2DTool::None : Image2DTool::LineAngle;
            EnterView2DMode();
            SetStatus(lineAngleOn ? u8"已退出两线夹角" : u8"两线夹角：点击两条已提取线段");
        }
        const bool circleGapOn = image2DTool_ == Image2DTool::CircleGap;
        if (ImGui::MenuItem(u8"圆心距 / 圆间隙", nullptr, circleGapOn)) {
            image2DTool_ = circleGapOn ? Image2DTool::None : Image2DTool::CircleGap;
            EnterView2DMode();
            SetStatus(circleGapOn ? u8"已退出圆间隙" : u8"圆间隙：点击两个已拟合圆");
        }
        const bool arcLengthOn = image2DTool_ == Image2DTool::ArcLength;
        if (ImGui::MenuItem(u8"弧长 / 弦长", nullptr, arcLengthOn)) {
            image2DTool_ = arcLengthOn ? Image2DTool::None : Image2DTool::ArcLength;
            ClearArcLength();
            EnterView2DMode();
            SetStatus(arcLengthOn ? u8"已退出弧长测量" : u8"弧长：点击一条已提取圆弧");
        }
        toggleTool(u8"投影点 / 垂足", Image2DTool::PointProjection, u8"投影点：先点选再选线段",
                   u8"已退出投影点");
        ImGui::EndMenu();
    }

    if (ImGui::BeginMenu(u8"形位与区域")) {
        toggleTool(u8"同心度", Image2DTool::Concentricity, u8"同心度：点击两个拟合圆", u8"已退出同心度");
        toggleTool(u8"圆度", Image2DTool::Roundness, u8"圆度：点击圆后计算", u8"已退出圆度");
        toggleTool(u8"区域面积 / 质心", Image2DTool::RegionBlob, u8"区域分析：拖拽矩形",
                   u8"已退出区域分析");
        ImGui::EndMenu();
    }

    if (ImGui::BeginMenu(u8"深度与匹配")) {
        toggleTool(u8"两点高度差", Image2DTool::DepthHeightDiff, u8"高度差：深度图点击两点",
                   u8"已退出高度差");
        if (ImGui::MenuItem(u8"剖面高度曲线", nullptr, image2DTool_ == Image2DTool::DepthProfile)) {
            if (image2DTool_ == Image2DTool::DepthProfile) {
                image2DTool_ = Image2DTool::None;
                SetStatus(u8"已退出深度剖面");
            } else {
                image2DTool_ = Image2DTool::DepthProfile;
                EnterView2DMode();
                imagePanelTab_ = 0;
                showDepthProfilePanel_ = true;
                SetStatus(u8"深度 Profile：在深度图上单击扫描行，或切换为任意线段后拖拽");
            }
        }
        if (ImGui::MenuItem(u8"模板匹配（亮度图）",
                            nullptr, image2DTool_ == Image2DTool::TemplateMatch)) {
            if (image2DTool_ == Image2DTool::TemplateMatch) {
                image2DTool_ = Image2DTool::None;
                SetStatus(u8"已退出模板匹配");
            } else {
                image2DTool_ = Image2DTool::TemplateMatch;
                ResetTemplateMatchPick();
                EnterView2DMode();
                imagePanelTab_ = 1;
                if (!brightnessImage_.valid()) {
                    SetStatus(u8"请先打开亮度图，再框选模板");
                } else {
                    SetStatus(u8"模板匹配：在亮度图上拖拽框选模板");
                }
            }
        }
        ImGui::EndMenu();
    }

    ImGui::Separator();
    if (ImGui::MenuItem(u8"清除全部线段", nullptr, false, !measuredLines_.empty())) {
        ClearMeasuredLinesOnly();
        SetStatus(u8"已清除全部线段");
    }
    if (ImGui::MenuItem(u8"清除全部圆弧", nullptr, false, !measuredArcs_.empty())) {
        ClearMeasuredArcsOnly();
        SetStatus(u8"已清除全部圆弧及关联圆/椭圆拟合");
    }
    if (ImGui::MenuItem(u8"清除全部 2D 测量", nullptr, false,
                        depthImage_.valid() || brightnessImage_.valid())) {
        ClearAllMeasuredLines();
        SetStatus(u8"已清除全部 2D 测量叠加");
    }
    if (image2DTool_ != Image2DTool::None) {
        ImGui::Separator();
        ImGui::TextDisabled(u8"参数与线段列表见左侧工具面板");
    }
}

void Application::DrawFilterMenuItems() {
    ImGui::TextDisabled(u8"体素下采样");
    ImGui::SetNextItemWidth(160.f);
    ImGui::DragFloat(u8"体素边长##v", &filterVoxelLeaf_, 0.01f, 1e-4f, 1e6f, "%.4f");
    if (ImGui::MenuItem(u8"预览体素滤波")) RunFilterPreview(0, algoBackend_);
    ImGui::Separator();
    ImGui::TextDisabled(u8"半径离群点");
    ImGui::SetNextItemWidth(160.f);
    ImGui::DragFloat(u8"搜索半径##r", &filterRadius_, 0.01f, 1e-4f, 1e6f, "%.4f");
    ImGui::SetNextItemWidth(160.f);
    ImGui::DragInt(u8"最少邻居##r", &filterRadiusMinNeighbors_, 1, 1, 200);
    if (ImGui::MenuItem(u8"预览半径滤波")) RunFilterPreview(1, algoBackend_);
    ImGui::Separator();
    ImGui::TextDisabled(u8"统计离群点");
    ImGui::SetNextItemWidth(160.f);
    ImGui::DragInt(u8"邻域点数 K##s", &filterStatMeanK_, 1, 2, 200);
    ImGui::SetNextItemWidth(160.f);
    ImGui::DragFloat(u8"标准差倍数##s", &filterStatStdMul_, 0.05f, 0.1f, 10.f, "%.2f");
    if (ImGui::MenuItem(u8"预览统计滤波")) RunFilterPreview(2, algoBackend_);
    ImGui::Separator();
    if (filterCompareActive_) {
        DrawFilterCompareViewControls();
        ImGui::Text(u8"保留 %d / 滤除 %d", filterLastKept_, filterLastRemoved_);
        if (ImGui::MenuItem(u8"应用滤波到点云")) ApplyFilterResult();
        if (ImGui::MenuItem(u8"取消预览")) ClearFilterCompare();
    } else {
        ImGui::TextDisabled(u8"先预览，再切换对比/原始/滤波后/差异");
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
    const UiPalette& pal = GetUiPalette();
    if (image2DTool_ != Image2DTool::None || view2DMode_) {
        if (view2DMode_) DrawImageViewControls();
        if (image2DTool_ != Image2DTool::None) {
            UiSectionHeader(Image2DToolLabel(image2DTool_), u8"2D 算子 — 参数与结果", &pal.tool2D);
        } else {
            UiSectionHeader(u8"2D 模式", u8"请从「2D算子」启用卡尺提线等工具", &pal.tool2D);
        }
        ImGui::PushStyleColor(ImGuiCol_ChildBg, pal.panelRaised);
        ImGui::PushStyleColor(ImGuiCol_Border, pal.border);
        const float toolH2d = std::max(80.f, ImGui::GetContentRegionAvail().y);
        ImGui::BeginChild(u8"##toolhelp", ImVec2(0.f, toolH2d), ImGuiChildFlags_Borders,
                          ImGuiWindowFlags_AlwaysVerticalScrollbar);
        if (image2DTool_ != Image2DTool::None) {
            DrawImage2DToolPanel();
        } else {
            if (!depthImage_.gray.empty()) {
                DrawDepthToCloudControls();
            } else {
                ImGui::TextWrapped(u8"当前为 2D 模式，点云视区已隐藏。\n"
                                   u8"在菜单「2D算子」中启用卡尺提线等工具。");
            }
        }
        ImGui::EndChild();
        ImGui::PopStyleColor(2);
        return;
    }

    const bool icpMode = measure_.mode == ToolMode::Icp;
    if (!view2DMode_ && !depthImage_.gray.empty()) {
        UiSectionHeader(u8"深度转点云", u8"已加载深度图，可重新设置参数", &pal.tool2D);
        if (ImGui::Button(u8"设置参数并生成点云…", ImVec2(-1.f, 32.f))) {
            OpenDepthToCloudDialog();
        }
        ImGui::Spacing();
    }
    UiSectionHeader(ToolModeLabel(measure_.mode),
                    icpMode ? nullptr : u8"当前工具参数与说明", &pal.tool3D);

    ImGui::PushStyleColor(ImGuiCol_ChildBg, pal.panelRaised);
    ImGui::PushStyleColor(ImGuiCol_Border, pal.border);
    const float toolH = std::max(120.f, ImGui::GetContentRegionAvail().y);
    ImGui::BeginChild(u8"##toolhelp", ImVec2(0.f, toolH), ImGuiChildFlags_Borders,
                      ImGuiWindowFlags_AlwaysVerticalScrollbar);

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
                const Vec3 w = MainCloud().ToWorld(*measure_.picked);
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
                SetStatus(u8"已清除测距标记");
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
                    SetStatus(u8"请先框选区域，或改用“对全部可见点拟合”");
                } else {
                    std::string error;
                    PlaneModel plane;
                    if (FitPlaneWithBackend(measure_.roiIndices, plane, error, algoBackend_)) {
                        measure_.plane = plane;
                        UpdatePlaneFitProfile(plane, measure_.roiIndices);
                        UpdateOverlays();
                        SetStatus(std::string(AlgorithmBackendLabel(EffectiveAlgoBackend())) +
                                  u8" 框选拟合完成");
                    } else {
                        SetStatus(error);
                    }
                }
            }
            if (ImGui::Button(u8"对全部可见点拟合", ImVec2(-1, 0))) {
                std::string error;
                PlaneModel plane;
                std::vector<std::size_t> empty;
                if (FitPlaneWithBackend(empty, plane, error, algoBackend_)) {
                    measure_.plane = plane;
                    UpdatePlaneFitProfile(plane, measure_.roiIndices);
                    UpdateOverlays();
                    SetStatus(std::string(AlgorithmBackendLabel(EffectiveAlgoBackend())) +
                              u8" 全点云拟合完成");
                } else {
                    SetStatus(error);
                }
            }
            if (measure_.plane && ImGui::Button(u8"清除拟合平面显示", ImVec2(-1, 0))) {
                measure_.plane.reset();
                measure_.planeDistProfile = {};
                UpdateOverlays();
            }
            if (measure_.planeDistProfile.valid) {
                DrawPlaneDistanceHistogram(measure_.planeDistProfile);
            }
            break;
        case ToolMode::PlaneAlign:
            ImGui::TextWrapped(
                u8"线扫倾斜校正：\n"
                u8"① 在工件平整区域拖拽框选基准面\n"
                u8"② 选择摆正目标（通常选水平面 +Z）\n"
                u8"③ 点击执行，整体旋转点云使基准面水平\n"
                u8"④ 橙色面为拟合预览，可 Ctrl+Z 撤销");
            ImGui::Text(u8"当前框选: %zu 点", measure_.roiIndices.size());
            ImGui::Spacing();
            ImGui::TextDisabled(u8"摆正目标（基准面法向对齐）");
            if (ImGui::RadioButton(u8"水平面 (+Z)", planeAlignTarget_ == 0)) planeAlignTarget_ = 0;
            ImGui::SameLine();
            if (ImGui::RadioButton(u8"+Y", planeAlignTarget_ == 1)) planeAlignTarget_ = 1;
            ImGui::SameLine();
            if (ImGui::RadioButton(u8"+X", planeAlignTarget_ == 2)) planeAlignTarget_ = 2;
            ImGui::Spacing();
            if (ImGui::Button(u8"执行平面校准", ImVec2(-1, 40.f))) {
                AlignCloudToReferencePlane(&measure_.roiIndices, nullptr);
            }
            if (ImGui::Button(u8"仅预览拟合平面（不旋转）", ImVec2(-1, 0))) {
                if (measure_.roiIndices.empty()) {
                    SetStatus(u8"请先框选基准平面区域");
                } else {
                    std::string error;
                    PlaneModel plane;
                    if (FitPlaneWithBackend(measure_.roiIndices, plane, error, algoBackend_)) {
                        measure_.plane = plane;
                        UpdateOverlays();
                        char buf[160];
                        std::snprintf(buf, sizeof(buf), u8"基准面预览 RMS=%.4f mm，确认后点「执行平面校准」",
                                      plane.rms);
                        SetStatus(buf);
                    } else {
                        SetStatus(error);
                    }
                }
            }
            if (measure_.plane &&
                ImGui::Button(u8"以当前预览平面执行校准", ImVec2(-1, 0))) {
                AlignCloudToReferencePlane(nullptr, &*measure_.plane);
            }
            if (measure_.plane && ImGui::Button(u8"清除平面预览", ImVec2(-1, 0))) {
                measure_.plane.reset();
                UpdateOverlays();
            }
            break;
        case ToolMode::SphereFit:
        case ToolMode::SphereBodyFit: {
            ImGui::TextWrapped(
                u8"① 左键拖拽框选可见表面\n"
                u8"② 点击下方按钮拟合球体\n"
                u8"③ 橙色半透明实体球为拟合结果，可与原始点云对比\n"
                u8"④ 青蓝=拟合内点，灰蓝=外点（对比色）\n"
                u8"⑤ 拟合后会在 DB 树生成「<拟合:球面>」，可隐藏原始点云单独查看");
            ImGui::Text(u8"当前框选: %zu 点", measure_.roiIndices.size());
            ImGui::Spacing();
            if (ImGui::Button(u8"对框选区域拟合", ImVec2(-1, 32.f))) {
                if (measure_.roiIndices.empty()) {
                    SetStatus(u8"请先框选区域，或改用“对全部可见点拟合”");
                } else {
                    std::string error;
                    SphereModel sphere;
                    if (FitSphereWithBackend(measure_.roiIndices, sphere, error, algoBackend_)) {
                        measure_.sphere = sphere;
                        measure_.circle.reset();
                        measure_.cylinder.reset();
                        SyncFittedSphereLayer(sphere);
                        UpdateOverlays();
                        sceneGpuDirty_ = true;
                        char buf[192];
                        std::snprintf(buf, sizeof(buf),
                                      u8"%s 完成 R=%.6f RMS=%.6f（内点 %zu）",
                                      AlgorithmBackendLabel(EffectiveAlgoBackend()), sphere.radius,
                                      sphere.rms, sphere.inlierIndices.size());
                        SetStatus(buf);
                    } else {
                        SetStatus(error);
                    }
                }
            }
            if (ImGui::Button(u8"对全部可见点拟合", ImVec2(-1, 0))) {
                std::string error;
                SphereModel sphere;
                std::vector<std::size_t> empty;
                if (FitSphereWithBackend(empty, sphere, error, algoBackend_)) {
                    measure_.sphere = sphere;
                    measure_.circle.reset();
                    measure_.cylinder.reset();
                    SyncFittedSphereLayer(sphere);
                    UpdateOverlays();
                    sceneGpuDirty_ = true;
                    char buf[192];
                    std::snprintf(buf, sizeof(buf), u8"%s 全点云拟合 R=%.6f RMS=%.6f（内点 %zu）",
                                  AlgorithmBackendLabel(EffectiveAlgoBackend()), sphere.radius,
                                  sphere.rms, sphere.inlierIndices.size());
                    SetStatus(buf);
                } else {
                    SetStatus(error);
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
                RemoveFittedSphereLayers();
                UpdateOverlays();
                sceneGpuDirty_ = true;
            }
            break;
        }
        case ToolMode::CircleFit:
            ImGui::TextWrapped(
                u8"① 执行「投影并填充」后，左键拖拽框选青色填充点\n"
                u8"② 灰色为原始参考（仅显示，不参与框选与拟合）\n"
                u8"③ 橙色线框为拟合圆；拟合后近圆点为橙色");
            DrawFitRoiShapeControls();
            ImGui::Spacing();
            if (ImGui::Button(u8"对框选区域拟合圆", ImVec2(-1, 32.f))) {
                if (measure_.roiIndices.empty()) {
                    SetStatus(u8"请先框选区域，或改用“对全部可见点拟合”");
                } else {
                    std::string error;
                    CircleModel circle;
                    if (FitCircleWithBackend(measure_.roiIndices, circle, error, algoBackend_)) {
                        measure_.circle = circle;
                        measure_.sphere.reset();
                        measure_.cylinder.reset();
                        UpdateOverlays();
                        if (activeCloudPane_ == 1 && DualCloudViewActive()) {
                            needUploadFilled_ = true;
                        } else {
                            sceneGpuDirty_ = true;
                        }
                        char buf[192];
                        std::snprintf(buf, sizeof(buf),
                                      u8"%s 圆拟合 R=%.6f RMS=%.6f（近圆点 %zu）",
                                      AlgorithmBackendLabel(EffectiveAlgoBackend()), circle.radius,
                                      circle.rms, circle.inlierIndices.size());
                        SetStatus(buf);
                    } else {
                        SetStatus(error);
                    }
                }
            }
            if (ImGui::Button(u8"对全部可见点拟合", ImVec2(-1, 0))) {
                std::string error;
                CircleModel circle;
                std::vector<std::size_t> empty;
                if (FitCircleWithBackend(empty, circle, error, algoBackend_)) {
                    measure_.circle = circle;
                    measure_.sphere.reset();
                    measure_.cylinder.reset();
                    UpdateOverlays();
                    if (activeCloudPane_ == 1 && DualCloudViewActive()) {
                        needUploadFilled_ = true;
                    } else {
                        sceneGpuDirty_ = true;
                    }
                    char buf[192];
                    std::snprintf(buf, sizeof(buf), u8"%s 全点云圆拟合 R=%.6f RMS=%.6f（近圆点 %zu）",
                                  AlgorithmBackendLabel(EffectiveAlgoBackend()), circle.radius,
                                  circle.rms, circle.inlierIndices.size());
                    SetStatus(buf);
                } else {
                    SetStatus(error);
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
                if (activeCloudPane_ == 1 && DualCloudViewActive()) {
                    needUploadFilled_ = true;
                } else {
                    sceneGpuDirty_ = true;
                }
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
                    SetStatus(u8"请先框选区域，或改用“对全部可见点拟合”");
                } else {
                    std::string error;
                    CylinderModel cyl;
                    if (FitCylinderWithBackend(measure_.roiIndices, cyl, error, algoBackend_)) {
                        measure_.cylinder = cyl;
                        measure_.sphere.reset();
                        measure_.circle.reset();
                        UpdateOverlays();
                        char buf[160];
                        std::snprintf(buf, sizeof(buf),
                                      u8"%s 圆柱拟合完成 R=%.6f RMS=%.6f（%d 点）",
                                      AlgorithmBackendLabel(EffectiveAlgoBackend()), cyl.radius,
                                      cyl.rms, cyl.pointCount);
                        SetStatus(buf);
                    } else {
                        SetStatus(error);
                    }
                }
            }
            if (ImGui::Button(u8"对全部可见点拟合", ImVec2(-1, 0))) {
                std::string error;
                CylinderModel cyl;
                std::vector<std::size_t> empty;
                if (FitCylinderWithBackend(empty, cyl, error, algoBackend_)) {
                    measure_.cylinder = cyl;
                    measure_.sphere.reset();
                    measure_.circle.reset();
                    UpdateOverlays();
                    char buf[160];
                    std::snprintf(buf, sizeof(buf), u8"%s 全点云圆柱拟合 R=%.6f RMS=%.6f",
                                  AlgorithmBackendLabel(EffectiveAlgoBackend()), cyl.radius,
                                  cyl.rms);
                    SetStatus(buf);
                } else {
                    SetStatus(error);
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
                if (ComputeFlatnessWithBackend(measure_.roiIndices, fr, error, algoBackend_)) {
                    measure_.flatness = std::move(fr);
                    measure_.plane = measure_.flatness.plane;
                    sceneGpuDirty_ = true;
                    UpdateOverlays();
                    char buf[192];
                    std::snprintf(buf, sizeof(buf),
                                  u8"%s 平面度 PV=%.6f mm，RMS=%.6f（%d 点）",
                                  AlgorithmBackendLabel(EffectiveAlgoBackend()),
                                  measure_.flatness.peakToValley, measure_.flatness.rms,
                                  measure_.flatness.plane.pointCount);
                    SetStatus(buf);
                } else {
                    SetStatus(error);
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
                sceneGpuDirty_ = true;
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
                    SetStatus(u8"请先框选区域 A");
                } else {
                    std::string error;
                    PlaneModel plane;
                    if (FitPlaneWithBackend(sg.regionA, plane, error, algoBackend_)) {
                        sg.planeA = plane;
                        sg.hasPlane = true;
                        sg.hasDistances = false;
                        sg.phase = StepGapPhase::SelectB;
                        measure_.plane = plane;
                        UpdateOverlays();
                        SetStatus(std::string(AlgorithmBackendLabel(EffectiveAlgoBackend())) + u8" 区域 A 平面已拟合，请框选区域 B");
                        sceneGpuDirty_ = true;
                    } else {
                        SetStatus(error);
                    }
                }
            }
            if (ImGui::Button(u8"计算段差 ΔZ", ImVec2(-1, 32.f))) {
                if (sg.regionA.empty()) {
                    SetStatus(u8"请先框选区域 A");
                } else if (sg.regionB.empty()) {
                    SetStatus(u8"请先框选区域 B");
                } else {
                    std::string error;
                    if (ComputeStepGapZHeightWithBackend(sg.regionA, sg.regionB, sg, error,
                                                         algoBackend_)) {
                        sceneGpuDirty_ = true;
                        UpdateOverlays();
                        char buf[192];
                        std::snprintf(buf, sizeof(buf),
                                      u8"段差 ΔZ=%.6f（中位 %.6f，B=%zu 点）", sg.mean, sg.median,
                                      sg.regionB.size());
                        SetStatus(buf);
                    } else {
                        SetStatus(error);
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
                sceneGpuDirty_ = true;
                UpdateOverlays();
                SetStatus(u8"段差：先框选基准区域 A");
            }
            break;
        }
        case ToolMode::Roi: {
            ImGui::TextWrapped(
                u8"框选形状：矩形/圆形拖拽；自由多边形逐点点击后「完成多边形」。\n"
                u8"勾选「世界尺寸」后：按下并拖动可移动框选位置，松手确认；按下方半径或边长（mm）框选。");
            int shape = static_cast<int>(measure_.roiShape);
            if (ImGui::RadioButton(u8"矩形", shape == 0)) measure_.roiShape = RoiShape::Rect;
            ImGui::SameLine();
            if (ImGui::RadioButton(u8"圆形", shape == 1)) measure_.roiShape = RoiShape::Circle;
            ImGui::SameLine();
            if (ImGui::RadioButton(u8"自由多边形", shape == 2)) {
                measure_.roiShape = RoiShape::FreePolygon;
            }
            ImGui::Checkbox(u8"使用世界尺寸 (mm)", &measure_.roiUseWorldSize);
            if (measure_.roiShape == RoiShape::Circle) {
                ImGui::DragFloat(u8"圆半径 (mm)", &measure_.roiWorldRadius, 0.1f, 0.1f, 1e6f,
                                 "%.3f");
            } else if (measure_.roiShape == RoiShape::Rect) {
                ImGui::DragFloat(u8"矩形宽 (mm)", &measure_.roiWorldWidth, 0.1f, 0.1f, 1e6f,
                                 "%.3f");
                ImGui::DragFloat(u8"矩形高 (mm)", &measure_.roiWorldHeight, 0.1f, 0.1f, 1e6f,
                                 "%.3f");
            }
            if (measure_.roiShape == RoiShape::FreePolygon) {
                ImGui::Text(u8"顶点数: %zu", measure_.roiPolyX.size());
                if (ImGui::Button(u8"完成多边形", ImVec2(-1, 0))) FinishRoiPolygon();
                if (ImGui::Button(u8"清除多边形顶点", ImVec2(-1, 0))) {
                    measure_.roiPolyX.clear();
                    measure_.roiPolyY.clear();
                    measure_.roiPolyBuilding = false;
                }
            }
            ImGui::Spacing();
            ImGui::Text(u8"已选 %zu 点", measure_.roiIndices.size());
            ImGui::Spacing();
            if (ImGui::Button(u8"清除框选内的点", ImVec2(-1, 32.f))) {
                if (measure_.roiIndices.empty()) {
                    SetStatus(u8"请先框选区域");
                } else {
                    PushHistory(u8"清除框内点");
                    ApplyRoiDeleteWithBackend(measure_.roiIndices, true);
                    SetStatus(std::string(u8"已清除框内点，可见 ") + std::to_string(EditableCloud().VisibleCount()));
                    measure_.roiIndices.clear();
                    if (activeCloudPane_ == 1 && DualCloudViewActive()) {
                        needUploadFilled_ = true;
                    } else {
                        sceneGpuDirty_ = true;
                    }
                }
            }
            if (ImGui::Button(u8"清除框选外的点（只留框内）", ImVec2(-1, 32.f))) {
                if (measure_.roiIndices.empty()) {
                    SetStatus(u8"请先框选区域");
                } else {
                    PushHistory(u8"清除框外点");
                    ApplyRoiDeleteWithBackend(measure_.roiIndices, false);
                    SetStatus(std::string(u8"已清除框外点，可见 ") + std::to_string(EditableCloud().VisibleCount()));
                    measure_.roiIndices.clear();
                    if (activeCloudPane_ == 1 && DualCloudViewActive()) {
                        needUploadFilled_ = true;
                    } else {
                        sceneGpuDirty_ = true;
                    }
                }
            }
            if (ImGui::Button(u8"恢复全部点显示", ImVec2(-1, 0))) {
                PushHistory(u8"恢复全部点前");
                RestoreAllPointsWithBackend();
                SetStatus(u8"已恢复全部点");
                sceneGpuDirty_ = true;
            }
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::TextDisabled(u8"投影 → 填充");
            ImGui::DragFloat(u8"填充网格步长 (mm)", &measure_.roiFillGridStep, 0.05f, 0.f, 100.f,
                             measure_.roiFillGridStep <= 0.f ? u8"自动" : "%.3f");
            if (ImGui::Button(u8"执行：投影并填充", ImVec2(-1, 36.f))) {
                std::string err;
                if (!RunRoiProjectFill(err)) SetStatus(err);
            }
            if (DualCloudViewActive()) {
                ImGui::Text(u8"填充 %zu 点（叠加原始参考 %zu 点，仅显示）", filledCloud_.points.size(),
                            MainCloud().VisibleCount());
                if (ImGui::Button(u8"关闭填充视图", ImVec2(-1, 0))) {
                    CloseDualCloudView();
                    SetStatus(u8"已关闭填充视图");
                }
            }
            ImGui::TextWrapped(
                u8"流程：圆形 ROI 框选挖孔区域 →「投影并填充」→ 单视区显示填充结果"
                u8"（灰色为原始参考，仅显示）。用「拟合→圆拟合」在青色填充点上测量。");
            break;
        }
        case ToolMode::StepHeight:
            ImGui::TextWrapped(
                u8"工业台阶/高度差测量：\n"
                u8"① 点击基准面点 A\n"
                u8"② 再点击测量面点 B\n"
                u8"③ 显示 ΔZ = B.Z − A.Z（单位 mm）");
            if (measure_.stepA && measure_.stepB) {
                ImGui::Spacing();
                ImGui::Text(u8"ΔZ = %.4f mm", measure_.stepDeltaZ);
                const Vec3 wa = MainCloud().ToWorld(*measure_.stepA);
                const Vec3 wb = MainCloud().ToWorld(*measure_.stepB);
                ImGui::Text(u8"A.Z = %.4f mm", wa.z);
                ImGui::Text(u8"B.Z = %.4f mm", wb.z);
            }
            if (ImGui::Button(u8"清除台阶测量", ImVec2(-1, 0))) {
                measure_.stepA.reset();
                measure_.stepB.reset();
                measure_.stepDeltaZ = 0.f;
                UpdateOverlays();
                SetStatus(u8"已清除台阶测量");
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
                if (MainCloud().bounds.Valid()) {
                    measure_.section.position = measure_.section.cutAlongX
                                                    ? MainCloud().bounds.Center().x
                                                    : MainCloud().bounds.Center().y;
                }
                SyncSectionCutPlane();
            }

            float amin = 0.f, amax = 1.f;
            if (MainCloud().bounds.Valid()) {
                if (measure_.section.cutAlongX) {
                    amin = MainCloud().bounds.min.x;
                    amax = MainCloud().bounds.max.x;
                } else {
                    amin = MainCloud().bounds.min.y;
                    amax = MainCloud().bounds.max.y;
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
            if (!showSectionPanel_) {
                if (ImGui::Button(u8"显示 2D 轮廓窗口", ImVec2(-1, 0))) {
                    showSectionPanel_ = true;
                }
            }
            if (ImGui::Button(u8"清除截面结果", ImVec2(-1, 0))) {
                measure_.section.points.clear();
                measure_.section.pickA.reset();
                measure_.section.pickB.reset();
                measure_.section.lineDistance = 0.f;
                measure_.section.zDistance = 0.f;
                SyncSectionCutPlane();
                SetStatus(u8"已清除截面轮廓（切面仍可拖拽）");
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
        case ToolMode::Icp:
            DrawIcpPanel();
            break;
    }

    ImGui::EndChild();
    ImGui::PopStyleColor(2);
}

void Application::DrawUi() {
    const UiPalette& pal = GetUiPalette();
    const ImGuiViewport* vp = ImGui::GetMainViewport();
    const float sidebarW = SidebarWidth();
    const float consoleH = ConsoleHeight();
    const float toolbarH = 44.f;

    const float menuBottom = DrawMenuBar();

    if (algoEditor_.IsVisible()) {
        AlgoHost host;
        host.currentCloud = &MainCloud();
        host.publishCloud = [this](PointCloud&& c, const char* status) {
            ApplyCloud(std::move(c), status, false);
        };
        algoEditor_.SetHost(host);
        DrawAboutPopup();
        DrawNativeAlgoPasswordPopup();
        DrawCreatePopups();
        algoEditor_.Draw(menuBottom, consoleH);
        DrawConsolePanel();
        shapeTemplateWindow_.Draw();
        return;
    }

    if (halconMatchWindow_.IsVisible()) {
        DrawAboutPopup();
        DrawNativeAlgoPasswordPopup();
        DrawCreatePopups();
        halconMatchWindow_.Draw(menuBottom, consoleH);
        DrawConsolePanel();
        return;
    }

    if (ocrWindow_.IsVisible()) {
        DrawAboutPopup();
        DrawNativeAlgoPasswordPopup();
        DrawCreatePopups();
        ocrWindow_.Draw(menuBottom, consoleH);
        DrawConsolePanel();
        return;
    }

    if (camera2DCalibWindow_.IsVisible()) {
        DrawAboutPopup();
        DrawNativeAlgoPasswordPopup();
        DrawCreatePopups();
        camera2DCalibWindow_.Draw(menuBottom, consoleH);
        DrawConsolePanel();
        return;
    }

    if (cameraIntrinsicsCalibWindow_.IsVisible()) {
        DrawAboutPopup();
        DrawNativeAlgoPasswordPopup();
        DrawCreatePopups();
        cameraIntrinsicsCalibWindow_.Draw(menuBottom, consoleH);
        DrawConsolePanel();
        return;
    }

    if (multiViewGeometryWindow_.IsVisible()) {
        DrawAboutPopup();
        DrawNativeAlgoPasswordPopup();
        DrawCreatePopups();
        multiViewGeometryWindow_.Draw(menuBottom, consoleH);
        DrawConsolePanel();
        return;
    }

    DrawToolbar(menuBottom, toolbarH);
    const float contentTop = menuBottom + toolbarH;
    const float contentH = vp->Pos.y + vp->Size.y - contentTop;
    const float viewH = std::max(contentH - consoleH, 1.f);
    const float minToolH = 120.f;
    const float maxDbH = std::max(100.f, contentH - minToolH - 10.f);
    dbTreePreferredH_ = std::clamp(dbTreePreferredH_, 100.f, maxDbH);

    UpdateView3dLayout(contentTop, viewH, sidebarW);
    cachedContentTop_ = contentTop;
    cachedContentH_ = viewH;
    cachedSidebarW_ = sidebarW;
    cachedViewLayoutValid_ = true;

    ImGui::SetNextWindowPos(ImVec2(vp->Pos.x, contentTop));
    ImGui::SetNextWindowSize(ImVec2(sidebarW, contentH));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.f);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, pal.panel);
    ImGui::Begin(u8"##侧栏", nullptr,
                 ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                     ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoBringToFrontOnFocus);

    DrawDbTreePanel(dbTreePreferredH_);
    DrawDbTreeSplitter(maxDbH);
    DrawToolPanel();

    ImGui::End();
    ImGui::PopStyleColor();
    ImGui::PopStyleVar(2);

    DrawSidebarSplitter(contentTop, contentH);

    if (measure_.mode == ToolMode::Section && showSectionPanel_) {
        DrawSectionPanel();
    }
    if (measure_.mode == ToolMode::StepGap && measure_.stepGap.hasDistances) {
        DrawStepGapPanel();
    }
    if (image2DTool_ == Image2DTool::DepthProfile && depthProfileValid_ && showDepthProfilePanel_) {
        DrawDepthProfilePanel();
    }

    DrawEmptyCloudPrompt(contentTop, viewH);

    if (!view2DMode_) DrawViewAxisWidget(contentTop, contentTop + viewH, SidebarWidth());
    DrawImagePanel();
    UpdateView3dLayout(contentTop, viewH, SidebarWidth());
    DrawAboutPopup();
    DrawDepthToCloudWindow();
    DrawNativeAlgoPasswordPopup();
    DrawCreatePopups();
    if (!view2DMode_) {
        DrawDualCloudPaneLabels();
        DrawIcpOverlayLabel();
        DrawOverlays();
    }

    DrawConsolePanel();
    shapeTemplateWindow_.Draw();
}

void Application::Run() {
    while (!glfwWindowShouldClose(window_)) {
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        HandleInput();
        DrawUi();

        if (sceneGpuDirty_ && !algoEditor_.IsVisible() && !halconMatchWindow_.IsVisible() &&
            !ocrWindow_.IsVisible())
            RefreshGpu();
        if (needUploadFilled_ && !algoEditor_.IsVisible() && !halconMatchWindow_.IsVisible() &&
            !ocrWindow_.IsVisible())
            RefreshGpuFilled();
        if (needUploadIcp_ && !algoEditor_.IsVisible() && !halconMatchWindow_.IsVisible() &&
            !ocrWindow_.IsVisible())
            RefreshIcpGpu();

        glfwGetFramebufferSize(window_, &fbW_, &fbH_);
        glDisable(GL_SCISSOR_TEST);
        glViewport(0, 0, fbW_, fbH_);
        {
            const UiPalette& pal = GetUiPalette();
            glClearColor(pal.bgDeep.x, pal.bgDeep.y, pal.bgDeep.z, pal.bgDeep.w);
        }
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        if (!algoEditor_.IsVisible() && !view2DMode_) {
            if (DualCloudViewActive()) {
                int vx = 0, vy = 0, vw = 0, vh = 0;
                GetCloudPaneGlViewport(1, vx, vy, vw, vh);
                glViewport(vx, vy, vw, vh);
                glEnable(GL_SCISSOR_TEST);
                glScissor(vx, vy, vw, vh);
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
                filledRenderer_.Draw(camera_, vw, vh, pointSize_, opacity_);
                glDisable(GL_SCISSOR_TEST);
                glViewport(0, 0, fbW_, fbH_);
            } else {
                int vx = 0, vy = 0, vw = 0, vh = 0;
                GetView3dGlViewport(vx, vy, vw, vh);
                glViewport(vx, vy, vw, vh);
                glEnable(GL_SCISSOR_TEST);
                glScissor(vx, vy, vw, vh);
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
                for (SceneCloudLayer& layer : sceneLayers_) {
                    if (!layer.visible || !layer.rendererReady) continue;
                    layer.renderer.Draw(camera_, vw, vh, pointSize_, opacity_);
                }
                if (icpOverlayActive_ && icpSourceLoaded_) {
                    icpRenderer_.Draw(camera_, vw, vh, pointSize_, opacity_);
                }
                glDisable(GL_SCISSOR_TEST);
                glViewport(0, 0, fbW_, fbH_);
            }
        }

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window_);
    }
}

void Application::Shutdown() {
    PaddleOcrTools::Shutdown();
    DestroyImageView(depthImage_);
    DestroyImageView(brightnessImage_);
    ShutdownSceneLayers();
    filledRenderer_.Shutdown();
    icpRenderer_.Shutdown();
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    if (window_) {
        glfwDestroyWindow(window_);
        window_ = nullptr;
    }
    glfwTerminate();
}
