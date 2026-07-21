#include "render/PointCloudRenderer.h"

#include <glad/gl.h>

#include <cmath>
#include <unordered_set>
#include <vector>

namespace {

const char* kVs = R"(#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;
uniform mat4 uMVP;
uniform float uPointSize;
out vec3 vColor;
void main() {
    vColor = aColor;
    gl_Position = uMVP * vec4(aPos, 1.0);
    gl_PointSize = uPointSize;
}
)";

const char* kFs = R"(#version 330 core
in vec3 vColor;
uniform float uOpacity;
out vec4 FragColor;
void main() {
    FragColor = vec4(vColor, uOpacity);
}
)";

struct Vertex {
    float x, y, z;
    float r, g, b;
};

}  // namespace

bool PointCloudRenderer::Init(std::string& error) {
    if (!shader_.Build(kVs, kFs, error)) return false;
    glGenVertexArrays(1, &vao_);
    glGenBuffers(1, &vbo_);
    glGenVertexArrays(1, &overlayVao_);
    glGenBuffers(1, &overlayVbo_);
    glGenVertexArrays(1, &planeVao_);
    glGenBuffers(1, &planeVbo_);
    glGenVertexArrays(1, &axesVao_);
    glGenBuffers(1, &axesVbo_);
    glGenVertexArrays(1, &fitWireVao_);
    glGenBuffers(1, &fitWireVbo_);
    RebuildAxesMesh(1.f);
    return true;
}

int PointCloudRenderer::Upload(const PointCloud& cloud, const UploadParams& params,
                               std::vector<std::size_t>* outDisplayIndices) {
    std::unordered_set<std::size_t> hi;
    if (params.highlightRoi) {
        hi.reserve(params.highlightRoi->size() * 2 + 1);
        hi.insert(params.highlightRoi->begin(), params.highlightRoi->end());
    }

    const bool useColors =
        params.usePointColors && cloud.colors.size() == cloud.points.size();

    std::size_t visible = 0;
    for (std::size_t i = 0; i < cloud.points.size(); ++i) {
        if (params.ignoreMask || cloud.mask.empty() || cloud.mask[i]) ++visible;
    }

    const int maxPts = std::max(params.maxDisplayPoints, 10000);
    const int stride = (visible > static_cast<std::size_t>(maxPts))
                           ? static_cast<int>((visible + static_cast<std::size_t>(maxPts) - 1) /
                                              static_cast<std::size_t>(maxPts))
                           : 1;

    float zMin = params.zMin;
    float zMax = params.zMax;
    if (zMax <= zMin) {
        zMin = cloud.bounds.min.z;
        zMax = cloud.bounds.max.z;
        if (zMax <= zMin) zMax = zMin + 1.f;
    }
    const float invZ = 1.f / (zMax - zMin);

    std::vector<Vertex> verts;
    verts.reserve(static_cast<std::size_t>(std::min(visible, static_cast<std::size_t>(maxPts)) +
                                           hi.size()));
    if (outDisplayIndices) {
        outDisplayIndices->clear();
        outDisplayIndices->reserve(verts.capacity());
    }

    auto pushPoint = [&](std::size_t idx, bool forceGreenRoi) {
        const Vec3& p = cloud.points[idx];
        float r, g, b;
        if (forceGreenRoi) {
            r = 0.15f;
            g = 1.0f;
            b = 0.45f;
        } else if (useColors) {
            r = cloud.colors[idx].x;
            g = cloud.colors[idx].y;
            b = cloud.colors[idx].z;
        } else {
            const float t = (p.z - zMin) * invZ;
            const Vec3 c = HeightToColor(t);
            r = c.x;
            g = c.y;
            b = c.z;
        }
        verts.push_back({p.x, p.y, p.z, r, g, b});
        if (outDisplayIndices) outDisplayIndices->push_back(idx);
    };

    // Always include ROI highlights fully (unless using custom colors for analysis maps).
    if (!useColors) {
        for (std::size_t idx : hi) {
            if (idx >= cloud.points.size()) continue;
            if (!params.ignoreMask && !cloud.mask.empty() && !cloud.mask[idx]) continue;
            pushPoint(idx, true);
        }
    }

    int kept = 0;
    for (std::size_t i = 0; i < cloud.points.size(); ++i) {
        if (!params.ignoreMask && !cloud.mask.empty() && !cloud.mask[i]) continue;
        if (!useColors && hi.count(i)) continue;
        if ((kept++ % stride) != 0) continue;
        if (static_cast<int>(verts.size()) >= maxPts + static_cast<int>(hi.size())) break;
        pushPoint(i, false);
    }

    vertexCount_ = static_cast<int>(verts.size());
    glBindVertexArray(vao_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(verts.size() * sizeof(Vertex)),
                 verts.empty() ? nullptr : verts.data(), GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(0));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                          reinterpret_cast<void*>(sizeof(float) * 3));
    glBindVertexArray(0);
    return vertexCount_;
}

void PointCloudRenderer::SetDistanceOverlay(const std::optional<Vec3>& a,
                                            const std::optional<Vec3>& b) {
    std::vector<Vertex> verts;
    markerCount_ = 0;
    lineVertexCount_ = 0;

    auto pushMarker = [&](const Vec3& p, const Vec3& color) {
        verts.push_back({p.x, p.y, p.z, color.x, color.y, color.z});
        ++markerCount_;
    };

    if (a) pushMarker(*a, {1.f, 0.92f, 0.2f});
    if (b) pushMarker(*b, {1.f, 0.35f, 0.15f});
    if (a && b) {
        verts.push_back({a->x, a->y, a->z, 1.f, 0.95f, 0.3f});
        verts.push_back({b->x, b->y, b->z, 1.f, 0.95f, 0.3f});
        lineVertexCount_ = 2;
    }

    glBindVertexArray(overlayVao_);
    glBindBuffer(GL_ARRAY_BUFFER, overlayVbo_);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(verts.size() * sizeof(Vertex)),
                 verts.empty() ? nullptr : verts.data(), GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(0));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                          reinterpret_cast<void*>(sizeof(float) * 3));
    glBindVertexArray(0);
}

void PointCloudRenderer::SetPickOverlay(const std::optional<Vec3>& p) {
    std::vector<Vertex> verts;
    markerCount_ = 0;
    lineVertexCount_ = 0;
    if (p) {
        verts.push_back({p->x, p->y, p->z, 0.2f, 0.95f, 1.f});
        markerCount_ = 1;
    }
    glBindVertexArray(overlayVao_);
    glBindBuffer(GL_ARRAY_BUFFER, overlayVbo_);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(verts.size() * sizeof(Vertex)),
                 verts.empty() ? nullptr : verts.data(), GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(0));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                          reinterpret_cast<void*>(sizeof(float) * 3));
    glBindVertexArray(0);
}

void PointCloudRenderer::RebuildPlaneMesh(const PlaneModel& plane) {
    Vec3 n = plane.normal.Normalized();
    Vec3 tmp = (std::fabs(n.x) < 0.9f) ? Vec3{1, 0, 0} : Vec3{0, 1, 0};
    Vec3 u = n.Cross(tmp).Normalized();
    Vec3 v = n.Cross(u).Normalized();
    const float su = std::max(plane.halfExtentU > 0.f ? plane.halfExtentU : plane.halfSize, 0.01f);
    const float sv = std::max(plane.halfExtentV > 0.f ? plane.halfExtentV : plane.halfSize, 0.01f);
    const int div = 12;
    const Vec3 color{1.f, 0.45f, 0.12f};

    std::vector<Vertex> verts;
    verts.reserve(static_cast<std::size_t>(div * div * 6));
    for (int i = 0; i < div; ++i) {
        for (int j = 0; j < div; ++j) {
            const float u0 = -su + 2.f * su * (static_cast<float>(i) / div);
            const float u1 = -su + 2.f * su * (static_cast<float>(i + 1) / div);
            const float v0 = -sv + 2.f * sv * (static_cast<float>(j) / div);
            const float v1 = -sv + 2.f * sv * (static_cast<float>(j + 1) / div);
            const Vec3 p00 = plane.centroid + u * u0 + v * v0;
            const Vec3 p10 = plane.centroid + u * u1 + v * v0;
            const Vec3 p11 = plane.centroid + u * u1 + v * v1;
            const Vec3 p01 = plane.centroid + u * u0 + v * v1;
            auto push = [&](const Vec3& p) {
                verts.push_back({p.x, p.y, p.z, color.x, color.y, color.z});
            };
            push(p00);
            push(p10);
            push(p11);
            push(p00);
            push(p11);
            push(p01);
        }
    }

    planeVertexCount_ = static_cast<int>(verts.size());
    glBindVertexArray(planeVao_);
    glBindBuffer(GL_ARRAY_BUFFER, planeVbo_);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(verts.size() * sizeof(Vertex)),
                 verts.data(), GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(0));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                          reinterpret_cast<void*>(sizeof(float) * 3));
    glBindVertexArray(0);
    hasPlane_ = true;
}

void PointCloudRenderer::SetPlaneOverlay(const std::optional<PlaneModel>& plane) {
    if (!plane) {
        hasPlane_ = false;
        planeVertexCount_ = 0;
        return;
    }
    RebuildPlaneMesh(*plane);
}

void PointCloudRenderer::UploadFitWire(const std::vector<float>& posRgb) {
    fitWireVertexCount_ = static_cast<int>(posRgb.size() / 6);
    if (fitWireVertexCount_ <= 0) {
        fitWireVertexCount_ = 0;
        return;
    }
    glBindVertexArray(fitWireVao_);
    glBindBuffer(GL_ARRAY_BUFFER, fitWireVbo_);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(posRgb.size() * sizeof(float)),
                 posRgb.data(), GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 6,
                          reinterpret_cast<void*>(0));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 6,
                          reinterpret_cast<void*>(sizeof(float) * 3));
    glBindVertexArray(0);
}

void PointCloudRenderer::ClearFitWireOverlay() {
    fitWireVertexCount_ = 0;
}

namespace {
void PushSeg(std::vector<float>& v, const Vec3& a, const Vec3& b, const Vec3& col) {
    v.push_back(a.x);
    v.push_back(a.y);
    v.push_back(a.z);
    v.push_back(col.x);
    v.push_back(col.y);
    v.push_back(col.z);
    v.push_back(b.x);
    v.push_back(b.y);
    v.push_back(b.z);
    v.push_back(col.x);
    v.push_back(col.y);
    v.push_back(col.z);
}

Vec3 OrthoU(const Vec3& nIn) {
    const Vec3 n = nIn.Normalized();
    const Vec3 tmp = (std::fabs(n.x) < 0.9f) ? Vec3{1, 0, 0} : Vec3{0, 1, 0};
    return n.Cross(tmp).Normalized();
}
}  // namespace

void PointCloudRenderer::SetSphereOverlay(const std::optional<SphereModel>& sphere) {
    if (!sphere || sphere->radius <= 0.f) {
        ClearFitWireOverlay();
        return;
    }
    const Vec3 c = sphere->center;
    const float R = sphere->radius;
    const Vec3 col{1.f, 0.55f, 0.15f};
    std::vector<float> verts;
    verts.reserve(64 * 3 * 2 * 6);
    constexpr int N = 48;
    constexpr float kPi = 3.14159265f;
    // Equator + two meridians (XZ, YZ) + a few latitude rings
    auto ring = [&](float elev) {
        const float ce = std::cos(elev);
        const float se = std::sin(elev);
        Vec3 prev;
        for (int i = 0; i <= N; ++i) {
            const float a = 2.f * kPi * static_cast<float>(i) / N;
            const Vec3 p = c + Vec3{R * ce * std::cos(a), R * se, R * ce * std::sin(a)};
            if (i > 0) PushSeg(verts, prev, p, col);
            prev = p;
        }
    };
    ring(0.f);
    ring(kPi / 4.f);
    ring(-kPi / 4.f);
    // Meridians
    for (int m = 0; m < 4; ++m) {
        const float lon = 0.5f * kPi * static_cast<float>(m);
        Vec3 prev;
        for (int i = 0; i <= N; ++i) {
            const float lat = -0.5f * kPi + kPi * static_cast<float>(i) / N;
            const Vec3 p =
                c + Vec3{R * std::cos(lat) * std::cos(lon), R * std::sin(lat),
                         R * std::cos(lat) * std::sin(lon)};
            if (i > 0) PushSeg(verts, prev, p, col);
            prev = p;
        }
    }
    UploadFitWire(verts);
}

void PointCloudRenderer::SetCircleOverlay(const std::optional<CircleModel>& circle) {
    if (!circle || circle->radius <= 0.f) {
        ClearFitWireOverlay();
        return;
    }
    const Vec3 n = circle->normal.Normalized();
    const Vec3 u = OrthoU(n);
    const Vec3 v = n.Cross(u).Normalized();
    const Vec3 c = circle->center;
    const float R = circle->radius;
    const Vec3 col{0.25f, 0.95f, 0.75f};
    std::vector<float> verts;
    constexpr int N = 64;
    constexpr float kPi = 3.14159265f;
    Vec3 prev;
    for (int i = 0; i <= N; ++i) {
        const float a = 2.f * kPi * static_cast<float>(i) / N;
        const Vec3 p = c + u * (R * std::cos(a)) + v * (R * std::sin(a));
        if (i > 0) PushSeg(verts, prev, p, col);
        prev = p;
    }
    // Normal tick
    PushSeg(verts, c, c + n * (R * 0.25f), Vec3{0.95f, 0.85f, 0.20f});
    UploadFitWire(verts);
}

void PointCloudRenderer::SetCylinderOverlay(const std::optional<CylinderModel>& cylinder) {
    if (!cylinder || cylinder->radius <= 0.f) {
        ClearFitWireOverlay();
        return;
    }
    const Vec3 axis = cylinder->axisDir.Normalized();
    const Vec3 u = OrthoU(axis);
    const Vec3 v = axis.Cross(u).Normalized();
    const Vec3 mid = cylinder->axisPoint;
    const float R = cylinder->radius;
    const float H = std::max(cylinder->halfHeight, R * 0.2f);
    const Vec3 col{0.95f, 0.45f, 0.85f};
    std::vector<float> verts;
    constexpr int N = 48;
    constexpr float kPi = 3.14159265f;

    auto ringAt = [&](float t) {
        const Vec3 c = mid + axis * t;
        Vec3 prev;
        for (int i = 0; i <= N; ++i) {
            const float a = 2.f * kPi * static_cast<float>(i) / N;
            const Vec3 p = c + u * (R * std::cos(a)) + v * (R * std::sin(a));
            if (i > 0) PushSeg(verts, prev, p, col);
            prev = p;
        }
    };
    ringAt(-H);
    ringAt(0.f);
    ringAt(H);
    // Generatrices
    for (int i = 0; i < 8; ++i) {
        const float a = 2.f * kPi * static_cast<float>(i) / 8.f;
        const Vec3 offset = u * (R * std::cos(a)) + v * (R * std::sin(a));
        PushSeg(verts, mid + axis * (-H) + offset, mid + axis * H + offset, col);
    }
    // Axis
    PushSeg(verts, mid + axis * (-H), mid + axis * H, Vec3{1.f, 0.9f, 0.3f});
    UploadFitWire(verts);
}

void PointCloudRenderer::RebuildAxesMesh(float axisLength) {
    axesLength_ = std::max(axisLength, 1e-4f);
    const float L = axesLength_;
    // 3 colored axes from origin
    Vertex verts[] = {
        {0, 0, 0, 1, 0.2f, 0.2f}, {L, 0, 0, 1, 0.2f, 0.2f},          // X red
        {0, 0, 0, 0.2f, 1, 0.3f}, {0, L, 0, 0.2f, 1, 0.3f},          // Y green
        {0, 0, 0, 0.3f, 0.5f, 1}, {0, 0, L, 0.3f, 0.5f, 1},          // Z blue
    };
    glBindVertexArray(axesVao_);
    glBindBuffer(GL_ARRAY_BUFFER, axesVbo_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(0));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                          reinterpret_cast<void*>(sizeof(float) * 3));
    glBindVertexArray(0);
}

void PointCloudRenderer::SetAxes(bool enabled, float axisLength) {
    axesEnabled_ = enabled;
    if (std::fabs(axisLength - axesLength_) > 1e-6f || axesVbo_ == 0) {
        RebuildAxesMesh(axisLength);
    }
}

void PointCloudRenderer::DrawAxes(const Mat4& mvp) const {
    if (!axesEnabled_) return;
    shader_.Use();
    shader_.SetMat4("uMVP", mvp.m);
    shader_.SetFloat("uOpacity", 1.f);
    glDisable(GL_DEPTH_TEST);
    glLineWidth(2.5f);
    glBindVertexArray(axesVao_);
    glDrawArrays(GL_LINES, 0, 6);
    glBindVertexArray(0);
    glEnable(GL_DEPTH_TEST);
}

void PointCloudRenderer::DrawMarkersAndLines(const Mat4& mvp) const {
    if (markerCount_ <= 0 && lineVertexCount_ <= 0) return;

    shader_.Use();
    shader_.SetMat4("uMVP", mvp.m);

    glDisable(GL_DEPTH_TEST);
    glBindVertexArray(overlayVao_);

    if (markerCount_ > 0) {
        shader_.SetFloat("uPointSize", 14.f);
        shader_.SetFloat("uOpacity", 1.f);
        glDrawArrays(GL_POINTS, 0, markerCount_);
    }
    if (lineVertexCount_ >= 2) {
        shader_.SetFloat("uOpacity", 1.f);
        glLineWidth(3.f);
        glDrawArrays(GL_LINES, markerCount_, lineVertexCount_);
    }

    glBindVertexArray(0);
    glEnable(GL_DEPTH_TEST);
}

void PointCloudRenderer::DrawPlane(const Mat4& mvp) const {
    if (!hasPlane_ || planeVertexCount_ <= 0) return;

    shader_.Use();
    shader_.SetMat4("uMVP", mvp.m);
    shader_.SetFloat("uOpacity", 0.45f);
    shader_.SetFloat("uPointSize", 1.f);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDisable(GL_CULL_FACE);
    glDepthMask(GL_FALSE);
    glEnable(GL_POLYGON_OFFSET_FILL);
    glPolygonOffset(-1.f, -1.f);

    glBindVertexArray(planeVao_);
    glDrawArrays(GL_TRIANGLES, 0, planeVertexCount_);
    glBindVertexArray(0);

    glDisable(GL_POLYGON_OFFSET_FILL);
    glDepthMask(GL_TRUE);
}

void PointCloudRenderer::DrawFitWire(const Mat4& mvp) const {
    if (fitWireVertexCount_ < 2) return;
    shader_.Use();
    shader_.SetMat4("uMVP", mvp.m);
    shader_.SetFloat("uOpacity", 1.f);
    glDisable(GL_DEPTH_TEST);
    glLineWidth(2.5f);
    glBindVertexArray(fitWireVao_);
    glDrawArrays(GL_LINES, 0, fitWireVertexCount_);
    glBindVertexArray(0);
    glEnable(GL_DEPTH_TEST);
}

void PointCloudRenderer::Draw(const Camera& camera, int fbWidth, int fbHeight, float pointSize,
                              float opacity) const {
    if (fbWidth <= 0 || fbHeight <= 0) return;

    const float aspect = static_cast<float>(fbWidth) / static_cast<float>(fbHeight);
    const Mat4 mvp = camera.ProjMatrix(aspect) * camera.ViewMatrix();

    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    if (vertexCount_ > 0) {
        shader_.Use();
        shader_.SetMat4("uMVP", mvp.m);
        shader_.SetFloat("uPointSize", pointSize);
        shader_.SetFloat("uOpacity", opacity);
        glBindVertexArray(vao_);
        glDrawArrays(GL_POINTS, 0, vertexCount_);
        glBindVertexArray(0);
    }

    DrawAxes(mvp);
    DrawPlane(mvp);
    DrawFitWire(mvp);
    DrawMarkersAndLines(mvp);
}

void PointCloudRenderer::Shutdown() {
    if (vbo_) glDeleteBuffers(1, &vbo_);
    if (vao_) glDeleteVertexArrays(1, &vao_);
    if (overlayVbo_) glDeleteBuffers(1, &overlayVbo_);
    if (overlayVao_) glDeleteVertexArrays(1, &overlayVao_);
    if (planeVbo_) glDeleteBuffers(1, &planeVbo_);
    if (planeVao_) glDeleteVertexArrays(1, &planeVao_);
    if (fitWireVbo_) glDeleteBuffers(1, &fitWireVbo_);
    if (fitWireVao_) glDeleteVertexArrays(1, &fitWireVao_);
    if (axesVbo_) glDeleteBuffers(1, &axesVbo_);
    if (axesVao_) glDeleteVertexArrays(1, &axesVao_);
    vbo_ = vao_ = overlayVbo_ = overlayVao_ = planeVbo_ = planeVao_ = axesVbo_ = axesVao_ = 0;
    fitWireVbo_ = fitWireVao_ = 0;
    vertexCount_ = markerCount_ = lineVertexCount_ = planeVertexCount_ = fitWireVertexCount_ = 0;
    hasPlane_ = false;
}
