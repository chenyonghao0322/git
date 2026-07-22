#pragma once

// PointCloudRenderer — OpenGL 点云渲染
//
// 将 PointCloud 上传到 GPU，支持 Z 着色、逐点颜色、ROI 高亮、拟合线框叠加等。

#include "core/MathTypes.h"
#include "core/PointCloud.h"
#include "render/Camera.h"
#include "render/Shader.h"
#include "tools/MeasureTools.h"

#include <optional>
#include <vector>

struct UploadParams {
    const std::vector<std::size_t>* highlightRoi = nullptr;
    int maxDisplayPoints = 1500000;
    float zMin = 0.f;
    float zMax = 1.f;
    bool usePointColors = false;   // 使用 cloud.colors
    bool ignoreMask = false;       // 滤波对比：同时显示被滤除点
};


class PointCloudRenderer {
public:
    bool Init(std::string& error);
    int Upload(const PointCloud& cloud, const UploadParams& params,
               std::vector<std::size_t>* outDisplayIndices = nullptr);  // 上传 GPU，返回实际绘制点数
    void SetDistanceOverlay(const std::optional<Vec3>& a, const std::optional<Vec3>& b);  // 测距两点
    void SetPickOverlay(const std::optional<Vec3>& p);       // 点选高亮
    void SetPlaneOverlay(const std::optional<PlaneModel>& plane);
    void SetSphereOverlay(const std::optional<SphereModel>& sphere);
    void SetCircleOverlay(const std::optional<CircleModel>& circle);
    void SetCylinderOverlay(const std::optional<CylinderModel>& cylinder);
    void ClearFitWireOverlay();
    void SetAxes(bool enabled, float axisLength);            // 世界坐标轴
    void Draw(const Camera& camera, int fbWidth, int fbHeight, float pointSize, float opacity) const;
    void Shutdown();
    int DisplayedCount() const { return vertexCount_; }

private:
    void DrawMarkersAndLines(const Mat4& mvp) const;
    void DrawPlane(const Mat4& mvp) const;
    void DrawFitWire(const Mat4& mvp) const;
    void DrawAxes(const Mat4& mvp) const;
    void RebuildPlaneMesh(const PlaneModel& plane);
    void RebuildAxesMesh(float axisLength);
    void UploadFitWire(const std::vector<float>& posRgb);  // interleaved x,y,z,r,g,b

    unsigned int vao_ = 0;
    unsigned int vbo_ = 0;
    int vertexCount_ = 0;
    Shader shader_;

    unsigned int overlayVao_ = 0;
    unsigned int overlayVbo_ = 0;
    int markerCount_ = 0;
    int lineVertexCount_ = 0;

    unsigned int planeVao_ = 0;
    unsigned int planeVbo_ = 0;
    int planeVertexCount_ = 0;
    bool hasPlane_ = false;

    unsigned int fitWireVao_ = 0;
    unsigned int fitWireVbo_ = 0;
    int fitWireVertexCount_ = 0;

    unsigned int axesVao_ = 0;
    unsigned int axesVbo_ = 0;
    bool axesEnabled_ = true;
    float axesLength_ = 1.f;
};
