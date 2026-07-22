#pragma once

// OpenCv2D — 2D 图像算子（卡尺提线等，类似 Halcon/VisionPro）

#include <cstdint>
#include <string>
#include <vector>

namespace OpenCv2D {

struct LineSegment {
    float x1 = 0.f;
    float y1 = 0.f;
    float x2 = 0.f;
    float y2 = 0.f;
};

enum class EdgePolarity {
    All = 0,         // 最大梯度幅值
    DarkToLight = 1, // 由暗到亮
    LightToDark = 2, // 由亮到暗
};

struct CaliperLineParams {
    int numCalipers = 20;        // 垂直测量线数量
    float caliperHalfLength = 30.f;  // 每条测量线半长（像素）
    int caliperWidth = 3;        // 沿测量方向平均宽度（像素）
    float minContrast = 1.f;     // 最小梯度阈值
    EdgePolarity polarity = EdgePolarity::All;
    bool skipZero = true;        // 深度图跳过 0 值
};

struct CaliperEdgePoint {
    float x = 0.f;
    float y = 0.f;
    bool valid = false;
};

struct CaliperLineResult {
    float roiX0 = 0.f;
    float roiY0 = 0.f;
    float roiX1 = 0.f;
    float roiY1 = 0.f;
    float fitX1 = 0.f;
    float fitY1 = 0.f;
    float fitX2 = 0.f;
    float fitY2 = 0.f;
    std::vector<CaliperEdgePoint> edgePoints;
    std::vector<LineSegment> calipers;
    int validCount = 0;
    float fitRms = 0.f;
    bool ok = false;
};

// 在浮点灰度图上执行卡尺提线；roi 为测量方向线（像素坐标，左上原点）
bool MeasureLineWithCalipers(const std::vector<float>& gray, int width, int height,
                             float roiX0, float roiY0, float roiX1, float roiY1,
                             const CaliperLineParams& params, CaliperLineResult& result,
                             std::string& error);

// 从 RGB8 转灰度后执行卡尺提线
bool MeasureLineWithCalipersRgb(const std::vector<uint8_t>& rgb, int width, int height,
                                float roiX0, float roiY0, float roiX1, float roiY1,
                                const CaliperLineParams& params, CaliperLineResult& result,
                                std::string& error);

struct CaliperArcResult {
    float roiP0X = 0.f;
    float roiP0Y = 0.f;
    float roiP1X = 0.f;
    float roiP1Y = 0.f;
    float roiP2X = 0.f;
    float roiP2Y = 0.f;
    float roiCenterX = 0.f;
    float roiCenterY = 0.f;
    float roiRadius = 0.f;
    float roiStartAngle = 0.f;
    float roiEndAngle = 0.f;
    float fitCenterX = 0.f;
    float fitCenterY = 0.f;
    float fitRadius = 0.f;
    float fitStartAngle = 0.f;
    float fitEndAngle = 0.f;
    std::vector<CaliperEdgePoint> edgePoints;
    std::vector<LineSegment> calipers;
    int validCount = 0;
    float fitRms = 0.f;
    bool ok = false;
};

using CaliperArcParams = CaliperLineParams;

// 三点定义圆弧 ROI，沿弧布置径向卡尺并拟合圆
bool MeasureArcWithCalipers(const std::vector<float>& gray, int width, int height, float roiP0X,
                            float roiP0Y, float roiP1X, float roiP1Y, float roiP2X, float roiP2Y,
                            const CaliperArcParams& params, CaliperArcResult& result,
                            std::string& error);

bool MeasureArcWithCalipersRgb(const std::vector<uint8_t>& rgb, int width, int height, float roiP0X,
                               float roiP0Y, float roiP1X, float roiP1Y, float roiP2X, float roiP2Y,
                               const CaliperArcParams& params, CaliperArcResult& result,
                               std::string& error);

// 生成圆弧折线点（像素坐标），用于绘制
void SampleArcPolyline(float cx, float cy, float radius, float startAngle, float endAngle,
                       int segments, std::vector<float>& outX, std::vector<float>& outY);

// 三点定圆 / 过中间点的弧跨度（用于 ROI 预览）
bool CircleFromThreePoints(float x1, float y1, float x2, float y2, float x3, float y3, float& cx,
                           float& cy, float& r);
bool ArcSpanThroughMiddle(float cx, float cy, float x0, float y0, float x1, float y1, float xm,
                          float ym, float& startAngle, float& endAngle);

// 两线段最短距离（像素）；可选输出最近点
float SegmentSegmentDistance(float ax1, float ay1, float ax2, float ay2, float bx1, float by1,
                             float bx2, float by2, float* closestAx = nullptr,
                             float* closestAy = nullptr, float* closestBx = nullptr,
                             float* closestBy = nullptr);

struct GapSample {
    float ax = 0.f;
    float ay = 0.f;
    float bx = 0.f;
    float by = 0.f;
    float dist = 0.f;
};

struct AverageGapResult {
    float average = 0.f;
    float minDist = 0.f;
    float maxDist = 0.f;
    int validCount = 0;
    int totalSamples = 0;
    std::vector<GapSample> samples;  // 仅含有效交点
};

// 线段 A 上均匀采样，过采样点作垂直于 A 的直线与线段 B 求交；无交点不计入，有效距离取平均
bool AverageGapDistance(float ax1, float ay1, float ax2, float ay2, float bx1, float by1,
                        float bx2, float by2, int numSamples, AverageGapResult& out);

// 圆弧 A 上均匀采样，过采样点作法向（径向）直线与圆弧 B 求交；无交点不计入，有效距离取平均
bool AverageArcGapDistance(float acx, float acy, float ar, float aStart, float aEnd, float bcx,
                           float bcy, float br, float bStart, float bEnd, int numSamples,
                           AverageGapResult& out);

struct CircleFitResult {
    float centerX = 0.f;
    float centerY = 0.f;
    float radius = 0.f;
    float rms = 0.f;
    int pointCount = 0;
    bool ok = false;
};

// 由有效边缘点最小二乘拟合圆
bool FitCircleFromEdgePoints(const std::vector<CaliperEdgePoint>& points, CircleFitResult& result);

struct ArcMetrics {
    float arcLength = 0.f;
    float chordLength = 0.f;
    float sagitta = 0.f;
    float spanRadians = 0.f;
};

// 由圆弧参数计算弧长、弦长、弓高
bool ComputeArcMetrics(float cx, float cy, float radius, float startAngle, float endAngle,
                       ArcMetrics& out);

float PointPointDistance(float x1, float y1, float x2, float y2, float* outDx = nullptr,
                         float* outDy = nullptr);

// 两线段夹角（度）；acuteOnly=true 时返回 0~90°
float AngleBetweenSegments(float ax1, float ay1, float ax2, float ay2, float bx1, float by1,
                           float bx2, float by2, bool acuteOnly = true);

// 点到线段垂直距离，输出垂足
bool PointToSegmentDistance(float px, float py, float x1, float y1, float x2, float y2,
                            float& outDist, float& footX, float& footY);

struct CircleGapResult {
    float centerDistance = 0.f;
    float surfaceGap = 0.f;  // 圆心距 - r1 - r2（负值表示相交）
};

bool ComputeCircleGap(float cx1, float cy1, float r1, float cx2, float cy2, float r2,
                      CircleGapResult& out);

struct CaliperCircleResult {
    float roiCenterX = 0.f;
    float roiCenterY = 0.f;
    float roiRadius = 0.f;
    float fitCenterX = 0.f;
    float fitCenterY = 0.f;
    float fitRadius = 0.f;
    std::vector<CaliperEdgePoint> edgePoints;
    std::vector<LineSegment> calipers;
    int validCount = 0;
    float fitRms = 0.f;
    bool ok = false;
};

using CaliperCircleParams = CaliperLineParams;

// 沿整圆 ROI 布置径向卡尺并拟合圆
bool MeasureCircleWithCalipers(const std::vector<float>& gray, int width, int height, float roiCenterX,
                               float roiCenterY, float roiRadius, const CaliperCircleParams& params,
                               CaliperCircleResult& result, std::string& error);

bool MeasureCircleWithCalipersRgb(const std::vector<uint8_t>& rgb, int width, int height,
                                  float roiCenterX, float roiCenterY, float roiRadius,
                                  const CaliperCircleParams& params, CaliperCircleResult& result,
                                  std::string& error);

// 线段 B 中点到线段 A 的垂直距离（平行线间距）
float ParallelLineDistance(float ax1, float ay1, float ax2, float ay2, float bx1, float by1,
                           float bx2, float by2);

struct PointProjectionResult {
    float footX = 0.f;
    float footY = 0.f;
    float perpDist = 0.f;
    float alongT = 0.f;  // 在线段上的参数 0~1
};

bool ProjectPointOntoSegment(float px, float py, float x1, float y1, float x2, float y2,
                             PointProjectionResult& out);

struct CaliperRectResult {
    float roiX0 = 0.f;
    float roiY0 = 0.f;
    float roiX1 = 0.f;
    float roiY1 = 0.f;
    float centerX = 0.f;
    float centerY = 0.f;
    float width = 0.f;
    float height = 0.f;
    float angleDeg = 0.f;
    std::vector<CaliperEdgePoint> edgePoints;
    std::vector<LineSegment> calipers;
    int validCount = 0;
    bool ok = false;
};

bool MeasureRectWithCalipers(const std::vector<float>& gray, int width, int height, float roiX0,
                             float roiY0, float roiX1, float roiY1, const CaliperLineParams& params,
                             CaliperRectResult& result, std::string& error);

bool MeasureRectWithCalipersRgb(const std::vector<uint8_t>& rgb, int width, int height, float roiX0,
                                float roiY0, float roiX1, float roiY1,
                                const CaliperLineParams& params, CaliperRectResult& result,
                                std::string& error);

struct EllipseFitResult {
    float centerX = 0.f;
    float centerY = 0.f;
    float axisA = 0.f;  // 长半轴
    float axisB = 0.f;  // 短半轴
    float angleDeg = 0.f;
    float rms = 0.f;
    int pointCount = 0;
    bool ok = false;
};

bool FitEllipseFromEdgePoints(const std::vector<CaliperEdgePoint>& points, EllipseFitResult& result);

struct ProfileWidthResult {
    float roiX0 = 0.f;
    float roiY0 = 0.f;
    float roiX1 = 0.f;
    float roiY1 = 0.f;
    float edge1X = 0.f;
    float edge1Y = 0.f;
    float edge2X = 0.f;
    float edge2Y = 0.f;
    float width = 0.f;
    bool ok = false;
};

bool MeasureProfileWidth(const std::vector<float>& gray, int width, int height, float roiX0,
                         float roiY0, float roiX1, float roiY1, const CaliperLineParams& params,
                         ProfileWidthResult& result, std::string& error);

bool MeasureProfileWidthRgb(const std::vector<uint8_t>& rgb, int width, int height, float roiX0,
                            float roiY0, float roiX1, float roiY1, const CaliperLineParams& params,
                            ProfileWidthResult& result, std::string& error);

struct RoundnessResult {
    float rms = 0.f;
    float maxDev = 0.f;
    float minDev = 0.f;
    bool ok = false;
};

bool ComputeRoundness(float centerX, float centerY, float radius,
                      const std::vector<CaliperEdgePoint>& points, RoundnessResult& out);

struct RegionBlobResult {
    float roiX0 = 0.f;
    float roiY0 = 0.f;
    float roiX1 = 0.f;
    float roiY1 = 0.f;
    float centroidX = 0.f;
    float centroidY = 0.f;
    int pixelCount = 0;
    float areaPx = 0.f;
    bool ok = false;
};

bool ComputeRegionBlob(const std::vector<float>& gray, int width, int height, float roiX0,
                       float roiY0, float roiX1, float roiY1, float threshold, bool greaterThan,
                       RegionBlobResult& out, std::string& error);

bool ComputeRegionBlobRgb(const std::vector<uint8_t>& rgb, int width, int height, float roiX0,
                          float roiY0, float roiX1, float roiY1, float threshold, bool greaterThan,
                          RegionBlobResult& out, std::string& error);

struct ConcentricityResult {
    float offsetX = 0.f;
    float offsetY = 0.f;
    float offsetDist = 0.f;
};

bool ComputeConcentricity(float cx1, float cy1, float cx2, float cy2, ConcentricityResult& out);

struct LineProfileSample {
    float x = 0.f;
    float y = 0.f;
    float value = 0.f;
};

bool SampleLineProfile(const std::vector<float>& gray, int width, int height, float x0, float y0,
                       float x1, float y1, int numSamples, std::vector<LineProfileSample>& out,
                       bool skipZero);

}  // namespace OpenCv2D
