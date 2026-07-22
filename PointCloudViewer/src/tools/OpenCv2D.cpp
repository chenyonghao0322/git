#include "tools/OpenCv2D.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace OpenCv2D {
namespace {

float SampleBilinear(const std::vector<float>& gray, int width, int height, float x, float y,
                     bool skipZero) {
    if (width <= 0 || height <= 0 || gray.empty()) return 0.f;
    x = std::clamp(x, 0.f, static_cast<float>(width - 1));
    y = std::clamp(y, 0.f, static_cast<float>(height - 1));
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const int x1 = std::min(x0 + 1, width - 1);
    const int y1 = std::min(y0 + 1, height - 1);
    const float fx = x - static_cast<float>(x0);
    const float fy = y - static_cast<float>(y0);

    auto at = [&](int c, int r) -> float {
        const float v = gray[static_cast<std::size_t>(r) * static_cast<std::size_t>(width) +
                             static_cast<std::size_t>(c)];
        if (skipZero && v == 0.f) return std::numeric_limits<float>::quiet_NaN();
        return v;
    };

    const float v00 = at(x0, y0);
    const float v10 = at(x1, y0);
    const float v01 = at(x0, y1);
    const float v11 = at(x1, y1);

    float sum = 0.f;
    float wsum = 0.f;
    const float w00 = (1.f - fx) * (1.f - fy);
    const float w10 = fx * (1.f - fy);
    const float w01 = (1.f - fx) * fy;
    const float w11 = fx * fy;
    const float vals[4] = {v00, v10, v01, v11};
    const float ws[4] = {w00, w10, w01, w11};
    for (int i = 0; i < 4; ++i) {
        if (!std::isnan(vals[i])) {
            sum += vals[i] * ws[i];
            wsum += ws[i];
        }
    }
    if (wsum < 1e-6f) return 0.f;
    return sum / wsum;
}

float SampleAveraged(const std::vector<float>& gray, int width, int height, float cx, float cy,
                     float dx, float dy, int halfWidth, bool skipZero) {
    if (halfWidth < 0) halfWidth = 0;
    float sum = 0.f;
    int count = 0;
    for (int k = -halfWidth; k <= halfWidth; ++k) {
        const float v =
            SampleBilinear(gray, width, height, cx + dx * static_cast<float>(k),
                           cy + dy * static_cast<float>(k), skipZero);
        if (skipZero && v == 0.f) continue;
        sum += v;
        ++count;
    }
    return (count > 0) ? (sum / static_cast<float>(count)) : 0.f;
}

void RgbToGrayFloat(const std::vector<uint8_t>& rgb, int width, int height,
                    std::vector<float>& outGray) {
    const std::size_t n = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    outGray.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t bi = i * 3u;
        const float r = static_cast<float>(rgb[bi]);
        const float g = static_cast<float>(rgb[bi + 1]);
        const float b = static_cast<float>(rgb[bi + 2]);
        outGray[i] = 0.299f * r + 0.587f * g + 0.114f * b;
    }
}

bool MeasureLineWithCalipersImpl(const std::vector<float>& gray, int width, int height,
                                 float roiX0, float roiY0, float roiX1, float roiY1,
                                 const CaliperLineParams& params, CaliperLineResult& result,
                                 std::string& error) {
    result = {};
    result.roiX0 = roiX0;
    result.roiY0 = roiY0;
    result.roiX1 = roiX1;
    result.roiY1 = roiY1;

    if (gray.empty() || width <= 0 || height <= 0 ||
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height) != gray.size()) {
        error = u8"图像数据无效";
        return false;
    }

    const float dx = roiX1 - roiX0;
    const float dy = roiY1 - roiY0;
    const float lineLen = std::sqrt(dx * dx + dy * dy);
    if (lineLen < 2.f) {
        error = u8"测量方向线过短，请重新拖拽";
        return false;
    }

    const int numCalipers = std::max(2, params.numCalipers);
    const float halfLen = std::max(2.f, params.caliperHalfLength);
    const int halfWidth = std::max(0, params.caliperWidth / 2);
    const int profileSteps = std::max(3, static_cast<int>(std::ceil(halfLen * 2.f)) + 1);

    const float dirX = dx / lineLen;
    const float dirY = dy / lineLen;
    const float normX = -dirY;
    const float normY = dirX;

    std::vector<cv::Point2f> fitPts;
    result.edgePoints.resize(static_cast<std::size_t>(numCalipers));
    result.calipers.resize(static_cast<std::size_t>(numCalipers));

    for (int i = 0; i < numCalipers; ++i) {
        const float t = (numCalipers == 1) ? 0.5f
                                           : static_cast<float>(i) / static_cast<float>(numCalipers - 1);
        const float cx = roiX0 + dx * t;
        const float cy = roiY0 + dy * t;

        LineSegment cal;
        cal.x1 = cx - normX * halfLen;
        cal.y1 = cy - normY * halfLen;
        cal.x2 = cx + normX * halfLen;
        cal.y2 = cy + normY * halfLen;
        result.calipers[static_cast<std::size_t>(i)] = cal;

        std::vector<float> profile(static_cast<std::size_t>(profileSteps));
        for (int j = 0; j < profileSteps; ++j) {
            const float s = -halfLen + (2.f * halfLen * static_cast<float>(j)) /
                                           static_cast<float>(profileSteps - 1);
            profile[static_cast<std::size_t>(j)] =
                SampleAveraged(gray, width, height, cx + normX * s, cy + normY * s, dirX, dirY,
                               halfWidth, params.skipZero);
        }

        int bestIdx = -1;
        float bestScore = -1.f;
        for (int j = 1; j < profileSteps - 1; ++j) {
            const float g = 0.5f * (profile[static_cast<std::size_t>(j + 1)] -
                                    profile[static_cast<std::size_t>(j - 1)]);
            float score = 0.f;
            switch (params.polarity) {
                case EdgePolarity::DarkToLight:
                    score = g;
                    break;
                case EdgePolarity::LightToDark:
                    score = -g;
                    break;
                case EdgePolarity::All:
                default:
                    score = std::abs(g);
                    break;
            }
            if (score > bestScore) {
                bestScore = score;
                bestIdx = j;
            }
        }

        CaliperEdgePoint& ep = result.edgePoints[static_cast<std::size_t>(i)];
        if (bestIdx < 0 || bestScore < params.minContrast) {
            ep.valid = false;
            continue;
        }

        const float s0 = -halfLen + (2.f * halfLen * static_cast<float>(bestIdx - 1)) /
                                        static_cast<float>(profileSteps - 1);
        const float s1 = -halfLen + (2.f * halfLen * static_cast<float>(bestIdx)) /
                                        static_cast<float>(profileSteps - 1);
        const float s2 = -halfLen + (2.f * halfLen * static_cast<float>(bestIdx + 1)) /
                                        static_cast<float>(profileSteps - 1);
        const float g0 = profile[static_cast<std::size_t>(bestIdx - 1)];
        const float g1 = profile[static_cast<std::size_t>(bestIdx)];
        const float g2 = profile[static_cast<std::size_t>(bestIdx + 1)];
        float subS = s1;
        const float denom = g0 - 2.f * g1 + g2;
        if (std::abs(denom) > 1e-6f) {
            subS = s1 + 0.5f * (g0 - g2) / denom;
            subS = std::clamp(subS, s0, s2);
        }

        ep.x = cx + normX * subS;
        ep.y = cy + normY * subS;
        ep.valid = true;
        fitPts.emplace_back(ep.x, ep.y);
    }

    result.validCount = static_cast<int>(fitPts.size());
    if (fitPts.size() < 2) {
        error = u8"有效边缘点不足，请调整卡尺参数或测量位置";
        return false;
    }

    cv::Vec4f line;
    cv::fitLine(fitPts, line, cv::DIST_L2, 0, 0.01, 0.01);
    const float vx = line[0];
    const float vy = line[1];
    const float px = line[2];
    const float py = line[3];

    float minProj = std::numeric_limits<float>::max();
    float maxProj = std::numeric_limits<float>::lowest();
    for (const cv::Point2f& p : fitPts) {
        const float proj = (p.x - px) * vx + (p.y - py) * vy;
        minProj = std::min(minProj, proj);
        maxProj = std::max(maxProj, proj);
    }

    result.fitX1 = px + vx * minProj;
    result.fitY1 = py + vy * minProj;
    result.fitX2 = px + vx * maxProj;
    result.fitY2 = py + vy * maxProj;

    float sqErr = 0.f;
    for (const cv::Point2f& p : fitPts) {
        const float cross = vx * (p.y - py) - vy * (p.x - px);
        sqErr += cross * cross;
    }
    result.fitRms = std::sqrt(sqErr / static_cast<float>(fitPts.size()));
    result.ok = true;
    return true;
}

}  // namespace

bool MeasureLineWithCalipers(const std::vector<float>& gray, int width, int height, float roiX0,
                             float roiY0, float roiX1, float roiY1, const CaliperLineParams& params,
                             CaliperLineResult& result, std::string& error) {
    return MeasureLineWithCalipersImpl(gray, width, height, roiX0, roiY0, roiX1, roiY1, params,
                                       result, error);
}

bool MeasureLineWithCalipersRgb(const std::vector<uint8_t>& rgb, int width, int height, float roiX0,
                                float roiY0, float roiX1, float roiY1,
                                const CaliperLineParams& params, CaliperLineResult& result,
                                std::string& error) {
    if (rgb.empty() || width <= 0 || height <= 0 ||
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3u != rgb.size()) {
        error = u8"图像数据无效";
        return false;
    }
    std::vector<float> gray;
    RgbToGrayFloat(rgb, width, height, gray);
    CaliperLineParams p = params;
    p.skipZero = false;
    return MeasureLineWithCalipersImpl(gray, width, height, roiX0, roiY0, roiX1, roiY1, p, result,
                                       error);
}

namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kTwoPi = kPi * 2.f;

float NormAnglePos(float a) {
    while (a < 0.f) a += kTwoPi;
    while (a >= kTwoPi) a -= kTwoPi;
    return a;
}

float DeltaCCW(float from, float to) {
    return NormAnglePos(to - from);
}

bool CircleFromThreePoints(float x1, float y1, float x2, float y2, float x3, float y3, float& cx,
                           float& cy, float& r) {
    const float d = 2.f * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2));
    if (std::fabs(d) < 1e-6f) return false;
    const float x1s = x1 * x1 + y1 * y1;
    const float x2s = x2 * x2 + y2 * y2;
    const float x3s = x3 * x3 + y3 * y3;
    cx = (x1s * (y2 - y3) + x2s * (y3 - y1) + x3s * (y1 - y2)) / d;
    cy = (x1s * (x3 - x2) + x2s * (x1 - x3) + x3s * (x2 - x1)) / d;
    const float dx = x1 - cx;
    const float dy = y1 - cy;
    r = std::sqrt(dx * dx + dy * dy);
    return r > 1e-3f;
}

bool ArcSpanThroughMiddle(float cx, float cy, float x0, float y0, float x1, float y1, float xm,
                          float ym, float& startAngle, float& endAngle) {
    const float a0 = std::atan2(y0 - cy, x0 - cx);
    const float a1 = std::atan2(y1 - cy, x1 - cx);
    const float am = std::atan2(ym - cy, xm - cx);
    const float d0m = DeltaCCW(a0, am);
    const float d01 = DeltaCCW(a0, a1);
    if (d0m <= d01) {
        startAngle = a0;
        endAngle = a0 + d01;
    } else {
        startAngle = a1;
        endAngle = a1 + DeltaCCW(a1, a0);
    }
    return (endAngle - startAngle) > 1e-4f;
}

bool FindEdgeOnProfile(const std::vector<float>& profile, float halfLen, int profileSteps,
                       const CaliperLineParams& params, float cx, float cy, float normX,
                       float normY, float& outX, float& outY) {
    int bestIdx = -1;
    float bestScore = -1.f;
    for (int j = 1; j < profileSteps - 1; ++j) {
        const float g =
            0.5f * (profile[static_cast<std::size_t>(j + 1)] - profile[static_cast<std::size_t>(j - 1)]);
        float score = 0.f;
        switch (params.polarity) {
            case EdgePolarity::DarkToLight:
                score = g;
                break;
            case EdgePolarity::LightToDark:
                score = -g;
                break;
            case EdgePolarity::All:
            default:
                score = std::abs(g);
                break;
        }
        if (score > bestScore) {
            bestScore = score;
            bestIdx = j;
        }
    }
    if (bestIdx < 0 || bestScore < params.minContrast) return false;

    const float s0 = -halfLen + (2.f * halfLen * static_cast<float>(bestIdx - 1)) /
                                    static_cast<float>(profileSteps - 1);
    const float s1 = -halfLen + (2.f * halfLen * static_cast<float>(bestIdx)) /
                                    static_cast<float>(profileSteps - 1);
    const float s2 = -halfLen + (2.f * halfLen * static_cast<float>(bestIdx + 1)) /
                                    static_cast<float>(profileSteps - 1);
    const float g0 = profile[static_cast<std::size_t>(bestIdx - 1)];
    const float g1 = profile[static_cast<std::size_t>(bestIdx)];
    const float g2 = profile[static_cast<std::size_t>(bestIdx + 1)];
    float subS = s1;
    const float denom = g0 - 2.f * g1 + g2;
    if (std::abs(denom) > 1e-6f) {
        subS = s1 + 0.5f * (g0 - g2) / denom;
        subS = std::clamp(subS, s0, s2);
    }

    outX = cx + normX * subS;
    outY = cy + normY * subS;
    return true;
}

bool FitCircleLeastSquares(const std::vector<cv::Point2f>& pts, float& cx, float& cy, float& r,
                           float& rms) {
    if (pts.size() < 3) return false;
    double sx = 0.0;
    double sy = 0.0;
    double sxx = 0.0;
    double syy = 0.0;
    double sxy = 0.0;
    double sx3 = 0.0;
    double sy3 = 0.0;
    double sxy2 = 0.0;
    double sx2y = 0.0;
    const double n = static_cast<double>(pts.size());
    for (const cv::Point2f& p : pts) {
        const double x = p.x;
        const double y = p.y;
        const double x2 = x * x;
        const double y2 = y * y;
        sx += x;
        sy += y;
        sxx += x2;
        syy += y2;
        sxy += x * y;
        sx3 += x2 * x;
        sy3 += y2 * y;
        sxy2 += x * y2;
        sx2y += x2 * y;
    }
    const double a = n * sxx - sx * sx;
    const double b = n * sxy - sx * sy;
    const double c = n * syy - sy * sy;
    const double d = 0.5 * (n * sx3 + n * sxy2 - sx * (sxx + syy));
    const double e = 0.5 * (n * sy3 + n * sx2y - sy * (sxx + syy));
    const double det = a * c - b * b;
    if (std::fabs(det) < 1e-9) return false;
    cx = static_cast<float>((d * c - b * e) / det);
    cy = static_cast<float>((a * e - b * d) / det);

  double sumR = 0.0;
    double sq = 0.0;
    for (const cv::Point2f& p : pts) {
        const double dx = p.x - cx;
        const double dy = p.y - cy;
        const double ri = std::sqrt(dx * dx + dy * dy);
        sumR += ri;
        sq += ri * ri;
    }
    r = static_cast<float>(sumR / n);
    const double meanSq = sq / n;
    const double rMean = static_cast<double>(r);
    rms = static_cast<float>(std::sqrt(std::max(0.0, meanSq - rMean * rMean)));
    return r > 1e-3f;
}

bool MeasureArcWithCalipersImpl(const std::vector<float>& gray, int width, int height, float roiP0X,
                                float roiP0Y, float roiP1X, float roiP1Y, float roiP2X,
                                float roiP2Y, const CaliperArcParams& params,
                                CaliperArcResult& result, std::string& error) {
    result = {};
    result.roiP0X = roiP0X;
    result.roiP0Y = roiP0Y;
    result.roiP1X = roiP1X;
    result.roiP1Y = roiP1Y;
    result.roiP2X = roiP2X;
    result.roiP2Y = roiP2Y;

    if (gray.empty() || width <= 0 || height <= 0) {
        error = u8"图像数据无效";
        return false;
    }

    float cx = 0.f;
    float cy = 0.f;
    float radius = 0.f;
    if (!CircleFromThreePoints(roiP0X, roiP0Y, roiP1X, roiP1Y, roiP2X, roiP2Y, cx, cy, radius)) {
        error = u8"三点近乎共线，请沿圆弧轨迹拖拽拉出弧度";
        return false;
    }

    float startAngle = 0.f;
    float endAngle = 0.f;
    if (!ArcSpanThroughMiddle(cx, cy, roiP0X, roiP0Y, roiP1X, roiP1Y, roiP2X, roiP2Y, startAngle,
                              endAngle)) {
        error = u8"圆弧跨度无效";
        return false;
    }

    result.roiCenterX = cx;
    result.roiCenterY = cy;
    result.roiRadius = radius;
    result.roiStartAngle = startAngle;
    result.roiEndAngle = endAngle;

    const int numCalipers = std::max(3, params.numCalipers);
    const float halfLen = std::max(2.f, params.caliperHalfLength);
    const int halfWidth = std::max(0, params.caliperWidth / 2);
    const int profileSteps = std::max(3, static_cast<int>(std::ceil(halfLen * 2.f)) + 1);
    const float span = endAngle - startAngle;

    std::vector<cv::Point2f> fitPts;
    result.edgePoints.resize(static_cast<std::size_t>(numCalipers));
    result.calipers.resize(static_cast<std::size_t>(numCalipers));

    for (int i = 0; i < numCalipers; ++i) {
        const float t = (numCalipers == 1)
                            ? 0.5f
                            : static_cast<float>(i) / static_cast<float>(numCalipers - 1);
        const float ang = startAngle + span * t;
        const float cosA = std::cos(ang);
        const float sinA = std::sin(ang);
        const float px = cx + radius * cosA;
        const float py = cy + radius * sinA;
        const float normX = cosA;
        const float normY = sinA;
        const float tanX = -sinA;
        const float tanY = cosA;

        LineSegment cal;
        cal.x1 = px - normX * halfLen;
        cal.y1 = py - normY * halfLen;
        cal.x2 = px + normX * halfLen;
        cal.y2 = py + normY * halfLen;
        result.calipers[static_cast<std::size_t>(i)] = cal;

        std::vector<float> profile(static_cast<std::size_t>(profileSteps));
        for (int j = 0; j < profileSteps; ++j) {
            const float s = -halfLen + (2.f * halfLen * static_cast<float>(j)) /
                                           static_cast<float>(profileSteps - 1);
            profile[static_cast<std::size_t>(j)] =
                SampleAveraged(gray, width, height, px + normX * s, py + normY * s, tanX, tanY,
                               halfWidth, params.skipZero);
        }

        CaliperEdgePoint& ep = result.edgePoints[static_cast<std::size_t>(i)];
        float ex = 0.f;
        float ey = 0.f;
        if (!FindEdgeOnProfile(profile, halfLen, profileSteps, params, px, py, normX, normY, ex,
                               ey)) {
            ep.valid = false;
            continue;
        }
        ep.x = ex;
        ep.y = ey;
        ep.valid = true;
        fitPts.emplace_back(ex, ey);
    }

    result.validCount = static_cast<int>(fitPts.size());
    if (fitPts.size() < 3) {
        error = u8"有效边缘点不足，请调整卡尺参数或圆弧位置";
        return false;
    }

    if (!FitCircleLeastSquares(fitPts, result.fitCenterX, result.fitCenterY, result.fitRadius,
                               result.fitRms)) {
        error = u8"圆弧拟合失败";
        return false;
    }

    const float fa0 = std::atan2(fitPts.front().y - result.fitCenterY, fitPts.front().x - result.fitCenterX);
    const float fa1 = std::atan2(fitPts.back().y - result.fitCenterY, fitPts.back().x - result.fitCenterX);
    const float fam = std::atan2(fitPts[fitPts.size() / 2].y - result.fitCenterY,
                                 fitPts[fitPts.size() / 2].x - result.fitCenterX);
    ArcSpanThroughMiddle(result.fitCenterX, result.fitCenterY, fitPts.front().x, fitPts.front().y,
                         fitPts.back().x, fitPts.back().y,
                         fitPts[fitPts.size() / 2].x, fitPts[fitPts.size() / 2].y,
                         result.fitStartAngle, result.fitEndAngle);
    (void)fa0;
    (void)fa1;
    (void)fam;

    result.ok = true;
    return true;
}

bool MeasureCircleWithCalipersImpl(const std::vector<float>& gray, int width, int height,
                                   float roiCenterX, float roiCenterY, float roiRadius,
                                   const CaliperCircleParams& params, CaliperCircleResult& result,
                                   std::string& error) {
    result = {};
    result.roiCenterX = roiCenterX;
    result.roiCenterY = roiCenterY;
    result.roiRadius = roiRadius;

    if (gray.empty() || width <= 0 || height <= 0) {
        error = u8"图像数据无效";
        return false;
    }
    if (roiRadius < 3.f) {
        error = u8"圆 ROI 半径过小，请重新拖拽";
        return false;
    }

    const int numCalipers = std::max(8, params.numCalipers);
    const float halfLen = std::max(2.f, params.caliperHalfLength);
    const int halfWidth = std::max(0, params.caliperWidth / 2);
    const int profileSteps = std::max(3, static_cast<int>(std::ceil(halfLen * 2.f)) + 1);

    std::vector<cv::Point2f> fitPts;
    result.edgePoints.resize(static_cast<std::size_t>(numCalipers));
    result.calipers.resize(static_cast<std::size_t>(numCalipers));

    for (int i = 0; i < numCalipers; ++i) {
        const float ang = kTwoPi * static_cast<float>(i) / static_cast<float>(numCalipers);
        const float cosA = std::cos(ang);
        const float sinA = std::sin(ang);
        const float px = roiCenterX + roiRadius * cosA;
        const float py = roiCenterY + roiRadius * sinA;
        const float normX = cosA;
        const float normY = sinA;
        const float tanX = -sinA;
        const float tanY = cosA;

        LineSegment cal;
        cal.x1 = px - normX * halfLen;
        cal.y1 = py - normY * halfLen;
        cal.x2 = px + normX * halfLen;
        cal.y2 = py + normY * halfLen;
        result.calipers[static_cast<std::size_t>(i)] = cal;

        std::vector<float> profile(static_cast<std::size_t>(profileSteps));
        for (int j = 0; j < profileSteps; ++j) {
            const float s = -halfLen + (2.f * halfLen * static_cast<float>(j)) /
                                           static_cast<float>(profileSteps - 1);
            profile[static_cast<std::size_t>(j)] =
                SampleAveraged(gray, width, height, px + normX * s, py + normY * s, tanX, tanY,
                               halfWidth, params.skipZero);
        }

        CaliperEdgePoint& ep = result.edgePoints[static_cast<std::size_t>(i)];
        float ex = 0.f;
        float ey = 0.f;
        if (!FindEdgeOnProfile(profile, halfLen, profileSteps, params, px, py, normX, normY, ex,
                               ey)) {
            ep.valid = false;
            continue;
        }
        ep.x = ex;
        ep.y = ey;
        ep.valid = true;
        fitPts.emplace_back(ex, ey);
    }

    result.validCount = static_cast<int>(fitPts.size());
    if (fitPts.size() < 3) {
        error = u8"有效边缘点不足，请调整卡尺参数或圆位置";
        return false;
    }

    if (!FitCircleLeastSquares(fitPts, result.fitCenterX, result.fitCenterY, result.fitRadius,
                               result.fitRms)) {
        error = u8"圆拟合失败";
        return false;
    }

    result.ok = true;
    return true;
}

}  // namespace

bool FitCircleFromEdgePoints(const std::vector<CaliperEdgePoint>& points, CircleFitResult& result) {
    result = {};
    std::vector<cv::Point2f> pts;
    pts.reserve(points.size());
    for (const CaliperEdgePoint& p : points) {
        if (!p.valid) continue;
        pts.emplace_back(p.x, p.y);
    }
    if (pts.size() < 3) return false;

    double sx = 0.0;
    double sy = 0.0;
    double sxx = 0.0;
    double syy = 0.0;
    double sxy = 0.0;
    double sx3 = 0.0;
    double sy3 = 0.0;
    double sxy2 = 0.0;
    double sx2y = 0.0;
    const double n = static_cast<double>(pts.size());
    for (const cv::Point2f& p : pts) {
        const double x = p.x;
        const double y = p.y;
        const double x2 = x * x;
        const double y2 = y * y;
        sx += x;
        sy += y;
        sxx += x2;
        syy += y2;
        sxy += x * y;
        sx3 += x2 * x;
        sy3 += y2 * y;
        sxy2 += x * y2;
        sx2y += x2 * y;
    }
    const double a = n * sxx - sx * sx;
    const double b = n * sxy - sx * sy;
    const double c = n * syy - sy * sy;
    const double d = 0.5 * (n * sx3 + n * sxy2 - sx * (sxx + syy));
    const double e = 0.5 * (n * sy3 + n * sx2y - sy * (sxx + syy));
    const double det = a * c - b * b;
    if (std::fabs(det) < 1e-9) return false;
    result.centerX = static_cast<float>((d * c - b * e) / det);
    result.centerY = static_cast<float>((a * e - b * d) / det);

    double sumR = 0.0;
    double sq = 0.0;
    for (const cv::Point2f& p : pts) {
        const double dx = p.x - result.centerX;
        const double dy = p.y - result.centerY;
        const double ri = std::sqrt(dx * dx + dy * dy);
        sumR += ri;
        sq += ri * ri;
    }
    result.radius = static_cast<float>(sumR / n);
    const double meanSq = sq / n;
    const double rMean = static_cast<double>(result.radius);
    result.rms = static_cast<float>(std::sqrt(std::max(0.0, meanSq - rMean * rMean)));
    if (result.radius <= 1e-3f) return false;
    result.pointCount = static_cast<int>(pts.size());
    result.ok = true;
    return true;
}

bool MeasureArcWithCalipers(const std::vector<float>& gray, int width, int height, float roiP0X,
                            float roiP0Y, float roiP1X, float roiP1Y, float roiP2X, float roiP2Y,
                            const CaliperArcParams& params, CaliperArcResult& result,
                            std::string& error) {
    return MeasureArcWithCalipersImpl(gray, width, height, roiP0X, roiP0Y, roiP1X, roiP1Y, roiP2X,
                                      roiP2Y, params, result, error);
}

bool MeasureArcWithCalipersRgb(const std::vector<uint8_t>& rgb, int width, int height, float roiP0X,
                               float roiP0Y, float roiP1X, float roiP1Y, float roiP2X, float roiP2Y,
                               const CaliperArcParams& params, CaliperArcResult& result,
                               std::string& error) {
    if (rgb.empty() || width <= 0 || height <= 0 ||
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3u != rgb.size()) {
        error = u8"图像数据无效";
        return false;
    }
    std::vector<float> gray;
    RgbToGrayFloat(rgb, width, height, gray);
    CaliperArcParams p = params;
    p.skipZero = false;
    return MeasureArcWithCalipersImpl(gray, width, height, roiP0X, roiP0Y, roiP1X, roiP1Y, roiP2X,
                                      roiP2Y, p, result, error);
}

void SampleArcPolyline(float cx, float cy, float radius, float startAngle, float endAngle,
                       int segments, std::vector<float>& outX, std::vector<float>& outY) {
    outX.clear();
    outY.clear();
    if (segments < 2 || radius <= 0.f) return;
    outX.reserve(static_cast<std::size_t>(segments));
    outY.reserve(static_cast<std::size_t>(segments));
    const float span = endAngle - startAngle;
    for (int i = 0; i < segments; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(segments - 1);
        const float ang = startAngle + span * t;
        outX.push_back(cx + radius * std::cos(ang));
        outY.push_back(cy + radius * std::sin(ang));
    }
}

namespace {
constexpr float kArcPi = 3.14159265358979323846f;
constexpr float kArcTwoPi = kArcPi * 2.f;

float ArcNormAnglePos(float a) {
    while (a < 0.f) a += kArcTwoPi;
    while (a >= kArcTwoPi) a -= kArcTwoPi;
    return a;
}

float ArcDeltaCCW(float from, float to) {
    return ArcNormAnglePos(to - from);
}
}  // namespace

bool CircleFromThreePoints(float x1, float y1, float x2, float y2, float x3, float y3, float& cx,
                           float& cy, float& r) {
    const float d = 2.f * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2));
    if (std::fabs(d) < 1e-6f) return false;
    const float x1s = x1 * x1 + y1 * y1;
    const float x2s = x2 * x2 + y2 * y2;
    const float x3s = x3 * x3 + y3 * y3;
    cx = (x1s * (y2 - y3) + x2s * (y3 - y1) + x3s * (y1 - y2)) / d;
    cy = (x1s * (x3 - x2) + x2s * (x1 - x3) + x3s * (x2 - x1)) / d;
    const float dx = x1 - cx;
    const float dy = y1 - cy;
    r = std::sqrt(dx * dx + dy * dy);
    return r > 1e-3f;
}

bool ArcSpanThroughMiddle(float cx, float cy, float x0, float y0, float x1, float y1, float xm,
                          float ym, float& startAngle, float& endAngle) {
    const float a0 = std::atan2(y0 - cy, x0 - cx);
    const float a1 = std::atan2(y1 - cy, x1 - cx);
    const float am = std::atan2(ym - cy, xm - cx);
    const float d0m = ArcDeltaCCW(a0, am);
    const float d01 = ArcDeltaCCW(a0, a1);
    if (d0m <= d01) {
        startAngle = a0;
        endAngle = a0 + d01;
    } else {
        startAngle = a1;
        endAngle = a1 + ArcDeltaCCW(a1, a0);
    }
    return (endAngle - startAngle) > 1e-4f;
}

float SegmentSegmentDistance(float ax1, float ay1, float ax2, float ay2, float bx1, float by1,
                             float bx2, float by2, float* closestAx, float* closestAy,
                             float* closestBx, float* closestBy) {
    auto dot = [](float x1, float y1, float x2, float y2) { return x1 * x2 + y1 * y2; };

    const float ux = ax2 - ax1;
    const float uy = ay2 - ay1;
    const float vx = bx2 - bx1;
    const float vy = by2 - by1;
    const float wx = ax1 - bx1;
    const float wy = ay1 - by1;

    const float a = dot(ux, uy, ux, uy);
    const float b = dot(ux, uy, vx, vy);
    const float c = dot(vx, vy, vx, vy);
    const float d = dot(ux, uy, wx, wy);
    const float e = dot(vx, vy, wx, wy);
    const float denom = a * c - b * b;

    float sc = 0.f;
    float tc = 0.f;
    if (denom < 1e-8f) {
        sc = 0.f;
        tc = (b > c ? d / b : e / c);
    } else {
        sc = (b * e - c * d) / denom;
        tc = (a * e - b * d) / denom;
    }
    sc = std::clamp(sc, 0.f, 1.f);
    tc = std::clamp(tc, 0.f, 1.f);

    const float px = ax1 + sc * ux;
    const float py = ay1 + sc * uy;
    const float qx = bx1 + tc * vx;
    const float qy = by1 + tc * vy;
    const float dx = px - qx;
    const float dy = py - qy;
    if (closestAx) *closestAx = px;
    if (closestAy) *closestAy = py;
    if (closestBx) *closestBx = qx;
    if (closestBy) *closestBy = qy;
    return std::sqrt(dx * dx + dy * dy);
}

namespace {

float SegmentLength(float x1, float y1, float x2, float y2) {
    const float dx = x2 - x1;
    const float dy = y2 - y1;
    return std::sqrt(dx * dx + dy * dy);
}

void SampleSegment(float x1, float y1, float x2, float y2, int index, int count, float& outX,
                   float& outY) {
    const float t = (count <= 1) ? 0.5f : static_cast<float>(index) / static_cast<float>(count - 1);
    outX = x1 + (x2 - x1) * t;
    outY = y1 + (y2 - y1) * t;
}

void PointToSegmentPerpendicularGap(float px, float py, float x1, float y1, float x2, float y2,
                                    float& footX, float& footY, float& perpDist) {
    const float dx = x2 - x1;
    const float dy = y2 - y1;
    const float len = std::sqrt(dx * dx + dy * dy);
    if (len < 1e-8f) {
        footX = x1;
        footY = y1;
        const float ex = px - x1;
        const float ey = py - y1;
        perpDist = std::sqrt(ex * ex + ey * ey);
        return;
    }

    perpDist = std::fabs((px - x1) * dy - (py - y1) * dx) / len;

    const float len2 = dx * dx + dy * dy;
    float t = ((px - x1) * dx + (py - y1) * dy) / len2;
    t = std::clamp(t, 0.f, 1.f);
    footX = x1 + t * dx;
    footY = y1 + t * dy;
}

// 过 P 垂直于线段 A 的直线与线段 B 求交；交点在 B 上则有效
bool PerpendicularFromAToSegmentB(float px, float py, float a1x, float a1y, float a2x, float a2y,
                                  float b1x, float b1y, float b2x, float b2y, float& outQx,
                                  float& outQy, float& outDist) {
    const float adx = a2x - a1x;
    const float ady = a2y - a1y;
    const float alen = std::sqrt(adx * adx + ady * ady);
    if (alen < 1e-8f) return false;

    const float nx = -ady / alen;
    const float ny = adx / alen;
    const float bdx = b2x - b1x;
    const float bdy = b2y - b1y;

    const float rhsX = b1x - px;
    const float rhsY = b1y - py;
    const float det = nx * (-bdy) - ny * (-bdx);
    if (std::fabs(det) < 1e-8f) return false;

    const float s = (nx * rhsY - ny * rhsX) / det;
    if (s < 0.f || s > 1.f) return false;

    outQx = b1x + s * bdx;
    outQy = b1y + s * bdy;
    const float ex = px - outQx;
    const float ey = py - outQy;
    outDist = std::sqrt(ex * ex + ey * ey);
    return true;
}

}  // namespace

bool AverageGapDistance(float ax1, float ay1, float ax2, float ay2, float bx1, float by1, float bx2,
                        float by2, int numSamples, AverageGapResult& out) {
    out = {};
    const int n = std::max(2, numSamples);
    if (SegmentLength(ax1, ay1, ax2, ay2) < 1e-3f || SegmentLength(bx1, by1, bx2, by2) < 1e-3f) {
        return false;
    }

    out.totalSamples = n;
    out.samples.reserve(static_cast<std::size_t>(n));
    float sum = 0.f;
    float minD = std::numeric_limits<float>::max();
    float maxD = 0.f;

    for (int i = 0; i < n; ++i) {
        float sampleX = 0.f;
        float sampleY = 0.f;
        SampleSegment(ax1, ay1, ax2, ay2, i, n, sampleX, sampleY);

        float footX = 0.f;
        float footY = 0.f;
        float dist = 0.f;
        if (!PerpendicularFromAToSegmentB(sampleX, sampleY, ax1, ay1, ax2, ay2, bx1, by1, bx2, by2,
                                          footX, footY, dist)) {
            continue;
        }

        GapSample gs;
        gs.ax = sampleX;
        gs.ay = sampleY;
        gs.bx = footX;
        gs.by = footY;
        gs.dist = dist;
        out.samples.push_back(gs);
        sum += dist;
        minD = std::min(minD, dist);
        maxD = std::max(maxD, dist);
        ++out.validCount;
    }

    if (out.validCount == 0) return false;

    out.average = sum / static_cast<float>(out.validCount);
    out.minDist = minD;
    out.maxDist = maxD;
    return true;
}

namespace {

float ArcSpanDelta(float startAngle, float endAngle) {
    return ArcDeltaCCW(startAngle, endAngle);
}

bool IsAngleOnArc(float angle, float startAngle, float endAngle) {
    return ArcSpanDelta(startAngle, angle) <= ArcSpanDelta(startAngle, endAngle) + 1e-4f;
}

bool IsPointOnArc(float qx, float qy, float cx, float cy, float radius, float startAngle,
                  float endAngle, float eps = 0.25f) {
    const float dx = qx - cx;
    const float dy = qy - cy;
    const float dist = std::sqrt(dx * dx + dy * dy);
    if (std::fabs(dist - radius) > eps) return false;
    const float ang = std::atan2(dy, dx);
    return IsAngleOnArc(ang, startAngle, endAngle);
}

void SampleArc(float cx, float cy, float radius, float startAngle, float endAngle, int index,
               int count, float& outX, float& outY) {
    const float t = (count <= 1) ? 0.5f : static_cast<float>(index) / static_cast<float>(count - 1);
    const float ang = startAngle + (endAngle - startAngle) * t;
    outX = cx + radius * std::cos(ang);
    outY = cy + radius * std::sin(ang);
}

bool LineCircleIntersectTs(float px, float py, float dx, float dy, float cx, float cy, float radius,
                           float& t1, float& t2) {
    const float fx = px - cx;
    const float fy = py - cy;
    const float a = dx * dx + dy * dy;
    if (a < 1e-10f) return false;
    const float b = 2.f * (fx * dx + fy * dy);
    const float c = fx * fx + fy * fy - radius * radius;
    float disc = b * b - 4.f * a * c;
    if (disc < 0.f) return false;
    disc = std::sqrt(disc);
    const float inv = 1.f / (2.f * a);
    t1 = (-b - disc) * inv;
    t2 = (-b + disc) * inv;
    return true;
}

// 过弧 A 上点 P 作法向（径向）直线，与弧 B 求最近有效交点
bool PerpendicularFromArcAToArcB(float px, float py, float acx, float acy, float bcx, float bcy,
                                 float br, float bStart, float bEnd, float& outQx, float& outQy,
                                 float& outDist) {
    float rdx = px - acx;
    float rdy = py - acy;
    const float rlen = std::sqrt(rdx * rdx + rdy * rdy);
    if (rlen < 1e-6f) return false;
    rdx /= rlen;
    rdy /= rlen;

    float t1 = 0.f;
    float t2 = 0.f;
    if (!LineCircleIntersectTs(px, py, rdx, rdy, bcx, bcy, br, t1, t2)) return false;

    float bestDist = std::numeric_limits<float>::max();
    bool found = false;
    const float ts[2] = {t1, t2};
    for (float t : ts) {
        if (std::fabs(t) < 1e-3f) continue;
        const float qx = px + t * rdx;
        const float qy = py + t * rdy;
        if (!IsPointOnArc(qx, qy, bcx, bcy, br, bStart, bEnd)) continue;
        const float d = std::hypot(px - qx, py - qy);
        if (d < bestDist) {
            bestDist = d;
            outQx = qx;
            outQy = qy;
            outDist = d;
            found = true;
        }
    }
    return found;
}

}  // namespace

bool AverageArcGapDistance(float acx, float acy, float ar, float aStart, float aEnd, float bcx,
                           float bcy, float br, float bStart, float bEnd, int numSamples,
                           AverageGapResult& out) {
    out = {};
    if (ar < 1e-3f || br < 1e-3f) return false;
    if (ArcSpanDelta(aStart, aEnd) < 1e-4f || ArcSpanDelta(bStart, bEnd) < 1e-4f) return false;

    const int n = std::max(2, numSamples);
    out.totalSamples = n;
    out.samples.reserve(static_cast<std::size_t>(n));
    float sum = 0.f;
    float minD = std::numeric_limits<float>::max();
    float maxD = 0.f;

    for (int i = 0; i < n; ++i) {
        float sampleX = 0.f;
        float sampleY = 0.f;
        SampleArc(acx, acy, ar, aStart, aEnd, i, n, sampleX, sampleY);

        float hitX = 0.f;
        float hitY = 0.f;
        float dist = 0.f;
        if (!PerpendicularFromArcAToArcB(sampleX, sampleY, acx, acy, bcx, bcy, br, bStart, bEnd,
                                       hitX, hitY, dist)) {
            continue;
        }

        GapSample gs;
        gs.ax = sampleX;
        gs.ay = sampleY;
        gs.bx = hitX;
        gs.by = hitY;
        gs.dist = dist;
        out.samples.push_back(gs);
        sum += dist;
        minD = std::min(minD, dist);
        maxD = std::max(maxD, dist);
        ++out.validCount;
    }

    if (out.validCount == 0) return false;

    out.average = sum / static_cast<float>(out.validCount);
    out.minDist = minD;
    out.maxDist = maxD;
    return true;
}

bool ComputeArcMetrics(float cx, float cy, float radius, float startAngle, float endAngle,
                       ArcMetrics& out) {
    out = {};
    if (radius < 1e-3f) return false;
    const float span = endAngle - startAngle;
    if (span < 1e-4f) return false;
    out.spanRadians = span;
    out.arcLength = radius * span;
    const float x0 = cx + radius * std::cos(startAngle);
    const float y0 = cy + radius * std::sin(startAngle);
    const float x1 = cx + radius * std::cos(endAngle);
    const float y1 = cy + radius * std::sin(endAngle);
    out.chordLength = std::hypot(x1 - x0, y1 - y0);
    out.sagitta = radius * (1.f - std::cos(span * 0.5f));
    return true;
}

float PointPointDistance(float x1, float y1, float x2, float y2, float* outDx, float* outDy) {
    const float dx = x2 - x1;
    const float dy = y2 - y1;
    if (outDx) *outDx = dx;
    if (outDy) *outDy = dy;
    return std::hypot(dx, dy);
}

float AngleBetweenSegments(float ax1, float ay1, float ax2, float ay2, float bx1, float by1,
                           float bx2, float by2, bool acuteOnly) {
    const float adx = ax2 - ax1;
    const float ady = ay2 - ay1;
    const float bdx = bx2 - bx1;
    const float bdy = by2 - by1;
    const float alen = std::hypot(adx, ady);
    const float blen = std::hypot(bdx, bdy);
    if (alen < 1e-6f || blen < 1e-6f) return 0.f;
    float cosA = (adx * bdx + ady * bdy) / (alen * blen);
    cosA = std::clamp(cosA, -1.f, 1.f);
    float deg = std::acos(cosA) * 180.f / 3.14159265358979323846f;
    if (acuteOnly && deg > 90.f) deg = 180.f - deg;
    return deg;
}

bool PointToSegmentDistance(float px, float py, float x1, float y1, float x2, float y2,
                            float& outDist, float& footX, float& footY) {
    const float dx = x2 - x1;
    const float dy = y2 - y1;
    const float len2 = dx * dx + dy * dy;
    if (len2 < 1e-6f) {
        footX = x1;
        footY = y1;
        outDist = std::hypot(px - x1, py - y1);
        return true;
    }
    float t = ((px - x1) * dx + (py - y1) * dy) / len2;
    t = std::clamp(t, 0.f, 1.f);
    footX = x1 + t * dx;
    footY = y1 + t * dy;
    outDist = std::hypot(px - footX, py - footY);
    return true;
}

bool ComputeCircleGap(float cx1, float cy1, float r1, float cx2, float cy2, float r2,
                      CircleGapResult& out) {
    out = {};
    if (r1 < 1e-3f || r2 < 1e-3f) return false;
    out.centerDistance = std::hypot(cx2 - cx1, cy2 - cy1);
    out.surfaceGap = out.centerDistance - r1 - r2;
    return true;
}

bool MeasureCircleWithCalipers(const std::vector<float>& gray, int width, int height,
                               float roiCenterX, float roiCenterY, float roiRadius,
                               const CaliperCircleParams& params, CaliperCircleResult& result,
                               std::string& error) {
    return MeasureCircleWithCalipersImpl(gray, width, height, roiCenterX, roiCenterY, roiRadius,
                                         params, result, error);
}

bool MeasureCircleWithCalipersRgb(const std::vector<uint8_t>& rgb, int width, int height,
                                  float roiCenterX, float roiCenterY, float roiRadius,
                                  const CaliperCircleParams& params, CaliperCircleResult& result,
                                  std::string& error) {
    if (rgb.empty() || width <= 0 || height <= 0 ||
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3u != rgb.size()) {
        error = u8"图像数据无效";
        return false;
    }
    std::vector<float> gray;
    RgbToGrayFloat(rgb, width, height, gray);
    CaliperCircleParams p = params;
    p.skipZero = false;
    return MeasureCircleWithCalipersImpl(gray, width, height, roiCenterX, roiCenterY, roiRadius, p,
                                         result, error);
}

float ParallelLineDistance(float ax1, float ay1, float ax2, float ay2, float bx1, float by1,
                           float bx2, float by2) {
    const float dx = ax2 - ax1;
    const float dy = ay2 - ay1;
    const float len = std::hypot(dx, dy);
    if (len < 1e-6f) return 0.f;
    const float bxm = (bx1 + bx2) * 0.5f;
    const float bym = (by1 + by2) * 0.5f;
    return std::fabs(dx * (ay1 - bym) - (ax1 - bxm) * dy) / len;
}

bool ProjectPointOntoSegment(float px, float py, float x1, float y1, float x2, float y2,
                             PointProjectionResult& out) {
    out = {};
    float footX = 0.f;
    float footY = 0.f;
    float dist = 0.f;
    if (!PointToSegmentDistance(px, py, x1, y1, x2, y2, dist, footX, footY)) return false;
    out.footX = footX;
    out.footY = footY;
    out.perpDist = dist;
    const float dx = x2 - x1;
    const float dy = y2 - y1;
    const float len2 = dx * dx + dy * dy;
    if (len2 < 1e-6f) {
        out.alongT = 0.f;
    } else {
        out.alongT = ((footX - x1) * dx + (footY - y1) * dy) / len2;
    }
    return true;
}

namespace {

bool MeasureRectWithCalipersImpl(const std::vector<float>& gray, int width, int height, float roiX0,
                                 float roiY0, float roiX1, float roiY1,
                                 const CaliperLineParams& params, CaliperRectResult& result,
                                 std::string& error) {
    result = {};
    const float left = std::min(roiX0, roiX1);
    const float right = std::max(roiX0, roiX1);
    const float top = std::min(roiY0, roiY1);
    const float bottom = std::max(roiY0, roiY1);
    result.roiX0 = left;
    result.roiY0 = top;
    result.roiX1 = right;
    result.roiY1 = bottom;
    if (right - left < 5.f || bottom - top < 5.f) {
        error = u8"矩形 ROI 过小";
        return false;
    }

    CaliperLineParams edgeParams = params;
    edgeParams.numCalipers = std::max(3, params.numCalipers);

    CaliperLineResult topR;
    CaliperLineResult botR;
    CaliperLineResult leftR;
    CaliperLineResult rightR;
    std::string e;
    if (!MeasureLineWithCalipersImpl(gray, width, height, left, top, right, top, edgeParams, topR,
                                     e) ||
        !topR.ok) {
        error = u8"上边卡尺失败";
        return false;
    }
    if (!MeasureLineWithCalipersImpl(gray, width, height, left, bottom, right, bottom, edgeParams,
                                     botR, e) ||
        !botR.ok) {
        error = u8"下边卡尺失败";
        return false;
    }
    if (!MeasureLineWithCalipersImpl(gray, width, height, left, top, left, bottom, edgeParams,
                                     leftR, e) ||
        !leftR.ok) {
        error = u8"左边卡尺失败";
        return false;
    }
    if (!MeasureLineWithCalipersImpl(gray, width, height, right, top, right, bottom, edgeParams,
                                     rightR, e) ||
        !rightR.ok) {
        error = u8"右边卡尺失败";
        return false;
    }

    const CaliperLineResult* edges[4] = {&topR, &botR, &leftR, &rightR};
    std::vector<cv::Point2f> fitPts;
    for (const CaliperLineResult* er : edges) {
        for (const LineSegment& c : er->calipers) {
            result.calipers.push_back(c);
        }
        for (const CaliperEdgePoint& ep : er->edgePoints) {
            result.edgePoints.push_back(ep);
            if (ep.valid) fitPts.emplace_back(ep.x, ep.y);
        }
    }
    result.validCount = static_cast<int>(fitPts.size());
    if (fitPts.size() < 4) {
        error = u8"矩形边缘点不足";
        return false;
    }

    cv::RotatedRect rr = cv::minAreaRect(fitPts);
    result.centerX = rr.center.x;
    result.centerY = rr.center.y;
    result.width = rr.size.width;
    result.height = rr.size.height;
    result.angleDeg = rr.angle;
    result.ok = true;
    return true;
}

bool MeasureProfileWidthImpl(const std::vector<float>& gray, int width, int height, float roiX0,
                             float roiY0, float roiX1, float roiY1, const CaliperLineParams& params,
                             ProfileWidthResult& result, std::string& error) {
    result = {};
    result.roiX0 = roiX0;
    result.roiY0 = roiY0;
    result.roiX1 = roiX1;
    result.roiY1 = roiY1;

    const float dx = roiX1 - roiX0;
    const float dy = roiY1 - roiY0;
    const float lineLen = std::sqrt(dx * dx + dy * dy);
    if (lineLen < 2.f) {
        error = u8"测量线过短";
        return false;
    }

    const float mx = (roiX0 + roiX1) * 0.5f;
    const float my = (roiY0 + roiY1) * 0.5f;
    const float dirX = dx / lineLen;
    const float dirY = dy / lineLen;
    const float normX = -dirY;
    const float normY = dirX;
    const float halfLen = std::max(8.f, params.caliperHalfLength);
    const int halfWidth = std::max(0, params.caliperWidth / 2);
    const int profileSteps = std::max(5, static_cast<int>(std::ceil(halfLen * 2.f)) + 1);

    std::vector<float> profile(static_cast<std::size_t>(profileSteps));
    for (int j = 0; j < profileSteps; ++j) {
        const float s = -halfLen + (2.f * halfLen * static_cast<float>(j)) /
                                       static_cast<float>(profileSteps - 1);
        profile[static_cast<std::size_t>(j)] =
            SampleAveraged(gray, width, height, mx + normX * s, my + normY * s, dirX, dirY,
                           halfWidth, params.skipZero);
    }

    int bestPos = -1;
    int bestNeg = -1;
    float bestPosScore = -1.f;
    float bestNegScore = -1.f;
    for (int j = 1; j < profileSteps - 1; ++j) {
        const float g =
            0.5f * (profile[static_cast<std::size_t>(j + 1)] - profile[static_cast<std::size_t>(j - 1)]);
        if (g > bestPosScore) {
            bestPosScore = g;
            bestPos = j;
        }
        if (-g > bestNegScore) {
            bestNegScore = -g;
            bestNeg = j;
        }
    }

    auto edgePos = [&](int idx, float& ex, float& ey) {
        const float s = -halfLen + (2.f * halfLen * static_cast<float>(idx)) /
                                       static_cast<float>(profileSteps - 1);
        ex = mx + normX * s;
        ey = my + normY * s;
    };

    if (bestPos >= 0 && bestPosScore >= params.minContrast) {
        edgePos(bestPos, result.edge1X, result.edge1Y);
    } else {
        error = u8"未检测到正梯度边缘";
        return false;
    }
    if (bestNeg >= 0 && bestNegScore >= params.minContrast) {
        edgePos(bestNeg, result.edge2X, result.edge2Y);
    } else {
        error = u8"未检测到负梯度边缘";
        return false;
    }

    result.width = std::hypot(result.edge2X - result.edge1X, result.edge2Y - result.edge1Y);
    if (result.width < 1.f) {
        error = u8"宽度过小";
        return false;
    }
    result.ok = true;
    return true;
}

bool SampleLineProfileImpl(const std::vector<float>& gray, int width, int height, float x0, float y0,
                           float x1, float y1, int numSamples,
                           std::vector<LineProfileSample>& out, bool skipZero) {
    out.clear();
    if (gray.empty() || width <= 0 || height <= 0 || numSamples < 2) return false;
    out.reserve(static_cast<std::size_t>(numSamples));
    for (int i = 0; i < numSamples; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(numSamples - 1);
        const float x = x0 + (x1 - x0) * t;
        const float y = y0 + (y1 - y0) * t;
        LineProfileSample s;
        s.x = x;
        s.y = y;
        s.value = SampleBilinear(gray, width, height, x, y, skipZero);
        if (skipZero && s.value == 0.f) s.value = std::numeric_limits<float>::quiet_NaN();
        out.push_back(s);
    }
    return !out.empty();
}

}  // namespace

bool MeasureRectWithCalipers(const std::vector<float>& gray, int width, int height, float roiX0,
                             float roiY0, float roiX1, float roiY1, const CaliperLineParams& params,
                             CaliperRectResult& result, std::string& error) {
    return MeasureRectWithCalipersImpl(gray, width, height, roiX0, roiY0, roiX1, roiY1, params,
                                       result, error);
}

bool MeasureRectWithCalipersRgb(const std::vector<uint8_t>& rgb, int width, int height, float roiX0,
                                float roiY0, float roiX1, float roiY1,
                                const CaliperLineParams& params, CaliperRectResult& result,
                                std::string& error) {
    if (rgb.empty() || width <= 0 || height <= 0) {
        error = u8"图像数据无效";
        return false;
    }
    std::vector<float> gray;
    RgbToGrayFloat(rgb, width, height, gray);
    CaliperLineParams p = params;
    p.skipZero = false;
    return MeasureRectWithCalipersImpl(gray, width, height, roiX0, roiY0, roiX1, roiY1, p, result,
                                       error);
}

bool FitEllipseFromEdgePoints(const std::vector<CaliperEdgePoint>& points, EllipseFitResult& result) {
    result = {};
    std::vector<cv::Point2f> pts;
    pts.reserve(points.size());
    for (const CaliperEdgePoint& p : points) {
        if (p.valid) pts.emplace_back(p.x, p.y);
    }
    if (pts.size() < 5) return false;

    cv::RotatedRect er = cv::fitEllipse(pts);
    result.centerX = er.center.x;
    result.centerY = er.center.y;
    result.axisA = std::max(er.size.width, er.size.height) * 0.5f;
    result.axisB = std::min(er.size.width, er.size.height) * 0.5f;
    result.angleDeg = er.angle;
    result.pointCount = static_cast<int>(pts.size());

    float sq = 0.f;
    float maxD = 0.f;
    float minD = std::numeric_limits<float>::max();
    const float cosA = std::cos(er.angle * 3.14159265358979323846f / 180.f);
    const float sinA = std::sin(er.angle * 3.14159265358979323846f / 180.f);
    const float a = result.axisA;
    const float b = result.axisB;
    for (const cv::Point2f& p : pts) {
        const float lx = p.x - result.centerX;
        const float ly = p.y - result.centerY;
        const float rx = lx * cosA + ly * sinA;
        const float ry = -lx * sinA + ly * cosA;
        const float norm = (rx * rx) / (a * a) + (ry * ry) / (b * b);
        const float ri = std::sqrt(norm) * std::sqrt(a * b);
        const float dev = std::fabs(ri - std::sqrt(a * b));
        sq += dev * dev;
        maxD = std::max(maxD, dev);
        minD = std::min(minD, dev);
    }
    result.rms = std::sqrt(sq / static_cast<float>(pts.size()));
    (void)maxD;
    (void)minD;
    result.ok = true;
    return true;
}

bool MeasureProfileWidth(const std::vector<float>& gray, int width, int height, float roiX0,
                         float roiY0, float roiX1, float roiY1, const CaliperLineParams& params,
                         ProfileWidthResult& result, std::string& error) {
    return MeasureProfileWidthImpl(gray, width, height, roiX0, roiY0, roiX1, roiY1, params, result,
                                   error);
}

bool MeasureProfileWidthRgb(const std::vector<uint8_t>& rgb, int width, int height, float roiX0,
                            float roiY0, float roiX1, float roiY1, const CaliperLineParams& params,
                            ProfileWidthResult& result, std::string& error) {
    if (rgb.empty() || width <= 0 || height <= 0) {
        error = u8"图像数据无效";
        return false;
    }
    std::vector<float> gray;
    RgbToGrayFloat(rgb, width, height, gray);
    CaliperLineParams p = params;
    p.skipZero = false;
    return MeasureProfileWidthImpl(gray, width, height, roiX0, roiY0, roiX1, roiY1, p, result,
                                    error);
}

bool ComputeRoundness(float centerX, float centerY, float radius,
                      const std::vector<CaliperEdgePoint>& points, RoundnessResult& out) {
    out = {};
    if (radius < 1e-3f) return false;
    float sq = 0.f;
    int count = 0;
    float maxD = 0.f;
    float minD = std::numeric_limits<float>::max();
    for (const CaliperEdgePoint& p : points) {
        if (!p.valid) continue;
        const float ri = std::hypot(p.x - centerX, p.y - centerY);
        const float dev = ri - radius;
        sq += dev * dev;
        maxD = std::max(maxD, dev);
        minD = std::min(minD, dev);
        ++count;
    }
    if (count < 3) return false;
    out.rms = std::sqrt(sq / static_cast<float>(count));
    out.maxDev = maxD;
    out.minDev = minD;
    out.ok = true;
    return true;
}

bool ComputeRegionBlob(const std::vector<float>& gray, int width, int height, float roiX0,
                       float roiY0, float roiX1, float roiY1, float threshold, bool greaterThan,
                       RegionBlobResult& out, std::string& error) {
    out = {};
    if (gray.empty() || width <= 0 || height <= 0) {
        error = u8"图像数据无效";
        return false;
    }
    const int x0 = std::clamp(static_cast<int>(std::floor(std::min(roiX0, roiX1))), 0, width - 1);
    const int x1 = std::clamp(static_cast<int>(std::ceil(std::max(roiX0, roiX1))), 0, width - 1);
    const int y0 = std::clamp(static_cast<int>(std::floor(std::min(roiY0, roiY1))), 0, height - 1);
    const int y1 = std::clamp(static_cast<int>(std::ceil(std::max(roiY0, roiY1))), 0, height - 1);
    out.roiX0 = static_cast<float>(x0);
    out.roiY0 = static_cast<float>(y0);
    out.roiX1 = static_cast<float>(x1);
    out.roiY1 = static_cast<float>(y1);
    if (x1 <= x0 || y1 <= y0) {
        error = u8"区域过小";
        return false;
    }

    double sumX = 0.0;
    double sumY = 0.0;
    int count = 0;
    for (int y = y0; y <= y1; ++y) {
        for (int x = x0; x <= x1; ++x) {
            const float v = gray[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                                 static_cast<std::size_t>(x)];
            const bool hit = greaterThan ? (v > threshold) : (v < threshold);
            if (!hit) continue;
            sumX += x;
            sumY += y;
            ++count;
        }
    }
    if (count == 0) {
        error = u8"区域内无满足阈值的像素";
        return false;
    }
    out.pixelCount = count;
    out.areaPx = static_cast<float>(count);
    out.centroidX = static_cast<float>(sumX / static_cast<double>(count));
    out.centroidY = static_cast<float>(sumY / static_cast<double>(count));
    out.ok = true;
    return true;
}

bool ComputeRegionBlobRgb(const std::vector<uint8_t>& rgb, int width, int height, float roiX0,
                          float roiY0, float roiX1, float roiY1, float threshold, bool greaterThan,
                          RegionBlobResult& out, std::string& error) {
    if (rgb.empty() || width <= 0 || height <= 0) {
        error = u8"图像数据无效";
        return false;
    }
    std::vector<float> gray;
    RgbToGrayFloat(rgb, width, height, gray);
    return ComputeRegionBlob(gray, width, height, roiX0, roiY0, roiX1, roiY1, threshold, greaterThan,
                             out, error);
}

bool ComputeConcentricity(float cx1, float cy1, float cx2, float cy2, ConcentricityResult& out) {
    out = {};
    out.offsetX = cx2 - cx1;
    out.offsetY = cy2 - cy1;
    out.offsetDist = std::hypot(out.offsetX, out.offsetY);
    return true;
}

bool SampleLineProfile(const std::vector<float>& gray, int width, int height, float x0, float y0,
                       float x1, float y1, int numSamples, std::vector<LineProfileSample>& out,
                       bool skipZero) {
    return SampleLineProfileImpl(gray, width, height, x0, y0, x1, y1, numSamples, out, skipZero);
}

}  // namespace OpenCv2D
