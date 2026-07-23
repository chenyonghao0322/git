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
    const float lineLen = std::hypot(x1 - x0, y1 - y0);
    out.reserve(static_cast<std::size_t>(numSamples));
    for (int i = 0; i < numSamples; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(numSamples - 1);
        const float x = x0 + (x1 - x0) * t;
        const float y = y0 + (y1 - y0) * t;
        LineProfileSample s;
        s.x = x;
        s.y = y;
        s.distance = lineLen * t;
        s.value = SampleBilinear(gray, width, height, x, y, skipZero);
        if (skipZero && s.value == 0.f) s.value = std::numeric_limits<float>::quiet_NaN();
        out.push_back(s);
    }
    return !out.empty();
}

bool SampleRowProfileImpl(const std::vector<float>& gray, int width, int height, int row,
                          std::vector<LineProfileSample>& out, bool skipZero) {
    out.clear();
    if (gray.empty() || width <= 0 || height <= 0) return false;
    row = std::clamp(row, 0, height - 1);
    out.reserve(static_cast<std::size_t>(width));
    for (int col = 0; col < width; ++col) {
        LineProfileSample s;
        s.x = static_cast<float>(col);
        s.y = static_cast<float>(row);
        s.distance = static_cast<float>(col);
        s.value = gray[static_cast<std::size_t>(row) * static_cast<std::size_t>(width) +
                       static_cast<std::size_t>(col)];
        if (skipZero && s.value == 0.f) s.value = std::numeric_limits<float>::quiet_NaN();
        out.push_back(s);
    }
    return !out.empty();
}

bool SampleColumnProfileImpl(const std::vector<float>& gray, int width, int height, int col,
                             std::vector<LineProfileSample>& out, bool skipZero) {
    out.clear();
    if (gray.empty() || width <= 0 || height <= 0) return false;
    col = std::clamp(col, 0, width - 1);
    out.reserve(static_cast<std::size_t>(height));
    for (int row = 0; row < height; ++row) {
        LineProfileSample s;
        s.x = static_cast<float>(col);
        s.y = static_cast<float>(row);
        s.distance = static_cast<float>(row);
        s.value = gray[static_cast<std::size_t>(row) * static_cast<std::size_t>(width) +
                       static_cast<std::size_t>(col)];
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

    const int roiW = x1 - x0 + 1;
    const int roiH = y1 - y0 + 1;
    out.roiWidth = roiW;
    out.roiHeight = roiH;
    out.hitMask.assign(static_cast<std::size_t>(roiW * roiH), 0);

    double sumX = 0.0;
    double sumY = 0.0;
    int count = 0;
    for (int y = y0; y <= y1; ++y) {
        for (int x = x0; x <= x1; ++x) {
            const float v = gray[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                                 static_cast<std::size_t>(x)];
            const bool hit = greaterThan ? (v > threshold) : (v < threshold);
            if (!hit) continue;
            out.hitMask[static_cast<std::size_t>(y - y0) * static_cast<std::size_t>(roiW) +
                        static_cast<std::size_t>(x - x0)] = 1;
            sumX += x;
            sumY += y;
            ++count;
        }
    }
    out.pixelCount = count;
    out.areaPx = static_cast<float>(count);
    if (count > 0) {
        out.centroidX = static_cast<float>(sumX / static_cast<double>(count));
        out.centroidY = static_cast<float>(sumY / static_cast<double>(count));
    } else {
        out.centroidX = (out.roiX0 + out.roiX1) * 0.5f;
        out.centroidY = (out.roiY0 + out.roiY1) * 0.5f;
    }
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

bool SampleRowProfile(const std::vector<float>& gray, int width, int height, int row,
                      std::vector<LineProfileSample>& out, bool skipZero) {
    return SampleRowProfileImpl(gray, width, height, row, out, skipZero);
}

bool SampleColumnProfile(const std::vector<float>& gray, int width, int height, int col,
                         std::vector<LineProfileSample>& out, bool skipZero) {
    return SampleColumnProfileImpl(gray, width, height, col, out, skipZero);
}

namespace {

void NormalizeRoi(float& x0, float& y0, float& x1, float& y1) {
    if (x0 > x1) std::swap(x0, x1);
    if (y0 > y1) std::swap(y0, y1);
}

int RoiPixelSpan(float a0, float a1) {
    const int i0 = static_cast<int>(std::ceil(a0));
    const int i1 = static_cast<int>(std::floor(a1));
    return std::max(0, i1 - i0 + 1);
}

void BuildSearchRange(float vmin, float vmax, float step, std::vector<float>& out) {
    out.clear();
    if (vmax < vmin) std::swap(vmin, vmax);
    if (step <= 0.f) step = 1.f;
    if (std::fabs(vmax - vmin) < 1e-4f) {
        out.push_back(vmin);
        return;
    }
    for (float v = vmin; v <= vmax + step * 0.5f; v += step) out.push_back(v);
    if (out.empty()) out.push_back(vmin);
}

void AppendValueIfMissing(std::vector<float>& values, float value, float tolerance) {
    for (float v : values) {
        if (std::fabs(v - value) <= tolerance) return;
    }
    values.push_back(value);
    std::sort(values.begin(), values.end());
}

void BuildScaleSearchRange(float vmin, float vmax, float step, std::vector<float>& out) {
    BuildSearchRange(vmin, vmax, step, out);
    if (vmin <= 1.f && vmax >= 1.f) {
        AppendValueIfMissing(out, 1.f, std::max(step * 0.25f, 0.02f));
    }
}

cv::Mat TransformTemplateGray(const cv::Mat& grayTpl, float scale, float angleDeg, float& outTplW,
                              float& outTplH) {
    cv::Mat scaled;
    if (std::fabs(scale - 1.f) > 1e-4f) {
        cv::resize(grayTpl, scaled, cv::Size(), static_cast<double>(scale),
                   static_cast<double>(scale), cv::INTER_LINEAR);
    } else {
        scaled = grayTpl;
    }
    outTplW = static_cast<float>(scaled.cols);
    outTplH = static_cast<float>(scaled.rows);
    if (scaled.cols < 4 || scaled.rows < 4) return {};

    if (std::fabs(angleDeg) < 1e-3f) return scaled;

    const cv::Point2f center(scaled.cols * 0.5f, scaled.rows * 0.5f);
    cv::Mat rotMat = cv::getRotationMatrix2D(center, static_cast<double>(angleDeg), 1.0);
    const cv::Rect2f bbox =
        cv::RotatedRect(cv::Point2f(), scaled.size(), angleDeg).boundingRect2f();
    rotMat.at<double>(0, 2) += static_cast<double>(bbox.width * 0.5f - center.x);
    rotMat.at<double>(1, 2) += static_cast<double>(bbox.height * 0.5f - center.y);
    cv::Mat rotated;
    cv::warpAffine(scaled, rotated, rotMat, bbox.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT,
                   cv::Scalar(0));
    return rotated;
}

void BuildGrayPyramid(const cv::Mat& gray, int maxLevels, std::vector<cv::Mat>& pyramid) {
    pyramid.clear();
    pyramid.push_back(gray);
    const int cap = std::clamp(maxLevels, 1, 5);
    while (static_cast<int>(pyramid.size()) < cap && pyramid.back().cols >= 16 &&
           pyramid.back().rows >= 16) {
        cv::Mat down;
        cv::pyrDown(pyramid.back(), down);
        pyramid.push_back(down);
    }
}

int AutoPyramidLevels(int tplW, int tplH, int requested) {
    if (requested > 0) return std::clamp(requested, 1, 5);
    int levels = 1;
    int w = tplW;
    int h = tplH;
    while (w >= 20 && h >= 20 && levels < 4) {
        w /= 2;
        h /= 2;
        ++levels;
    }
    return levels;
}

bool RefineSubPixelPeak(const cv::Mat& scoreMap, const cv::Point& peak, float& subX, float& subY) {
    subX = static_cast<float>(peak.x);
    subY = static_cast<float>(peak.y);
    if (scoreMap.empty() || scoreMap.type() != CV_32FC1) return false;
    const int x = peak.x;
    const int y = peak.y;
    if (x < 1 || y < 1 || x >= scoreMap.cols - 1 || y >= scoreMap.rows - 1) return false;

    auto at = [&](int px, int py) -> float { return scoreMap.at<float>(py, px); };
    const float c = at(x, y);
    const float denomX = at(x - 1, y) - 2.f * c + at(x + 1, y);
    const float denomY = at(x, y - 1) - 2.f * c + at(x, y + 1);
    float dx = 0.f;
    float dy = 0.f;
    if (std::fabs(denomX) > 1e-6f) dx = 0.5f * (at(x - 1, y) - at(x + 1, y)) / denomX;
    if (std::fabs(denomY) > 1e-6f) dy = 0.5f * (at(x, y - 1) - at(x, y + 1)) / denomY;
    subX = static_cast<float>(x) + std::clamp(dx, -0.5f, 0.5f);
    subY = static_cast<float>(y) + std::clamp(dy, -0.5f, 0.5f);
    return true;
}

struct TemplateMatchCandidate {
    float score = 0.f;
    float centerX = 0.f;
    float centerY = 0.f;
    float templateWidth = 0.f;
    float templateHeight = 0.f;
    float angleDeg = 0.f;
    float scale = 1.f;
    float bboxX = 0.f;
    float bboxY = 0.f;
    float bboxW = 0.f;
    float bboxH = 0.f;
};

float CandidateIoU(const TemplateMatchCandidate& a, const TemplateMatchCandidate& b) {
    const float ax1 = a.bboxX + a.bboxW;
    const float ay1 = a.bboxY + a.bboxH;
    const float bx1 = b.bboxX + b.bboxW;
    const float by1 = b.bboxY + b.bboxH;
    const float ix0 = std::max(a.bboxX, b.bboxX);
    const float iy0 = std::max(a.bboxY, b.bboxY);
    const float ix1 = std::min(ax1, bx1);
    const float iy1 = std::min(ay1, by1);
    const float iw = std::max(0.f, ix1 - ix0);
    const float ih = std::max(0.f, iy1 - iy0);
    const float inter = iw * ih;
    const float areaA = std::max(a.bboxW * a.bboxH, 1.f);
    const float areaB = std::max(b.bboxW * b.bboxH, 1.f);
    return inter / (areaA + areaB - inter);
}

bool RejectBorderHit(const TemplateMatchCandidate& hit, int searchGlobalOffX, int searchGlobalOffY,
                     const TemplateMatchParams& params) {
    if (params.borderIntersect || params.searchGlobalW <= 0 || params.searchGlobalH <= 0) {
        return false;
    }
    const float gx0 = static_cast<float>(searchGlobalOffX);
    const float gy0 = static_cast<float>(searchGlobalOffY);
    const float gx1 = gx0 + static_cast<float>(params.searchGlobalW);
    const float gy1 = gy0 + static_cast<float>(params.searchGlobalH);
    const float eps = 1.5f;
    if (hit.bboxX <= gx0 + eps || hit.bboxY <= gy0 + eps) return true;
    if (hit.bboxX + hit.bboxW >= gx1 - eps || hit.bboxY + hit.bboxH >= gy1 - eps) return true;
    return false;
}

bool PickMatchCandidates(const std::vector<TemplateMatchCandidate>& candidates, float minScore,
                         int maxMatches, float maxOverlap,
                         std::vector<TemplateMatchCandidate>& picked) {
    picked.clear();
    const int want = std::clamp(maxMatches, 1, 32);
    picked.reserve(static_cast<std::size_t>(want));
    for (const TemplateMatchCandidate& c : candidates) {
        if (c.score < minScore) break;
        bool overlap = false;
        for (const TemplateMatchCandidate& p : picked) {
            if (CandidateIoU(c, p) > maxOverlap) {
                overlap = true;
                break;
            }
            const float dx = c.centerX - p.centerX;
            const float dy = c.centerY - p.centerY;
            const float centerDist = std::sqrt(dx * dx + dy * dy);
            const float minDim =
                std::min(std::min(c.bboxW, c.bboxH), std::min(p.bboxW, p.bboxH));
            if (centerDist < minDim * (1.f - maxOverlap * 0.5f)) {
                overlap = true;
                break;
            }
        }
        if (overlap) continue;
        picked.push_back(c);
        if (static_cast<int>(picked.size()) >= want) break;
    }
    return !picked.empty();
}

float CoarseMinScore(float minScore, float greediness) {
    const float g = std::clamp(greediness, 0.f, 1.f);
    return minScore * (0.55f + 0.45f * (1.f - g));
}

bool MatchWarpedTemplate(const cv::Mat& searchImg, int searchGlobalOffX, int searchGlobalOffY,
                         const cv::Mat& warpedTpl, float scaledTplW, float scaledTplH,
                         float angleDeg, float scale, float minScore, bool subPixel,
                         const cv::Rect* localRoi, int pyramidLevel,
                         const TemplateMatchParams* policy, TemplateMatchCandidate& out) {
    if (searchImg.empty() || warpedTpl.empty()) return false;

    cv::Mat tpl = warpedTpl;
    for (int i = 0; i < pyramidLevel; ++i) {
        if (tpl.cols < 8 || tpl.rows < 8) return false;
        cv::Mat down;
        cv::pyrDown(tpl, down);
        tpl = down;
    }
    if (tpl.cols > searchImg.cols || tpl.rows > searchImg.rows) return false;

    cv::Mat searchRoi = searchImg;
    int roiOffX = 0;
    int roiOffY = 0;
    if (localRoi) {
        cv::Rect clipped = *localRoi;
        clipped &= cv::Rect(0, 0, searchImg.cols, searchImg.rows);
        if (clipped.width < tpl.cols || clipped.height < tpl.rows) return false;
        searchRoi = searchImg(clipped);
        roiOffX = clipped.x;
        roiOffY = clipped.y;
    }

    cv::Mat scoreMap;
    cv::matchTemplate(searchRoi, tpl, scoreMap, cv::TM_CCOEFF_NORMED);
    if (scoreMap.empty()) return false;

    double minVal = 0.0;
    double maxVal = 0.0;
    cv::Point minLoc;
    cv::Point maxLoc;
    cv::minMaxLoc(scoreMap, &minVal, &maxVal, &minLoc, &maxLoc);
    if (static_cast<float>(maxVal) < minScore) return false;

    float peakX = static_cast<float>(maxLoc.x);
    float peakY = static_cast<float>(maxLoc.y);
    if (subPixel && pyramidLevel == 0) {
        RefineSubPixelPeak(scoreMap, maxLoc, peakX, peakY);
    }

    const float mul = static_cast<float>(1 << std::max(pyramidLevel, 0));
    out.score = static_cast<float>(maxVal);
    out.bboxX = static_cast<float>(searchGlobalOffX) + (static_cast<float>(roiOffX) + peakX) * mul;
    out.bboxY = static_cast<float>(searchGlobalOffY) + (static_cast<float>(roiOffY) + peakY) * mul;
    out.bboxW = static_cast<float>(tpl.cols) * mul;
    out.bboxH = static_cast<float>(tpl.rows) * mul;
    out.centerX = out.bboxX + out.bboxW * 0.5f;
    out.centerY = out.bboxY + out.bboxH * 0.5f;
    out.templateWidth = scaledTplW;
    out.templateHeight = scaledTplH;
    out.angleDeg = angleDeg;
    out.scale = scale;
    if (policy && RejectBorderHit(out, searchGlobalOffX, searchGlobalOffY, *policy)) {
        return false;
    }
    return true;
}

void SearchAngleScaleGrid(const cv::Mat& searchImg, int searchGlobalOffX, int searchGlobalOffY,
                          const cv::Mat& grayTpl, const std::vector<float>& angles,
                          const std::vector<float>& scales, float minScore, bool subPixel,
                          const cv::Rect* localRoi, int pyramidLevel,
                          const TemplateMatchParams* policy,
                          std::vector<TemplateMatchCandidate>& outCandidates) {
    for (float scale : scales) {
        for (float angleDeg : angles) {
            float scaledTplW = 0.f;
            float scaledTplH = 0.f;
            const cv::Mat warped =
                TransformTemplateGray(grayTpl, scale, angleDeg, scaledTplW, scaledTplH);
            if (warped.empty() || warped.cols < 4 || warped.rows < 4) continue;

            TemplateMatchCandidate c;
            if (!MatchWarpedTemplate(searchImg, searchGlobalOffX, searchGlobalOffY, warped,
                                     scaledTplW, scaledTplH, angleDeg, scale, minScore, subPixel,
                                     localRoi, pyramidLevel, policy, c)) {
                continue;
            }
            outCandidates.push_back(c);
        }
    }
}

cv::Rect LocalRefineRoi(const TemplateMatchCandidate& cand, int searchOffX, int searchOffY,
                         int lvl, int searchW, int searchH, int tplW, int tplH) {
    const float inv = 1.f / static_cast<float>(1 << lvl);
    const float cx = (cand.centerX - static_cast<float>(searchOffX)) * inv;
    const float cy = (cand.centerY - static_cast<float>(searchOffY)) * inv;
    const int margin = std::max(6, std::max(tplW, tplH) / 4 + 2);
    const int halfW = static_cast<int>(cand.bboxW * inv * 0.5f) + margin;
    const int halfH = static_cast<int>(cand.bboxH * inv * 0.5f) + margin;
    int x0 = static_cast<int>(std::floor(cx)) - halfW;
    int y0 = static_cast<int>(std::floor(cy)) - halfH;
    int x1 = static_cast<int>(std::ceil(cx)) + halfW;
    int y1 = static_cast<int>(std::ceil(cy)) + halfH;
    x0 = std::clamp(x0, 0, searchW - 1);
    y0 = std::clamp(y0, 0, searchH - 1);
    x1 = std::clamp(x1, x0 + 1, searchW);
    y1 = std::clamp(y1, y0 + 1, searchH);
    return cv::Rect(x0, y0, x1 - x0, y1 - y0);
}

void PickTopCandidates(std::vector<TemplateMatchCandidate>& candidates, int keepCount) {
    if (keepCount <= 0 || static_cast<int>(candidates.size()) <= keepCount) return;
    std::partial_sort(candidates.begin(), candidates.begin() + keepCount, candidates.end(),
                      [](const TemplateMatchCandidate& a, const TemplateMatchCandidate& b) {
                          return a.score > b.score;
                      });
    candidates.resize(static_cast<std::size_t>(keepCount));
}

}  // namespace

bool MatchTemplateRgb(const std::vector<uint8_t>& rgb, int width, int height, float templateX0,
                      float templateY0, float templateX1, float templateY1, float searchX0,
                      float searchY0, float searchX1, float searchY1,
                      const TemplateMatchParams& params, TemplateMatchResult& result,
                      std::string& error) {
    result = {};
    if (rgb.empty() || width <= 0 || height <= 0) {
        error = u8"图像无效";
        return false;
    }
    const std::size_t expect = static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3u;
    if (rgb.size() < expect) {
        error = u8"图像数据长度不足";
        return false;
    }

    NormalizeRoi(templateX0, templateY0, templateX1, templateY1);
    if (!params.searchFullImage) {
        NormalizeRoi(searchX0, searchY0, searchX1, searchY1);
    } else {
        searchX0 = 0.f;
        searchY0 = 0.f;
        searchX1 = static_cast<float>(width - 1);
        searchY1 = static_cast<float>(height - 1);
    }

    const int tplW = RoiPixelSpan(templateX0, templateX1);
    const int tplH = RoiPixelSpan(templateY0, templateY1);
    if (tplW < 4 || tplH < 4) {
        error = u8"模板区域过小（至少 4×4 像素）";
        return false;
    }

    const int tplX0 = static_cast<int>(std::ceil(templateX0));
    const int tplY0 = static_cast<int>(std::ceil(templateY0));
    const int searchX0i = static_cast<int>(std::ceil(searchX0));
    const int searchY0i = static_cast<int>(std::ceil(searchY0));
    const int searchW = RoiPixelSpan(searchX0, searchX1);
    const int searchH = RoiPixelSpan(searchY0, searchY1);
    if (searchW < 4 || searchH < 4) {
        error = u8"搜索区域过小";
        return false;
    }

    cv::Mat image(height, width, CV_8UC3, const_cast<uint8_t*>(rgb.data()));
    const cv::Rect tplRect(tplX0, tplY0, tplW, tplH);
    if (tplRect.x + tplRect.width > width || tplRect.y + tplRect.height > height) {
        error = u8"模板区域超出图像范围";
        return false;
    }
    const cv::Mat tpl = image(tplRect).clone();

    cv::Mat grayTpl;
    cv::cvtColor(tpl, grayTpl, cv::COLOR_RGB2GRAY);

    cv::Mat graySearch;
    const cv::Rect searchRect(searchX0i, searchY0i, searchW, searchH);
    if (searchRect.x + searchRect.width > width || searchRect.y + searchRect.height > height) {
        error = u8"搜索区域超出图像范围";
        return false;
    }
    cv::cvtColor(image(searchRect), graySearch, cv::COLOR_RGB2GRAY);

    result.templateX0 = templateX0;
    result.templateY0 = templateY0;
    result.templateX1 = templateX1;
    result.templateY1 = templateY1;
    result.searchX0 = searchX0;
    result.searchY0 = searchY0;
    result.searchX1 = searchX1;
    result.searchY1 = searchY1;

    const bool subPixel = params.subPixelRefine;
    const float minScore = std::clamp(params.minScore, 0.f, 1.f);

    std::vector<TemplateMatchCandidate> candidates;

    if (!params.usePyramid) {
        std::vector<float> angles;
        std::vector<float> scales;
        BuildSearchRange(params.angleMinDeg, params.angleMaxDeg, params.angleStepDeg, angles);
        BuildScaleSearchRange(params.scaleMin, params.scaleMax, params.scaleStep, scales);
        SearchAngleScaleGrid(graySearch, searchX0i, searchY0i, grayTpl, angles, scales, minScore,
                             subPixel, nullptr, 0, &params, candidates);
    } else {
        const int numLevels =
            AutoPyramidLevels(tplW, tplH, params.pyramidLevels);
        std::vector<cv::Mat> searchPyramid;
        BuildGrayPyramid(graySearch, numLevels, searchPyramid);
        const int topLevel = static_cast<int>(searchPyramid.size()) - 1;

        std::vector<float> coarseAngles;
        std::vector<float> coarseScales;
        const float coarseAngleStep = params.angleStepDeg * static_cast<float>(1 << topLevel);
        const float coarseScaleStep = params.scaleStep * static_cast<float>(1 << topLevel);
        BuildSearchRange(params.angleMinDeg, params.angleMaxDeg, coarseAngleStep, coarseAngles);
        BuildScaleSearchRange(params.scaleMin, params.scaleMax, coarseScaleStep, coarseScales);

        SearchAngleScaleGrid(searchPyramid[static_cast<std::size_t>(topLevel)], searchX0i,
                             searchY0i, grayTpl, coarseAngles, coarseScales,
                             CoarseMinScore(minScore, params.greediness), false, nullptr, topLevel,
                             &params, candidates);

        if (candidates.empty()) {
            error = u8"粗搜未找到候选（可放宽角度/缩放范围或降低最小得分）";
            return false;
        }

        PickTopCandidates(candidates, std::max(params.maxMatches * 2, 6));

        for (int lvl = topLevel - 1; lvl >= 0; --lvl) {
            const cv::Mat& lvlSearch = searchPyramid[static_cast<std::size_t>(lvl)];
            const int lvlW = lvlSearch.cols;
            const int lvlH = lvlSearch.rows;
            const float angleStep = params.angleStepDeg * static_cast<float>(1 << lvl);
            const float scaleStep = params.scaleStep * static_cast<float>(1 << lvl);

            std::vector<TemplateMatchCandidate> refined;
            refined.reserve(candidates.size() * 9u);

            for (const TemplateMatchCandidate& seed : candidates) {
                std::vector<float> angles;
                std::vector<float> scales;
                BuildSearchRange(seed.angleDeg - angleStep * 2.f, seed.angleDeg + angleStep * 2.f,
                                 std::max(angleStep, 0.5f), angles);
                BuildScaleSearchRange(seed.scale - scaleStep * 2.f, seed.scale + scaleStep * 2.f,
                                 std::max(scaleStep, 0.01f), scales);
                const cv::Rect localRoi =
                    LocalRefineRoi(seed, searchX0i, searchY0i, lvl, lvlW, lvlH, tplW, tplH);
                SearchAngleScaleGrid(lvlSearch, searchX0i, searchY0i, grayTpl, angles, scales,
                                     minScore * (lvl == 0 ? 1.f : 0.9f),
                                     subPixel && lvl == 0, &localRoi, lvl, &params, refined);
            }

            if (refined.empty()) continue;
            candidates = std::move(refined);
            PickTopCandidates(candidates, std::max(params.maxMatches * 2, 6));
        }
    }

    if (candidates.empty()) {
        error = u8"未找到满足得分阈值的匹配（可放宽角度/缩放范围或降低最小得分）";
        return false;
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const TemplateMatchCandidate& a, const TemplateMatchCandidate& b) {
                  return a.score > b.score;
              });

    std::vector<TemplateMatchCandidate> picked;
    if (!PickMatchCandidates(candidates, minScore, params.maxMatches, params.maxOverlap, picked)) {
        error = u8"未找到满足得分阈值的匹配";
        return false;
    }

    result.hits.reserve(picked.size());
    for (const TemplateMatchCandidate& c : picked) {
        TemplateMatchHit hit;
        hit.centerX = c.centerX;
        hit.centerY = c.centerY;
        hit.templateWidth = c.templateWidth;
        hit.templateHeight = c.templateHeight;
        hit.angleDeg = c.angleDeg;
        hit.scale = c.scale;
        hit.score = c.score;
        hit.bboxX = c.bboxX;
        hit.bboxY = c.bboxY;
        hit.bboxW = c.bboxW;
        hit.bboxH = c.bboxH;
        result.hits.push_back(hit);
    }

    result.ok = true;
    return true;
}

bool MatchTemplateGrayPatch(const cv::Mat& graySearch, int searchGlobalX, int searchGlobalY,
                            const cv::Mat& grayTpl, const TemplateMatchParams& params,
                            TemplateMatchResult& result, std::string& error) {
    result = {};
    if (graySearch.empty() || grayTpl.empty()) {
        error = u8"模板或搜索图像无效";
        return false;
    }
    const int tplW = grayTpl.cols;
    const int tplH = grayTpl.rows;
    if (tplW < 4 || tplH < 4) {
        error = u8"模板过小";
        return false;
    }
    const int searchW = graySearch.cols;
    const int searchH = graySearch.rows;
    if (searchW < tplW || searchH < tplH) {
        error = u8"搜索区域小于模板";
        return false;
    }

    result.templateX0 = 0.f;
    result.templateY0 = 0.f;
    result.templateX1 = static_cast<float>(tplW - 1);
    result.templateY1 = static_cast<float>(tplH - 1);
    result.searchX0 = static_cast<float>(searchGlobalX);
    result.searchY0 = static_cast<float>(searchGlobalY);
    result.searchX1 = static_cast<float>(searchGlobalX + searchW - 1);
    result.searchY1 = static_cast<float>(searchGlobalY + searchH - 1);

    const bool subPixel = params.subPixelRefine;
    const float minScore = std::clamp(params.minScore, 0.f, 1.f);
    std::vector<TemplateMatchCandidate> candidates;

    if (!params.usePyramid) {
        std::vector<float> angles;
        std::vector<float> scales;
        BuildSearchRange(params.angleMinDeg, params.angleMaxDeg, params.angleStepDeg, angles);
        BuildScaleSearchRange(params.scaleMin, params.scaleMax, params.scaleStep, scales);
        SearchAngleScaleGrid(graySearch, searchGlobalX, searchGlobalY, grayTpl, angles, scales,
                             minScore, subPixel, nullptr, 0, &params, candidates);
    } else {
        const int numLevels = AutoPyramidLevels(tplW, tplH, params.pyramidLevels);
        std::vector<cv::Mat> searchPyramid;
        BuildGrayPyramid(graySearch, numLevels, searchPyramid);
        const int topLevel = static_cast<int>(searchPyramid.size()) - 1;

        std::vector<float> coarseAngles;
        std::vector<float> coarseScales;
        const float coarseAngleStep = params.angleStepDeg * static_cast<float>(1 << topLevel);
        const float coarseScaleStep = params.scaleStep * static_cast<float>(1 << topLevel);
        BuildSearchRange(params.angleMinDeg, params.angleMaxDeg, coarseAngleStep, coarseAngles);
        BuildScaleSearchRange(params.scaleMin, params.scaleMax, coarseScaleStep, coarseScales);

        SearchAngleScaleGrid(searchPyramid[static_cast<std::size_t>(topLevel)], searchGlobalX,
                             searchGlobalY, grayTpl, coarseAngles, coarseScales,
                             CoarseMinScore(minScore, params.greediness), false, nullptr, topLevel,
                             &params, candidates);

        if (candidates.empty()) {
            error = u8"粗搜未找到候选（可放宽角度/缩放范围或降低最小得分）";
            return false;
        }

        PickTopCandidates(candidates, std::max(params.maxMatches * 2, 6));

        for (int lvl = topLevel - 1; lvl >= 0; --lvl) {
            const cv::Mat& lvlSearch = searchPyramid[static_cast<std::size_t>(lvl)];
            const int lvlW = lvlSearch.cols;
            const int lvlH = lvlSearch.rows;
            const float angleStep = params.angleStepDeg * static_cast<float>(1 << lvl);
            const float scaleStep = params.scaleStep * static_cast<float>(1 << lvl);

            std::vector<TemplateMatchCandidate> refined;
            refined.reserve(candidates.size() * 9u);

            for (const TemplateMatchCandidate& seed : candidates) {
                std::vector<float> angles;
                std::vector<float> scales;
                BuildSearchRange(seed.angleDeg - angleStep * 2.f, seed.angleDeg + angleStep * 2.f,
                                 std::max(angleStep, 0.5f), angles);
                BuildScaleSearchRange(seed.scale - scaleStep * 2.f, seed.scale + scaleStep * 2.f,
                                 std::max(scaleStep, 0.01f), scales);
                const cv::Rect localRoi =
                    LocalRefineRoi(seed, searchGlobalX, searchGlobalY, lvl, lvlW, lvlH, tplW,
                                   tplH);
                SearchAngleScaleGrid(lvlSearch, searchGlobalX, searchGlobalY, grayTpl, angles,
                                     scales, minScore * (lvl == 0 ? 1.f : 0.9f),
                                     subPixel && lvl == 0, &localRoi, lvl, &params, refined);
            }

            if (refined.empty()) continue;
            candidates = std::move(refined);
            PickTopCandidates(candidates, std::max(params.maxMatches * 2, 6));
        }
    }

    if (candidates.empty()) {
        error = u8"未找到满足得分阈值的匹配";
        return false;
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const TemplateMatchCandidate& a, const TemplateMatchCandidate& b) {
                  return a.score > b.score;
              });

    std::vector<TemplateMatchCandidate> picked;
    if (!PickMatchCandidates(candidates, minScore, params.maxMatches, params.maxOverlap, picked)) {
        error = u8"未找到满足得分阈值的匹配";
        return false;
    }

    result.hits.reserve(picked.size());
    for (const TemplateMatchCandidate& c : picked) {
        TemplateMatchHit hit;
        hit.centerX = c.centerX;
        hit.centerY = c.centerY;
        hit.templateWidth = c.templateWidth;
        hit.templateHeight = c.templateHeight;
        hit.angleDeg = c.angleDeg;
        hit.scale = c.scale;
        hit.score = c.score;
        hit.bboxX = c.bboxX;
        hit.bboxY = c.bboxY;
        hit.bboxW = c.bboxW;
        hit.bboxH = c.bboxH;
        result.hits.push_back(hit);
    }

    result.ok = true;
    return true;
}

}  // namespace OpenCv2D
