#include "tools/Camera2DCalibration.h"

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>

namespace Camera2DCalibration {
namespace {

bool CollectValidPairs(const std::vector<PointPair>& pairs, std::vector<cv::Point2f>& imagePts,
                       std::vector<cv::Point2f>& robotPts) {
    imagePts.clear();
    robotPts.clear();
    imagePts.reserve(pairs.size());
    robotPts.reserve(pairs.size());
    for (const PointPair& p : pairs) {
        if (!p.hasImagePoint || !p.hasRobotCoord) continue;
        imagePts.emplace_back(p.imageU, p.imageV);
        robotPts.emplace_back(p.robotX, p.robotY);
    }
    return imagePts.size() >= 3;
}

double ComputeRms(const std::vector<cv::Point2f>& imagePts,
                  const std::vector<cv::Point2f>& robotPts, const AffineResult& result) {
    if (imagePts.empty()) return 0.0;
    double sumSq = 0.0;
    for (std::size_t i = 0; i < imagePts.size(); ++i) {
        const float predX = static_cast<float>(result.a * imagePts[i].x + result.b * imagePts[i].y +
                                               result.c);
        const float predY = static_cast<float>(result.d * imagePts[i].x + result.e * imagePts[i].y +
                                               result.f);
        const double dx = predX - robotPts[i].x;
        const double dy = predY - robotPts[i].y;
        sumSq += dx * dx + dy * dy;
    }
    return std::sqrt(sumSq / static_cast<double>(imagePts.size()));
}

}  // namespace

bool ComputeAffine(const std::vector<PointPair>& pairs, AffineResult& result, std::string& error) {
    result = {};
    std::vector<cv::Point2f> imagePts;
    std::vector<cv::Point2f> robotPts;
    if (!CollectValidPairs(pairs, imagePts, robotPts)) {
        error = u8"至少需要 3 组完整对应点（图像坐标 + 机器人坐标）";
        return false;
    }

    cv::Mat M = cv::estimateAffine2D(imagePts, robotPts);
    if (M.empty() || M.rows != 2 || M.cols != 3) {
        error = u8"标定求解失败，请检查点位是否共线或数据重复";
        return false;
    }

    result.a = M.at<double>(0, 0);
    result.b = M.at<double>(0, 1);
    result.c = M.at<double>(0, 2);
    result.d = M.at<double>(1, 0);
    result.e = M.at<double>(1, 1);
    result.f = M.at<double>(1, 2);
    result.pointCount = static_cast<int>(imagePts.size());
    result.rms = ComputeRms(imagePts, robotPts, result);
    result.valid = true;
    return true;
}

void ImageToRobot(float u, float v, const AffineResult& result, float& outX, float& outY) {
    outX = static_cast<float>(result.a * u + result.b * v + result.c);
    outY = static_cast<float>(result.d * u + result.e * v + result.f);
}

bool ComputeErrorStats(const std::vector<PointPair>& pairs, const AffineResult& result,
                       ErrorStats& stats, std::string& error) {
    stats = {};
    std::vector<double> errs;
    errs.reserve(pairs.size());
    for (const PointPair& p : pairs) {
        if (!p.hasImagePoint || !p.hasRobotCoord) continue;
        float qx = 0.f;
        float qy = 0.f;
        ImageToRobot(p.imageU, p.imageV, result, qx, qy);
        const double dx = static_cast<double>(p.robotX) - static_cast<double>(qx);
        const double dy = static_cast<double>(p.robotY) - static_cast<double>(qy);
        errs.push_back(std::sqrt(dx * dx + dy * dy));
    }
    if (errs.empty()) {
        error = u8"无有效标定点，无法统计误差";
        return false;
    }

    double sum = 0.0;
    double errMax = 0.0;
    for (double e : errs) {
        sum += e;
        errMax = std::max(errMax, e);
    }
    stats.perPointError = std::move(errs);
    stats.pointCount = static_cast<int>(stats.perPointError.size());
    stats.errMean = sum / static_cast<double>(stats.pointCount);
    stats.errMax = errMax;
    stats.valid = true;
    return true;
}

bool DetectCalibrationDotCenterRgb(const std::vector<uint8_t>& rgb, int width, int height,
                                   float roiX0, float roiY0, float roiX1, float roiY1,
                                   DotDetectResult& result, std::string& error) {
    result = {};
    if (rgb.empty() || width <= 0 || height <= 0) {
        error = u8"图像无效";
        return false;
    }

    const int x0 = std::clamp(static_cast<int>(std::floor(std::min(roiX0, roiX1))), 0, width - 1);
    const int y0 = std::clamp(static_cast<int>(std::floor(std::min(roiY0, roiY1))), 0, height - 1);
    const int x1 = std::clamp(static_cast<int>(std::ceil(std::max(roiX0, roiX1))), x0 + 1, width);
    const int y1 = std::clamp(static_cast<int>(std::ceil(std::max(roiY0, roiY1))), y0 + 1, height);
    const int roiW = x1 - x0;
    const int roiH = y1 - y0;
    if (roiW < 8 || roiH < 8) {
        error = u8"框选区域过小，请重新框选";
        return false;
    }

    cv::Mat full(height, width, CV_8UC3, const_cast<uint8_t*>(rgb.data()));
    cv::Mat roiBgr = full(cv::Rect(x0, y0, roiW, roiH)).clone();
    cv::Mat gray;
    cv::cvtColor(roiBgr, gray, cv::COLOR_RGB2GRAY);
    cv::GaussianBlur(gray, gray, cv::Size(5, 5), 0);

    const float roiCx = static_cast<float>(roiW) * 0.5f;
    const float roiCy = static_cast<float>(roiH) * 0.5f;
    const double minArea = std::max(16.0, static_cast<double>(roiW * roiH) * 0.005);
    const double maxArea = static_cast<double>(roiW * roiH) * 0.95;

    DotDetectResult best;
    double bestScore = -1.0;
    for (int inv : {0, 1}) {
        cv::Mat bin;
        const int flags = cv::THRESH_BINARY | cv::THRESH_OTSU | (inv ? cv::THRESH_BINARY_INV : 0);
        cv::threshold(gray, bin, 0, 255, flags);
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(bin, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
        for (const std::vector<cv::Point>& c : contours) {
            const double area = cv::contourArea(c);
            if (area < minArea || area > maxArea) continue;
            const double per = cv::arcLength(c, true);
            if (per < 1e-3) continue;
            const double circularity = 4.0 * CV_PI * area / (per * per);
            if (circularity < 0.55) continue;

            cv::Point2f center;
            float radius = 0.f;
            cv::minEnclosingCircle(c, center, radius);
            if (radius < 2.f) continue;

            const double dist = std::hypot(center.x - roiCx, center.y - roiCy);
            const double distScore = 1.0 / (1.0 + dist * 0.05);
            const double score = circularity * distScore;
            if (score > bestScore) {
                bestScore = score;
                best.centerX = center.x + static_cast<float>(x0);
                best.centerY = center.y + static_cast<float>(y0);
                best.radius = radius;
                best.ok = true;
            }
        }
    }

    if (!best.ok) {
        error = u8"未在框选区域内找到圆形标定点，请调整框选范围";
        return false;
    }

    result = best;
    return true;
}

}  // namespace Camera2DCalibration
