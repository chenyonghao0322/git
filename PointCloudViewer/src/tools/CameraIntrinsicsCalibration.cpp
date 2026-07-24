#include "tools/CameraIntrinsicsCalibration.h"

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>

namespace CameraIntrinsicsCalibration {
namespace {

bool PatternValid(const PatternConfig& pattern, std::string& error) {
    if (pattern.innerCols < 2 || pattern.innerRows < 2) {
        error = u8"棋盘内角点行列数至少为 2";
        return false;
    }
    if (pattern.squareSizeMm <= 0.0) {
        error = u8"方格边长必须大于 0";
        return false;
    }
    return true;
}

std::vector<cv::Point3f> BuildObjectPoints(const PatternConfig& pattern) {
    std::vector<cv::Point3f> pts;
    pts.reserve(static_cast<std::size_t>(pattern.innerCols * pattern.innerRows));
    for (int r = 0; r < pattern.innerRows; ++r) {
        for (int c = 0; c < pattern.innerCols; ++c) {
            pts.emplace_back(static_cast<float>(c * pattern.squareSizeMm),
                             static_cast<float>(r * pattern.squareSizeMm), 0.f);
        }
    }
    return pts;
}

void StoreCorners(const std::vector<cv::Point2f>& corners, CornerDetectResult& result) {
    result.cornersX.clear();
    result.cornersY.clear();
    result.cornersX.reserve(corners.size());
    result.cornersY.reserve(corners.size());
    for (const cv::Point2f& p : corners) {
        result.cornersX.push_back(p.x);
        result.cornersY.push_back(p.y);
    }
    result.ok = !corners.empty();
}

}  // namespace

bool DetectChessboardCorners(const std::vector<uint8_t>& rgb, int width, int height,
                             const PatternConfig& pattern, CornerDetectResult& result,
                             std::string& error) {
    result = {};
    if (!PatternValid(pattern, error)) return false;
    if (rgb.empty() || width <= 0 || height <= 0) {
        error = u8"图像无效";
        return false;
    }

    cv::Mat bgr(height, width, CV_8UC3, const_cast<uint8_t*>(rgb.data()));
    cv::Mat gray;
    cv::cvtColor(bgr, gray, cv::COLOR_RGB2GRAY);

    const cv::Size boardSize(pattern.innerCols, pattern.innerRows);
    std::vector<cv::Point2f> corners;
    const int flags = cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE;
    const bool found = cv::findChessboardCorners(gray, boardSize, corners, flags);
    if (!found || corners.size() != static_cast<std::size_t>(pattern.innerCols * pattern.innerRows)) {
        error = u8"未检测到棋盘角点，请检查行列数设置或图像质量";
        return false;
    }

    cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
                     cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.001));
    StoreCorners(corners, result);
    return true;
}

bool CalibrateIntrinsics(const std::vector<ImageObservation>& images, const PatternConfig& pattern,
                         IntrinsicsResult& result, std::string& error) {
    result = {};
    if (!PatternValid(pattern, error)) return false;

    const std::vector<cv::Point3f> objectTemplate = BuildObjectPoints(pattern);
    const int expectedCorners = pattern.innerCols * pattern.innerRows;

    std::vector<std::vector<cv::Point3f>> objectPoints;
    std::vector<std::vector<cv::Point2f>> imagePoints;
    cv::Size imageSize;
    int validCount = 0;

    for (const ImageObservation& img : images) {
        if (!img.hasCorners || !img.cornersX || !img.cornersY) continue;
        if (img.cornersX->size() != img.cornersY->size()) continue;
        if (static_cast<int>(img.cornersX->size()) != expectedCorners) continue;
        if (!img.rgb || img.width <= 0 || img.height <= 0) continue;

        std::vector<cv::Point2f> pts;
        pts.reserve(img.cornersX->size());
        for (std::size_t i = 0; i < img.cornersX->size(); ++i) {
            pts.emplace_back((*img.cornersX)[i], (*img.cornersY)[i]);
        }

        objectPoints.push_back(objectTemplate);
        imagePoints.push_back(std::move(pts));
        imageSize = cv::Size(img.width, img.height);
        ++validCount;
    }

    if (validCount < 3) {
        error = u8"至少需要 3 张成功检测角点的图像";
        return false;
    }

    cv::Mat cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat distCoeffs;
    std::vector<cv::Mat> rvecs;
    std::vector<cv::Mat> tvecs;
    const double rms = cv::calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix,
                                           distCoeffs, rvecs, tvecs);

    double totalErr = 0.0;
    int totalPts = 0;
    double maxErr = 0.0;
    result.perImageReprojError.resize(imagePoints.size());

    for (std::size_t i = 0; i < imagePoints.size(); ++i) {
        std::vector<cv::Point2f> projected;
        cv::projectPoints(objectPoints[i], rvecs[i], tvecs[i], cameraMatrix, distCoeffs,
                          projected);
        double imgSum = 0.0;
        double imgMax = 0.0;
        for (std::size_t j = 0; j < projected.size(); ++j) {
            const double dx = projected[j].x - imagePoints[i][j].x;
            const double dy = projected[j].y - imagePoints[i][j].y;
            const double e = std::sqrt(dx * dx + dy * dy);
            imgSum += e;
            imgMax = std::max(imgMax, e);
            maxErr = std::max(maxErr, e);
        }
        result.perImageReprojError[i] = imgSum / static_cast<double>(projected.size());
        totalErr += imgSum;
        totalPts += static_cast<int>(projected.size());
    }

    result.fx = cameraMatrix.at<double>(0, 0);
    result.fy = cameraMatrix.at<double>(1, 1);
    result.cx = cameraMatrix.at<double>(0, 2);
    result.cy = cameraMatrix.at<double>(1, 2);
    if (distCoeffs.total() > 0) result.k1 = distCoeffs.at<double>(0);
    if (distCoeffs.total() > 1) result.k2 = distCoeffs.at<double>(1);
    if (distCoeffs.total() > 2) result.p1 = distCoeffs.at<double>(2);
    if (distCoeffs.total() > 3) result.p2 = distCoeffs.at<double>(3);
    if (distCoeffs.total() > 4) result.k3 = distCoeffs.at<double>(4);

    result.imageWidth = imageSize.width;
    result.imageHeight = imageSize.height;
    result.imageCount = validCount;
    result.cornerCountPerImage = expectedCorners;
    result.reprojMean = totalPts > 0 ? totalErr / static_cast<double>(totalPts) : rms;
    result.reprojMax = maxErr;
    result.valid = true;
    return true;
}

}  // namespace CameraIntrinsicsCalibration
