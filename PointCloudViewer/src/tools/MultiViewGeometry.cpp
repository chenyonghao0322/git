#include "tools/MultiViewGeometry.h"

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>

namespace MultiViewGeometry {
namespace {

cv::Mat IntrinsicsToMat(const CameraIntrinsics& k) {
    cv::Mat m = cv::Mat::eye(3, 3, CV_64F);
    m.at<double>(0, 0) = k.fx;
    m.at<double>(1, 1) = k.fy;
    m.at<double>(0, 2) = k.cx;
    m.at<double>(1, 2) = k.cy;
    return m;
}

double SymmetricEpipolarDistance(const cv::Mat& F, const cv::Point2f& p1, const cv::Point2f& p2) {
    const cv::Matx31d x1(p1.x, p1.y, 1.0);
    const cv::Matx31d x2(p2.x, p2.y, 1.0);
    const cv::Matx<double, 3, 3> Fm = F;
    const cv::Matx31d Fx1 = Fm * x1;
    const cv::Matx31d Ftx2 = Fm.t() * x2;
    const double x2tFx1 = x2(0) * Fx1(0) + x2(1) * Fx1(1) + x2(2) * Fx1(2);
    const double denom = Fx1(0) * Fx1(0) + Fx1(1) * Fx1(1) + Ftx2(0) * Ftx2(0) + Ftx2(1) * Ftx2(1);
    if (denom < 1e-12) return 0.0;
    return std::abs(x2tFx1) / std::sqrt(denom);
}

void FillEpipolarStats(const cv::Mat& F, const std::vector<Correspondence>& pairs,
                       const std::vector<uint8_t>& mask, FundamentalResult& result) {
    double sum = 0.0;
    double maxErr = 0.0;
    int count = 0;
    for (std::size_t i = 0; i < pairs.size(); ++i) {
        if (!mask.empty() && i < mask.size() && mask[i] == 0) continue;
        const Correspondence& p = pairs[i];
        const double err = SymmetricEpipolarDistance(
            F, cv::Point2f(p.u1, p.v1), cv::Point2f(p.u2, p.v2));
        sum += err;
        maxErr = std::max(maxErr, err);
        ++count;
    }
    result.inlierCount = count;
    result.meanEpipolarError = count > 0 ? sum / static_cast<double>(count) : 0.0;
    result.maxEpipolarError = maxErr;
}

}  // namespace

bool ComputeFundamental(const std::vector<Correspondence>& pairs, FundamentalResult& result,
                        std::string& error, double ransacThresh) {
    result = {};
    if (pairs.size() < 8) {
        error = u8"估计基础矩阵至少需要 8 组对应点";
        return false;
    }

    std::vector<cv::Point2f> pts1;
    std::vector<cv::Point2f> pts2;
    pts1.reserve(pairs.size());
    pts2.reserve(pairs.size());
    for (const Correspondence& p : pairs) {
        pts1.emplace_back(p.u1, p.v1);
        pts2.emplace_back(p.u2, p.v2);
    }

    cv::Mat mask;
    const cv::Mat F = cv::findFundamentalMat(pts1, pts2, cv::FM_RANSAC, ransacThresh, 0.99, mask);
    if (F.empty() || F.rows != 3 || F.cols != 3) {
        error = u8"基础矩阵求解失败，请检查对应点分布";
        return false;
    }

    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            result.f[r * 3 + c] = F.at<double>(r, c);
        }
    }
    result.inlierMask.assign(static_cast<std::size_t>(mask.rows), 0);
    for (int i = 0; i < mask.rows; ++i) {
        result.inlierMask[static_cast<std::size_t>(i)] = mask.at<uchar>(i);
    }
    FillEpipolarStats(F, pairs, result.inlierMask, result);
    result.valid = true;
    return true;
}

bool EpipolarLineInImage2(const FundamentalResult& fundamental, float u1, float v1, double& a,
                          double& b, double& c) {
    if (!fundamental.valid) return false;
    cv::Mat F(3, 3, CV_64F);
    for (int r = 0; r < 3; ++r) {
        for (int col = 0; col < 3; ++col) {
            F.at<double>(r, col) = fundamental.f[r * 3 + col];
        }
    }
    std::vector<cv::Point2f> pts1 = {cv::Point2f(u1, v1)};
    std::vector<cv::Point3f> lines;
    cv::computeCorrespondEpilines(pts1, 1, F, lines);
    if (lines.empty()) return false;
    a = lines[0].x;
    b = lines[0].y;
    c = lines[0].z;
    return true;
}

bool TriangulateWithIntrinsics(const std::vector<Correspondence>& pairs,
                               const CameraIntrinsics& intrinsics,
                               const FundamentalResult& fundamental,
                               TriangulationResult& result, std::string& error) {
    result = {};
    if (!fundamental.valid) {
        error = u8"请先计算基础矩阵";
        return false;
    }
    if (pairs.size() < 8) {
        error = u8"三角化至少需要 8 组对应点";
        return false;
    }
    if (intrinsics.fx <= 0.0 || intrinsics.fy <= 0.0) {
        error = u8"相机内参 fx/fy 必须大于 0";
        return false;
    }

    std::vector<cv::Point2f> pts1;
    std::vector<cv::Point2f> pts2;
    pts1.reserve(pairs.size());
    pts2.reserve(pairs.size());
    for (std::size_t i = 0; i < pairs.size(); ++i) {
        if (!fundamental.inlierMask.empty() && fundamental.inlierMask[i] == 0) continue;
        pts1.emplace_back(pairs[i].u1, pairs[i].v1);
        pts2.emplace_back(pairs[i].u2, pairs[i].v2);
    }
    if (pts1.size() < 8) {
        error = u8"内点数量不足，无法三角化";
        return false;
    }

    cv::Mat F(3, 3, CV_64F);
    for (int r = 0; r < 3; ++r) {
        for (int col = 0; col < 3; ++col) {
            F.at<double>(r, col) = fundamental.f[r * 3 + col];
        }
    }
    const cv::Mat K = IntrinsicsToMat(intrinsics);
    cv::Mat E = K.t() * F * K;
    cv::Mat R, t;
    cv::Mat maskPose;
    const int inliers = cv::recoverPose(E, pts1, pts2, K, R, t, 1.0, maskPose);
    if (inliers < 5) {
        error = u8"位姿恢复失败，请检查内参或对应点";
        return false;
    }

    cv::Mat P1 = K * cv::Mat::eye(3, 4, CV_64F);
    cv::Mat Rt(3, 4, CV_64F);
    R.copyTo(Rt(cv::Rect(0, 0, 3, 3)));
    t.copyTo(Rt(cv::Rect(3, 0, 1, 3)));
    cv::Mat P2 = K * Rt;

    cv::Mat pts4d;
    cv::triangulatePoints(P1, P2, pts1, pts2, pts4d);

    result.points.reserve(pts4d.cols);
    result.reprojectionErrors.reserve(pts4d.cols);
    double sumErr = 0.0;
    double maxErr = 0.0;
    for (int i = 0; i < pts4d.cols; ++i) {
        const double w = pts4d.at<float>(3, i);
        if (std::abs(w) < 1e-8) continue;
        const double x = pts4d.at<float>(0, i) / w;
        const double y = pts4d.at<float>(1, i) / w;
        const double z = pts4d.at<float>(2, i) / w;
        if (z <= 1e-6) continue;

        cv::Mat p3 = (cv::Mat_<double>(4, 1) << x, y, z, 1.0);
        cv::Mat proj1 = P1 * p3;
        cv::Mat proj2 = P2 * p3;
        const double u1p = proj1.at<double>(0) / proj1.at<double>(2);
        const double v1p = proj1.at<double>(1) / proj1.at<double>(2);
        const double u2p = proj2.at<double>(0) / proj2.at<double>(2);
        const double v2p = proj2.at<double>(1) / proj2.at<double>(2);
        const double err = std::hypot(u1p - pts1[static_cast<std::size_t>(i)].x,
                                    v1p - pts1[static_cast<std::size_t>(i)].y) +
                           std::hypot(u2p - pts2[static_cast<std::size_t>(i)].x,
                                      v2p - pts2[static_cast<std::size_t>(i)].y);

        result.points.push_back(Vec3{static_cast<float>(x), static_cast<float>(y),
                                     static_cast<float>(z)});
        result.reprojectionErrors.push_back(err);
        sumErr += err;
        maxErr = std::max(maxErr, err);
    }

    if (result.points.empty()) {
        error = u8"三角化未得到有效 3D 点";
        return false;
    }
    result.meanReprojError = sumErr / static_cast<double>(result.points.size());
    result.maxReprojError = maxErr;
    result.valid = true;
    return true;
}

}  // namespace MultiViewGeometry
