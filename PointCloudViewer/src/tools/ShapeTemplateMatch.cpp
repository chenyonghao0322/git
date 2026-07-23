#include "tools/ShapeTemplateMatch.h"

#include "tools/OpenCv2D.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

namespace ShapeTemplateMatch {
namespace {

void NormalizeRoi(float& x0, float& y0, float& x1, float& y1) {
    if (x0 > x1) std::swap(x0, x1);
    if (y0 > y1) std::swap(y0, y1);
}

int RoiSpan(float a, float b) {
    return std::max(1, static_cast<int>(std::floor(b)) - static_cast<int>(std::ceil(a)) + 1);
}

cv::Mat RgbToGray(const std::vector<uint8_t>& rgb, int width, int height) {
    cv::Mat image(height, width, CV_8UC3, const_cast<uint8_t*>(rgb.data()));
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_RGB2GRAY);
    return gray;
}

cv::Mat BuildEdgeGray(const cv::Mat& gray, const CreateParams& params) {
    cv::Mat blurred;
    const double sigmaX =
        params.semiAxisLong > 0.f ? static_cast<double>(params.semiAxisLong) : 0.0;
    const double sigmaY = params.semiAxisShort > 0.f ? static_cast<double>(params.semiAxisShort)
                                                     : (sigmaX > 0.0 ? sigmaX : 0.0);
    if (sigmaX > 0.0) {
        const int kx = std::max(3, static_cast<int>(std::ceil(sigmaX * 6.0)) | 1);
        const int ky = std::max(3, static_cast<int>(std::ceil(sigmaY * 6.0)) | 1);
        cv::GaussianBlur(gray, blurred, cv::Size(kx, ky), sigmaX, sigmaY);
    } else {
        cv::GaussianBlur(gray, blurred, cv::Size(3, 3), 0.0);
    }
    cv::Mat edges;
    int low = std::clamp(params.contrastLow, 1, 255);
    int high = std::clamp(std::max(params.contrastHigh, low + 1), low + 1, 255);
    if (params.minContrast > 0) {
        low = std::max(low, params.minContrast);
    }
    cv::Canny(blurred, edges, static_cast<double>(low), static_cast<double>(high));
    if (params.minComponentSize > 1) {
        cv::Mat labels;
        cv::Mat stats;
        cv::Mat centroids;
        const int n = cv::connectedComponentsWithStats(edges, labels, stats, centroids, 8, CV_32S);
        cv::Mat filtered = cv::Mat::zeros(edges.size(), CV_8UC1);
        for (int i = 1; i < n; ++i) {
            if (stats.at<int>(i, cv::CC_STAT_AREA) >= params.minComponentSize) {
                filtered.setTo(255, labels == i);
            }
        }
        edges = filtered;
    }
    return edges;
}

void ExtractAllContours(const cv::Mat& edges, std::vector<std::vector<cv::Point>>& contours,
                        int minPoints = 6) {
    contours.clear();
    std::vector<std::vector<cv::Point>> raw;
    cv::findContours(edges, raw, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
    for (auto& c : raw) {
        if (static_cast<int>(c.size()) >= minPoints) contours.push_back(std::move(c));
    }
}

std::vector<cv::Point2f> FlattenContours(const std::vector<std::vector<cv::Point>>& contours) {
    std::vector<cv::Point2f> flat;
    for (const auto& c : contours) {
        for (const cv::Point& p : c) {
            flat.emplace_back(static_cast<float>(p.x), static_cast<float>(p.y));
        }
    }
    return flat;
}

void BuildShapePreview(const cv::Mat& edges, std::vector<uint8_t>& previewRgb) {
    cv::Mat preview(edges.rows, edges.cols, CV_8UC3, cv::Scalar(28, 28, 30));
    for (int y = 0; y < edges.rows; ++y) {
        const uint8_t* row = edges.ptr<uint8_t>(y);
        uint8_t* out = preview.ptr<uint8_t>(y);
        for (int x = 0; x < edges.cols; ++x) {
            if (row[x] > 0) {
                out[x * 3 + 0] = 40;
                out[x * 3 + 1] = 220;
                out[x * 3 + 2] = 80;
            }
        }
    }
    previewRgb.resize(static_cast<std::size_t>(preview.total() * preview.channels()));
    std::memcpy(previewRgb.data(), preview.data, previewRgb.size());
}

void StoreContours(const std::vector<std::vector<cv::Point>>& contours, float offX, float offY,
                   int minPoints, std::vector<float>& outX, std::vector<float>& outY,
                   std::vector<int>& outStarts) {
    outX.clear();
    outY.clear();
    outStarts.clear();
    for (const auto& c : contours) {
        if (static_cast<int>(c.size()) < minPoints) continue;
        outStarts.push_back(static_cast<int>(outX.size()));
        for (const cv::Point& p : c) {
            outX.push_back(static_cast<float>(p.x) + offX);
            outY.push_back(static_cast<float>(p.y) + offY);
        }
    }
}

OpenCv2D::TemplateMatchParams ToOpenCvParams(const FindParams& params, const ShapeModel& model,
                                             int searchW, int searchH) {
    OpenCv2D::TemplateMatchParams p;
    p.minScore = params.minScore;
    const int want = params.numMatches > 0 ? std::clamp(params.numMatches, 1, 32) : 8;
    p.maxMatches = want;
    p.searchFullImage = false;
    p.angleMinDeg = params.angleStart;
    p.angleMaxDeg = params.angleStart + params.angleRange;

    int pyramidLevels = 2;
    if (params.endPyramid > 0) {
        pyramidLevels = std::clamp(params.endPyramid, 1, 4);
    } else if (model.createParams.pyramid > 0) {
        pyramidLevels = std::clamp(model.createParams.pyramid, 1, 4);
    }
    const int tplW = model.templateW;
    const int tplH = model.templateH;
    if (tplW * tplH > 120 * 120) {
        pyramidLevels = std::min(pyramidLevels, 2);
    }
    p.pyramidLevels = pyramidLevels;

    // 金字塔层数少时必须用更细的步长，否则角度/缩放严重偏离
    const bool fineSearch = pyramidLevels <= 1;
    if (params.angleRange <= 30.f) {
        p.angleStepDeg = fineSearch ? 2.f : 4.f;
    } else if (params.angleRange <= 90.f) {
        p.angleStepDeg = fineSearch ? 4.f : 8.f;
    } else if (params.angleRange >= 180.f) {
        p.angleStepDeg = fineSearch ? 5.f : 12.f;
    } else {
        p.angleStepDeg = fineSearch ? 5.f : 8.f;
    }

    p.scaleMin = params.scaleMin;
    p.scaleMax = params.isotropicScale ? params.scaleMax : params.scale2Max;
    const float scaleSpan = std::max(p.scaleMax - p.scaleMin, 0.1f);
    if (scaleSpan <= 0.35f) {
        p.scaleStep = fineSearch ? 0.04f : 0.06f;
    } else if (scaleSpan <= 0.8f) {
        p.scaleStep = fineSearch ? 0.06f : 0.1f;
    } else {
        p.scaleStep = fineSearch ? 0.08f : 0.15f;
    }

    p.usePyramid = true;
    p.subPixelRefine = params.subPixel;
    p.greediness = std::clamp(params.greediness, 0.f, 1.f);
    p.maxOverlap = std::clamp(params.maxOverlap, 0.f, 0.95f);
    p.borderIntersect = params.borderIntersect;
    p.searchGlobalW = searchW;
    p.searchGlobalH = searchH;
    return p;
}

void NormalizeMatchGray(cv::Mat& gray) {
    if (gray.empty()) return;
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    clahe->apply(gray, gray);
}

cv::Rect TightEdgeRect(const cv::Mat& edges, int pad = 4) {
    std::vector<cv::Point> pts;
    cv::findNonZero(edges, pts);
    if (pts.empty()) return cv::Rect(0, 0, edges.cols, edges.rows);
    cv::Rect tight = cv::boundingRect(pts);
    tight.x = std::max(0, tight.x - pad);
    tight.y = std::max(0, tight.y - pad);
    tight.width = std::min(edges.cols - tight.x, tight.width + pad * 2);
    tight.height = std::min(edges.rows - tight.y, tight.height + pad * 2);
    return tight;
}

bool RefineBestHit(const cv::Mat& searchPatch, int searchX0i, int searchY0i,
                   const cv::Mat& tplGray, const OpenCv2D::TemplateMatchParams& baseParams,
                   OpenCv2D::TemplateMatchHit& hit) {
    OpenCv2D::TemplateMatchParams fine = baseParams;
    fine.usePyramid = false;
    fine.subPixelRefine = true;
    fine.maxMatches = 1;
    fine.angleMinDeg = hit.angleDeg - 6.f;
    fine.angleMaxDeg = hit.angleDeg + 6.f;
    fine.angleStepDeg = 1.f;
    fine.scaleMin = std::max(0.1f, hit.scale - 0.12f);
    fine.scaleMax = hit.scale + 0.12f;
    fine.scaleStep = 0.02f;

    OpenCv2D::TemplateMatchResult refined;
    std::string err;
    if (!OpenCv2D::MatchTemplateGrayPatch(searchPatch, searchX0i, searchY0i, tplGray, fine,
                                          refined, err) ||
        refined.hits.empty()) {
        return false;
    }
    if (refined.hits[0].score >= hit.score) {
        hit = refined.hits[0];
        return true;
    }
    return false;
}

}  // namespace

void TransformContour(const ShapeModel& model, float bboxX, float bboxY, float angleDeg,
                      float scale, float scale2, std::vector<float>& outX,
                      std::vector<float>& outY) {
    outX.resize(model.contourX.size());
    outY.resize(model.contourY.size());
    if (model.contourX.empty()) return;

    const float originX = static_cast<float>(model.templateOriginX);
    const float originY = static_cast<float>(model.templateOriginY);
    const float scaledW = static_cast<float>(model.templateW) * scale;
    const float scaledH = static_cast<float>(model.templateH) * scale2;

    const cv::Point2f center(scaledW * 0.5f, scaledH * 0.5f);
    cv::Mat rotMat = cv::getRotationMatrix2D(center, static_cast<double>(angleDeg), 1.0);
    const cv::Rect2f bbox =
        cv::RotatedRect(cv::Point2f(), cv::Size2f(scaledW, scaledH), angleDeg).boundingRect2f();
    rotMat.at<double>(0, 2) += static_cast<double>(bbox.width * 0.5f - center.x);
    rotMat.at<double>(1, 2) += static_cast<double>(bbox.height * 0.5f - center.y);

    const double m00 = rotMat.at<double>(0, 0);
    const double m01 = rotMat.at<double>(0, 1);
    const double m02 = rotMat.at<double>(0, 2);
    const double m10 = rotMat.at<double>(1, 0);
    const double m11 = rotMat.at<double>(1, 1);
    const double m12 = rotMat.at<double>(1, 2);

    for (std::size_t i = 0; i < model.contourX.size(); ++i) {
        const float sx = (model.contourX[i] - originX) * scale;
        const float sy = (model.contourY[i] - originY) * scale2;
        const float wx = static_cast<float>(m00 * sx + m01 * sy + m02);
        const float wy = static_cast<float>(m10 * sx + m11 * sy + m12);
        outX[i] = bboxX + wx;
        outY[i] = bboxY + wy;
    }
}

bool CreateModel(const std::vector<uint8_t>& rgb, int width, int height, float roiX0, float roiY0,
                 float roiX1, float roiY1, const CreateParams& params, ShapeModel& model,
                 std::string& error) {
    model = {};
    if (rgb.empty() || width <= 0 || height <= 0) {
        error = u8"模板图像无效";
        return false;
    }

    NormalizeRoi(roiX0, roiY0, roiX1, roiY1);
    const int roiW = RoiSpan(roiX0, roiX1);
    const int roiH = RoiSpan(roiY0, roiY1);
    if (roiW < 8 || roiH < 8) {
        error = u8"模板 ROI 过小（至少 8×8）";
        return false;
    }

    const int x0 = static_cast<int>(std::ceil(roiX0));
    const int y0 = static_cast<int>(std::ceil(roiY0));
    if (x0 + roiW > width || y0 + roiH > height) {
        error = u8"模板 ROI 超出图像范围";
        return false;
    }

    cv::Mat gray = RgbToGray(rgb, width, height);
    const cv::Rect roiRect(x0, y0, roiW, roiH);
    const cv::Mat fullEdges = BuildEdgeGray(gray, params);
    const cv::Mat edgesRoi = fullEdges(roiRect);
    const cv::Rect tight = TightEdgeRect(edgesRoi);
    const cv::Mat edges = edgesRoi(tight);

    std::vector<std::vector<cv::Point>> localContours;
    ExtractAllContours(edges, localContours);
    const std::vector<cv::Point2f> localContour = FlattenContours(localContours);
    if (localContour.size() < 8) {
        error = u8"未提取到足够边缘，请调低对比度阈值或缩小 ROI";
        return false;
    }

    cv::Moments m = cv::moments(localContour);
    const float cx = m.m00 > 1e-6 ? static_cast<float>(m.m10 / m.m00) : tight.width * 0.5f;
    const float cy = m.m00 > 1e-6 ? static_cast<float>(m.m01 / m.m00) : tight.height * 0.5f;

    cv::Mat image(height, width, CV_8UC3, const_cast<uint8_t*>(rgb.data()));
    const cv::Mat tplRgb = image(roiRect)(tight).clone();
    const cv::Mat tplGray = gray(roiRect)(tight).clone();

    const int tx0 = x0 + tight.x;
    const int ty0 = y0 + tight.y;
    const int tplW = tight.width;
    const int tplH = tight.height;

    model.valid = true;
    model.templateX0 = static_cast<float>(tx0);
    model.templateY0 = static_cast<float>(ty0);
    model.templateX1 = static_cast<float>(tx0 + tplW - 1);
    model.templateY1 = static_cast<float>(ty0 + tplH - 1);
    model.templateW = tplW;
    model.templateH = tplH;
    model.templateOriginX = tx0;
    model.templateOriginY = ty0;
    model.centerX = static_cast<float>(tx0) + cx;
    model.centerY = static_cast<float>(ty0) + cy;
    model.imageWidth = width;
    model.imageHeight = height;
    model.createParams = params;

    model.templateRgb.resize(static_cast<std::size_t>(tplRgb.total() * tplRgb.channels()));
    std::memcpy(model.templateRgb.data(), tplRgb.data, model.templateRgb.size());

    model.templateGray.resize(static_cast<std::size_t>(tplGray.total()));
    std::memcpy(model.templateGray.data(), tplGray.data, model.templateGray.size());

    model.templateEdge.resize(static_cast<std::size_t>(edges.total()));
    std::memcpy(model.templateEdge.data(), edges.data, model.templateEdge.size());

    BuildShapePreview(edges, model.previewRgb);
    StoreContours(localContours, static_cast<float>(tx0), static_cast<float>(ty0), 6,
                  model.contourX, model.contourY, model.contourStarts);
    return true;
}

bool FindModel(const std::vector<uint8_t>& rgb, int width, int height, float searchX0,
               float searchY0, float searchX1, float searchY1, bool useSearchRoi,
               const ShapeModel& model, const FindParams& params, MatchResult& result,
               std::string& error) {
    result = {};
    if (!model.valid) {
        error = u8"请先创建或加载模板";
        return false;
    }
    if (rgb.empty() || width <= 0 || height <= 0) {
        error = u8"源图像无效";
        return false;
    }
    if (model.templateEdge.empty() || model.templateW <= 0 || model.templateH <= 0) {
        error = u8"模板边缘数据无效，请重新创建模板";
        return false;
    }

    const auto t0 = std::chrono::steady_clock::now();

    cv::Mat gray = RgbToGray(rgb, width, height);
    NormalizeMatchGray(gray);

    cv::Mat tplGray;
    if (!model.templateGray.empty()) {
        tplGray = cv::Mat(model.templateH, model.templateW, CV_8UC1,
                          const_cast<uint8_t*>(model.templateGray.data())).clone();
    } else if (!model.templateRgb.empty()) {
        cv::Mat tplRgb(model.templateH, model.templateW, CV_8UC3,
                       const_cast<uint8_t*>(model.templateRgb.data()));
        cv::cvtColor(tplRgb, tplGray, cv::COLOR_RGB2GRAY);
    } else {
        error = u8"模板灰度数据无效，请重新创建模板";
        return false;
    }
    NormalizeMatchGray(tplGray);

    float sx0 = 0.f;
    float sy0 = 0.f;
    float sx1 = static_cast<float>(width - 1);
    float sy1 = static_cast<float>(height - 1);
    if (useSearchRoi) {
        sx0 = searchX0;
        sy0 = searchY0;
        sx1 = searchX1;
        sy1 = searchY1;
    }
    NormalizeRoi(sx0, sy0, sx1, sy1);
    const int searchX0i = static_cast<int>(std::ceil(sx0));
    const int searchY0i = static_cast<int>(std::ceil(sy0));
    const int searchW = RoiSpan(sx0, sx1);
    const int searchH = RoiSpan(sy0, sy1);
    if (searchW < model.templateW || searchH < model.templateH) {
        error = u8"搜索区域小于模板";
        return false;
    }
    const cv::Rect searchRect(searchX0i, searchY0i, searchW, searchH);
    if (searchRect.x + searchRect.width > width || searchRect.y + searchRect.height > height) {
        error = u8"搜索区域超出图像范围";
        return false;
    }
    const cv::Mat searchPatch = gray(searchRect);

    OpenCv2D::TemplateMatchParams matchParams =
        ToOpenCvParams(params, model, searchW, searchH);

    OpenCv2D::TemplateMatchResult cvResult;
    const bool ok = OpenCv2D::MatchTemplateGrayPatch(searchPatch, searchX0i, searchY0i, tplGray,
                                                    matchParams, cvResult, error);
    if (!ok) return false;

    if (!cvResult.hits.empty()) {
        RefineBestHit(searchPatch, searchX0i, searchY0i, tplGray, matchParams, cvResult.hits[0]);
        const float bestScore = cvResult.hits[0].score;
        const float scoreFloor = std::max(params.minScore, bestScore * 0.88f);
        cvResult.hits.erase(
            std::remove_if(cvResult.hits.begin(), cvResult.hits.end(),
                           [scoreFloor](const OpenCv2D::TemplateMatchHit& h) {
                               return h.score < scoreFloor;
                           }),
            cvResult.hits.end());
        if (cvResult.hits.empty()) {
            error = u8"精修后无满足得分阈值的目标";
            return false;
        }
    }

    result.hits.reserve(cvResult.hits.size());
    for (const OpenCv2D::TemplateMatchHit& h : cvResult.hits) {
        MatchHit hit;
        hit.centerX = h.centerX;
        hit.centerY = h.centerY;
        hit.angleDeg = h.angleDeg;
        hit.scale = h.scale;
        hit.scale2 = h.scale;
        hit.score = h.score;
        TransformContour(model, h.bboxX, h.bboxY, hit.angleDeg, hit.scale, hit.scale2,
                         hit.contourX, hit.contourY);
        result.hits.push_back(std::move(hit));
    }

    const auto t1 = std::chrono::steady_clock::now();
    result.elapsedMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    result.ok = !result.hits.empty();
    return result.ok;
}

bool SaveModel(const std::string& path, const ShapeModel& model, std::string& error) {
    if (!model.valid) {
        error = u8"没有可保存的模板";
        return false;
    }
    cv::FileStorage fs(path, cv::FileStorage::WRITE);
    if (!fs.isOpened()) {
        error = u8"无法写入模板文件";
        return false;
    }
    fs << "version" << 2;
    fs << "templateX0" << model.templateX0;
    fs << "templateY0" << model.templateY0;
    fs << "templateX1" << model.templateX1;
    fs << "templateY1" << model.templateY1;
    fs << "centerX" << model.centerX;
    fs << "centerY" << model.centerY;
    fs << "templateW" << model.templateW;
    fs << "templateH" << model.templateH;
    fs << "templateOriginX" << model.templateOriginX;
    fs << "templateOriginY" << model.templateOriginY;
    fs << "imageWidth" << model.imageWidth;
    fs << "imageHeight" << model.imageHeight;
    fs << "contrastLow" << model.createParams.contrastLow;
    fs << "contrastHigh" << model.createParams.contrastHigh;
    fs << "pyramid" << model.createParams.pyramid;
    fs << "minContrast" << model.createParams.minContrast;
    fs << "minComponentSize" << model.createParams.minComponentSize;
    fs << "semiAxisLong" << model.createParams.semiAxisLong;
    fs << "semiAxisShort" << model.createParams.semiAxisShort;
    fs << "usePolarity" << (model.createParams.usePolarity ? 1 : 0);
    fs << "contourX" << model.contourX;
    fs << "contourY" << model.contourY;
    fs << "contourStarts" << model.contourStarts;

    cv::Mat tpl(model.templateH, model.templateW, CV_8UC3,
                const_cast<uint8_t*>(model.templateRgb.data()));
    fs << "templateImage" << tpl;
    cv::Mat edge(model.templateH, model.templateW, CV_8UC1,
                 const_cast<uint8_t*>(model.templateEdge.data()));
    fs << "templateEdge" << edge;
    cv::Mat prev(model.templateH, model.templateW, CV_8UC3,
                 const_cast<uint8_t*>(model.previewRgb.data()));
    fs << "previewImage" << prev;
    fs.release();
    return true;
}

bool LoadModel(const std::string& path, ShapeModel& model, std::string& error) {
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        error = u8"无法打开模板文件";
        return false;
    }
    int version = 0;
    fs["version"] >> version;
    if (version != 1 && version != 2) {
        error = u8"不支持的模板文件版本";
        return false;
    }

    model = {};
    fs["templateX0"] >> model.templateX0;
    fs["templateY0"] >> model.templateY0;
    fs["templateX1"] >> model.templateX1;
    fs["templateY1"] >> model.templateY1;
    fs["centerX"] >> model.centerX;
    fs["centerY"] >> model.centerY;
    if (version >= 2) {
        fs["templateW"] >> model.templateW;
        fs["templateH"] >> model.templateH;
        fs["templateOriginX"] >> model.templateOriginX;
        fs["templateOriginY"] >> model.templateOriginY;
    }
    fs["imageWidth"] >> model.imageWidth;
    fs["imageHeight"] >> model.imageHeight;
    fs["contrastLow"] >> model.createParams.contrastLow;
    fs["contrastHigh"] >> model.createParams.contrastHigh;
    fs["pyramid"] >> model.createParams.pyramid;
    fs["minContrast"] >> model.createParams.minContrast;
    fs["minComponentSize"] >> model.createParams.minComponentSize;
    fs["semiAxisLong"] >> model.createParams.semiAxisLong;
    fs["semiAxisShort"] >> model.createParams.semiAxisShort;
    int usePolarity = 1;
    fs["usePolarity"] >> usePolarity;
    model.createParams.usePolarity = usePolarity != 0;
    fs["contourX"] >> model.contourX;
    fs["contourY"] >> model.contourY;
    fs["contourStarts"] >> model.contourStarts;
    if (model.contourStarts.empty() && !model.contourX.empty()) {
        model.contourStarts.push_back(0);
    }

    cv::Mat tpl;
    fs["templateImage"] >> tpl;
    if (tpl.empty()) {
        error = u8"模板图像数据缺失";
        return false;
    }
    if (model.templateW <= 0) model.templateW = tpl.cols;
    if (model.templateH <= 0) model.templateH = tpl.rows;
    if (version < 2) {
        model.templateOriginX = static_cast<int>(std::ceil(model.templateX0));
        model.templateOriginY = static_cast<int>(std::ceil(model.templateY0));
    }
    model.templateRgb.resize(static_cast<std::size_t>(tpl.total() * tpl.channels()));
    std::memcpy(model.templateRgb.data(), tpl.data, model.templateRgb.size());

    cv::Mat tplG;
    cv::cvtColor(tpl, tplG, cv::COLOR_RGB2GRAY);
    model.templateGray.resize(static_cast<std::size_t>(tplG.total()));
    std::memcpy(model.templateGray.data(), tplG.data, model.templateGray.size());

    cv::Mat edge;
    fs["templateEdge"] >> edge;
    if (!edge.empty()) {
        model.templateEdge.resize(static_cast<std::size_t>(edge.total()));
        std::memcpy(model.templateEdge.data(), edge.data, model.templateEdge.size());
    } else {
        cv::Mat gray;
        cv::cvtColor(tpl, gray, cv::COLOR_RGB2GRAY);
        edge = BuildEdgeGray(gray, model.createParams);
        model.templateEdge.resize(static_cast<std::size_t>(edge.total()));
        std::memcpy(model.templateEdge.data(), edge.data, model.templateEdge.size());
    }

    cv::Mat prev;
    fs["previewImage"] >> prev;
    if (!prev.empty()) {
        model.previewRgb.resize(static_cast<std::size_t>(prev.total() * prev.channels()));
        std::memcpy(model.previewRgb.data(), prev.data, model.previewRgb.size());
    } else {
        cv::Mat edgeMat(model.templateH, model.templateW, CV_8UC1, model.templateEdge.data());
        BuildShapePreview(edgeMat, model.previewRgb);
    }

    model.valid = !model.contourX.empty();
    if (!model.valid) {
        error = u8"模板轮廓数据无效";
        return false;
    }
    return true;
}

}  // namespace ShapeTemplateMatch
