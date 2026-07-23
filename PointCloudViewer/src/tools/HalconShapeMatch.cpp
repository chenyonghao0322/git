#include "tools/HalconShapeMatch.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

#ifdef POINTCLOUDVIEWER_USE_HALCON
#include "HalconCpp.h"
#endif

namespace HalconShapeMatch {
namespace {

constexpr double kLocalPi = 3.14159265358979323846;

double LocalDegToRad(double deg) { return deg * kLocalPi / 180.0; }

double LocalRadToDeg(double rad) { return rad * 180.0 / kLocalPi; }

void NormalizeRoi(float& x0, float& y0, float& x1, float& y1) {
    if (x0 > x1) std::swap(x0, x1);
    if (y0 > y1) std::swap(y0, y1);
}

#ifdef POINTCLOUDVIEWER_USE_HALCON
using namespace HalconCpp;

struct HalconModelHolder {
    HShapeModel model;
    HXLDCont contours;
};

HalconModelHolder* AsHolder(ShapeModel& model) {
    return static_cast<HalconModelHolder*>(model.halconModel);
}

const HalconModelHolder* AsHolder(const ShapeModel& model) {
    return static_cast<const HalconModelHolder*>(model.halconModel);
}

HImage RgbToHImage(const std::vector<uint8_t>& rgb, int width, int height) {
    HImage image;
    if (rgb.empty() || width <= 0 || height <= 0) return image;
    image.GenImageInterleaved(const_cast<uint8_t*>(rgb.data()), "rgb", static_cast<Hlong>(width),
                              static_cast<Hlong>(height), 0, "byte", static_cast<Hlong>(width),
                              static_cast<Hlong>(height), 0, 0, 8, 0);
    return image;
}

HImage RgbToGrayImage(const std::vector<uint8_t>& rgb, int width, int height) {
    const HImage rgbImg = RgbToHImage(rgb, width, height);
    if (!rgbImg.IsInitialized()) return {};
    return rgbImg.Rgb1ToGray();
}

void ReadModelScaleRange(const HShapeModel& shapeModel, double& scaleMin, double& scaleMax) {
    double angleStart = 0.0;
    double angleExtent = 0.0;
    double angleStep = 0.0;
    double scaleStep = 0.0;
    HString metric;
    Hlong minContrast = 0;
    scaleMin = 0.5;
    scaleMax = 2.0;
    try {
        shapeModel.GetShapeModelParams(&angleStart, &angleExtent, &angleStep, &scaleMin, &scaleMax,
                                       &scaleStep, &metric, &minContrast);
    } catch (...) {
    }
}

double ComputeScaleStep(double scaleMin, double scaleMax) {
    if (scaleMin <= 0.0 || scaleMax <= scaleMin) return 0.01;
    const double ratio = scaleMax / scaleMin;
    const int nSteps = std::max(12, static_cast<int>(std::ceil(std::log(ratio) / std::log(1.04))));
    return std::max(0.01, (scaleMax - scaleMin) / static_cast<double>(nSteps));
}

const char* PickOptimization(double scaleMin, double scaleMax) {
    if (scaleMin <= 0.0) return "auto";
    return (scaleMax / scaleMin > 2.5) ? "none" : "auto";
}

Hlong AutoNumLevelsForTemplate(const HImage& templ, double scaleMin) {
    Hlong w = 0;
    Hlong h = 0;
    templ.GetImageSize(&w, &h);
    if (w <= 0 || h <= 0) return 4;
    const double minDim = std::min(static_cast<double>(w), static_cast<double>(h)) *
                          std::max(scaleMin, 0.05);
    Hlong levels = 1;
    while (levels < 10 && (minDim / static_cast<double>(1LL << levels)) >= 6.0) {
        ++levels;
    }
    return std::max<Hlong>(levels, 4);
}

Hlong ShapeModelBorderPad(Hlong w, Hlong h) {
    if (w <= 0 || h <= 0) return 32;
    const Hlong base = std::max<Hlong>(32, std::min(w, h) / 6);
    return std::min<Hlong>(base, 128);
}

HImage ReplicatePadGrayImage(const HImage& src) {
    Hlong w = 0;
    Hlong h = 0;
    src.GetImageSize(&w, &h);
    if (w <= 0 || h <= 0) return src;

    const Hlong pad = ShapeModelBorderPad(w, h);
    const Hlong newW = w + 2 * pad;
    const Hlong newH = h + 2 * pad;

    HString type;
    Hlong srcW = w;
    Hlong srcH = h;
    void* ptr = src.GetImagePointer1(&type, &srcW, &srcH);
    if (type != "byte" || ptr == nullptr) {
        return src;
    }
    const auto* srcBytes = static_cast<const unsigned char*>(ptr);

    std::vector<unsigned char> buf(static_cast<std::size_t>(newW * newH));
    for (Hlong y = 0; y < newH; ++y) {
        const Hlong sy = std::clamp(y - pad, 0LL, srcH - 1);
        for (Hlong x = 0; x < newW; ++x) {
            const Hlong sx = std::clamp(x - pad, 0LL, srcW - 1);
            buf[static_cast<std::size_t>(y * newW + x)] =
                srcBytes[static_cast<std::size_t>(sy * srcW + sx)];
        }
    }

    HImage out;
    out.GenImage1("byte", newW, newH, buf.data());
    return out;
}

HImage ShrinkDomainInset(const HImage& src, Hlong inset) {
    if (inset <= 0) return src;
    Hlong w = 0;
    Hlong h = 0;
    src.GetImageSize(&w, &h);
    if (w <= inset * 2 + 8 || h <= inset * 2 + 8) return src;

    HRegion inner;
    inner.GenRectangle1(static_cast<double>(inset), static_cast<double>(inset),
                        static_cast<double>(h - 1 - inset), static_cast<double>(w - 1 - inset));
    return src.ReduceDomain(inner);
}

void ExtractXldPoints(const HXLDCont& xld, std::vector<float>& xs, std::vector<float>& ys,
                      std::vector<int>& starts) {
    xs.clear();
    ys.clear();
    starts.clear();
    if (!xld.IsInitialized()) return;

    Hlong count = xld.CountObj();
    for (Hlong i = 1; i <= count; ++i) {
        HXLDCont part = xld.SelectObj(i);
        HTuple rows;
        HTuple cols;
        part.GetContourXld(&rows, &cols);
        const Hlong n = rows.Length();
        if (n < 2) continue;
        starts.push_back(static_cast<int>(xs.size()));
        for (Hlong j = 0; j < n; ++j) {
            xs.push_back(static_cast<float>(cols[j].D()));
            ys.push_back(static_cast<float>(rows[j].D()));
        }
    }
}

void TransformXldToVectors(const HXLDCont& src, double row, double col, double angleRad,
                           double scale, std::vector<float>& xs, std::vector<float>& ys,
                           std::vector<int>& starts) {
    // Halcon 官方示例：vector_angle_to_rigid + hom_mat2d_scale(…, Row, Column)
    // 对 iconic 数据 Px=Row、Py=Column（见 hom_mat2d_scale 文档 Attention）
    HHomMat2D hom;
    hom.VectorAngleToRigid(0.0, 0.0, 0.0, row, col, angleRad);
    if (std::abs(scale - 1.0) > 1e-9) {
        hom = hom.HomMat2dScale(scale, scale, row, col);
    }
    const HXLDCont trans = src.AffineTransContourXld(hom);
    ExtractXldPoints(trans, xs, ys, starts);
}

void ComputeHitBBox(const MatchHit& hit, float& minX, float& minY, float& maxX, float& maxY) {
    if (hit.contourX.empty()) {
        minX = maxX = hit.centerX;
        minY = maxY = hit.centerY;
        return;
    }
    minX = maxX = hit.contourX[0];
    minY = maxY = hit.contourY[0];
    for (std::size_t i = 1; i < hit.contourX.size(); ++i) {
        minX = std::min(minX, hit.contourX[i]);
        maxX = std::max(maxX, hit.contourX[i]);
        minY = std::min(minY, hit.contourY[i]);
        maxY = std::max(maxY, hit.contourY[i]);
    }
}

bool CenterInsideBBox(float cx, float cy, float minX, float minY, float maxX, float maxY,
                      float margin) {
    return cx >= minX - margin && cx <= maxX + margin && cy >= minY - margin &&
           cy <= maxY + margin;
}

// Halcon 金字塔搜索阶段得分偏低，亚像素细化后才会升高；直接用用户阈值易漏检。
double HalconSearchMinScore(double userMinScore) {
    return std::max(0.15, userMinScore - 0.25);
}

void FilterNestedFalsePositives(std::vector<MatchHit>& hits) {
    if (hits.size() < 2) return;

    std::sort(hits.begin(), hits.end(),
              [](const MatchHit& a, const MatchHit& b) { return a.score > b.score; });

    std::vector<MatchHit> kept;
    kept.reserve(hits.size());
    for (const MatchHit& hit : hits) {
        bool reject = false;
        for (const MatchHit& k : kept) {
            float minX = 0.f;
            float minY = 0.f;
            float maxX = 0.f;
            float maxY = 0.f;
            ComputeHitBBox(k, minX, minY, maxX, maxY);
            const float margin = std::max(maxX - minX, maxY - minY) * 0.08f;
            if (CenterInsideBBox(hit.centerX, hit.centerY, minX, minY, maxX, maxY, margin) &&
                hit.scale < k.scale * 0.93f) {
                reject = true;
                break;
            }
        }
        if (!reject) kept.push_back(hit);
    }
    hits.swap(kept);
}

std::string HalconError(const HException& ex) {
    try {
        return ex.ErrorMessage().Text();
    } catch (...) {
        return "Halcon exception";
    }
}
#endif

}  // namespace

bool IsHalconAvailable() {
#ifdef POINTCLOUDVIEWER_USE_HALCON
    return true;
#else
    return false;
#endif
}

void DestroyModel(ShapeModel& model) {
#ifdef POINTCLOUDVIEWER_USE_HALCON
    if (model.halconModel) {
        auto* holder = AsHolder(model);
        try {
            if (holder->model.IsInitialized()) holder->model.ClearShapeModel();
            if (holder->contours.IsInitialized()) holder->contours.Clear();
        } catch (...) {
        }
        delete holder;
        model.halconModel = nullptr;
        model.halconContours = nullptr;
    }
#endif
    model = {};
}

bool CreateModel(const std::vector<uint8_t>& rgb, int width, int height, float roiCenterX,
                 float roiCenterY, float roiHalfW, float roiHalfH, float roiAngleDeg,
                 const CreateParams& params, ShapeModel& model, std::string& error) {
#ifndef POINTCLOUDVIEWER_USE_HALCON
    error = u8"未启用 Halcon 支持，请使用带 Halcon 的构建配置";
    return false;
#else
    DestroyModel(model);
    if (rgb.empty() || width <= 0 || height <= 0) {
        error = u8"模板图像无效";
        return false;
    }
    if (roiHalfW < 4.f || roiHalfH < 4.f) {
        error = u8"模板 ROI 过小，请框选更大区域";
        return false;
    }

    try {
        const HImage image = RgbToGrayImage(rgb, width, height);
        const double row = static_cast<double>(roiCenterY);
        const double col = static_cast<double>(roiCenterX);
        const double phi = LocalDegToRad(static_cast<double>(roiAngleDeg));
        // Halcon: Phi=0 时 Length1 沿列方向(宽)，Length2 沿行方向(高)
        const double len1 = static_cast<double>(roiHalfW);
        const double len2 = static_cast<double>(roiHalfH);

        const bool fullAxisAligned =
            std::abs(phi) < 1e-6 && len1 >= static_cast<double>(width) * 0.5 - 0.5 &&
            len2 >= static_cast<double>(height) * 0.5 - 0.5;

        // 整图创建直接用原图；ROI 创建才裁剪。之后复制扩边，避免贴边丢轮廓/出现矩形伪边缘。
        const HImage cropped =
            fullAxisAligned ? image
                            : image.CropRectangle2(row, col, phi, len1, len2, "true", "bilinear");

        Hlong logicW = 0;
        Hlong logicH = 0;
        cropped.GetImageSize(&logicW, &logicH);
        if (logicW < 8 || logicH < 8) {
            error = u8"模板 ROI 裁剪结果过小";
            return false;
        }

        const HImage padded = ReplicatePadGrayImage(cropped);
        const HImage templ = ShrinkDomainInset(padded, 4);

        HTuple numLevels;
        if (params.numLevels > 0) {
            numLevels = params.numLevels;
        } else {
            numLevels = AutoNumLevelsForTemplate(templ, params.scaleMin);
        }
        const double angleStart = LocalDegToRad(params.angleStartDeg);
        const double angleExtent = LocalDegToRad(params.angleExtentDeg);
        const double scaleStep = ComputeScaleStep(params.scaleMin, params.scaleMax);
        const char* optimization = PickOptimization(params.scaleMin, params.scaleMax);
        const HTuple contrast =
            HTuple(params.contrastLow)
                .Append(params.contrastHigh)
                .Append(params.minComponentSize);

        HShapeModel shapeModel = templ.CreateScaledShapeModel(
            numLevels, angleStart, angleExtent, "auto", params.scaleMin, params.scaleMax, scaleStep,
            optimization, params.metric, contrast, params.minContrast);

        HXLDCont contours = shapeModel.GetShapeModelContours(1);

        auto* holder = new HalconModelHolder();
        holder->model = std::move(shapeModel);
        holder->contours = std::move(contours);

        model.valid = true;
        model.templateOriginX = static_cast<int>(std::round(col - len1));
        model.templateOriginY = static_cast<int>(std::round(row - len2));
        model.templateW = static_cast<int>(logicW);
        model.templateH = static_cast<int>(logicH);
        model.createParams = params;
        ReadModelScaleRange(holder->model, model.modelScaleMin, model.modelScaleMax);
        model.halconModel = holder;
        model.halconContours = &holder->contours;
        ExtractXldPoints(holder->contours, model.previewContourX, model.previewContourY,
                         model.previewContourStarts);
        return true;
    } catch (const HException& ex) {
        error = HalconError(ex);
        DestroyModel(model);
        return false;
    }
#endif
}

bool FindModel(const std::vector<uint8_t>& rgb, int width, int height, float searchX0,
               float searchY0, float searchX1, float searchY1, bool useSearchRoi,
               const ShapeModel& model, const FindParams& params, MatchResult& result,
               std::string& error) {
#ifndef POINTCLOUDVIEWER_USE_HALCON
    error = u8"未启用 Halcon 支持";
    return false;
#else
    result = {};
    if (!model.valid || !model.halconModel) {
        error = u8"请先创建或加载 Halcon 模板";
        return false;
    }
    const HalconModelHolder* holder = AsHolder(model);
    if (!holder || !holder->model.IsInitialized()) {
        error = u8"Halcon 模板无效";
        return false;
    }

    try {
        const auto t0 = std::chrono::steady_clock::now();
        const HImage image = RgbToGrayImage(rgb, width, height);

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
        const int ix0 = static_cast<int>(std::floor(sx0));
        const int iy0 = static_cast<int>(std::floor(sy0));
        const int ix1 = static_cast<int>(std::ceil(sx1));
        const int iy1 = static_cast<int>(std::ceil(sy1));

        HImage searchImg = image;
        if (useSearchRoi) {
            HRegion searchRoi;
            searchRoi.GenRectangle1(static_cast<double>(iy0), static_cast<double>(ix0),
                                    static_cast<double>(iy1), static_cast<double>(ix1));
            searchImg = image.ReduceDomain(searchRoi);
        }

        const double angleStart = LocalDegToRad(params.angleStartDeg);
        const double angleExtent = LocalDegToRad(params.angleExtentDeg);
        const Hlong numMatches = params.numMatches > 0 ? static_cast<Hlong>(params.numMatches) : 0;
        const Hlong numLevels =
            params.numLevels > 0 ? static_cast<Hlong>(params.numLevels) : static_cast<Hlong>(0);
        const char* subPixel = params.subPixel ? "least_squares" : "none";

        double modelScaleMin = model.modelScaleMin;
        double modelScaleMax = model.modelScaleMax;
        ReadModelScaleRange(holder->model, modelScaleMin, modelScaleMax);

        double findScaleMin = std::max(params.scaleMin, modelScaleMin);
        double findScaleMax = std::min(params.scaleMax, modelScaleMax);
        if (findScaleMin >= findScaleMax) {
            error = u8"查找缩放范围与模板创建范围无交集；请增大创建时最大缩放或减小查找最小缩放";
            return false;
        }
        if (params.scaleMin < modelScaleMin - 1e-6 || params.scaleMax > modelScaleMax + 1e-6) {
            result.note = u8"查找缩放已自动限制在模板创建范围 ["
                          + std::to_string(modelScaleMin) + ", " + std::to_string(modelScaleMax) +
                          u8"] 内";
        }

        HTuple rows;
        HTuple cols;
        HTuple angles;
        HTuple scales;
        HTuple scores;

        const double userMinScore = params.minScore;
        const double halconMinScore = HalconSearchMinScore(userMinScore);
        if (halconMinScore < userMinScore - 1e-6) {
            const std::string scoreNote = u8"搜索阶段阈值 " + std::to_string(halconMinScore) +
                                          u8"，结果按最小得分 " + std::to_string(userMinScore) +
                                          u8" 过滤";
            if (!result.note.empty()) {
                result.note += u8" | " + scoreNote;
            } else {
                result.note = scoreNote;
            }
        }

        holder->model.FindScaledShapeModel(
            searchImg, angleStart, angleExtent, findScaleMin, findScaleMax, halconMinScore,
            numMatches, params.maxOverlap, subPixel, numLevels, params.greediness, &rows, &cols,
            &angles, &scales, &scores);

        const Hlong n = scores.Length();
        if (n <= 0) {
            error = u8"Halcon 未找到满足条件的匹配";
            return false;
        }

        result.hits.reserve(static_cast<std::size_t>(n));
        for (Hlong i = 0; i < n; ++i) {
            const double score = scores[i].D();
            if (score < userMinScore - 1e-6) continue;

            MatchHit hit;
            hit.centerY = static_cast<float>(rows[i].D());
            hit.centerX = static_cast<float>(cols[i].D());
            hit.angleDeg = static_cast<float>(LocalRadToDeg(angles[i].D()));
            hit.scale = static_cast<float>(scales[i].D());
            hit.score = static_cast<float>(score);
            TransformXldToVectors(holder->contours, rows[i].D(), cols[i].D(), angles[i].D(),
                                  scales[i].D(), hit.contourX, hit.contourY, hit.contourStarts);
            if (hit.contourStarts.empty() && !hit.contourX.empty()) {
                hit.contourStarts.push_back(0);
            }
            result.hits.push_back(std::move(hit));
        }

        if (result.hits.empty()) {
            error = u8"有候选目标但最终得分均低于最小得分阈值（可略降低最小得分或贪婪度）";
            return false;
        }

        FilterNestedFalsePositives(result.hits);
        if (result.hits.empty()) {
            error = u8"Halcon 匹配后过滤误检，无有效目标（可提高最小得分或缩小缩放范围）";
            return false;
        }

        const auto t1 = std::chrono::steady_clock::now();
        result.elapsedMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
        result.ok = !result.hits.empty();
        return result.ok;
    } catch (const HException& ex) {
        error = HalconError(ex);
        return false;
    }
#endif
}

bool SaveModel(const std::string& path, const ShapeModel& model, std::string& error) {
#ifndef POINTCLOUDVIEWER_USE_HALCON
    error = u8"未启用 Halcon 支持";
    return false;
#else
    if (!model.valid || !model.halconModel) {
        error = u8"没有可保存的模板";
        return false;
    }
    try {
        AsHolder(model)->model.WriteShapeModel(path.c_str());
        return true;
    } catch (const HException& ex) {
        error = HalconError(ex);
        return false;
    }
#endif
}

bool LoadModel(const std::string& path, ShapeModel& model, std::string& error) {
#ifndef POINTCLOUDVIEWER_USE_HALCON
    error = u8"未启用 Halcon 支持";
    return false;
#else
    DestroyModel(model);
    try {
        auto* holder = new HalconModelHolder();
        holder->model.ReadShapeModel(path.c_str());
        holder->contours = holder->model.GetShapeModelContours(1);
        model.valid = true;
        ReadModelScaleRange(holder->model, model.modelScaleMin, model.modelScaleMax);
        model.halconModel = holder;
        model.halconContours = &holder->contours;
        ExtractXldPoints(holder->contours, model.previewContourX, model.previewContourY,
                         model.previewContourStarts);
        return true;
    } catch (const HException& ex) {
        error = HalconError(ex);
        DestroyModel(model);
        return false;
    }
#endif
}

}  // namespace HalconShapeMatch
