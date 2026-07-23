#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ShapeTemplateMatch {

struct CreateParams {
    int contrastLow = 50;
    int contrastHigh = 150;
    int pyramid = 0;
    int minContrast = 10;
    int minComponentSize = 5;
    float semiAxisLong = 0.f;   // 高斯平滑长半轴，0=默认
    float semiAxisShort = 0.f;  // 高斯平滑短半轴，0=与长半轴相同
    bool usePolarity = true;    // 梯度极性（预留）
};

struct FindParams {
    float angleStart = 0.f;
    float angleRange = 360.f;
    float scaleMin = 0.5f;
    float scaleMax = 2.0f;
    float scale2Min = 0.5f;
    float scale2Max = 2.0f;
    bool isotropicScale = true;
    float minScore = 0.5f;
    float maxOverlap = 0.5f;
    float greediness = 0.75f;
    int numMatches = 0;  // 0 = 自动（最多 32）
    int endPyramid = 2;
    bool subPixel = true;
    bool borderIntersect = false;
    int displayMode = 0;  // 0=全部, 1=仅最佳
};

struct ShapeModel {
    bool valid = false;
    float templateX0 = 0.f;
    float templateY0 = 0.f;
    float templateX1 = 0.f;
    float templateY1 = 0.f;
    float centerX = 0.f;
    float centerY = 0.f;
    int templateW = 0;
    int templateH = 0;
    int templateOriginX = 0;
    int templateOriginY = 0;
    int imageWidth = 0;
    int imageHeight = 0;
    std::vector<uint8_t> templateRgb;
    std::vector<uint8_t> templateGray;
    std::vector<uint8_t> templateEdge;
    std::vector<uint8_t> previewRgb;
    std::vector<float> contourX;
    std::vector<float> contourY;
    std::vector<int> contourStarts;
    CreateParams createParams;
};

struct MatchHit {
    float centerX = 0.f;
    float centerY = 0.f;
    float angleDeg = 0.f;
    float scale = 1.f;
    float scale2 = 1.f;
    float score = 0.f;
    std::vector<float> contourX;
    std::vector<float> contourY;
};

struct MatchResult {
    std::vector<MatchHit> hits;
    double elapsedMs = 0.0;
    bool ok = false;
};

bool CreateModel(const std::vector<uint8_t>& rgb, int width, int height, float roiX0, float roiY0,
                 float roiX1, float roiY1, const CreateParams& params, ShapeModel& model,
                 std::string& error);

bool FindModel(const std::vector<uint8_t>& rgb, int width, int height, float searchX0,
               float searchY0, float searchX1, float searchY1, bool useSearchRoi,
               const ShapeModel& model, const FindParams& params, MatchResult& result,
               std::string& error);

bool SaveModel(const std::string& path, const ShapeModel& model, std::string& error);
bool LoadModel(const std::string& path, ShapeModel& model, std::string& error);

void TransformContour(const ShapeModel& model, float bboxX, float bboxY, float angleDeg,
                      float scale, float scale2, std::vector<float>& outX,
                      std::vector<float>& outY);

}  // namespace ShapeTemplateMatch
