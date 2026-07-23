#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace HalconShapeMatch {

struct CreateParams {
    int numLevels = 0;  // 0 = auto
    double angleStartDeg = 0.0;
    double angleExtentDeg = 360.0;
    double scaleMin = 0.5;
    double scaleMax = 2.0;
    int contrastLow = 30;       // Contrast 元组第 1 项：边缘提取低阈值
    int contrastHigh = 45;      // Contrast 元组第 2 项：边缘提取高阈值
    int minComponentSize = 10;  // Contrast 元组第 3 项：最小组件尺寸
    int minContrast = 10;       // 搜索时最小对比度（与创建边缘阈值不同）
    const char* metric = "use_polarity";
};

struct FindParams {
    double angleStartDeg = 0.0;
    double angleExtentDeg = 360.0;
    double scaleMin = 0.5;
    double scaleMax = 2.0;
    double minScore = 0.75;
    double maxOverlap = 0.5;
    double greediness = 0.5;
    int numMatches = 0;  // 0 = 全部
    int numLevels = 0;   // 0 = 使用模型层数
    bool subPixel = true;
};

struct ShapeModel {
    bool valid = false;
    int templateOriginX = 0;
    int templateOriginY = 0;
    int templateW = 0;
    int templateH = 0;
    std::vector<float> previewContourX;
    std::vector<float> previewContourY;
    std::vector<int> previewContourStarts;
    CreateParams createParams;
    double modelScaleMin = 0.5;
    double modelScaleMax = 2.0;
#ifdef POINTCLOUDVIEWER_USE_HALCON
    void* halconModel = nullptr;  // HalconCpp::HShapeModel*
    void* halconContours = nullptr;  // HalconCpp::HXLDCont*
#endif
};

struct MatchHit {
    float centerX = 0.f;
    float centerY = 0.f;
    float angleDeg = 0.f;
    float scale = 1.f;
    float score = 0.f;
    std::vector<float> contourX;
    std::vector<float> contourY;
    std::vector<int> contourStarts;
};

struct MatchResult {
    std::vector<MatchHit> hits;
    double elapsedMs = 0.0;
    bool ok = false;
    std::string note;
};

void DestroyModel(ShapeModel& model);

bool CreateModel(const std::vector<uint8_t>& rgb, int width, int height, float roiCenterX,
                 float roiCenterY, float roiHalfW, float roiHalfH, float roiAngleDeg,
                 const CreateParams& params, ShapeModel& model, std::string& error);

bool FindModel(const std::vector<uint8_t>& rgb, int width, int height, float searchX0,
               float searchY0, float searchX1, float searchY1, bool useSearchRoi,
               const ShapeModel& model, const FindParams& params, MatchResult& result,
               std::string& error);

bool SaveModel(const std::string& path, const ShapeModel& model, std::string& error);
bool LoadModel(const std::string& path, ShapeModel& model, std::string& error);

bool IsHalconAvailable();

}  // namespace HalconShapeMatch
