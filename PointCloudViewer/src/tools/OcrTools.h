#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace OcrTools {

enum class OcrEngine { Tesseract = 0, PaddleOcr = 1 };

enum class BinarizeMode { None = 0, Otsu = 1, Adaptive = 2 };

struct PreprocessParams {
    float scale = 1.f;
    int blurKernel = 0;
    BinarizeMode binarize = BinarizeMode::None;
    bool invert = false;
    int morphClose = 0;
    int minHeight = 0;
};

struct RecognizeParams {
    std::string lang = "chi_sim";
    int oem = 3;
    int psm = 3;
    std::string whitelist;
    std::string blacklist;
    float minConfidence = 0.f;
};

struct OcrWord {
    std::string text;
    float confidence = 0.f;
    float x0 = 0.f;
    float y0 = 0.f;
    float x1 = 0.f;
    float y1 = 0.f;
};

struct OcrResult {
    std::vector<OcrWord> words;
    std::string fullText;
    double elapsedMs = 0.;
    bool ok = false;
};

struct PreprocessPreview {
    int width = 0;
    int height = 0;
    std::vector<uint8_t> rgb;
};

bool IsAvailable();
bool IsEngineAvailable(OcrEngine engine);
bool IsAnyEngineAvailable();
const char* EngineLabel(OcrEngine engine);
std::string EngineAvailabilityHint(OcrEngine engine);

// 车牌常用字符白名单（省份简称 + 字母 + 数字）
const char* PlateWhitelist();

bool RecognizeTesseract(const std::vector<uint8_t>& rgb, int width, int height, float roiX0,
                        float roiY0, float roiX1, float roiY1, bool useRoi,
                        const PreprocessParams& prep, const RecognizeParams& rec, OcrResult& result,
                        PreprocessPreview* preview, std::string& error);

// 兼容旧调用
inline bool Recognize(const std::vector<uint8_t>& rgb, int width, int height, float roiX0,
                      float roiY0, float roiX1, float roiY1, bool useRoi,
                      const PreprocessParams& prep, const RecognizeParams& rec, OcrResult& result,
                      PreprocessPreview* preview, std::string& error) {
    return RecognizeTesseract(rgb, width, height, roiX0, roiY0, roiX1, roiY1, useRoi, prep, rec,
                              result, preview, error);
}

}  // namespace OcrTools
