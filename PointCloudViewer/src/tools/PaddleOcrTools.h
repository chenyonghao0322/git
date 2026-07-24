#pragma once

#include "tools/OcrTools.h"

#include <string>
#include <vector>

namespace PaddleOcrTools {

struct PaddleParams {
    std::string lang = "ch";
    bool useAngleCls = false;
    float minConfidence = 0.f;
    // ROI 已框选单行文字（如车牌）时跳过检测，仅识别，显著加速
    bool skipDet = true;
};

bool IsAvailable();
std::string AvailabilityHint();

bool Recognize(const std::vector<uint8_t>& rgb, int width, int height, float roiX0, float roiY0,
               float roiX1, float roiY1, bool useRoi, const PaddleParams& params,
               OcrTools::OcrResult& result, std::string& error);

// 预加载常驻 Python 服务（首次约数秒，后续识别更快）
bool Warmup(std::string& error);
void Shutdown();

}  // namespace PaddleOcrTools
