#include "app/OcrWindow.h"

#include "app/FileDialog.h"
#include "io/ImageIO.h"
#include "tools/PaddleOcrTools.h"

#include <glad/gl.h>
#include <imgui.h>

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace {
constexpr ImU32 kRoiCol = IM_COL32(80, 180, 255, 220);
constexpr ImU32 kBoxCol = IM_COL32(40, 220, 80, 255);
constexpr ImU32 kTextCol = IM_COL32(255, 60, 60, 255);

void OcrHelpMarker(const char* desc) {
    ImGui::SameLine();
    ImGui::TextDisabled("(?)");
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
        ImGui::BeginTooltip();
        ImGui::PushTextWrapPos(380.f);
        ImGui::TextUnformatted(desc);
        ImGui::PopTextWrapPos();
        ImGui::EndTooltip();
    }
}
}  // namespace

void OcrWindow::SetVisible(bool visible) {
    if (visible && !visible_) focusOnOpen_ = true;
    visible_ = visible;
}

void OcrWindow::ToggleVisible() { SetVisible(!visible_); }

void OcrWindow::SetStatus(const char* msg) {
    statusText_ = msg ? msg : "";
    if (onStatus_) onStatus_(msg);
}

void OcrWindow::DestroyImageSlot(ImageSlot& slot) {
    if (slot.texId) {
        glDeleteTextures(1, &slot.texId);
        slot.texId = 0;
    }
    slot.width = 0;
    slot.height = 0;
    slot.rgb.clear();
    slot.path.clear();
}

void OcrWindow::DestroyPreviewTexture() {
    if (previewTexId_) {
        glDeleteTextures(1, &previewTexId_);
        previewTexId_ = 0;
    }
    previewW_ = 0;
    previewH_ = 0;
}

void OcrWindow::RefreshPreviewTexture() {
    DestroyPreviewTexture();
    if (lastPreview_.rgb.empty() || lastPreview_.width <= 0 || lastPreview_.height <= 0) return;

    glGenTextures(1, &previewTexId_);
    glBindTexture(GL_TEXTURE_2D, previewTexId_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, lastPreview_.width, lastPreview_.height, 0, GL_RGB,
                  GL_UNSIGNED_BYTE, lastPreview_.rgb.data());
    glBindTexture(GL_TEXTURE_2D, 0);
    previewW_ = lastPreview_.width;
    previewH_ = lastPreview_.height;
}

bool OcrWindow::UploadTexture(ImageSlot& slot) {
    if (slot.rgb.empty() || slot.width <= 0 || slot.height <= 0) return false;
    if (slot.texId == 0) glGenTextures(1, &slot.texId);
    glBindTexture(GL_TEXTURE_2D, slot.texId);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, slot.width, slot.height, 0, GL_RGB, GL_UNSIGNED_BYTE,
                 slot.rgb.data());
    glBindTexture(GL_TEXTURE_2D, 0);
    return true;
}

bool OcrWindow::LoadImageSlot(ImageSlot& slot, const std::string& path, std::string& error) {
    ImageIO::RgbImage img;
    if (!ImageIO::LoadRgb(path, img, error)) return false;
    DestroyImageSlot(slot);
    slot.width = img.width;
    slot.height = img.height;
    slot.rgb = std::move(img.rgb);
    slot.path = path;
    return UploadTexture(slot);
}

bool OcrWindow::ImageToPixel(float imgPosX, float imgPosY, float drawW, float drawH, float mouseX,
                             float mouseY, float& outX, float& outY) const {
    if (!image_.valid()) return false;
    if (mouseX < imgPosX || mouseY < imgPosY || mouseX > imgPosX + drawW ||
        mouseY > imgPosY + drawH) {
        return false;
    }
    outX = (mouseX - imgPosX) / drawW * static_cast<float>(image_.width);
    outY = (mouseY - imgPosY) / drawH * static_cast<float>(image_.height);
    outX = std::clamp(outX, 0.f, static_cast<float>(image_.width - 1));
    outY = std::clamp(outY, 0.f, static_cast<float>(image_.height - 1));
    return true;
}

void OcrWindow::NormalizeRoi() {
    if (roiX1_ < roiX0_) std::swap(roiX0_, roiX1_);
    if (roiY1_ < roiY0_) std::swap(roiY0_, roiY1_);
}

bool OcrWindow::HasValidRoi() const {
    return std::abs(roiX1_ - roiX0_) > 2.f && std::abs(roiY1_ - roiY0_) > 2.f;
}

void OcrWindow::DrawOcrOverlays(ImDrawList* dl, const ImVec2& imgPos, float drawW, float drawH) {
    if (!image_.valid()) return;
    const float sx = drawW / static_cast<float>(image_.width);
    const float sy = drawH / static_cast<float>(image_.height);

    if (roiDragging_ || useRoi_ || HasValidRoi()) {
        const float x0 = std::min(roiX0_, roiX1_);
        const float y0 = std::min(roiY0_, roiY1_);
        const float x1 = std::max(roiX0_, roiX1_);
        const float y1 = std::max(roiY0_, roiY1_);
        dl->AddRect(ImVec2(imgPos.x + x0 * sx, imgPos.y + y0 * sy),
                    ImVec2(imgPos.x + x1 * sx, imgPos.y + y1 * sy), kRoiCol, 0.f, 0, 2.f);
    }

    if (!showBoxes_ || !lastResult_.ok) return;

    for (const OcrTools::OcrWord& word : lastResult_.words) {
        dl->AddRect(ImVec2(imgPos.x + word.x0 * sx, imgPos.y + word.y0 * sy),
                    ImVec2(imgPos.x + word.x1 * sx, imgPos.y + word.y1 * sy), kBoxCol, 0.f, 0, 2.f);
        if (showConfidence_) {
            char buf[64];
            std::snprintf(buf, sizeof(buf), "%s (%.0f)", word.text.c_str(), word.confidence);
            dl->AddText(ImVec2(imgPos.x + word.x0 * sx, imgPos.y + word.y0 * sy - 18.f), kTextCol,
                        buf);
        } else {
            dl->AddText(ImVec2(imgPos.x + word.x0 * sx, imgPos.y + word.y0 * sy - 18.f), kTextCol,
                        word.text.c_str());
        }
    }
}

void OcrWindow::DrawImageCanvas() {
    if (!image_.valid()) {
        ImGui::TextDisabled(u8"尚未加载图像");
        return;
    }

    const ImVec2 avail = ImGui::GetContentRegionAvail();
    const float areaH = std::max(avail.y, 200.f);
    ImGui::BeginChild("##ocr_canvas", ImVec2(avail.x, areaH), ImGuiChildFlags_Borders);

    const float imgAspect =
        (image_.height > 0) ? static_cast<float>(image_.width) / static_cast<float>(image_.height)
                            : 1.f;
    float fitW = std::max(avail.x - 8.f, 1.f);
    float fitH = fitW / imgAspect;
    if (fitH > areaH - 8.f) {
        fitH = std::max(areaH - 8.f, 1.f);
        fitW = fitH * imgAspect;
    }
    const float drawW = std::max(fitW * zoom_, 1.f);
    const float drawH = std::max(fitH * zoom_, 1.f);
    const float basePanX = std::max((avail.x - drawW) * 0.5f, 0.f);
    const float basePanY = std::max((areaH - drawH) * 0.5f, 0.f);

    ImGuiIO& io = ImGui::GetIO();
    if (ImGui::IsWindowHovered() && io.KeyShift && io.MouseWheel != 0.f) {
        zoom_ = std::clamp(zoom_ * (1.f + io.MouseWheel * 0.12f), 0.1f, 16.f);
    }

    ImGui::SetCursorPos(ImVec2(basePanX, basePanY));
    const ImVec2 imgPos = ImGui::GetCursorScreenPos();
    ImGui::InvisibleButton("##ocr_hit", ImVec2(drawW, drawH));
    const bool imgHovered = ImGui::IsItemHovered();

    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddImage((ImTextureID)(intptr_t)image_.texId, imgPos,
                 ImVec2(imgPos.x + drawW, imgPos.y + drawH));
    DrawOcrOverlays(dl, imgPos, drawW, drawH);

    const bool trackMouse = imgHovered || roiDragging_;
    if (trackMouse) {
        const float mx = std::clamp(io.MousePos.x, imgPos.x, imgPos.x + drawW);
        const float my = std::clamp(io.MousePos.y, imgPos.y, imgPos.y + drawH);
        float px = 0.f;
        float py = 0.f;
        if (ImageToPixel(imgPos.x, imgPos.y, drawW, drawH, mx, my, px, py)) {
            hoverPx_ = static_cast<int>(px);
            hoverPy_ = static_cast<int>(py);
            const std::size_t idx =
                (static_cast<std::size_t>(hoverPy_) * static_cast<std::size_t>(image_.width) +
                 static_cast<std::size_t>(hoverPx_)) *
                3u;
            if (idx + 2 < image_.rgb.size()) {
                hoverR_ = image_.rgb[idx];
                hoverG_ = image_.rgb[idx + 1];
                hoverB_ = image_.rgb[idx + 2];
            }
        }

        if (ImGui::IsMouseClicked(ImGuiMouseButton_Left) && imgHovered) {
            roiDragging_ = true;
            float px0 = 0.f;
            float py0 = 0.f;
            ImageToPixel(imgPos.x, imgPos.y, drawW, drawH, mx, my, px0, py0);
            roiX0_ = roiX1_ = px0;
            roiY0_ = roiY1_ = py0;
        }
    }

    if (roiDragging_ && ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
        const float mx = std::clamp(io.MousePos.x, imgPos.x, imgPos.x + drawW);
        const float my = std::clamp(io.MousePos.y, imgPos.y, imgPos.y + drawH);
        float px1 = 0.f;
        float py1 = 0.f;
        ImageToPixel(imgPos.x, imgPos.y, drawW, drawH, mx, my, px1, py1);
        roiX1_ = px1;
        roiY1_ = py1;
    }
    if (roiDragging_ && ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
        roiDragging_ = false;
        NormalizeRoi();
        if (HasValidRoi()) {
            useRoi_ = true;
        }
    }

    ImGui::EndChild();
}

void OcrWindow::DrawEngineBar() {
    const auto drawEngineBtn = [&](OcrTools::OcrEngine engine) {
        const bool active = engine_ == engine;
        if (active) {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.26f, 0.59f, 0.98f, 0.55f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.26f, 0.59f, 0.98f, 0.75f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.06f, 0.53f, 0.98f, 0.85f));
        }
        if (ImGui::Button(OcrTools::EngineLabel(engine), ImVec2(110.f, 0.f))) {
            engine_ = engine;
            if (engine == OcrTools::OcrEngine::PaddleOcr) {
                SetStatus(u8"正在预加载 PaddleOCR 模型...");
                std::string warmupError;
                if (PaddleOcrTools::Warmup(warmupError)) {
                    SetStatus(u8"PaddleOCR 模型已就绪");
                } else if (!warmupError.empty()) {
                    SetStatus(warmupError.c_str());
                }
            }
        }
        if (active) ImGui::PopStyleColor(3);
        ImGui::SameLine();
    };

    drawEngineBtn(OcrTools::OcrEngine::Tesseract);
    drawEngineBtn(OcrTools::OcrEngine::PaddleOcr);
    ImGui::NewLine();

    if (!OcrTools::IsEngineAvailable(engine_)) {
        const std::string hint = OcrTools::EngineAvailabilityHint(engine_);
        ImGui::TextColored(ImVec4(1.f, 0.45f, 0.45f, 1.f), "%s", hint.c_str());
    } else {
        ImGui::TextDisabled(u8"当前引擎：%s", OcrTools::EngineLabel(engine_));
    }
    ImGui::Spacing();
}

void OcrWindow::DrawTesseractParamPanel() {
    ImGui::TextDisabled(u8"预处理（Tesseract）");
    if (prepParams_.invert) {
        ImGui::TextColored(ImVec4(1.f, 0.75f, 0.35f, 1.f),
                           u8"提示：蓝牌已自动白字提取，建议关闭「反色」");
    }
    if (ImGui::BeginTable("##ocr_prep", 4,
                          ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_PadOuterX)) {
        ImGui::TableSetupColumn("lbl1", ImGuiTableColumnFlags_WidthFixed, 78.f);
        ImGui::TableSetupColumn("val1", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("lbl2", ImGuiTableColumnFlags_WidthFixed, 78.f);
        ImGui::TableSetupColumn("val2", ImGuiTableColumnFlags_WidthStretch);

        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(u8"缩放");
        OcrHelpMarker(u8"放大图像后再识别。字太小可试 2.0；已清晰保持 1.0。");
        ImGui::TableNextColumn();
        ImGui::SetNextItemWidth(-1);
        ImGui::DragFloat("##ocr_scale", &prepParams_.scale, 0.1f, 1.f, 8.f, "%.1f");
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(u8"平滑核");
        OcrHelpMarker(u8"高斯模糊核大小（奇数）。去噪用 3/5，清晰图保持 0。");
        ImGui::TableNextColumn();
        ImGui::SetNextItemWidth(-1);
        ImGui::DragInt("##ocr_blur", &prepParams_.blurKernel, 1, 0, 15);

        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(u8"二值化");
        OcrHelpMarker(u8"转黑白图。铭牌/激光打标可用 Otsu 或自适应；蓝牌保持「无」。");
        ImGui::TableNextColumn();
        ImGui::SetNextItemWidth(-1);
        const char* binItems[] = {u8"无", u8"Otsu", u8"自适应"};
        int binMode = static_cast<int>(prepParams_.binarize);
        if (ImGui::Combo("##ocr_bin", &binMode, binItems, 3)) {
            prepParams_.binarize = static_cast<OcrTools::BinarizeMode>(binMode);
        }
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(u8"闭运算");
        OcrHelpMarker(u8"形态学闭运算，连接断裂笔画。一般保持 0。");
        ImGui::TableNextColumn();
        ImGui::SetNextItemWidth(-1);
        ImGui::DragInt("##ocr_morph", &prepParams_.morphClose, 1, 0, 15);

        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::Checkbox(u8"反色##ocr_inv", &prepParams_.invert);
        OcrHelpMarker(u8"黑白反转。黑底白字铭牌可开；蓝牌请关闭（软件已自动处理）。");
        ImGui::TableNextColumn();
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(u8"最小组件");
        OcrHelpMarker(u8"二值化后过滤过小连通域（像素高度）。一般保持 0。");
        ImGui::TableNextColumn();
        ImGui::SetNextItemWidth(-1);
        ImGui::DragInt("##ocr_minh", &prepParams_.minHeight, 1, 0, 64);
        ImGui::EndTable();
    }

    ImGui::Spacing();
    ImGui::TextDisabled(u8"识别（Tesseract）");
    if (ImGui::BeginTable("##ocr_rec", 4,
                          ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_PadOuterX)) {
        ImGui::TableSetupColumn("lbl1", ImGuiTableColumnFlags_WidthFixed, 78.f);
        ImGui::TableSetupColumn("val1", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("lbl2", ImGuiTableColumnFlags_WidthFixed, 78.f);
        ImGui::TableSetupColumn("val2", ImGuiTableColumnFlags_WidthStretch);

        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(u8"语言");
        OcrHelpMarker(u8"Tesseract 语言包。中文车牌用 chi_sim；中英混排用 chi_sim+eng。");
        ImGui::TableNextColumn();
        ImGui::SetNextItemWidth(-1);
        ImGui::InputText("##ocr_lang", langBuf_, sizeof(langBuf_));
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(u8"最小置信");
        OcrHelpMarker(u8"过滤低置信度结果。建议 30~50；调试可设 0 看全部候选。");
        ImGui::TableNextColumn();
        ImGui::SetNextItemWidth(-1);
        ImGui::DragFloat("##ocr_conf", &recParams_.minConfidence, 1.f, 0.f, 100.f, "%.0f");

        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(u8"PSM");
        OcrHelpMarker(
            u8"版面分割模式。3=自动；7=单行；8=单词；13=原始行（蓝牌推荐）。"
            u8"软件会自动多模式尝试并选最优。");
        ImGui::TableNextColumn();
        ImGui::SetNextItemWidth(-1);
        const char* psmItems[] = {u8"3 自动", u8"6 文本块", u8"7 单行", u8"8 单词", u8"11 稀疏", u8"13 原始行"};
        const int psmValues[] = {3, 6, 7, 8, 11, 13};
        int psmIndex = 0;
        for (int i = 0; i < 6; ++i) {
            if (recParams_.psm == psmValues[i]) {
                psmIndex = i;
                break;
            }
        }
        if (ImGui::Combo("##ocr_psm", &psmIndex, psmItems, 6)) {
            recParams_.psm = psmValues[psmIndex];
        }
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(u8"OEM");
        OcrHelpMarker(u8"OCR 引擎。一般保持「3 默认」即可。");
        ImGui::TableNextColumn();
        ImGui::SetNextItemWidth(-1);
        const char* oemItems[] = {u8"3 默认", u8"1 LSTM", u8"0 传统", u8"2 组合"};
        const int oemValues[] = {3, 1, 0, 2};
        int oemIndex = 0;
        for (int i = 0; i < 4; ++i) {
            if (recParams_.oem == oemValues[i]) {
                oemIndex = i;
                break;
            }
        }
        if (ImGui::Combo("##ocr_oem", &oemIndex, oemItems, 4)) {
            recParams_.oem = oemValues[oemIndex];
        }

        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(u8"白名单");
        OcrHelpMarker(u8"只允许出现的字符。车牌可填省份简称+字母+数字，减少误识。");
        ImGui::TableNextColumn();
        ImGui::SetNextItemWidth(-1);
        ImGui::InputText("##ocr_white", whiteBuf_, sizeof(whiteBuf_));
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(u8"黑名单");
        OcrHelpMarker(u8"禁止出现的字符。一般留空。");
        ImGui::TableNextColumn();
        ImGui::SetNextItemWidth(-1);
        ImGui::InputText("##ocr_black", blackBuf_, sizeof(blackBuf_));

        recParams_.lang = langBuf_;
        recParams_.whitelist = whiteBuf_;
        recParams_.blacklist = blackBuf_;
        ImGui::EndTable();
    }

    if (ImGui::CollapsingHeader(u8"参数说明##tess")) {
        ImGui::TextWrapped(
            u8"【框选】左键拖拽 ROI，勾选「仅识别 ROI」。车牌框紧贴牌照，略留边距即可。\n"
            u8"【蓝牌】软件自动白字提取+多阈值尝试，无需手动二值化/反色。\n"
            u8"【误识原因】Tesseract 非车牌专用引擎，「京」与「区」、「6」与「9」易混淆；"
            u8"置信度低（<50%）时结果不可靠。可点「车牌模式」或切换 PaddleOCR。\n"
            u8"【调试图】%TEMP%\\pcv_ocr_debug_input.png 为送入识别的预处理图（黑字白底）。");
    }
}

void OcrWindow::DrawPaddleParamPanel() {
    ImGui::TextDisabled(u8"PaddleOCR 参数");
    ImGui::TextWrapped(u8"框选 ROI 后推荐开启「快速识别」，车牌场景可再点「车牌模式」。");
    ImGui::Spacing();

    if (ImGui::BeginTable("##ocr_paddle", 4,
                          ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_PadOuterX)) {
        ImGui::TableSetupColumn("lbl1", ImGuiTableColumnFlags_WidthFixed, 78.f);
        ImGui::TableSetupColumn("val1", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("lbl2", ImGuiTableColumnFlags_WidthFixed, 78.f);
        ImGui::TableSetupColumn("val2", ImGuiTableColumnFlags_WidthStretch);

        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(u8"语言");
        OcrHelpMarker(u8"PaddleOCR 语言：ch 中文、en 英文。");
        ImGui::TableNextColumn();
        ImGui::SetNextItemWidth(-1);
        ImGui::InputText("##ocr_paddle_lang", paddleLangBuf_, sizeof(paddleLangBuf_));
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(u8"最小置信");
        OcrHelpMarker(u8"过滤低置信度行。建议 30~50。");
        ImGui::TableNextColumn();
        ImGui::SetNextItemWidth(-1);
        ImGui::DragFloat("##ocr_paddle_conf", &paddleParams_.minConfidence, 1.f, 0.f, 100.f,
                         "%.0f");

        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::Checkbox(u8"方向分类", &paddleParams_.useAngleCls);
        OcrHelpMarker(u8"纠正旋转文字。车牌/水平文字建议关闭。");
        ImGui::TableNextColumn();
        ImGui::Checkbox(u8"快速识别", &paddleParams_.skipDet);
        OcrHelpMarker(u8"勾选且已框选 ROI 时跳过文字检测，仅识别，速度更快（适合车牌等单行）。");
        ImGui::EndTable();
    }

    paddleParams_.lang = paddleLangBuf_;

    if (ImGui::CollapsingHeader(u8"参数说明##paddle")) {
        ImGui::TextWrapped(
            u8"PaddleOCR 为通用深度学习 OCR，中文/车牌效果通常优于 Tesseract。\n"
            u8"首次识别会加载模型（约数秒），之后识别通常在 1 秒内完成。\n"
            u8"安装：pip install paddlepaddle paddleocr");
    }
}

void OcrWindow::DrawParamPanel() {
    DrawEngineBar();

    if (!OcrTools::IsAnyEngineAvailable()) {
        ImGui::TextColored(ImVec4(1.f, 0.4f, 0.4f, 1.f), u8"未检测到可用 OCR 引擎");
        ImGui::TextWrapped(
            u8"Tesseract：安装 Tesseract-OCR 并重新编译。\n"
            u8"PaddleOCR：pip install paddlepaddle paddleocr");
        return;
    }

    if (engine_ == OcrTools::OcrEngine::PaddleOcr) {
        DrawPaddleParamPanel();
    } else {
        DrawTesseractParamPanel();
    }

    ImGui::Spacing();
    ImGui::Checkbox(u8"显示识别框", &showBoxes_);
    ImGui::SameLine();
    ImGui::Checkbox(u8"显示置信度", &showConfidence_);
    if (engine_ == OcrTools::OcrEngine::Tesseract) {
        ImGui::SameLine();
        ImGui::Checkbox(u8"二值化预览", &showPreprocessPreview_);
    }
}

void OcrWindow::DrawResultPanel() {
    ImGui::TextDisabled(u8"识别结果");
    ImGui::BeginChild("##ocr_result", ImVec2(0, 140.f), ImGuiChildFlags_Borders);
    if (!lastResult_.ok) {
        ImGui::TextDisabled(u8"（点击「识别」后显示结果）");
    } else {
        ImGui::TextWrapped("%s", lastResult_.fullText.c_str());
        ImGui::Separator();
        for (const OcrTools::OcrWord& word : lastResult_.words) {
            if (word.text.empty()) continue;
            ImGui::BulletText("%s  (%.0f%%)", word.text.c_str(), word.confidence);
        }
        ImGui::TextDisabled(u8"耗时 %.1f ms", lastResult_.elapsedMs);
    }
    ImGui::EndChild();

    if (showPreprocessPreview_) {
        ImGui::Spacing();
        ImGui::TextDisabled(u8"预处理预览");
        ImGui::BeginChild("##ocr_prev", ImVec2(0, 120.f), ImGuiChildFlags_Borders);
        if (previewTexId_ != 0) {
            const float prevW = ImGui::GetContentRegionAvail().x;
            const float aspect =
                static_cast<float>(previewW_) / static_cast<float>(std::max(previewH_, 1));
            float drawW = prevW;
            float drawH = drawW / aspect;
            if (drawH > 110.f) {
                drawH = 110.f;
                drawW = drawH * aspect;
            }
            ImGui::Image((ImTextureID)(intptr_t)previewTexId_, ImVec2(drawW, drawH));
        } else {
            ImGui::TextDisabled(u8"识别后显示");
        }
        ImGui::EndChild();
    }
}

void OcrWindow::ReadImage() {
    const std::string path = FileDialog::OpenImageFile(u8"读取图像");
    if (path.empty()) return;
    std::string error;
    if (!LoadImageSlot(image_, path, error)) {
        SetStatus(error.c_str());
        return;
    }
    roiX0_ = roiY0_ = 0.f;
    roiX1_ = static_cast<float>(image_.width - 1);
    roiY1_ = static_cast<float>(image_.height - 1);
    useRoi_ = false;
    ClearResults();
    char buf[256];
    std::snprintf(buf, sizeof(buf), u8"已读取图像 %dx%d", image_.width, image_.height);
    SetStatus(buf);
}

void OcrWindow::RunPlateRecognize() {
    if (OcrTools::IsEngineAvailable(OcrTools::OcrEngine::PaddleOcr)) {
        engine_ = OcrTools::OcrEngine::PaddleOcr;
        paddleParams_ = {};
        paddleParams_.lang = "ch";
        paddleParams_.useAngleCls = false;
        paddleParams_.skipDet = true;
        paddleParams_.minConfidence = 30.f;
        std::snprintf(paddleLangBuf_, sizeof(paddleLangBuf_), "ch");
        RunRecognize();
        return;
    }
    prepParams_ = {};
    prepParams_.invert = false;
    recParams_ = {};
    recParams_.lang = "chi_sim";
    recParams_.psm = 13;
    recParams_.oem = 3;
    recParams_.minConfidence = 30.f;
    std::snprintf(langBuf_, sizeof(langBuf_), "chi_sim");
    std::snprintf(whiteBuf_, sizeof(whiteBuf_), "%s", OcrTools::PlateWhitelist());
    blackBuf_[0] = '\0';
    RunRecognize();
}

void OcrWindow::RunCliRecognize() {
    prepParams_ = {};
    recParams_ = {};
    recParams_.lang = "chi_sim";
    recParams_.psm = 3;
    recParams_.oem = 3;
    recParams_.minConfidence = 0.f;
    std::snprintf(langBuf_, sizeof(langBuf_), "chi_sim");
    whiteBuf_[0] = '\0';
    blackBuf_[0] = '\0';
    useRoi_ = false;
    RunRecognize();
}

void OcrWindow::RunRecognize() {
    if (!image_.valid()) {
        SetStatus(u8"请先读取图像");
        return;
    }
    if (!OcrTools::IsEngineAvailable(engine_)) {
        SetStatus(OcrTools::EngineAvailabilityHint(engine_).c_str());
        return;
    }

    std::string error;
    lastPreview_ = {};
    bool ok = false;
    if (engine_ == OcrTools::OcrEngine::PaddleOcr) {
        paddleParams_.lang = paddleLangBuf_;
        ok = PaddleOcrTools::Recognize(image_.rgb, image_.width, image_.height, roiX0_, roiY0_,
                                       roiX1_, roiY1_, useRoi_, paddleParams_, lastResult_, error);
    } else {
        ok = OcrTools::RecognizeTesseract(image_.rgb, image_.width, image_.height, roiX0_, roiY0_,
                                        roiX1_, roiY1_, useRoi_, prepParams_, recParams_,
                                        lastResult_, &lastPreview_, error);
    }

    if (!ok) {
        lastResult_ = {};
        RefreshPreviewTexture();
        SetStatus(error.c_str());
        return;
    }

    RefreshPreviewTexture();
    char buf[256];
    std::snprintf(buf, sizeof(buf), u8"[%s] 完成：%zu 项，耗时 %.1f ms",
                  OcrTools::EngineLabel(engine_), lastResult_.words.size(), lastResult_.elapsedMs);
    SetStatus(buf);
}

void OcrWindow::ClearRoi() {
    if (!image_.valid()) return;
    roiX0_ = roiY0_ = 0.f;
    roiX1_ = static_cast<float>(image_.width - 1);
    roiY1_ = static_cast<float>(image_.height - 1);
    useRoi_ = false;
    roiDragging_ = false;
}

void OcrWindow::ClearResults() {
    lastResult_ = {};
    lastPreview_ = {};
    DestroyPreviewTexture();
}

void OcrWindow::Draw(float menuBottomY, float bottomInset) {
    if (!visible_) return;

    ImGuiViewport* vp = ImGui::GetMainViewport();
    const float windowH = std::max(vp->Size.y - menuBottomY - bottomInset, 1.f);
    ImGui::SetNextWindowPos(ImVec2(vp->Pos.x, menuBottomY), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(vp->Size.x, windowH), ImGuiCond_Always);
    ImGui::SetNextWindowBgAlpha(1.f);
    if (focusOnOpen_) {
        ImGui::SetNextWindowFocus();
        focusOnOpen_ = false;
    }

    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.f);

    const ImGuiWindowFlags flags =
        ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoSavedSettings;

    bool open = visible_;
    if (!ImGui::Begin(u8"OCR 识别", &open, flags)) {
        ImGui::End();
        ImGui::PopStyleVar(2);
        visible_ = open;
        return;
    }
    visible_ = open;
    if (!visible_) {
        ImGui::End();
        ImGui::PopStyleVar(2);
        return;
    }

    if (ImGui::Button(u8"关闭")) {
        SetVisible(false);
    }
    ImGui::SameLine();
    ImGui::TextDisabled(u8"左键框选 ROI | 顶部切换 Tesseract / PaddleOCR 引擎");

    const float rightW =
        std::clamp(ImGui::GetContentRegionAvail().x * 0.32f, 320.f, 480.f);
    const float bottomH = 24.f;
    const ImVec2 contentAvail = ImGui::GetContentRegionAvail();

    ImGui::BeginChild("##ocr_main_row", ImVec2(0, contentAvail.y - bottomH), false);
    ImGui::BeginChild("##ocr_left", ImVec2(ImGui::GetContentRegionAvail().x - rightW, 0), false);
    ImGui::TextDisabled(u8"图像（框选 ROI 后识别）");
    DrawImageCanvas();
    ImGui::EndChild();

    ImGui::SameLine();
    ImGui::BeginChild("##ocr_right", ImVec2(rightW, 0), ImGuiChildFlags_Borders);
    DrawParamPanel();

    ImGui::Spacing();
    ImGui::Checkbox(u8"仅识别 ROI", &useRoi_);
    if (!useRoi_) {
        ImGui::SameLine();
        ImGui::TextDisabled(u8"（未勾选时识别整图）");
    }
    ImGui::SameLine();
    if (ImGui::Button(u8"清除 ROI")) ClearRoi();

    ImGui::Spacing();
    const float btnW = (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x) * 0.5f;
    const ImVec2 btnSize(btnW, 28.f);
    if (ImGui::Button(u8"读取图像", btnSize)) ReadImage();
    ImGui::SameLine();
    if (ImGui::Button(u8"识别", btnSize)) RunRecognize();
    ImGui::SameLine();
    if (ImGui::Button(u8"车牌模式", btnSize)) RunPlateRecognize();
    if (ImGui::Button(u8"命令行模式", btnSize)) RunCliRecognize();
    ImGui::SameLine();
    if (ImGui::Button(u8"清除结果", btnSize)) ClearResults();

    ImGui::Spacing();
    DrawResultPanel();
    ImGui::EndChild();
    ImGui::EndChild();

    ImGui::Separator();
    if (image_.valid()) {
        if (hoverPx_ >= 0 && hoverPy_ >= 0) {
            ImGui::TextDisabled(u8"W:%d  H:%d   X:%d  Y:%d   R:%d G:%d B:%d", image_.width,
                                image_.height, hoverPx_, hoverPy_, hoverR_, hoverG_, hoverB_);
        } else {
            ImGui::TextDisabled(u8"W:%d  H:%d", image_.width, image_.height);
        }
    }
    if (!statusText_.empty()) {
        ImGui::SameLine(ImGui::GetWindowWidth() - 360.f);
        ImGui::TextColored(ImVec4(0.35f, 0.9f, 0.55f, 1.f), "%s", statusText_.c_str());
    }

    ImGui::End();
    ImGui::PopStyleVar(2);
}
