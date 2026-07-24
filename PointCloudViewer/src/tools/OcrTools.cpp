#include "tools/OcrTools.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "tools/PaddleOcrTools.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

#ifdef POINTCLOUDVIEWER_USE_TESSERACT
#include <tesseract/baseapi.h>
#endif

namespace OcrTools {
namespace {

cv::Mat RgbToBgr(const std::vector<uint8_t>& rgb, int width, int height) {
    cv::Mat bgr(height, width, CV_8UC3);
    for (int y = 0; y < height; ++y) {
        const uint8_t* src = rgb.data() + static_cast<std::size_t>(y * width * 3);
        uint8_t* dst = bgr.ptr<uint8_t>(y);
        for (int x = 0; x < width; ++x) {
            dst[x * 3 + 0] = src[x * 3 + 2];
            dst[x * 3 + 1] = src[x * 3 + 1];
            dst[x * 3 + 2] = src[x * 3 + 0];
        }
    }
    return bgr;
}

void GrayToRgbPreview(const cv::Mat& gray, PreprocessPreview& preview) {
    preview.width = gray.cols;
    preview.height = gray.rows;
    preview.rgb.resize(static_cast<std::size_t>(gray.cols * gray.rows * 3));
    for (int y = 0; y < gray.rows; ++y) {
        const uint8_t* row = gray.ptr<uint8_t>(y);
        for (int x = 0; x < gray.cols; ++x) {
            const uint8_t v = row[x];
            const std::size_t i = (static_cast<std::size_t>(y * gray.cols + x)) * 3u;
            preview.rgb[i] = v;
            preview.rgb[i + 1] = v;
            preview.rgb[i + 2] = v;
        }
    }
}

cv::Mat PreprocessImage(const cv::Mat& bgr, const PreprocessParams& prep) {
    cv::Mat gray;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);

    if (prep.scale > 1.001f) {
        cv::resize(gray, gray, cv::Size(), prep.scale, prep.scale, cv::INTER_CUBIC);
    }

    if (prep.blurKernel >= 3) {
        const int k = prep.blurKernel | 1;
        cv::GaussianBlur(gray, gray, cv::Size(k, k), 0.);
    }

    switch (prep.binarize) {
        case BinarizeMode::Otsu:
            cv::threshold(gray, gray, 0., 255., cv::THRESH_BINARY | cv::THRESH_OTSU);
            break;
        case BinarizeMode::Adaptive:
            cv::adaptiveThreshold(gray, gray, 255., cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv::THRESH_BINARY, 31, 8.);
            break;
        case BinarizeMode::None:
        default:
            break;
    }

    if (prep.invert) {
        cv::bitwise_not(gray, gray);
    }

    if (prep.morphClose >= 3) {
        const int k = prep.morphClose | 1;
        const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(k, k));
        cv::morphologyEx(gray, gray, cv::MORPH_CLOSE, kernel);
    }

    if (prep.minHeight > 0 && prep.binarize != BinarizeMode::None) {
        cv::Mat labels;
        cv::Mat stats;
        cv::Mat centroids;
        const int n = cv::connectedComponentsWithStats(gray, labels, stats, centroids, 8, CV_32S);
        cv::Mat filtered = cv::Mat::zeros(gray.size(), CV_8UC1);
        for (int i = 1; i < n; ++i) {
            const int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
            if (h < prep.minHeight) continue;
            filtered.setTo(255, labels == i);
        }
        gray = filtered;
    }

    return gray;
}

void MapBoxToSource(float& x0, float& y0, float& x1, float& y1, float roiOx, float roiOy,
                    float scale) {
    const float inv = (scale > 1e-6f) ? (1.f / scale) : 1.f;
    x0 = x0 * inv + roiOx;
    y0 = y0 * inv + roiOy;
    x1 = x1 * inv + roiOx;
    y1 = y1 * inv + roiOy;
}

bool NeedsGrayPreprocess(const PreprocessParams& prep) {
    return prep.binarize != BinarizeMode::None || prep.invert || prep.blurKernel >= 3 ||
           prep.morphClose >= 3 || prep.minHeight > 0;
}

cv::Mat PrepareOcrImage(const cv::Mat& bgr, const PreprocessParams& prep, bool& isGray) {
    if (!NeedsGrayPreprocess(prep)) {
        if (prep.scale > 1.001f) {
            cv::Mat scaled;
            cv::resize(bgr, scaled, cv::Size(), prep.scale, prep.scale, cv::INTER_CUBIC);
            isGray = false;
            return scaled;
        }
        isGray = false;
        return bgr;
    }
    isGray = true;
    return PreprocessImage(bgr, prep);
}

void MatToRgbPreview(const cv::Mat& image, bool isGray, PreprocessPreview& preview) {
    preview.width = image.cols;
    preview.height = image.rows;
    preview.rgb.resize(static_cast<std::size_t>(image.cols * image.rows * 3));
    if (isGray) {
        GrayToRgbPreview(image, preview);
        return;
    }
    for (int y = 0; y < image.rows; ++y) {
        const uint8_t* row = image.ptr<uint8_t>(y);
        for (int x = 0; x < image.cols; ++x) {
            const std::size_t i = (static_cast<std::size_t>(y * image.cols + x)) * 3u;
            preview.rgb[i] = row[x * 3 + 2];
            preview.rgb[i + 1] = row[x * 3 + 1];
            preview.rgb[i + 2] = row[x * 3 + 0];
        }
    }
}

bool PassesConfidence(float conf, float minConfidence) {
    if (conf < 0.f) return true;
    return conf >= minConfidence;
}

std::string TrimText(std::string text) {
    while (!text.empty() && (static_cast<unsigned char>(text.front()) <= ' ')) {
        text.erase(text.begin());
    }
    while (!text.empty() && (static_cast<unsigned char>(text.back()) <= ' ')) {
        text.pop_back();
    }
    return text;
}

bool IsMeaningfulText(const std::string& text) {
    const std::string trimmed = TrimText(text);
    if (trimmed.empty()) return false;
    for (unsigned char c : trimmed) {
        if (c > ' ') return true;
    }
    return false;
}

bool WriteImagePng(const std::filesystem::path& path, const cv::Mat& image, std::string& error) {
    std::vector<uchar> buf;
    if (!cv::imencode(".png", image, buf)) {
        error = u8"图像编码失败";
        return false;
    }
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        error = u8"无法写入临时图像";
        return false;
    }
    out.write(reinterpret_cast<const char*>(buf.data()), static_cast<std::streamsize>(buf.size()));
    if (!out.good()) {
        error = u8"无法写入临时图像";
        return false;
    }
    return true;
}

void SaveDebugImage(const cv::Mat& image) {
    std::string err;
    WriteImagePng(std::filesystem::temp_directory_path() / "pcv_ocr_debug_input.png", image, err);
}

cv::Mat UpscaleIfSmall(cv::Mat image, float minHeight = 64.f) {
    if (image.empty() || image.rows >= static_cast<int>(minHeight)) return image;
    const float scale = minHeight / static_cast<float>(image.rows);
    cv::Mat scaled;
    cv::resize(image, scaled, cv::Size(), scale, scale, cv::INTER_CUBIC);
    return scaled;
}

cv::Mat UpscaleMinWidth(cv::Mat image, int minWidth = 900) {
    if (image.empty() || image.cols >= minWidth) return image;
    const float scale = static_cast<float>(minWidth) / static_cast<float>(image.cols);
    cv::Mat scaled;
    cv::resize(image, scaled, cv::Size(), scale, scale, cv::INTER_CUBIC);
    return scaled;
}

std::string CleanOcrText(std::string text) {
    text = TrimText(text);
    std::string out;
    out.reserve(text.size());
    for (unsigned char c : text) {
        if (c == '|' || c == '\n' || c == '\r' || c == '\t') continue;
        out.push_back(static_cast<char>(c));
    }
    return TrimText(out);
}

const char* kPlateWhitelist =
    u8"京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼"
    u8"ABCDEFGHJKLMNPQRSTUVWXYZ0123456789";

double ScoreOcrResult(const OcrResult& result) {
    if (!IsMeaningfulText(result.fullText) && result.words.empty()) return -1.0;

    double score = 0.0;
    int confCount = 0;
    double confSum = 0.0;
    for (const OcrWord& word : result.words) {
        if (!IsMeaningfulText(word.text)) continue;
        score += static_cast<double>(word.text.size()) * 10.0;
        if (word.confidence >= 0.f) {
            confSum += word.confidence;
            confCount++;
        }
    }
    if (confCount > 0) {
        score += (confSum / static_cast<double>(confCount)) * 3.0;
    }
    score += static_cast<double>(CleanOcrText(result.fullText).size()) * 6.0;
    return score;
}

cv::Mat MakePlateBinary(const cv::Mat& bgr, double threshold = 160.) {
    cv::Mat gray;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, gray, cv::Size(3, 3), 0.);

    cv::Mat bin;
    cv::threshold(gray, bin, threshold, 255., cv::THRESH_BINARY);
    const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(bin, bin, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(bin, bin, cv::MORPH_OPEN, kernel);
    cv::bitwise_not(bin, bin);
    return bin;
}

cv::Mat MakePlateBinaryOtsu(const cv::Mat& bgr) {
    cv::Mat gray;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, gray, cv::Size(3, 3), 0.);
    cv::Mat bin;
    cv::threshold(gray, bin, 0., 255., cv::THRESH_BINARY | cv::THRESH_OTSU);
    if (cv::mean(bin)[0] < 127.0) {
        cv::bitwise_not(bin, bin);
    }
    return bin;
}

cv::Mat ScaleImage(const cv::Mat& image, float scale) {
    if (scale <= 1.001f) return image;
    cv::Mat scaled;
    cv::resize(image, scaled, cv::Size(), scale, scale, cv::INTER_CUBIC);
    return scaled;
}

void ParseTsvFile(const std::string& tsvPath, float minConfidence, int ix0, int iy0, float scale,
                  OcrResult& result);
std::string ReadTextFile(const std::string& path);
#ifdef _WIN32
std::wstring Utf8ToWide(const std::string& text);
std::wstring PathToWide(const std::filesystem::path& path);
std::wstring BuildTesseractArgs(const std::wstring& imgPathW, const std::wstring& outBaseW,
                                const std::wstring& tessdataW, const RecognizeParams& rec,
                                const wchar_t* outputConfig);
bool RunTesseractProcess(const std::wstring& tessExe, const std::wstring& tessDir,
                         const std::wstring& cmdArgs, unsigned int& exitCode, std::string& error);
#endif

cv::Mat NormalizeForTesseract(const cv::Mat& image, bool isGray) {
    if (!isGray && image.channels() == 3) {
        return image;
    }
    cv::Mat gray;
    if (image.channels() == 1) {
        gray = image;
    } else {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    }
    // 仅当背景偏暗（白字黑底）时自动反色，避免与用户的「反色」设置重复翻转
    if (cv::mean(gray)[0] < 80.0) {
        cv::Mat inverted;
        cv::bitwise_not(gray, inverted);
        return inverted;
    }
    return gray;
}

struct OcrAttempt {
    cv::Mat image;
    bool isGray = false;
    bool fullImage = false;
};

bool RunSingleCliPass(const cv::Mat& image, int ix0, int iy0, float scale, const RecognizeParams& rec,
                      const std::string& tessdata, const std::string& tessExe, OcrResult& result,
                      std::string& error) {
    namespace fs = std::filesystem;

    const fs::path workDir =
        fs::temp_directory_path() /
        ("pcv_ocr_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::error_code ec;
    fs::create_directories(workDir, ec);
    if (ec) {
        error = u8"无法创建临时目录";
        return false;
    }

    const fs::path imgPathFs = workDir / "input.png";
    const fs::path outBaseFs = workDir / "out";
    const std::string outBase = outBaseFs.string();
    const std::string tsvPath = outBase + ".tsv";
    const std::string txtPath = outBase + ".txt";

    std::vector<int> pngParams = {cv::IMWRITE_PNG_COMPRESSION, 1};
    if (!WriteImagePng(imgPathFs, image, error)) {
        fs::remove_all(workDir, ec);
        return false;
    }

    unsigned int exitCode = 1;
#ifdef _WIN32
    const std::wstring tessExeW = Utf8ToWide(tessExe);
    const fs::path tessRoot = fs::path(tessExe).parent_path();
    const std::wstring tessDirW = PathToWide(tessRoot);
    const std::wstring imgPathW = PathToWide(imgPathFs);
    const std::wstring outBaseW = PathToWide(outBaseFs);
    const std::wstring tessdataW = Utf8ToWide(tessdata);

    const std::wstring txtArgs = BuildTesseractArgs(imgPathW, outBaseW, tessdataW, rec, nullptr);
    if (!RunTesseractProcess(tessExeW, tessDirW, txtArgs, exitCode, error)) {
        fs::remove_all(workDir, ec);
        return false;
    }

    if (!fs::exists(txtPath, ec)) {
        error = u8"Tesseract 未生成输出（退出码 " + std::to_string(exitCode) + u8"）";
        fs::remove_all(workDir, ec);
        return false;
    }

    result.fullText = ReadTextFile(txtPath);

    const std::wstring tsvArgs = BuildTesseractArgs(imgPathW, outBaseW, tessdataW, rec, L"tsv");
    RunTesseractProcess(tessExeW, tessDirW, tsvArgs, exitCode, error);
#else
    const std::string imgPath = imgPathFs.string();
    auto buildCmd = [&](const char* suffix) {
        std::ostringstream cmd;
        cmd << QuotePath(tessExe) << ' ' << QuotePath(imgPath) << ' ' << QuotePath(outBase)
            << " --tessdata-dir " << QuotePath(tessdata) << " -l \"" << rec.lang << "\" --psm "
            << rec.psm << " --oem " << rec.oem;
        if (!rec.whitelist.empty()) {
            cmd << " -c tessedit_char_whitelist=" << rec.whitelist;
        }
        if (!rec.blacklist.empty()) {
            cmd << " -c tessedit_char_blacklist=" << rec.blacklist;
        }
        if (suffix && suffix[0] != '\0') cmd << ' ' << suffix;
        return cmd.str();
    };

    if (std::system(buildCmd(nullptr).c_str()) != 0) {
        fs::remove_all(workDir, ec);
        error = u8"Tesseract 执行失败，请检查安装路径与语言包";
        return false;
    }
    result.fullText = ReadTextFile(txtPath);
    std::system(buildCmd("tsv").c_str());
#endif

  result.words.clear();
    ParseTsvFile(tsvPath, rec.minConfidence, ix0, iy0, scale, result);

    if (!IsMeaningfulText(result.fullText)) {
        for (std::size_t i = 0; i < result.words.size(); ++i) {
            if (i > 0) result.fullText += ' ';
            result.fullText += result.words[i].text;
        }
    }
    result.fullText = CleanOcrText(result.fullText);
    for (OcrWord& word : result.words) {
        word.text = CleanOcrText(std::move(word.text));
    }

    fs::remove_all(workDir, ec);
    return IsMeaningfulText(result.fullText) || !result.words.empty();
}

void AppendWordFromTsv(const std::vector<std::string>& cols, float minConfidence, int ix0, int iy0,
                       float scale, OcrResult& result) {
    if (cols.size() < 12) return;

    std::string text = TrimText(cols[11]);
    if (!IsMeaningfulText(text)) return;

    const float conf = std::strtof(cols[10].c_str(), nullptr);
    if (!PassesConfidence(conf, minConfidence)) return;

    const int left = std::atoi(cols[6].c_str());
    const int top = std::atoi(cols[7].c_str());
    const int w = std::atoi(cols[8].c_str());
    const int h = std::atoi(cols[9].c_str());

    OcrWord hit;
    hit.text = std::move(text);
    hit.confidence = conf;
    hit.x0 = static_cast<float>(left);
    hit.y0 = static_cast<float>(top);
    hit.x1 = static_cast<float>(left + w);
    hit.y1 = static_cast<float>(top + h);
    MapBoxToSource(hit.x0, hit.y0, hit.x1, hit.y1, static_cast<float>(ix0), static_cast<float>(iy0),
                   scale);
    result.words.push_back(std::move(hit));
}

std::vector<std::string> SplitTsv(const std::string& line);

void ParseTsvFile(const std::string& tsvPath, float minConfidence, int ix0, int iy0, float scale,
                  OcrResult& result) {
    std::ifstream tsv(tsvPath, std::ios::binary);
    if (!tsv.is_open()) return;

    std::string line;
    std::getline(tsv, line);
    std::ostringstream lineText;
    std::ostringstream symbolText;
    bool hasLineText = false;
    bool hasSymbolText = false;

    while (std::getline(tsv, line)) {
        if (line.empty()) continue;
        const std::vector<std::string> cols = SplitTsv(line);
        if (cols.size() < 12) continue;

        const std::string& level = cols[0];
        if (level == "5" || level == "6") {
            AppendWordFromTsv(cols, minConfidence, ix0, iy0, scale, result);
            if (level == "6" && IsMeaningfulText(cols[11])) {
                symbolText << TrimText(cols[11]);
                hasSymbolText = true;
            }
        } else if (level == "4" && IsMeaningfulText(cols[11])) {
            const float conf = std::strtof(cols[10].c_str(), nullptr);
            if (!PassesConfidence(conf, minConfidence)) continue;
            if (hasLineText) lineText << '\n';
            lineText << TrimText(cols[11]);
            hasLineText = true;
        }
    }

    if (result.fullText.empty()) {
        if (hasLineText) {
            result.fullText = lineText.str();
        } else if (hasSymbolText) {
            result.fullText = symbolText.str();
        }
    }
}

std::string FindTessdataDir() {
    namespace fs = std::filesystem;
    const fs::path candidates[] = {
#ifdef POINTCLOUDVIEWER_TESSERACT_ROOT
        fs::path(POINTCLOUDVIEWER_TESSERACT_ROOT) / "tessdata",
#endif
        fs::path("tessdata"),
        fs::path("assets/tessdata"),
        fs::path("D:/software/Tesseract-OCR/tessdata"),
        fs::path("C:/Program Files/Tesseract-OCR/tessdata"),
    };
    for (const fs::path& p : candidates) {
        std::error_code ec;
        if (fs::exists(p / "eng.traineddata", ec)) {
            return fs::absolute(p).string();
        }
    }
    return {};
}

std::string FindTesseractExe() {
    namespace fs = std::filesystem;
    const fs::path candidates[] = {
#ifdef POINTCLOUDVIEWER_TESSERACT_ROOT
        fs::path(POINTCLOUDVIEWER_TESSERACT_ROOT) / "tesseract.exe",
#endif
        fs::path("D:/software/Tesseract-OCR/tesseract.exe"),
        fs::path("C:/Program Files/Tesseract-OCR/tesseract.exe"),
    };
    for (const fs::path& p : candidates) {
        std::error_code ec;
        if (fs::exists(p, ec)) {
            return fs::absolute(p).string();
        }
    }
    return {};
}

std::string QuotePath(const std::string& path) { return "\"" + path + "\""; }

#ifdef _WIN32
std::wstring Utf8ToWide(const std::string& text) {
    if (text.empty()) return {};
    int n = MultiByteToWideChar(CP_UTF8, 0, text.c_str(), -1, nullptr, 0);
    if (n <= 0) {
        n = MultiByteToWideChar(CP_ACP, 0, text.c_str(), -1, nullptr, 0);
        if (n <= 0) return {};
        std::wstring wide(static_cast<std::size_t>(n - 1), L'\0');
        MultiByteToWideChar(CP_ACP, 0, text.c_str(), -1, wide.data(), n);
        return wide;
    }
    std::wstring wide(static_cast<std::size_t>(n - 1), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, text.c_str(), -1, wide.data(), n);
    return wide;
}

std::wstring PathToWide(const std::filesystem::path& path) {
    return path.wstring();
}

bool RunTesseractProcess(const std::wstring& tessExe, const std::wstring& tessDir,
                         const std::wstring& cmdArgs, unsigned int& exitCode, std::string& error) {
    std::wstring cmdLine = L'"' + tessExe + L'"' + cmdArgs;
    std::vector<wchar_t> cmdBuf(cmdLine.begin(), cmdLine.end());
    cmdBuf.push_back(L'\0');

    STARTUPINFOW si{};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESHOWWINDOW;
    si.wShowWindow = SW_HIDE;
    PROCESS_INFORMATION pi{};

    const BOOL ok = CreateProcessW(nullptr, cmdBuf.data(), nullptr, nullptr, FALSE,
                                   CREATE_NO_WINDOW, nullptr,
                                   tessDir.empty() ? nullptr : tessDir.c_str(), &si, &pi);
    if (!ok) {
        error = u8"无法启动 Tesseract（错误码 " + std::to_string(GetLastError()) + u8"）";
        return false;
    }

    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD code = 0;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    exitCode = code;
    return true;
}

bool RunTesseractProcessCaptureStdout(const std::wstring& tessExe, const std::wstring& tessDir,
                                      const std::wstring& cmdArgs, unsigned int& exitCode,
                                      std::string& output, std::string& error) {
    SECURITY_ATTRIBUTES sa{};
    sa.nLength = sizeof(sa);
    sa.bInheritHandle = TRUE;

    HANDLE readPipe = nullptr;
    HANDLE writePipe = nullptr;
    if (!CreatePipe(&readPipe, &writePipe, &sa, 0)) {
        error = u8"无法创建 Tesseract 输出管道";
        return false;
    }
    SetHandleInformation(readPipe, HANDLE_FLAG_INHERIT, 0);

    STARTUPINFOW si{};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES | STARTF_USESHOWWINDOW;
    si.hStdOutput = writePipe;
    si.hStdError = writePipe;
    si.wShowWindow = SW_HIDE;
    PROCESS_INFORMATION pi{};

    std::wstring cmdLine = L'"' + tessExe + L'"' + cmdArgs;
    std::vector<wchar_t> cmdBuf(cmdLine.begin(), cmdLine.end());
    cmdBuf.push_back(L'\0');

    const BOOL ok =
        CreateProcessW(tessExe.c_str(), cmdBuf.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW,
                       nullptr, tessDir.empty() ? nullptr : tessDir.c_str(), &si, &pi);
    CloseHandle(writePipe);
    if (!ok) {
        CloseHandle(readPipe);
        error = u8"无法启动 Tesseract（错误码 " + std::to_string(GetLastError()) + u8"）";
        return false;
    }

    output.clear();
    char buf[4096];
    DWORD bytesRead = 0;
    while (ReadFile(readPipe, buf, sizeof(buf), &bytesRead, nullptr) && bytesRead > 0) {
        output.append(buf, bytesRead);
    }
    CloseHandle(readPipe);

    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD code = 0;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    exitCode = code;

    output = TrimText(output);
    while (!output.empty() && (output.back() == '\n' || output.back() == '\r')) {
        output.pop_back();
    }
    return true;
}
#endif

std::string ReadTextFile(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) return {};
    std::ostringstream ss;
    ss << in.rdbuf();
    std::string text = ss.str();
    while (!text.empty() && (text.back() == '\n' || text.back() == '\r')) {
        text.pop_back();
    }
    return text;
}

#ifdef _WIN32
std::wstring BuildTesseractArgs(const std::wstring& imgPathW, const std::wstring& outBaseW,
                                const std::wstring& tessdataW, const RecognizeParams& rec,
                                const wchar_t* outputConfig) {
    const std::wstring langW = Utf8ToWide(rec.lang);
    std::wostringstream args;
    args << L' ' << L'"' << imgPathW << L'"' << L' ' << L'"' << outBaseW << L'"'
         << L" --tessdata-dir " << L'"' << tessdataW << L'"' << L" -l " << L'"' << langW << L'"'
         << L" --psm " << rec.psm << L" --oem " << rec.oem;
    if (!rec.whitelist.empty()) {
        args << L" -c tessedit_char_whitelist=" << Utf8ToWide(rec.whitelist);
    }
    if (!rec.blacklist.empty()) {
        args << L" -c tessedit_char_blacklist=" << Utf8ToWide(rec.blacklist);
    }
    if (outputConfig && outputConfig[0] != L'\0') {
        args << L' ' << outputConfig;
    }
    return args.str();
}
#endif

std::vector<std::string> SplitTsv(const std::string& line) {
    std::vector<std::string> cols;
    std::string cur;
    for (char c : line) {
        if (c == '\t') {
            cols.push_back(cur);
            cur.clear();
        } else {
            cur.push_back(c);
        }
    }
    cols.push_back(cur);
    return cols;
}

bool RecognizeWithApi(const cv::Mat& gray, int ix0, int iy0, float scale,
                      const RecognizeParams& rec, const std::string& tessdata, OcrResult& result,
                      std::string& error) {
#ifdef POINTCLOUDVIEWER_USE_TESSERACT
    tesseract::TessBaseAPI api;
    if (api.Init(tessdata.c_str(), rec.lang.c_str(), static_cast<tesseract::OEM>(rec.oem)) != 0) {
        error = u8"Tesseract 初始化失败，请检查语言包：" + rec.lang;
        return false;
    }

    api.SetPageSegMode(static_cast<tesseract::PageSegMode>(rec.psm));
    if (!rec.whitelist.empty()) {
        api.SetVariable("tessedit_char_whitelist", rec.whitelist.c_str());
    }
    if (!rec.blacklist.empty()) {
        api.SetVariable("tessedit_char_blacklist", rec.blacklist.c_str());
    }

    api.SetImage(gray.data, gray.cols, gray.rows, 1, gray.cols);

    std::string fullText;
    if (char* utf8 = api.GetUTF8Text()) {
        fullText = utf8;
        delete[] utf8;
    }

    tesseract::ResultIterator* ri = api.GetIterator();
    const std::unique_ptr<tesseract::ResultIterator> guard(ri);
    if (ri) {
        do {
            const char* word = ri->GetUTF8Text(tesseract::RIL_WORD);
            if (!word) continue;
            std::string text = word;
            delete[] word;
            if (text.empty()) continue;

            float conf = 0.f;
            ri->Confidence(tesseract::RIL_WORD, &conf);
            if (!PassesConfidence(conf, rec.minConfidence)) continue;

            int x1 = 0;
            int y1 = 0;
            int x2 = 0;
            int y2 = 0;
            if (!ri->BoundingBox(tesseract::RIL_WORD, &x1, &y1, &x2, &y2)) continue;

            OcrWord hit;
            hit.text = std::move(text);
            hit.confidence = conf;
            hit.x0 = static_cast<float>(x1);
            hit.y0 = static_cast<float>(y1);
            hit.x1 = static_cast<float>(x2);
            hit.y1 = static_cast<float>(y2);
            MapBoxToSource(hit.x0, hit.y0, hit.x1, hit.y1, static_cast<float>(ix0),
                           static_cast<float>(iy0), scale);
            result.words.push_back(std::move(hit));
        } while (ri->Next(tesseract::RIL_WORD));
    }

    result.fullText = fullText;
    return true;
#else
    (void)gray;
    (void)ix0;
    (void)iy0;
    (void)scale;
    (void)rec;
    (void)tessdata;
    (void)result;
    error = u8"未编译 Tesseract API 支持";
    return false;
#endif
}

bool RecognizeWithCli(const cv::Mat& image, bool /*isGray*/, int ix0, int iy0, float scale,
                      const RecognizeParams& rec, const std::string& tessdata,
                      const std::string& tessExe, OcrResult& result, std::string& error) {
    return RunSingleCliPass(image, ix0, iy0, scale, rec, tessdata, tessExe, result, error);
}

}  // namespace

bool IsTesseractAvailable() {
#if defined(POINTCLOUDVIEWER_USE_TESSERACT) || defined(POINTCLOUDVIEWER_USE_TESSERACT_CLI)
    return true;
#else
    return false;
#endif
}

bool IsAvailable() { return IsAnyEngineAvailable(); }

bool IsEngineAvailable(OcrEngine engine) {
    switch (engine) {
        case OcrEngine::Tesseract:
            return IsTesseractAvailable();
        case OcrEngine::PaddleOcr:
            return PaddleOcrTools::IsAvailable();
    }
    return false;
}

bool IsAnyEngineAvailable() {
    return IsEngineAvailable(OcrEngine::Tesseract) || IsEngineAvailable(OcrEngine::PaddleOcr);
}

const char* EngineLabel(OcrEngine engine) {
    switch (engine) {
        case OcrEngine::Tesseract:
            return "Tesseract";
        case OcrEngine::PaddleOcr:
            return "PaddleOCR";
    }
    return "?";
}

std::string EngineAvailabilityHint(OcrEngine engine) {
    if (IsEngineAvailable(engine)) return {};
    if (engine == OcrEngine::PaddleOcr) return PaddleOcrTools::AvailabilityHint();
    return u8"未启用 Tesseract，请安装 Tesseract-OCR 后重新编译";
}

const char* PlateWhitelist() { return kPlateWhitelist; }

bool RecognizeTesseract(const std::vector<uint8_t>& rgb, int width, int height, float roiX0,
                        float roiY0, float roiX1, float roiY1, bool useRoi,
                        const PreprocessParams& prep, const RecognizeParams& rec, OcrResult& result,
                        PreprocessPreview* preview, std::string& error) {
    result = {};
    if (rgb.empty() || width <= 0 || height <= 0) {
        error = u8"图像数据无效";
        return false;
    }

#if !defined(POINTCLOUDVIEWER_USE_TESSERACT) && !defined(POINTCLOUDVIEWER_USE_TESSERACT_CLI)
    error = u8"未启用 Tesseract，请安装 Tesseract 并重新编译";
    return false;
#else
    const float rx0 = useRoi ? std::min(roiX0, roiX1) : 0.f;
    const float ry0 = useRoi ? std::min(roiY0, roiY1) : 0.f;
    const float rx1 = useRoi ? std::max(roiX0, roiX1) : static_cast<float>(width - 1);
    const float ry1 = useRoi ? std::max(roiY0, roiY1) : static_cast<float>(height - 1);
    const int ix0 = std::clamp(static_cast<int>(std::floor(rx0)), 0, width - 1);
    const int iy0 = std::clamp(static_cast<int>(std::floor(ry0)), 0, height - 1);
    const int ix1 = std::clamp(static_cast<int>(std::ceil(rx1)), ix0 + 1, width);
    const int iy1 = std::clamp(static_cast<int>(std::ceil(ry1)), iy0 + 1, height);
    const int margin = useRoi ? std::max(4, std::min(ix1 - ix0, iy1 - iy0) / 15) : 0;
    const int cropX0 = std::max(0, ix0 - margin);
    const int cropY0 = std::max(0, iy0 - margin);
    const int cropX1 = std::min(width, ix1 + margin);
    const int cropY1 = std::min(height, iy1 + margin);
    const int cropW = cropX1 - cropX0;
    const int cropH = cropY1 - cropY0;
    if (cropW < 4 || cropH < 4) {
        error = u8"ROI 区域过小";
        return false;
    }

    cv::Mat bgr = RgbToBgr(rgb, width, height);
    cv::Mat crop = bgr(cv::Rect(cropX0, cropY0, cropW, cropH)).clone();

    bool previewGray = false;
    const cv::Mat plateBin = UpscaleMinWidth(MakePlateBinary(crop, 160.));
    SaveDebugImage(plateBin);
    if (preview) {
        MatToRgbPreview(plateBin, true, *preview);
    }

    std::vector<OcrAttempt> attempts;
    for (double th : {150., 160., 170.}) {
        attempts.push_back({UpscaleMinWidth(MakePlateBinary(crop, th)), true, false});
    }
    attempts.push_back({UpscaleMinWidth(MakePlateBinaryOtsu(crop)), true, false});
    const cv::Mat colorScaled = UpscaleIfSmall(ScaleImage(crop, prep.scale));
    attempts.push_back({colorScaled.clone(), false, false});
    if (NeedsGrayPreprocess(prep)) {
        bool isGray = false;
        cv::Mat ocrImage = PrepareOcrImage(crop, prep, isGray);
        attempts.push_back({UpscaleIfSmall(NormalizeForTesseract(ocrImage, isGray)),
                            isGray || ocrImage.channels() == 1, false});
    }
    {
        bool isGray = false;
        PreprocessParams grayOnly = prep;
        grayOnly.binarize = BinarizeMode::None;
        grayOnly.invert = false;
        cv::Mat grayImg = PrepareOcrImage(crop, grayOnly, isGray);
        attempts.push_back({UpscaleIfSmall(NormalizeForTesseract(grayImg, true)), true, false});
    }
    if (useRoi) {
        const cv::Mat fullScaled = UpscaleIfSmall(ScaleImage(bgr, prep.scale), 64.f);
        attempts.push_back({fullScaled.clone(), false, true});
    }

    const std::string tessdata = FindTessdataDir();
    if (tessdata.empty()) {
        error = u8"未找到 tessdata 语言包目录";
        return false;
    }

    const auto t0 = std::chrono::steady_clock::now();
    std::string lastError;
    bool ok = false;
    int fullOriginX = 0;
    int fullOriginY = 0;
    OcrResult bestResult;
    double bestScore = -1.0;

    auto tryAttempts = [&](const RecognizeParams& params) {
        for (const OcrAttempt& att : attempts) {
            OcrResult attemptResult;
            std::string attemptError;
            bool attemptOk = false;
            const int originX = att.fullImage ? fullOriginX : cropX0;
            const int originY = att.fullImage ? fullOriginY : cropY0;
#if defined(POINTCLOUDVIEWER_USE_TESSERACT_CLI)
            const std::string tessExe = FindTesseractExe();
            if (tessExe.empty()) {
                attemptError = u8"未找到 tesseract.exe";
            } else {
                attemptOk = RunSingleCliPass(att.image, originX, originY, prep.scale, params,
                                             tessdata, tessExe, attemptResult, attemptError);
            }
#elif defined(POINTCLOUDVIEWER_USE_TESSERACT)
            if (att.isGray || att.image.channels() == 1) {
                attemptOk = RecognizeWithApi(att.image, originX, originY, prep.scale, params,
                                             tessdata, attemptResult, attemptError);
            } else {
                cv::Mat gray;
                cv::cvtColor(att.image, gray, cv::COLOR_BGR2GRAY);
                attemptOk = RecognizeWithApi(NormalizeForTesseract(gray, true), originX, originY,
                                             prep.scale, params, tessdata, attemptResult,
                                             attemptError);
            }
#endif
            if (!attemptError.empty()) lastError = attemptError;
            if (!attemptOk) continue;
            const double score = ScoreOcrResult(attemptResult);
            if (score > bestScore) {
                bestScore = score;
                bestResult = std::move(attemptResult);
            }
        }
    };

    std::vector<RecognizeParams> paramSets;
    {
        RecognizeParams plate;
        plate.lang = "chi_sim";
        plate.psm = 13;
        plate.oem = 3;
        plate.minConfidence = rec.minConfidence;
        plate.whitelist = kPlateWhitelist;
        paramSets.push_back(plate);
    }
    {
        RecognizeParams plate;
        plate.lang = "chi_sim";
        plate.psm = 8;
        plate.oem = 3;
        plate.minConfidence = rec.minConfidence;
        plate.whitelist = kPlateWhitelist;
        paramSets.push_back(plate);
    }
    if (!rec.whitelist.empty()) {
        paramSets.push_back(rec);
    } else {
        RecognizeParams user = rec;
        user.whitelist = kPlateWhitelist;
        paramSets.push_back(user);
    }
    {
        RecognizeParams cli = rec;
        cli.lang = "chi_sim";
        cli.psm = 3;
        cli.oem = 3;
        bool exists = false;
        for (const RecognizeParams& p : paramSets) {
            if (p.lang == cli.lang && p.psm == cli.psm && p.oem == cli.oem) exists = true;
        }
        if (!exists) paramSets.push_back(cli);
    }
    if (rec.psm != 7) {
        RecognizeParams line = rec;
        line.psm = 7;
        paramSets.push_back(line);
    }

#if defined(POINTCLOUDVIEWER_USE_TESSERACT_CLI)
    if (FindTesseractExe().empty()) {
        error = u8"未找到 tesseract.exe";
        return false;
    }
#endif

    for (const RecognizeParams& params : paramSets) {
        tryAttempts(params);
    }

    ok = bestScore >= 0.0;
    if (ok) {
        result = std::move(bestResult);
    }

    const auto t1 = std::chrono::steady_clock::now();
    result.elapsedMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    result.ok = ok;

    if (!result.ok) {
        const std::string debugPath =
            (std::filesystem::temp_directory_path() / "pcv_ocr_debug_input.png").string();
        error = lastError.empty()
                    ? u8"未识别到文字，调试图已保存至 " + debugPath +
                          u8"（可用命令行 tesseract 该图验证）"
                    : lastError + u8"；调试图：" + debugPath;
        return false;
    }
    return true;
#endif
}

}  // namespace OcrTools
