#include "tools/PaddleOcrTools.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
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

namespace PaddleOcrTools {
namespace {

cv::Mat RgbToBgr(const std::vector<uint8_t>& rgb, int width, int height) {
    cv::Mat rgbMat(height, width, CV_8UC3, const_cast<uint8_t*>(rgb.data()));
    cv::Mat bgr;
    cv::cvtColor(rgbMat, bgr, cv::COLOR_RGB2BGR);
    return bgr;
}

cv::Mat RgbCropToBgr(const std::vector<uint8_t>& rgb, int width, int height, const cv::Rect& roi) {
    cv::Mat rgbMat(height, width, CV_8UC3, const_cast<uint8_t*>(rgb.data()));
    cv::Mat cropRgb = rgbMat(roi).clone();
    cv::Mat bgr;
    cv::cvtColor(cropRgb, bgr, cv::COLOR_RGB2BGR);
    return bgr;
}

bool WriteImageJpg(const std::filesystem::path& path, const cv::Mat& image) {
    std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 95};
    std::vector<uchar> buf;
    if (!cv::imencode(".jpg", image, buf, params)) return false;
    std::ofstream out(path, std::ios::binary);
    if (!out) return false;
    out.write(reinterpret_cast<const char*>(buf.data()), static_cast<std::streamsize>(buf.size()));
    return out.good();
}

std::string ReadTextFile(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) return {};
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

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

bool RunProcess(const std::wstring& exe, const std::wstring& args, unsigned int& exitCode) {
    std::wstring cmdLine = L'"' + exe + L'"' + args;
    std::vector<wchar_t> cmdBuf(cmdLine.begin(), cmdLine.end());
    cmdBuf.push_back(L'\0');

    STARTUPINFOW si{};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESHOWWINDOW;
    si.wShowWindow = SW_HIDE;
    PROCESS_INFORMATION pi{};

    const BOOL ok = CreateProcessW(nullptr, cmdBuf.data(), nullptr, nullptr, FALSE, CREATE_NO_WINDOW,
                                   nullptr, nullptr, &si, &pi);
    if (!ok) return false;

    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD code = 0;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    exitCode = code;
    return true;
}

std::string Base64Encode(const uint8_t* data, std::size_t len) {
    static constexpr char kTable[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve((len + 2) / 3 * 4);
    for (std::size_t i = 0; i < len; i += 3) {
        const unsigned int b0 = data[i];
        const unsigned int b1 = (i + 1 < len) ? data[i + 1] : 0;
        const unsigned int b2 = (i + 2 < len) ? data[i + 2] : 0;
        const unsigned int n = (b0 << 16) | (b1 << 8) | b2;
        out.push_back(kTable[(n >> 18) & 0x3F]);
        out.push_back(kTable[(n >> 12) & 0x3F]);
        out.push_back((i + 1 < len) ? kTable[(n >> 6) & 0x3F] : '=');
        out.push_back((i + 2 < len) ? kTable[n & 0x3F] : '=');
    }
    return out;
}

bool EncodeJpegBase64(const cv::Mat& image, std::string& outB64) {
    std::vector<uchar> buf;
    const std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 90};
    if (!cv::imencode(".jpg", image, buf, params)) return false;
    outB64 = Base64Encode(buf.data(), buf.size());
    return true;
}

std::string JsonEscape(const std::string& text) {
    std::string out;
    out.reserve(text.size() + 8);
    for (const char c : text) {
        switch (c) {
            case '\\':
                out += "\\\\";
                break;
            case '"':
                out += "\\\"";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                out.push_back(c);
                break;
        }
    }
    return out;
}

std::string PathToJson(const std::filesystem::path& path) {
    std::string s = path.u8string();
    for (char& c : s) {
        if (c == '\\') c = '/';
    }
    return JsonEscape(s);
}

bool ReadLineFromPipe(HANDLE pipe, std::string& line, DWORD timeoutMs) {
    line.clear();
    const DWORD start = GetTickCount();
    char ch = '\0';
    DWORD read = 0;
    while (true) {
        if (timeoutMs > 0 && GetTickCount() - start > timeoutMs) return false;

        DWORD avail = 0;
        if (!PeekNamedPipe(pipe, nullptr, 0, nullptr, &avail, nullptr)) return false;
        if (avail == 0) {
            Sleep(5);
            continue;
        }
        if (!ReadFile(pipe, &ch, 1, &read, nullptr) || read == 0) return false;
        if (ch == '\n') return true;
        if (ch != '\r') line.push_back(ch);
    }
}

bool ReadJsonLineFromPipe(HANDLE pipe, std::string& line, DWORD timeoutMs, const char* mustContain) {
    const DWORD start = GetTickCount();
    while (true) {
        const DWORD elapsed = GetTickCount() - start;
        if (timeoutMs > 0 && elapsed > timeoutMs) return false;
        const DWORD perLineTimeout = timeoutMs > 0 ? std::max<DWORD>(1000, timeoutMs - elapsed) : 0;
        if (!ReadLineFromPipe(pipe, line, perLineTimeout)) return false;
        if (line.find('{') == std::string::npos) continue;
        if (mustContain && line.find(mustContain) == std::string::npos) continue;
        return true;
    }
}

bool WriteLineToPipe(HANDLE pipe, const std::string& line) {
    std::string payload = line;
    if (payload.empty() || payload.back() != '\n') payload.push_back('\n');
    DWORD written = 0;
    return WriteFile(pipe, payload.data(), static_cast<DWORD>(payload.size()), &written, nullptr) != 0;
}

std::filesystem::path GetExeDirectory() {
    wchar_t buf[MAX_PATH]{};
    GetModuleFileNameW(nullptr, buf, MAX_PATH);
    return std::filesystem::path(buf).parent_path();
}
#endif

std::filesystem::path FindPaddleScript();
bool FindPython(std::wstring& pythonExe, std::wstring& prefixArgs);
bool ExtractJsonBool(const std::string& json, const std::string& key, bool fallback);
std::string ExtractJsonString(const std::string& json, const std::string& key, std::size_t from);

std::filesystem::path FindPaddleScript() {
    namespace fs = std::filesystem;
#ifdef _WIN32
    const fs::path candidates[] = {
        GetExeDirectory() / "assets" / "ocr" / "paddle_ocr.py",
        fs::path("assets/ocr/paddle_ocr.py"),
        fs::path("PointCloudViewer/assets/ocr/paddle_ocr.py"),
    };
#else
    const fs::path candidates[] = {
        fs::path("assets/ocr/paddle_ocr.py"),
    };
#endif
    for (const fs::path& p : candidates) {
        std::error_code ec;
        if (fs::exists(p, ec)) return fs::absolute(p);
    }
    return {};
}

bool ResolvePythonExe(std::wstring& pythonExe, std::wstring& prefixArgs) {
    namespace fs = std::filesystem;
#ifdef _WIN32
#ifdef POINTCLOUDVIEWER_PYTHON_EXE
    {
        const fs::path pyPath = POINTCLOUDVIEWER_PYTHON_EXE;
        std::error_code ec;
        if (fs::exists(pyPath, ec)) {
            pythonExe = pyPath.wstring();
            prefixArgs = L"";
            return true;
        }
    }
#endif
    const wchar_t* fullPaths[] = {
        L"D:/software/Python3.13.14/python.exe",
        L"D:/software/Python3.13/python.exe",
        L"D:/software/Python313/python.exe",
        L"C:/Python313/python.exe",
    };
    for (const wchar_t* path : fullPaths) {
        std::error_code ec;
        if (fs::exists(path, ec)) {
            pythonExe = path;
            prefixArgs = L"";
            return true;
        }
    }
#endif
    (void)pythonExe;
    (void)prefixArgs;
    return false;
}

bool FindPython(std::wstring& pythonExe, std::wstring& prefixArgs) {
    static bool cached = false;
    static std::wstring cachedExe;
    static std::wstring cachedPrefix;
    if (cached) {
        pythonExe = cachedExe;
        prefixArgs = cachedPrefix;
        return true;
    }

    if (!ResolvePythonExe(pythonExe, prefixArgs)) {
        return false;
    }

    cachedExe = pythonExe;
    cachedPrefix = prefixArgs;
    cached = true;
    return true;
}

std::string UnescapeJson(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (std::size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '\\' && i + 1 < s.size()) {
            const char c = s[i + 1];
            if (c == 'n') {
                out.push_back('\n');
                ++i;
            } else if (c == 'r') {
                out.push_back('\r');
                ++i;
            } else if (c == 't') {
                out.push_back('\t');
                ++i;
            } else if (c == '"' || c == '\\' || c == '/') {
                out.push_back(c);
                ++i;
            } else {
                out.push_back(s[i]);
            }
        } else {
            out.push_back(s[i]);
        }
    }
    return out;
}

std::string ExtractJsonString(const std::string& json, const std::string& key, std::size_t from = 0) {
    const std::string pat = "\"" + key + "\"";
    const std::size_t keyPos = json.find(pat, from);
    if (keyPos == std::string::npos) return {};
    const std::size_t colon = json.find(':', keyPos + pat.size());
    if (colon == std::string::npos) return {};
    const std::size_t q0 = json.find('"', colon + 1);
    if (q0 == std::string::npos) return {};
    std::string out;
    for (std::size_t i = q0 + 1; i < json.size(); ++i) {
        if (json[i] == '"' && (i == 0 || json[i - 1] != '\\')) break;
        out.push_back(json[i]);
    }
    return UnescapeJson(out);
}

bool ExtractJsonBool(const std::string& json, const std::string& key, bool fallback = false) {
    const std::string pat = "\"" + key + "\"";
    const std::size_t keyPos = json.find(pat);
    if (keyPos == std::string::npos) return fallback;
    const std::size_t colon = json.find(':', keyPos + pat.size());
    if (colon == std::string::npos) return fallback;
    const std::size_t t = json.find("true", colon);
    const std::size_t f = json.find("false", colon);
    if (t != std::string::npos && (f == std::string::npos || t < f)) return true;
    if (f != std::string::npos) return false;
    return fallback;
}

float ExtractJsonNumber(const std::string& json, const std::string& key, std::size_t from) {
    const std::string pat = "\"" + key + "\"";
    const std::size_t keyPos = json.find(pat, from);
    if (keyPos == std::string::npos) return 0.f;
    const std::size_t colon = json.find(':', keyPos + pat.size());
    if (colon == std::string::npos) return 0.f;
    return std::strtof(json.c_str() + colon + 1, nullptr);
}

bool ParsePaddleJson(const std::string& json, float minConfidence, int cropX0, int cropY0,
                     OcrTools::OcrResult& result, std::string& error) {
    result = {};
    if (!ExtractJsonBool(json, "ok", false)) {
        error = ExtractJsonString(json, "error");
        if (error.empty()) error = u8"PaddleOCR 返回失败";
        return false;
    }

    result.fullText = ExtractJsonString(json, "fullText");
    const std::size_t wordsPos = json.find("\"words\"");
    if (wordsPos == std::string::npos) {
        result.ok = !result.fullText.empty();
        return result.ok;
    }

    std::size_t pos = wordsPos;
    while ((pos = json.find("\"text\"", pos + 1)) != std::string::npos) {
        const std::string text = ExtractJsonString(json, "text", pos);
        if (text.empty()) continue;
        const float conf = ExtractJsonNumber(json, "confidence", pos);
        if (conf >= 0.f && conf < minConfidence) continue;
        OcrTools::OcrWord word;
        word.text = text;
        word.confidence = conf;
        word.x0 = ExtractJsonNumber(json, "x0", pos) + static_cast<float>(cropX0);
        word.y0 = ExtractJsonNumber(json, "y0", pos) + static_cast<float>(cropY0);
        word.x1 = ExtractJsonNumber(json, "x1", pos) + static_cast<float>(cropX0);
        word.y1 = ExtractJsonNumber(json, "y1", pos) + static_cast<float>(cropY0);
        result.words.push_back(std::move(word));
    }

    if (result.fullText.empty()) {
        for (std::size_t i = 0; i < result.words.size(); ++i) {
            if (i > 0) result.fullText += ' ';
            result.fullText += result.words[i].text;
        }
    }

    result.ok = !result.fullText.empty() || !result.words.empty();
    return result.ok;
}

#ifdef _WIN32
class PaddleWorker {
public:
    static PaddleWorker& Instance() {
        static PaddleWorker worker;
        return worker;
    }

    bool Recognize(const std::string& imageB64, const PaddleParams& params, bool recOnly,
                   std::string& jsonOut, std::string& error) {
        jsonOut.clear();
        if (!EnsureStarted(error)) return false;

        std::ostringstream req;
        req << "{\"image_b64\":\"" << imageB64 << "\",\"lang\":\"" << JsonEscape(params.lang)
            << "\",\"use_cls\":" << (params.useAngleCls ? "true" : "false") << ",\"rec_only\":"
            << (recOnly ? "true" : "false") << '}';

        if (!WriteLineToPipe(stdinWrite_, req.str())) {
            Reset();
            error = u8"PaddleOCR 服务通信失败";
            return false;
        }

        std::string line;
        if (!ReadJsonLineFromPipe(stdoutRead_, line, 180000, "\"ok\"")) {
            Reset();
            error = u8"PaddleOCR 服务超时";
            return false;
        }

        jsonOut = std::move(line);
        return !jsonOut.empty();
    }

    bool Warmup(std::string& error) {
        if (ready_) return true;
        return EnsureStarted(error);
    }

    void Shutdown() { Reset(); }

private:
    PaddleWorker() = default;
    ~PaddleWorker() { Reset(); }

    void Reset() {
        if (stdinWrite_) WriteLineToPipe(stdinWrite_, "quit");
        if (process_) {
            WaitForSingleObject(process_, 2000);
            DWORD code = STILL_ACTIVE;
            if (GetExitCodeProcess(process_, &code) && code == STILL_ACTIVE) {
                TerminateProcess(process_, 1);
            }
        }
        if (stdinWrite_) CloseHandle(stdinWrite_);
        if (stdoutRead_) CloseHandle(stdoutRead_);
        if (process_) CloseHandle(process_);
        stdinWrite_ = nullptr;
        stdoutRead_ = nullptr;
        process_ = nullptr;
        ready_ = false;
    }

    bool EnsureStarted(std::string& error) {
        if (ready_ && process_) {
            DWORD code = STILL_ACTIVE;
            if (GetExitCodeProcess(process_, &code) && code == STILL_ACTIVE) return true;
            Reset();
        }

        const std::filesystem::path script = FindPaddleScript();
        if (script.empty()) {
            error = u8"未找到 PaddleOCR 脚本 assets/ocr/paddle_ocr.py";
            return false;
        }

        std::wstring pythonExe;
        std::wstring prefixArgs;
        if (!FindPython(pythonExe, prefixArgs)) {
            error = u8"未找到 Python";
            return false;
        }

        SECURITY_ATTRIBUTES sa{};
        sa.nLength = sizeof(sa);
        sa.bInheritHandle = TRUE;

        HANDLE stdoutRead = nullptr;
        HANDLE stdoutWrite = nullptr;
        HANDLE stdinRead = nullptr;
        HANDLE stdinWrite = nullptr;
        if (!CreatePipe(&stdoutRead, &stdoutWrite, &sa, 0) ||
            !CreatePipe(&stdinRead, &stdinWrite, &sa, 0)) {
            error = u8"无法创建 PaddleOCR 服务管道";
            return false;
        }

        SetHandleInformation(stdoutRead, HANDLE_FLAG_INHERIT, 0);
        SetHandleInformation(stdinWrite, HANDLE_FLAG_INHERIT, 0);

        HANDLE stderrNull =
            CreateFileW(L"NUL", GENERIC_WRITE, FILE_SHARE_WRITE, &sa, OPEN_EXISTING, 0, nullptr);

        std::wstring cmdLine = L'"' + pythonExe + L'"' + prefixArgs + L" -u \"" +
                               Utf8ToWide(script.u8string()) + L"\" --server";
        std::vector<wchar_t> cmdBuf(cmdLine.begin(), cmdLine.end());
        cmdBuf.push_back(L'\0');

        STARTUPINFOW si{};
        si.cb = sizeof(si);
        si.dwFlags = STARTF_USESTDHANDLES | STARTF_USESHOWWINDOW;
        si.hStdInput = stdinRead;
        si.hStdOutput = stdoutWrite;
        si.hStdError = stderrNull != INVALID_HANDLE_VALUE ? stderrNull : stdoutWrite;
        si.wShowWindow = SW_HIDE;

        PROCESS_INFORMATION pi{};
        const BOOL ok =
            CreateProcessW(nullptr, cmdBuf.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW, nullptr,
                           nullptr, &si, &pi);
        CloseHandle(stdinRead);
        CloseHandle(stdoutWrite);
        if (stderrNull != INVALID_HANDLE_VALUE) CloseHandle(stderrNull);

        if (!ok) {
            CloseHandle(stdoutRead);
            CloseHandle(stdinWrite);
            error = u8"无法启动 PaddleOCR 服务";
            return false;
        }

        CloseHandle(pi.hThread);
        process_ = pi.hProcess;
        stdinWrite_ = stdinWrite;
        stdoutRead_ = stdoutRead;

        std::string line;
        if (!ReadJsonLineFromPipe(stdoutRead_, line, 180000, "\"ready\"")) {
            Reset();
            error = u8"PaddleOCR 模型加载超时";
            return false;
        }

        if (!ExtractJsonBool(line, "ready", false)) {
            error = ExtractJsonString(line, "error");
            if (error.empty()) error = u8"PaddleOCR 服务启动失败";
            Reset();
            return false;
        }

        ready_ = true;
        return true;
    }

    HANDLE process_ = nullptr;
    HANDLE stdinWrite_ = nullptr;
    HANDLE stdoutRead_ = nullptr;
    bool ready_ = false;
};

bool RecognizeViaCli(const std::filesystem::path& script, const std::wstring& pythonExe,
                     const std::wstring& prefixArgs, const std::filesystem::path& imgPath,
                     const std::filesystem::path& jsonPath, const PaddleParams& params,
                     std::string& jsonOut, std::string& error) {
    std::wostringstream args;
    args << prefixArgs << L" \"" << Utf8ToWide(script.u8string()) << L"\" \""
         << Utf8ToWide(imgPath.u8string()) << L"\" \"" << Utf8ToWide(jsonPath.u8string()) << L"\" "
         << Utf8ToWide(params.lang) << L' ' << (params.useAngleCls ? L"1" : L"0");

    unsigned int exitCode = 1;
    if (!RunProcess(pythonExe, args.str(), exitCode)) {
        error = u8"无法启动 Python 运行 PaddleOCR";
        return false;
    }

    jsonOut = ReadTextFile(jsonPath);
    if (jsonOut.empty()) {
        error = u8"PaddleOCR 未返回结果";
        return false;
    }
    return true;
}

std::filesystem::path PaddleWorkDir() {
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "pcv_paddle_work";
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    return dir;
}
#endif

}  // namespace

bool IsAvailable() {
    if (FindPaddleScript().empty()) return false;
    std::wstring pythonExe;
    std::wstring prefixArgs;
    return ResolvePythonExe(pythonExe, prefixArgs);
}

std::string AvailabilityHint() {
    if (FindPaddleScript().empty()) {
        return u8"未找到 assets/ocr/paddle_ocr.py";
    }
    std::wstring pythonExe;
    std::wstring prefixArgs;
    if (!ResolvePythonExe(pythonExe, prefixArgs)) {
        return u8"未找到 Python（期望路径 D:\\software\\Python3.13.14\\python.exe）";
    }
    return {};
}

bool Recognize(const std::vector<uint8_t>& rgb, int width, int height, float roiX0, float roiY0,
               float roiX1, float roiY1, bool useRoi, const PaddleParams& params,
               OcrTools::OcrResult& result, std::string& error) {
    result = {};
    if (rgb.empty() || width <= 0 || height <= 0) {
        error = u8"图像数据无效";
        return false;
    }

    const std::filesystem::path script = FindPaddleScript();
    if (script.empty()) {
        error = u8"未找到 PaddleOCR 脚本 assets/ocr/paddle_ocr.py";
        return false;
    }

    std::wstring pythonExe;
    std::wstring prefixArgs;
    if (!FindPython(pythonExe, prefixArgs)) {
        error = u8"未找到 Python 或未安装 paddleocr（pip install paddlepaddle paddleocr）";
        return false;
    }

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
    if (cropX1 - cropX0 < 4 || cropY1 - cropY0 < 4) {
        error = u8"ROI 区域过小";
        return false;
    }

    namespace fs = std::filesystem;
    const fs::path workDir = PaddleWorkDir();

    const cv::Rect cropRect(cropX0, cropY0, cropX1 - cropX0, cropY1 - cropY0);
    cv::Mat crop = RgbCropToBgr(rgb, width, height, cropRect);
    if (crop.cols > 640) {
        const int newH = std::max(1, static_cast<int>(std::lround(crop.rows * 640.0 / crop.cols)));
        cv::resize(crop, crop, cv::Size(640, newH), 0, 0, cv::INTER_AREA);
    }

    std::string imageB64;
    if (!EncodeJpegBase64(crop, imageB64)) {
        error = u8"无法编码图像";
        return false;
    }

    const bool recOnly = useRoi && params.skipDet;
    const fs::path imgPath = workDir / "input.jpg";
    if (!WriteImageJpg(imgPath, crop)) {
        error = u8"无法写入临时图像";
        return false;
    }

    const auto t0 = std::chrono::steady_clock::now();
    std::string json;
#ifdef _WIN32
    if (!PaddleWorker::Instance().Recognize(imageB64, params, recOnly, json, error)) {
        const fs::path jsonPath = workDir / "result.json";
        if (!RecognizeViaCli(script, pythonExe, prefixArgs, imgPath, jsonPath, params, json, error)) {
            return false;
        }
    }
#else
    const fs::path jsonPath = workDir / "result.json";
    std::ostringstream cmd;
    cmd << "python3 \"" << script.string() << "\" \"" << imgPath.string() << "\" \""
        << jsonPath.string() << "\" " << params.lang << ' ' << (params.useAngleCls ? "1" : "0");
    if (std::system(cmd.str().c_str()) != 0) {
        error = u8"PaddleOCR 脚本执行失败";
        return false;
    }
    json = ReadTextFile(jsonPath);
    if (json.empty()) {
        error = u8"PaddleOCR 未返回结果";
        return false;
    }
#endif

    const bool ok = ParsePaddleJson(json, params.minConfidence, cropX0, cropY0, result, error);
    const auto t1 = std::chrono::steady_clock::now();
    result.elapsedMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    result.ok = ok;
    if (!ok && error.empty()) error = u8"PaddleOCR 未识别到文字";
    return ok;
}

bool Warmup(std::string& error) {
#ifdef _WIN32
    return PaddleWorker::Instance().Warmup(error);
#else
    (void)error;
    return true;
#endif
}

void Shutdown() {
#ifdef _WIN32
    PaddleWorker::Instance().Shutdown();
#endif
}

}  // namespace PaddleOcrTools
