#include "app/FileDialog.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <commdlg.h>
#endif

#include <vector>
#include <cstring>
#include <string>
#include <cwchar>

namespace FileDialog {
namespace {

#ifdef _WIN32
std::wstring Utf8ToWide(const char* utf8) {
    if (!utf8 || !*utf8) return {};
    const int n = MultiByteToWideChar(CP_UTF8, 0, utf8, -1, nullptr, 0);
    if (n <= 0) return {};
    std::wstring out(static_cast<std::size_t>(n - 1), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, utf8, -1, out.data(), n);
    return out;
}

std::string WideToUtf8(const wchar_t* wide) {
    if (!wide || !*wide) return {};
    const int n = WideCharToMultiByte(CP_UTF8, 0, wide, -1, nullptr, 0, nullptr, nullptr);
    if (n <= 0) return {};
    std::string out(static_cast<std::size_t>(n - 1), '\0');
    WideCharToMultiByte(CP_UTF8, 0, wide, -1, out.data(), n, nullptr, nullptr);
    return out;
}
#endif

}  // namespace

std::string OpenPointCloudFile() {
#ifdef _WIN32
    char file[MAX_PATH] = {0};
    OPENFILENAMEA ofn{};
    ofn.lStructSize = sizeof(ofn);
    ofn.lpstrFile = file;
    ofn.nMaxFile = MAX_PATH;
    ofn.lpstrFilter =
        "点云文件\0*.ply;*.pcd;*.xyz;*.obj;*.txt\0"
        "PLY 文件\0*.ply\0"
        "PCD 文件\0*.pcd\0"
        "XYZ 文件\0*.xyz;*.txt\0"
        "OBJ 文件\0*.obj\0"
        "全部文件\0*.*\0";
    ofn.nFilterIndex = 1;
    ofn.lpstrTitle = "打开点云文件";
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;
    if (GetOpenFileNameA(&ofn) == TRUE) {
        return std::string(file);
    }
    return {};
#else
    return {};
#endif
}

std::string SavePointCloudFile() {
#ifdef _WIN32
    char file[MAX_PATH] = {0};
    OPENFILENAMEA ofn{};
    ofn.lStructSize = sizeof(ofn);
    ofn.lpstrFile = file;
    ofn.nMaxFile = MAX_PATH;
    ofn.lpstrFilter =
        "PLY 文件\0*.ply\0"
        "XYZ 文件\0*.xyz\0"
        "TXT 文件\0*.txt\0"
        "全部文件\0*.*\0";
    ofn.nFilterIndex = 1;
    ofn.lpstrTitle = "保存点云";
    ofn.lpstrDefExt = "ply";
    ofn.Flags = OFN_OVERWRITEPROMPT | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR;
    if (GetSaveFileNameA(&ofn) == TRUE) {
        std::string path(file);
        // Ensure extension when user typed bare name.
        if (path.find('.') == std::string::npos) {
            if (ofn.nFilterIndex == 1) path += ".ply";
            else if (ofn.nFilterIndex == 2) path += ".xyz";
            else if (ofn.nFilterIndex == 3) path += ".txt";
            else path += ".ply";
        }
        return path;
    }
    return {};
#else
    return {};
#endif
}

std::string OpenImageFile(const char* title) {
#ifdef _WIN32
    wchar_t file[MAX_PATH] = {0};
    OPENFILENAMEW ofn{};
    ofn.lStructSize = sizeof(ofn);
    ofn.lpstrFile = file;
    ofn.nMaxFile = MAX_PATH;
    const wchar_t* filter =
        L"图像文件\0*.png;*.bmp;*.jpg;*.jpeg;*.tif;*.tiff;*.gif;*.webp\0"
        L"PNG\0*.png\0"
        L"BMP\0*.bmp\0"
        L"JPEG\0*.jpg;*.jpeg\0"
        L"TIFF\0*.tif;*.tiff\0"
        L"全部文件\0*.*\0";
    ofn.lpstrFilter = filter;
    ofn.nFilterIndex = 1;
    const std::wstring titleW = title ? Utf8ToWide(title) : L"打开图像";
    ofn.lpstrTitle = titleW.c_str();
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;
    if (GetOpenFileNameW(&ofn) == TRUE) {
        return WideToUtf8(file);
    }
    return {};
#else
    (void)title;
    return {};
#endif
}

std::vector<std::string> OpenMultipleImageFiles(const char* title) {
#ifdef _WIN32
    std::vector<wchar_t> buf(65536, 0);
    OPENFILENAMEW ofn{};
    ofn.lStructSize = sizeof(ofn);
    ofn.lpstrFile = buf.data();
    ofn.nMaxFile = static_cast<DWORD>(buf.size());
    const wchar_t* filter =
        L"图像文件\0*.png;*.bmp;*.jpg;*.jpeg;*.tif;*.tiff;*.gif;*.webp\0"
        L"全部文件\0*.*\0";
    ofn.lpstrFilter = filter;
    ofn.nFilterIndex = 1;
    const std::wstring titleW = title ? Utf8ToWide(title) : L"批量选择图像";
    ofn.lpstrTitle = titleW.c_str();
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST | OFN_ALLOWMULTISELECT | OFN_EXPLORER |
                OFN_NOCHANGEDIR;
    if (GetOpenFileNameW(&ofn) != TRUE) return {};

    std::vector<std::string> paths;
    const wchar_t* p = buf.data();
    const std::wstring dir = p;
    p += dir.size() + 1;
    if (*p == L'\0') {
        paths.push_back(WideToUtf8(dir.c_str()));
        return paths;
    }
    while (*p != L'\0') {
        paths.push_back(WideToUtf8((dir + L"\\" + p).c_str()));
        p += std::wcslen(p) + 1;
    }
    return paths;
#else
    (void)title;
    return {};
#endif
}

std::string OpenShapeTemplateFile() {
#ifdef _WIN32
    char file[MAX_PATH] = {0};
    OPENFILENAMEA ofn{};
    ofn.lStructSize = sizeof(ofn);
    ofn.lpstrFile = file;
    ofn.nMaxFile = MAX_PATH;
    ofn.lpstrFilter = "形状模板\0*.stm;*.yaml;*.yml\0全部文件\0*.*\0";
    ofn.nFilterIndex = 1;
    ofn.lpstrTitle = "加载形状模板";
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;
    if (GetOpenFileNameA(&ofn) == TRUE) {
        return std::string(file);
    }
    return {};
#else
    return {};
#endif
}

std::string SaveShapeTemplateFile() {
#ifdef _WIN32
    char file[MAX_PATH] = {0};
    OPENFILENAMEA ofn{};
    ofn.lStructSize = sizeof(ofn);
    ofn.lpstrFile = file;
    ofn.nMaxFile = MAX_PATH;
    ofn.lpstrFilter = "形状模板\0*.stm\0YAML\0*.yaml;*.yml\0全部文件\0*.*\0";
    ofn.nFilterIndex = 1;
    ofn.lpstrTitle = "保存形状模板";
    ofn.lpstrDefExt = "stm";
    ofn.Flags = OFN_OVERWRITEPROMPT | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR;
    if (GetSaveFileNameA(&ofn) == TRUE) {
        std::string path(file);
        if (path.find('.') == std::string::npos) path += ".stm";
        return path;
    }
    return {};
#endif
}

std::string OpenHalconShapeModelFile() {
#ifdef _WIN32
    char file[MAX_PATH] = {0};
    OPENFILENAMEA ofn{};
    ofn.lStructSize = sizeof(ofn);
    ofn.lpstrFile = file;
    ofn.nMaxFile = MAX_PATH;
    ofn.lpstrFilter = "Halcon 形状模板\0*.shm\0全部文件\0*.*\0";
    ofn.nFilterIndex = 1;
    ofn.lpstrTitle = "加载 Halcon 形状模板";
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;
    if (GetOpenFileNameA(&ofn) == TRUE) {
        return std::string(file);
    }
    return {};
#else
    return {};
#endif
}

std::string SaveHalconShapeModelFile() {
#ifdef _WIN32
    char file[MAX_PATH] = {0};
    OPENFILENAMEA ofn{};
    ofn.lStructSize = sizeof(ofn);
    ofn.lpstrFile = file;
    ofn.nMaxFile = MAX_PATH;
    ofn.lpstrFilter = "Halcon 形状模板\0*.shm\0全部文件\0*.*\0";
    ofn.nFilterIndex = 1;
    ofn.lpstrTitle = "保存 Halcon 形状模板";
    ofn.lpstrDefExt = "shm";
    ofn.Flags = OFN_OVERWRITEPROMPT | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR;
    if (GetSaveFileNameA(&ofn) == TRUE) {
        std::string path(file);
        if (path.find('.') == std::string::npos) path += ".shm";
        return path;
    }
    return {};
#else
    return {};
#endif
}

}  // namespace FileDialog
