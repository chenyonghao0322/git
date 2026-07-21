#include "app/FileDialog.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <commdlg.h>
#endif

namespace FileDialog {

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
    char file[MAX_PATH] = {0};
    OPENFILENAMEA ofn{};
    ofn.lStructSize = sizeof(ofn);
    ofn.lpstrFile = file;
    ofn.nMaxFile = MAX_PATH;
    ofn.lpstrFilter =
        "图像文件\0*.png;*.bmp;*.jpg;*.jpeg;*.tif;*.tiff;*.gif;*.webp\0"
        "PNG\0*.png\0"
        "BMP\0*.bmp\0"
        "JPEG\0*.jpg;*.jpeg\0"
        "TIFF\0*.tif;*.tiff\0"
        "全部文件\0*.*\0";
    ofn.nFilterIndex = 1;
    ofn.lpstrTitle = title ? title : "打开图像";
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;
    if (GetOpenFileNameA(&ofn) == TRUE) {
        return std::string(file);
    }
    return {};
#else
    (void)title;
    return {};
#endif
}

}  // namespace FileDialog
