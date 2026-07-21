#pragma once

#include <string>

namespace FileDialog {

// Native Windows open-file dialog. Returns empty string if cancelled.
std::string OpenPointCloudFile();
std::string SavePointCloudFile();
std::string OpenImageFile(const char* title = nullptr);

}  // namespace FileDialog
