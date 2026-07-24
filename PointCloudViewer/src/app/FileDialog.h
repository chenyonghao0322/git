#pragma once

#include <string>
#include <vector>

namespace FileDialog {

// Native Windows open-file dialog. Returns empty string if cancelled.
std::string OpenPointCloudFile();
std::string SavePointCloudFile();
std::string OpenImageFile(const char* title = nullptr);
std::vector<std::string> OpenMultipleImageFiles(const char* title = nullptr);
std::string OpenShapeTemplateFile();
std::string SaveShapeTemplateFile();
std::string OpenHalconShapeModelFile();
std::string SaveHalconShapeModelFile();

}  // namespace FileDialog
