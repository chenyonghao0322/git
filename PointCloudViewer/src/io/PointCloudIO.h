#pragma once

#include "core/PointCloud.h"

#include <string>

namespace PointCloudIO {

struct DepthMapParams {
    float pixelSizeX = 0.05f;  // mm / pixel
    float pixelSizeY = 0.05f;
    float depthScale = 1.f;    // z = raw * depthScale (mm)
    float zOffset = 0.f;
    float invalidValue = 0.f;  // skip when |raw - invalidValue| <= invalidEps
    float invalidEps = 1e-6f;
    bool skipNonFinite = true;
    bool flipY = false;        // true: image row0 at +Y top
    int step = 1;              // subsample >= 1
};

bool Load(const std::string& path, PointCloud& out, std::string& error);

// Save PLY / XYZ / TXT. Writes world coordinates (display + originOffset).
// visibleOnly: skip mask==0 points when mask is present.
bool Save(const std::string& path, const PointCloud& cloud, std::string& error,
          bool visibleOnly = true);

// depthPath required; brightnessPath optional (empty = height coloring later).
// On success with brightness: out.colors filled with intensity RGB.
bool LoadDepthMaps(const std::string& depthPath, const std::string& brightnessPath,
                   const DepthMapParams& params, PointCloud& out, std::string& error);

}  // namespace PointCloudIO
