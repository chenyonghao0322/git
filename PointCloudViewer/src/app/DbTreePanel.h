#pragma once

// DbTreePanel — 侧栏点云 DB 树（仿 CloudCompare）
//
// 以树形列出当前会话中的点云实体，支持可见性开关与选中后查看属性。

#include "core/PointCloud.h"

#include <functional>

enum class DbEntityId : int {
    None = 0,
    FilledReference = -1,
    FilledResult = -2,
    IcpSource = -3,
    IcpAligned = -4,
};

struct DbSceneLayerRef {
    int id = 0;
    const PointCloud* cloud = nullptr;
    bool* visible = nullptr;
    int gpuCount = 0;
    bool isActive = false;
};

struct DbTreeHost {
    std::vector<DbSceneLayerRef> sceneLayers;

    const PointCloud* filledCloud = nullptr;
    const PointCloud* icpSourceCloud = nullptr;
    const PointCloud* icpAlignedCloud = nullptr;

    bool dualCloudActive = false;
    bool icpSourceLoaded = false;
    bool icpHasAligned = false;

    bool* filledRefVisible = nullptr;
    bool* filledResultVisible = nullptr;
    bool* icpOverlayVisible = nullptr;
    bool* icpShowAligned = nullptr;

    int* selectedLayerId = nullptr;       // >0 场景点云 id
    DbEntityId* selectedSpecial = nullptr;  // 填充/ICP 等特殊节点

    int filledGpuCount = 0;

    std::function<void(int layerId)> onSelectLayer;
    std::function<void(DbEntityId)> onSelectSpecial;
    std::function<void(int layerId)> onDeleteLayer;
    std::function<void()> onVisibilityChanged;
};

namespace DbTreePanel {

void Draw(const DbTreeHost& host);

}  // namespace DbTreePanel
