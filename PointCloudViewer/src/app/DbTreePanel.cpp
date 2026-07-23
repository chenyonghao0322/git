#include "app/DbTreePanel.h"

#include "app/UiTheme.h"

#include <imgui.h>

#include <algorithm>
#include <cstdio>
#include <string>

namespace DbTreePanel {
namespace {

std::string BaseName(const std::string& path) {
    const std::size_t p = path.find_last_of("/\\");
    return (p == std::string::npos) ? path : path.substr(p + 1);
}

std::string StemName(const std::string& path) {
    std::string base = BaseName(path);
    const std::size_t dot = base.find_last_of('.');
    if (dot != std::string::npos) base = base.substr(0, dot);
    return base.empty() ? std::string(u8"点云") : base;
}

std::string ShortPath(const std::string& path, std::size_t maxLen = 34) {
    if (path.size() <= maxLen) return path;
    const std::size_t head = maxLen / 2 - 1;
    const std::size_t tail = maxLen - head - 2;
    return path.substr(0, head) + ".." + path.substr(path.size() - tail);
}

std::string FileGroupLabel(const PointCloud& cloud) {
    if (cloud.sourcePath.empty()) return u8"点云";
    if (cloud.sourcePath[0] == '<') return cloud.sourcePath;
    const std::string base = BaseName(cloud.sourcePath);
    return base + " (" + ShortPath(cloud.sourcePath) + ")";
}

const PointCloud* CloudForSelection(const DbTreeHost& host) {
    if (host.selectedSpecial && *host.selectedSpecial != DbEntityId::None) {
        switch (*host.selectedSpecial) {
            case DbEntityId::FilledReference:
            case DbEntityId::FilledResult:
                return host.filledCloud;
            case DbEntityId::IcpSource:
                return host.icpSourceCloud;
            case DbEntityId::IcpAligned:
                return host.icpAlignedCloud;
            default:
                break;
        }
        return nullptr;
    }
    if (!host.selectedLayerId || *host.selectedLayerId <= 0) return nullptr;
    for (const DbSceneLayerRef& layer : host.sceneLayers) {
        if (layer.id == *host.selectedLayerId) return layer.cloud;
    }
    return nullptr;
}

const char* SpecialTypeLabel(DbEntityId id) {
    switch (id) {
        case DbEntityId::FilledReference:
            return u8"原始参考";
        case DbEntityId::FilledResult:
            return u8"填充结果";
        case DbEntityId::IcpSource:
            return u8"ICP 源点云";
        case DbEntityId::IcpAligned:
            return u8"ICP 配准后";
        default:
            return u8"点云";
    }
}

bool DrawVisibilityToggle(bool* visible, const char* id) {
    if (!visible) return false;
    ImGui::PushID(id);
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(2.f, 2.f));
    const bool changed = ImGui::Checkbox("##vis", visible);
    ImGui::PopStyleVar();
    ImGui::PopID();
    return changed;
}

bool DrawLayerLeaf(const DbTreeHost& host, int layerId, const char* label, bool* visible,
                   bool isActive) {
    ImGui::PushID(layerId);
    ImGui::AlignTextToFramePadding();

    bool changed = DrawVisibilityToggle(visible, "vis");
    ImGui::SameLine();

    const bool selected =
        host.selectedLayerId && *host.selectedLayerId == layerId &&
        (!host.selectedSpecial || *host.selectedSpecial == DbEntityId::None);
    ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen |
                               ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_Bullet;
    if (selected) flags |= ImGuiTreeNodeFlags_Selected;

    std::string text = label;
    if (isActive) text += u8"  [当前]";

    ImGui::TreeNodeEx("##leaf", flags, "%s", text.c_str());
    if (ImGui::IsItemClicked(ImGuiMouseButton_Left) && host.onSelectLayer) {
        host.onSelectLayer(layerId);
    }
    if (ImGui::BeginPopupContextItem()) {
        if (ImGui::MenuItem(u8"删除点云") && host.onDeleteLayer) {
            host.onDeleteLayer(layerId);
        }
        ImGui::EndPopup();
    }
    ImGui::PopID();
    return changed;
}

bool DrawSpecialLeaf(const DbTreeHost& host, DbEntityId id, const char* label, bool* visible) {
    ImGui::PushID(static_cast<int>(id));
    ImGui::AlignTextToFramePadding();

    bool changed = DrawVisibilityToggle(visible, "vis");
    ImGui::SameLine();

    const bool selected = host.selectedSpecial && *host.selectedSpecial == id;
    ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen |
                               ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_Bullet;
    if (selected) flags |= ImGuiTreeNodeFlags_Selected;

    ImGui::TreeNodeEx("##leaf", flags, "%s", label);
    if (ImGui::IsItemClicked(ImGuiMouseButton_Left) && host.onSelectSpecial) {
        host.onSelectSpecial(id);
    }
    ImGui::PopID();
    return changed;
}

bool DrawGroupHeader(const DbTreeHost& host, int layerId, const char* id, const char* label,
                     bool* groupVisible, bool defaultOpen) {
    ImGui::PushID(id);
    ImGui::AlignTextToFramePadding();

    bool changed = false;
    if (groupVisible) {
        changed = DrawVisibilityToggle(groupVisible, "grp_vis");
        ImGui::SameLine();
    }

    if (defaultOpen) ImGui::SetNextItemOpen(true, ImGuiCond_Once);
    ImGuiTreeNodeFlags flags =
        ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_DefaultOpen;
    const bool open = ImGui::TreeNodeEx("##grp", flags, "%s", label);
    if (layerId > 0 && ImGui::BeginPopupContextItem()) {
        if (ImGui::MenuItem(u8"删除点云") && host.onDeleteLayer) {
            host.onDeleteLayer(layerId);
        }
        ImGui::EndPopup();
    }
    ImGui::PopID();
    return open || changed;
}

void DrawProperties(const DbTreeHost& host) {
    const UiPalette& pal = GetUiPalette();
    const PointCloud* cloud = CloudForSelection(host);

    ImGui::TextDisabled(u8"属性");
    ImGui::Spacing();

    if (!cloud || cloud->points.empty()) {
        UiInfoRow(u8"状态", u8"未选中或无数据", &pal.textMuted);
        return;
    }

    char buf[96];
    if (host.selectedSpecial && *host.selectedSpecial != DbEntityId::None) {
        std::snprintf(buf, sizeof(buf), "%s", SpecialTypeLabel(*host.selectedSpecial));
    } else {
        std::snprintf(buf, sizeof(buf), "%s", u8"点云");
    }
    UiInfoRow(u8"类型", buf);

    std::snprintf(buf, sizeof(buf), "%d", static_cast<int>(cloud->points.size()));
    UiInfoRow(u8"总点数", buf);

    std::snprintf(buf, sizeof(buf), "%d", static_cast<int>(cloud->VisibleCount()));
    UiInfoRow(u8"可见点", buf, &pal.success);

    if (host.selectedLayerId && *host.selectedLayerId > 0) {
        for (const DbSceneLayerRef& layer : host.sceneLayers) {
            if (layer.id == *host.selectedLayerId) {
                std::snprintf(buf, sizeof(buf), "%d", layer.gpuCount);
                UiInfoRow(u8"GPU 显示", buf, &pal.accent);
                break;
            }
        }
    } else if (host.selectedSpecial && (*host.selectedSpecial == DbEntityId::FilledReference ||
                                         *host.selectedSpecial == DbEntityId::FilledResult)) {
        std::snprintf(buf, sizeof(buf), "%d", host.filledGpuCount);
        UiInfoRow(u8"GPU 显示", buf, &pal.accent);
    }

    if (cloud->bounds.Valid()) {
        const Vec3 e = cloud->bounds.Extent();
        std::snprintf(buf, sizeof(buf), u8"%.2f × %.2f × %.2f", e.x, e.y, e.z);
        UiInfoRow(u8"尺寸 mm", buf);
        const Vec3 off = cloud->originOffset;
        std::snprintf(buf, sizeof(buf), u8"%.3f, %.3f, %.3f", off.x, off.y, off.z);
        UiInfoRow(u8"原点偏移", buf);
    }

    if (!cloud->sourcePath.empty()) {
        UiInfoRow(u8"来源", ShortPath(cloud->sourcePath, 28).c_str(), &pal.textMuted);
    }
}

}  // namespace

void Draw(const DbTreeHost& host) {
    const float totalH = ImGui::GetContentRegionAvail().y;
    const float treeH = std::max(72.f, totalH * 0.58f);
    ImGui::BeginChild(u8"##db_tree_scroll", ImVec2(0, treeH), false);
    {
        bool visChanged = false;

        if (host.sceneLayers.empty()) {
            ImGui::TextDisabled(u8"（尚未加载点云）");
        } else {
            for (const DbSceneLayerRef& layer : host.sceneLayers) {
                if (!layer.cloud || layer.cloud->points.empty()) continue;

                bool groupVis = layer.visible ? *layer.visible : true;
                const std::string grpLabel = FileGroupLabel(*layer.cloud);
                const bool open = DrawGroupHeader(host, layer.id,
                                                  ("grp_" + std::to_string(layer.id)).c_str(),
                                                  grpLabel.c_str(),
                                                  layer.visible ? &groupVis : nullptr, true);
                if (layer.visible && groupVis != *layer.visible) {
                    *layer.visible = groupVis;
                    visChanged = true;
                }
                if (open) {
                    const std::string leafName = StemName(layer.cloud->sourcePath);
                    if (DrawLayerLeaf(host, layer.id, leafName.c_str(), layer.visible,
                                      layer.isActive)) {
                        visChanged = true;
                    }
                    ImGui::TreePop();
                }
            }
        }

        if (host.dualCloudActive && host.filledCloud && !host.filledCloud->points.empty()) {
            bool groupVis = true;
            if (host.filledRefVisible && host.filledResultVisible) {
                groupVis = *host.filledRefVisible || *host.filledResultVisible;
            }

            const char* fillLabel = host.filledCloud->sourcePath.empty()
                                        ? u8"<填充视图>"
                                        : host.filledCloud->sourcePath.c_str();
            bool grpToggle = groupVis;
            const bool open = DrawGroupHeader(host, 0, "fill_grp", fillLabel, &grpToggle, true);
            if (host.filledRefVisible && host.filledResultVisible && grpToggle != groupVis) {
                *host.filledRefVisible = grpToggle;
                *host.filledResultVisible = grpToggle;
                visChanged = true;
            }
            if (open) {
                if (DrawSpecialLeaf(host, DbEntityId::FilledReference, u8"原始参考",
                                    host.filledRefVisible)) {
                    visChanged = true;
                }
                if (DrawSpecialLeaf(host, DbEntityId::FilledResult, u8"填充结果",
                                    host.filledResultVisible)) {
                    visChanged = true;
                }
                ImGui::TreePop();
            }
        }

        if (host.icpSourceLoaded && host.icpSourceCloud && !host.icpSourceCloud->points.empty()) {
            bool icpVis = host.icpOverlayVisible ? *host.icpOverlayVisible : true;
            const std::string icpLabel = u8"ICP — " + FileGroupLabel(*host.icpSourceCloud);
            const bool open = DrawGroupHeader(host, 0, "icp_grp", icpLabel.c_str(),
                                              host.icpOverlayVisible ? &icpVis : nullptr, true);
            if (host.icpOverlayVisible && icpVis != *host.icpOverlayVisible) {
                *host.icpOverlayVisible = icpVis;
                visChanged = true;
            }
            if (open) {
                if (DrawSpecialLeaf(host, DbEntityId::IcpSource, u8"源点云",
                                    host.icpOverlayVisible)) {
                    visChanged = true;
                }
                if (host.icpHasAligned && host.icpAlignedCloud &&
                    !host.icpAlignedCloud->points.empty()) {
                    if (DrawSpecialLeaf(host, DbEntityId::IcpAligned, u8"配准后",
                                        host.icpOverlayVisible)) {
                        visChanged = true;
                    }
                }
                ImGui::TreePop();
            }
        }

        if (visChanged && host.onVisibilityChanged) host.onVisibilityChanged();
    }
    ImGui::EndChild();

    ImGui::Separator();
    DrawProperties(host);
}

}  // namespace DbTreePanel
