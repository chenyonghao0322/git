#pragma once

#include <imgui.h>

// 全局 UI 调色板（供各面板统一引用）
struct UiPalette {
    ImVec4 bgDeep;       // 最深背景（视区、控制台底）
    ImVec4 bgBar;        // 菜单栏 / 工具栏
    ImVec4 panel;        // 侧栏 / 面板
    ImVec4 panelRaised;  // 卡片 / Child 区域
    ImVec4 border;       // 分隔线 / 边框
    ImVec4 borderLight;  // 弱分隔

    ImVec4 accent;       // 主强调色
    ImVec4 accentHover;
    ImVec4 accentActive;

    ImVec4 text;         // 主文字
    ImVec4 textDim;      // 次要文字
    ImVec4 textMuted;    // 禁用 / 标签

    ImVec4 tool3D;       // 3D 工具状态
    ImVec4 tool2D;       // 2D 算子状态
    ImVec4 success;      // PCL / 成功
    ImVec4 warning;      // 警告 / 对比中
    ImVec4 native;       // 自研算法标签

    ImVec4 consoleTime;  // 控制台时间戳
    ImVec4 sectionTitle; // 区域标题
};

const UiPalette& GetUiPalette();
void ApplyAppTheme();

// 左侧色条 + 标题（用于侧栏分区；inlineRow=true 时不追加换行间距，便于 SameLine 放按钮）
void UiSectionHeader(const char* title, const char* subtitle = nullptr,
                     const ImVec4* titleColor = nullptr, bool inlineRow = false);

// 状态胶囊标签（工具栏当前工具 / 算法后端）
void UiStatusBadge(const char* label, const ImVec4& color, float height = 0.f);

// 信息卡片单行：标签 + 数值
void UiInfoRow(const char* label, const char* value, const ImVec4* valueColor = nullptr);
