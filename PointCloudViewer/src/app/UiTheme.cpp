#include "app/UiTheme.h"

namespace {

UiPalette gPalette;

}  // namespace

const UiPalette& GetUiPalette() { return gPalette; }

void ApplyAppTheme() {
    ImGuiStyle& style = ImGui::GetStyle();

    // 略紧凑、带细边框的工业风布局
    style.WindowRounding = 6.f;
    style.ChildRounding = 5.f;
    style.FrameRounding = 4.f;
    style.PopupRounding = 6.f;
    style.ScrollbarRounding = 6.f;
    style.GrabRounding = 3.f;
    style.TabRounding = 4.f;

    style.WindowPadding = ImVec2(12.f, 10.f);
    style.FramePadding = ImVec2(8.f, 5.f);
    style.ItemSpacing = ImVec2(8.f, 6.f);
    style.ItemInnerSpacing = ImVec2(6.f, 4.f);
    style.IndentSpacing = 14.f;
    style.ScrollbarSize = 10.f;
    style.GrabMinSize = 8.f;

    style.WindowBorderSize = 0.f;
    style.ChildBorderSize = 1.f;
    style.PopupBorderSize = 1.f;
    style.FrameBorderSize = 0.f;
    style.TabBorderSize = 0.f;

    style.WindowTitleAlign = ImVec2(0.5f, 0.5f);
    style.ButtonTextAlign = ImVec2(0.5f, 0.5f);

    // 深蓝灰底 + 青绿强调（预览版）
    gPalette = {};
    gPalette.bgDeep = ImVec4(0.055f, 0.065f, 0.085f, 1.00f);
    gPalette.bgBar = ImVec4(0.075f, 0.090f, 0.115f, 1.00f);
    gPalette.panel = ImVec4(0.095f, 0.110f, 0.140f, 0.98f);
    gPalette.panelRaised = ImVec4(0.115f, 0.135f, 0.165f, 1.00f);
    gPalette.border = ImVec4(0.200f, 0.240f, 0.300f, 0.45f);
    gPalette.borderLight = ImVec4(0.180f, 0.220f, 0.280f, 0.25f);

    gPalette.accent = ImVec4(0.24f, 0.72f, 0.82f, 1.00f);
    gPalette.accentHover = ImVec4(0.32f, 0.82f, 0.90f, 1.00f);
    gPalette.accentActive = ImVec4(0.16f, 0.58f, 0.68f, 1.00f);

    gPalette.text = ImVec4(0.93f, 0.95f, 0.97f, 1.00f);
    gPalette.textDim = ImVec4(0.62f, 0.70f, 0.78f, 1.00f);
    gPalette.textMuted = ImVec4(0.45f, 0.52f, 0.60f, 1.00f);

    gPalette.tool3D = ImVec4(0.45f, 0.82f, 0.92f, 1.00f);
    gPalette.tool2D = ImVec4(0.45f, 0.92f, 0.72f, 1.00f);
    gPalette.success = ImVec4(0.45f, 0.88f, 0.62f, 1.00f);
    gPalette.warning = ImVec4(0.95f, 0.62f, 0.32f, 1.00f);
    gPalette.native = ImVec4(0.72f, 0.78f, 0.88f, 1.00f);

    gPalette.consoleTime = ImVec4(0.50f, 0.68f, 0.78f, 1.00f);
    gPalette.sectionTitle = ImVec4(0.38f, 0.78f, 0.88f, 1.00f);

    const ImVec4 frame = ImVec4(0.14f, 0.17f, 0.22f, 1.00f);
    const ImVec4 frameHov = ImVec4(0.18f, 0.22f, 0.28f, 1.00f);
    const ImVec4 frameAct = ImVec4(0.14f, 0.40f, 0.48f, 1.00f);
    const ImVec4 bgSoft = ImVec4(0.10f, 0.12f, 0.16f, 1.00f);

    ImVec4* c = style.Colors;
    c[ImGuiCol_Text] = gPalette.text;
    c[ImGuiCol_TextDisabled] = gPalette.textMuted;
    c[ImGuiCol_WindowBg] = gPalette.panel;
    c[ImGuiCol_ChildBg] = ImVec4(0, 0, 0, 0);
    c[ImGuiCol_PopupBg] = bgSoft;
    c[ImGuiCol_Border] = gPalette.border;
    c[ImGuiCol_BorderShadow] = ImVec4(0, 0, 0, 0);
    c[ImGuiCol_FrameBg] = frame;
    c[ImGuiCol_FrameBgHovered] = frameHov;
    c[ImGuiCol_FrameBgActive] = frameAct;
    c[ImGuiCol_TitleBg] = gPalette.bgBar;
    c[ImGuiCol_TitleBgActive] = ImVec4(0.10f, 0.20f, 0.26f, 1.00f);
    c[ImGuiCol_TitleBgCollapsed] = gPalette.bgBar;
    c[ImGuiCol_MenuBarBg] = gPalette.bgBar;
    c[ImGuiCol_ScrollbarBg] = ImVec4(0.06f, 0.07f, 0.09f, 0.50f);
    c[ImGuiCol_ScrollbarGrab] = ImVec4(0.26f, 0.32f, 0.40f, 1.00f);
    c[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.34f, 0.42f, 0.50f, 1.00f);
    c[ImGuiCol_ScrollbarGrabActive] = gPalette.accent;
    c[ImGuiCol_CheckMark] = gPalette.accentHover;
    c[ImGuiCol_SliderGrab] = gPalette.accent;
    c[ImGuiCol_SliderGrabActive] = gPalette.accentHover;
    c[ImGuiCol_Button] = ImVec4(0.14f, 0.30f, 0.38f, 1.00f);
    c[ImGuiCol_ButtonHovered] = gPalette.accent;
    c[ImGuiCol_ButtonActive] = gPalette.accentActive;
    c[ImGuiCol_Header] = ImVec4(0.14f, 0.30f, 0.38f, 0.75f);
    c[ImGuiCol_HeaderHovered] = gPalette.accent;
    c[ImGuiCol_HeaderActive] = gPalette.accentActive;
    c[ImGuiCol_Separator] = gPalette.borderLight;
    c[ImGuiCol_SeparatorHovered] = gPalette.accent;
    c[ImGuiCol_SeparatorActive] = gPalette.accentHover;
    c[ImGuiCol_ResizeGrip] = ImVec4(gPalette.accent.x, gPalette.accent.y, gPalette.accent.z, 0.25f);
    c[ImGuiCol_ResizeGripHovered] =
        ImVec4(gPalette.accent.x, gPalette.accent.y, gPalette.accent.z, 0.55f);
    c[ImGuiCol_ResizeGripActive] = gPalette.accent;
    c[ImGuiCol_Tab] = frame;
    c[ImGuiCol_TabHovered] = gPalette.accent;
    c[ImGuiCol_TabActive] = gPalette.accentActive;
    c[ImGuiCol_TabUnfocused] = gPalette.bgBar;
    c[ImGuiCol_TabUnfocusedActive] = frame;
    c[ImGuiCol_PlotLines] = gPalette.accent;
    c[ImGuiCol_PlotLinesHovered] = gPalette.accentHover;
    c[ImGuiCol_PlotHistogram] = gPalette.accent;
    c[ImGuiCol_PlotHistogramHovered] = gPalette.accentHover;
    c[ImGuiCol_TableHeaderBg] = bgSoft;
    c[ImGuiCol_TableBorderStrong] = gPalette.border;
    c[ImGuiCol_TableBorderLight] = gPalette.borderLight;
    c[ImGuiCol_TextSelectedBg] =
        ImVec4(gPalette.accent.x, gPalette.accent.y, gPalette.accent.z, 0.32f);
    c[ImGuiCol_DragDropTarget] = gPalette.accentHover;
    c[ImGuiCol_NavHighlight] = gPalette.accent;
    c[ImGuiCol_NavWindowingHighlight] = ImVec4(1, 1, 1, 0.65f);
    c[ImGuiCol_NavWindowingDimBg] = ImVec4(0.15f, 0.15f, 0.18f, 0.25f);
    c[ImGuiCol_ModalWindowDimBg] = ImVec4(0.04f, 0.05f, 0.07f, 0.60f);
}

void UiSectionHeader(const char* title, const char* subtitle, const ImVec4* titleColor,
                     bool inlineRow) {
    const UiPalette& p = GetUiPalette();
    const ImVec4& col = titleColor ? *titleColor : p.sectionTitle;

    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImVec2 start = ImGui::GetCursorScreenPos();
    const float barH = ImGui::GetTextLineHeight() + (subtitle ? ImGui::GetTextLineHeight() + 4.f : 0.f);
    dl->AddRectFilled(start, ImVec2(start.x + 3.f, start.y + barH),
                      ImGui::ColorConvertFloat4ToU32(p.accent), 2.f);

    ImGui::Indent(10.f);
    ImGui::PushStyleColor(ImGuiCol_Text, col);
    ImGui::TextUnformatted(title);
    ImGui::PopStyleColor();
    if (subtitle && subtitle[0]) {
        ImGui::TextDisabled("%s", subtitle);
    }
    ImGui::Unindent(10.f);
    if (!inlineRow) {
        ImGui::Spacing();
    }
}

void UiStatusBadge(const char* label, const ImVec4& color, float height) {
    const ImVec2 textSz = ImGui::CalcTextSize(label);
    const float padX = 10.f;
    const float padY = 4.f;
    const float h = height > 0.f ? height : (textSz.y + padY * 2.f);
    const ImVec2 btn = ImVec2(textSz.x + padX * 2.f, h);

    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(color.x, color.y, color.z, 0.18f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(color.x, color.y, color.z, 0.28f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(color.x, color.y, color.z, 0.35f));
    ImGui::PushStyleColor(ImGuiCol_Text, color);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, h * 0.5f);
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(padX, padY));
    ImGui::Button(label, btn);
    ImGui::PopStyleVar(2);
    ImGui::PopStyleColor(4);
}

void UiInfoRow(const char* label, const char* value, const ImVec4* valueColor) {
    const UiPalette& p = GetUiPalette();
    ImGui::TextDisabled("%s", label);
    ImGui::SameLine(110.f);
    if (valueColor) {
        ImGui::TextColored(*valueColor, "%s", value);
    } else {
        ImGui::TextColored(p.text, "%s", value);
    }
}
