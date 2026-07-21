#include "app/UiTheme.h"

#include <imgui.h>

void ApplyAppTheme() {
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 8.f;
    style.ChildRounding = 6.f;
    style.FrameRounding = 5.f;
    style.PopupRounding = 6.f;
    style.ScrollbarRounding = 8.f;
    style.GrabRounding = 4.f;
    style.TabRounding = 5.f;

    style.WindowPadding = ImVec2(14.f, 12.f);
    style.FramePadding = ImVec2(10.f, 6.f);
    style.ItemSpacing = ImVec2(10.f, 8.f);
    style.ItemInnerSpacing = ImVec2(8.f, 6.f);
    style.IndentSpacing = 16.f;
    style.ScrollbarSize = 12.f;
    style.GrabMinSize = 10.f;

    style.WindowBorderSize = 0.f;
    style.ChildBorderSize = 0.f;
    style.PopupBorderSize = 1.f;
    style.FrameBorderSize = 0.f;
    style.TabBorderSize = 0.f;

    style.WindowTitleAlign = ImVec2(0.5f, 0.5f);
    style.ButtonTextAlign = ImVec2(0.5f, 0.5f);

    // Cool slate + teal accent (industrial tool look)
    ImVec4* c = style.Colors;
    const ImVec4 bg       = ImVec4(0.10f, 0.12f, 0.14f, 0.96f);
    const ImVec4 bgSoft   = ImVec4(0.13f, 0.15f, 0.18f, 1.00f);
    const ImVec4 panel    = ImVec4(0.14f, 0.16f, 0.19f, 0.98f);
    const ImVec4 frame    = ImVec4(0.18f, 0.21f, 0.25f, 1.00f);
    const ImVec4 frameHov = ImVec4(0.22f, 0.26f, 0.30f, 1.00f);
    const ImVec4 frameAct = ImVec4(0.16f, 0.42f, 0.48f, 1.00f);
    const ImVec4 accent   = ImVec4(0.20f, 0.62f, 0.68f, 1.00f);
    const ImVec4 accentH  = ImVec4(0.28f, 0.72f, 0.78f, 1.00f);
    const ImVec4 accentA  = ImVec4(0.14f, 0.50f, 0.56f, 1.00f);
    const ImVec4 text     = ImVec4(0.92f, 0.94f, 0.95f, 1.00f);
    const ImVec4 textDim  = ImVec4(0.62f, 0.68f, 0.72f, 1.00f);
    const ImVec4 border   = ImVec4(0.22f, 0.26f, 0.30f, 0.60f);

    c[ImGuiCol_Text]                  = text;
    c[ImGuiCol_TextDisabled]          = textDim;
    c[ImGuiCol_WindowBg]              = panel;
    c[ImGuiCol_ChildBg]               = ImVec4(0, 0, 0, 0);
    c[ImGuiCol_PopupBg]               = bgSoft;
    c[ImGuiCol_Border]                = border;
    c[ImGuiCol_BorderShadow]          = ImVec4(0, 0, 0, 0);
    c[ImGuiCol_FrameBg]               = frame;
    c[ImGuiCol_FrameBgHovered]        = frameHov;
    c[ImGuiCol_FrameBgActive]         = frameAct;
    c[ImGuiCol_TitleBg]               = bg;
    c[ImGuiCol_TitleBgActive]         = ImVec4(0.12f, 0.22f, 0.26f, 1.00f);
    c[ImGuiCol_TitleBgCollapsed]      = bg;
    c[ImGuiCol_MenuBarBg]             = bg;
    c[ImGuiCol_ScrollbarBg]           = ImVec4(0.10f, 0.11f, 0.13f, 0.60f);
    c[ImGuiCol_ScrollbarGrab]         = ImVec4(0.28f, 0.32f, 0.36f, 1.00f);
    c[ImGuiCol_ScrollbarGrabHovered]  = ImVec4(0.34f, 0.40f, 0.44f, 1.00f);
    c[ImGuiCol_ScrollbarGrabActive]   = accent;
    c[ImGuiCol_CheckMark]             = accentH;
    c[ImGuiCol_SliderGrab]            = accent;
    c[ImGuiCol_SliderGrabActive]      = accentH;
    c[ImGuiCol_Button]                = ImVec4(0.18f, 0.36f, 0.40f, 1.00f);
    c[ImGuiCol_ButtonHovered]         = accent;
    c[ImGuiCol_ButtonActive]          = accentA;
    c[ImGuiCol_Header]                = ImVec4(0.16f, 0.32f, 0.36f, 0.80f);
    c[ImGuiCol_HeaderHovered]         = accent;
    c[ImGuiCol_HeaderActive]          = accentA;
    c[ImGuiCol_Separator]             = border;
    c[ImGuiCol_SeparatorHovered]      = accent;
    c[ImGuiCol_SeparatorActive]       = accentH;
    c[ImGuiCol_ResizeGrip]            = ImVec4(accent.x, accent.y, accent.z, 0.30f);
    c[ImGuiCol_ResizeGripHovered]     = ImVec4(accent.x, accent.y, accent.z, 0.60f);
    c[ImGuiCol_ResizeGripActive]      = accent;
    c[ImGuiCol_Tab]                   = frame;
    c[ImGuiCol_TabHovered]            = accent;
    c[ImGuiCol_TabActive]             = accentA;
    c[ImGuiCol_TabUnfocused]          = bg;
    c[ImGuiCol_TabUnfocusedActive]    = frame;
    c[ImGuiCol_PlotLines]             = accent;
    c[ImGuiCol_PlotLinesHovered]      = accentH;
    c[ImGuiCol_PlotHistogram]         = accent;
    c[ImGuiCol_PlotHistogramHovered]  = accentH;
    c[ImGuiCol_TableHeaderBg]         = bgSoft;
    c[ImGuiCol_TableBorderStrong]     = border;
    c[ImGuiCol_TableBorderLight]      = ImVec4(0.20f, 0.24f, 0.28f, 0.40f);
    c[ImGuiCol_TextSelectedBg]        = ImVec4(accent.x, accent.y, accent.z, 0.35f);
    c[ImGuiCol_DragDropTarget]        = accentH;
    c[ImGuiCol_NavHighlight]          = accent;
    c[ImGuiCol_NavWindowingHighlight] = ImVec4(1, 1, 1, 0.70f);
    c[ImGuiCol_NavWindowingDimBg]     = ImVec4(0.2f, 0.2f, 0.2f, 0.20f);
    c[ImGuiCol_ModalWindowDimBg]      = ImVec4(0.05f, 0.06f, 0.07f, 0.55f);
}
