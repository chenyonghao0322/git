cmake_minimum_required(VERSION 3.16)
project(PointCloudViewer LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------- GLFW ----------
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/glfw)

# ---------- ImGui ----------
set(IMGUI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/imgui)
add_library(imgui STATIC
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
)
target_include_directories(imgui PUBLIC
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/glfw/deps
)
target_link_libraries(imgui PUBLIC glfw)
# Use ImGui embedded GL loader; app code uses GLFW's glad separately.

# ---------- App ----------
add_executable(PointCloudViewer
    src/main.cpp
    src/core/PointCloud.cpp
    src/io/PointCloudIO.cpp
    src/io/PointCloudGenerator.cpp
    src/io/ImageIO.cpp
    src/render/Camera.cpp
    src/render/Shader.cpp
    src/render/PointCloudRenderer.cpp
    src/render/GlLoader.cpp
    src/tools/MeasureTools.cpp
    src/tools/FilterTools.cpp
    src/tools/UndoHistory.cpp
    src/app/Application.cpp
    src/app/FileDialog.cpp
    src/app/UiTheme.cpp
    src/app/AlgorithmEditor.cpp
)

target_include_directories(PointCloudViewer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/glfw/deps
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/glfw/include
)

target_link_libraries(PointCloudViewer PRIVATE imgui glfw)

if(WIN32)
    target_link_libraries(PointCloudViewer PRIVATE opengl32 comdlg32 windowscodecs ole32)
    target_compile_options(PointCloudViewer PRIVATE /utf-8)
    # GUI app: no console/terminal window
    set_target_properties(PointCloudViewer PROPERTIES WIN32_EXECUTABLE TRUE)
    if(MSVC)
        target_link_options(PointCloudViewer PRIVATE "/ENTRY:mainCRTStartup")
    endif()
endif()

# Copy sample data next to exe
add_custom_command(TARGET PointCloudViewer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:PointCloudViewer>/assets
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/assets
        $<TARGET_FILE_DIR:PointCloudViewer>/assets
)
