cmake_minimum_required(VERSION 3.16)
project(PointCloudViewer LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options(/utf-8 /EHsc)
    add_compile_definitions(NOMINMAX _CRT_SECURE_NO_WARNINGS)
endif()

# ---------- PCL ----------
set(PCL_DIR "D:/software/PCL 1.14.0/cmake" CACHE PATH "PCL cmake directory")
find_package(PCL 1.14 REQUIRED COMPONENTS common filters sample_consensus segmentation search kdtree features)

# ---------- OpenCV ----------
set(OpenCV_DIR "D:/software/opencv-4.12.0/opencv/build/x64/vc16/lib" CACHE PATH "OpenCV cmake directory")
find_package(OpenCV 4.12 REQUIRED)

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
    src/tools/PclAdapter.cpp
    src/tools/PclTools.cpp
    src/tools/UndoHistory.cpp
    src/tools/OpenCv2D.cpp
    src/app/Application.cpp
    src/app/FileDialog.cpp
    src/app/UiTheme.cpp
    src/app/AlgorithmEditor.cpp
    src/app/AlgoToolsPanel.cpp
    src/app/PclPanel.cpp
    src/app/PclToolsPanel.cpp
)

target_include_directories(PointCloudViewer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/glfw/deps
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/glfw/include
    ${PCL_INCLUDE_DIRS}
)

target_link_directories(PointCloudViewer PRIVATE ${PCL_LIBRARY_DIRS})
target_link_libraries(PointCloudViewer PRIVATE imgui glfw ${PCL_LIBRARIES} ${OpenCV_LIBS})
target_include_directories(PointCloudViewer PRIVATE ${OpenCV_INCLUDE_DIRS})
target_compile_definitions(PointCloudViewer PRIVATE ${PCL_DEFINITIONS})

if(WIN32)
    target_link_libraries(PointCloudViewer PRIVATE opengl32 comdlg32 windowscodecs ole32)
    set_target_properties(PointCloudViewer PROPERTIES WIN32_EXECUTABLE TRUE)
    if(MSVC)
        target_link_options(PointCloudViewer PRIVATE "/ENTRY:mainCRTStartup")
    endif()

    # 运行时需要 PCL 及第三方 DLL
    set(_pcl_root "D:/software/PCL 1.14.0")
    add_custom_command(TARGET PointCloudViewer POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${_pcl_root}/bin"
            $<TARGET_FILE_DIR:PointCloudViewer>
    )

    set(_opencv_bin "D:/software/opencv-4.12.0/opencv/build/x64/vc16/bin")
    add_custom_command(TARGET PointCloudViewer POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${_opencv_bin}/opencv_world4120.dll"
            $<TARGET_FILE_DIR:PointCloudViewer>
    )
endif()

# Copy sample data next to exe
add_custom_command(TARGET PointCloudViewer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:PointCloudViewer>/assets
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/assets
        $<TARGET_FILE_DIR:PointCloudViewer>/assets
)
