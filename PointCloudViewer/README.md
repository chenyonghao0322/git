# PointCloudViewer 使用说明

个人用 Windows 桌面点云查看 / 测量工具（C++）。

## 功能（当前版本）

- 打开格式：`PLY` / `PCD` / `XYZ` / `OBJ`
- 按 **Z 高度伪彩** 显示
- 旋转 / 平移 / 缩放
- 点大小、透明度调节
- 点选坐标、两点测距
- ROI 框选
- 平面拟合（可基于 ROI 或全部可见点）
- 剖切平面（隐藏一侧点）
- 截面工具：暂未实现（按你的要求后续再加）

## 依赖

已放在 `third_party/`：

- GLFW
- Dear ImGui
- GLAD（使用 GLFW 自带的 header-only glad）

只需本机有：**CMake + Visual Studio 2022（MSVC）** 或带 OpenGL 的 MinGW。

## 编译（推荐 MSVC）

在 `PointCloudViewer` 目录下：

```bat
cmake -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

可执行文件：

```
build\Release\PointCloudViewer.exe
```

## 运行

1. 双击 `PointCloudViewer.exe`
2. 点 **Open File...** 选择点云，或点 **Load Sample** 加载自带样例
3. 左侧面板切换工具模式

### 操作

| 模式 | 操作 |
|------|------|
| Navigate | 左键拖拽旋转；中键或 Alt+左键平移；滚轮缩放；右键旋转 |
| Pick | 左键点选，状态栏显示 XYZ |
| Distance | 连续点两个点，显示距离 |
| Plane Fit | 可先 ROI，再点 **Fit Plane Now** |
| ROI Box | 左键拖矩形框选 |
| Clip Plane | 点一点设置剖切；可用拟合平面法向 |

## 目录结构

```
PointCloudViewer/
  src/
    app/        主窗口与 UI
    core/       点云数据结构
    io/         文件读写
    render/     OpenGL 渲染
    tools/      测量 / 拟合 / ROI / 剖切
  assets/sample/sample.xyz
  third_party/  glfw, imgui
```

## 后续可加

- 截面（剖面曲线）
- 直线 / 圆拟合
- 屏幕上标尺寸线
- 更大点云的八叉树加速拾取
