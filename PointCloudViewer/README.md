# PointCloudViewer 使用说明

个人用 Windows 桌面点云查看 / 测量工具（C++）。**当前版本：v0.5**

> 详细更新记录见 [CHANGELOG.md](CHANGELOG.md)

## 功能（v0.5）

### 3D 点云

- 打开格式：`PLY` / `PCD` / `XYZ` / `OBJ`
- 按 **Z 高度伪彩** 显示；支持亮度着色
- 旋转 / 平移 / 缩放
- 点选坐标、两点测距、ROI 框选
- 平面 / 球 / 圆 / 圆柱拟合，平面度、段差等测量
- 剖切平面、截面 2D 轮廓
- 深度图 / 亮度图生成点云

### 2D 图像算子

- 单独打开深度图 / 亮度图（不转点云）
- 线卡尺、弧卡尺、圆拟合等 **23 个 2D 测量算子**
- 2D 模式、深度/亮度联动、测量叠加显示

### 模板匹配

- **2D 模板匹配**：自研 OpenCV 形状模板（菜单「2D算子 → 2D模板匹配」）
- **Halcon 匹配**：Halcon 缩放形状模板（菜单「2D算子 → halcon匹配」）
  - 支持整图 / 旋转 ROI 框选创建模板
  - 参数对齐 Halcon 标准助手（对比度、金字塔、缩放、贪婪度等）
  - 需本机安装 Halcon 20.11（CMake 可选启用）

### 其他

- 算法编辑器（节点式流程）
- 点云数据库树面板（DbTree）
- PCL / 自研算法双后端
- 体素 / 半径 / 统计滤波预览对比

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

- 屏幕上标尺寸线
- 更大点云的八叉树加速拾取
- UI 主题切换（亮/暗）
