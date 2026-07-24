PaddleOCR 接入说明（PointCloudViewer）

一、Python 路径
  默认查找：D:\software\Python3.13.14\python.exe
  若安装在其他位置，CMake 配置时设置 PYTHON_ROOT 后重新编译。

二、安装依赖（在命令行执行一次）
  D:\software\Python3.13.14\python.exe -m pip install paddlepaddle paddleocr

  若识别报 oneDNN / ConvertPirAttribute 相关错误，可降级：
  D:\software\Python3.13.14\python.exe -m pip install paddlepaddle==3.2.2
  （脚本已默认 enable_mkldnn=False，一般无需降级）

三、手动测试
  D:\software\Python3.13.14\python.exe assets\ocr\paddle_ocr.py 测试图.png out.json ch 1
  type out.json

四、软件使用
  菜单 → OCR 识别 → 顶部切换「PaddleOCR」→ 框选 ROI → 识别
  首次运行会自动下载 OCR 模型，需等待片刻。

五、PATH 说明
  不强制把 Python 加入系统 PATH，软件会直接使用上述安装路径。
