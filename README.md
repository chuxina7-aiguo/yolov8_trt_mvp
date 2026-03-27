```bash

```

# 🚀 YOLOv8 TensorRT C++ 极速部署引擎 (工业级安全帽检测)



# **入门版**

[![C++](https://img.shields.io/badge/C++-17-blue.svg)](https://isocpp.org/)
[![TensorRT](https://img.shields.io/badge/TensorRT-10.0-green.svg)](https://developer.nvidia.com/tensorrt)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-red.svg)](https://opencv.org/)

这是一个专为**“零基础小白”**和**“初级 AI 工程师”**准备的工业级 C++ 部署脚手架。
它能将 Python 中极其臃肿的 YOLOv8 模型，转化为只能在显卡上极速狂飙的纯二进制引擎，**推理速度可达 90+ FPS！**

当前内置逻辑：**智慧工地安全帽检测 (支持实时红绿框预警)**。
- 🟢 戴安全帽 (Helmet) -> 绿色安全框
- 🔴 没戴安全帽 (Head) -> 红色预警框

---

## 🌟 核心特性
- **双模输入**：支持读取本地 `.mp4` 视频文件，也支持直接接入电脑/手机实时摄像头。
- **极致压榨**：摒弃 Python 环境，基于 C++ 和 TensorRT 10.0 直接调用显卡底层算力。
- **动态自适应**：智能读取模型的类别数，彻底杜绝内存越界。
- **性能雷达**：画面左上角实时显示毫秒级端到端耗时与 FPS。

---

## 🛠️ 第一步：准备“做饭的厨具” (环境依赖)

在运行本项目前，请确保你的 Windows 电脑上安装了以下工具：
1. **显卡与驱动**：你需要一张 NVIDIA 独立显卡（如 RTX 3060/4060），并安装了相应的驱动。
2. **Visual Studio 2022**：安装时请勾选“使用 C++ 的桌面开发”。
3. **CUDA 12.x & cuDNN**：英伟达官方的底层并行计算平台。
4. **TensorRT 10.x**：英伟达官方的模型加速神器（[下载并解压](https://developer.nvidia.com/tensorrt)）。
5. **OpenCV 4.x**：用于读取视频和画框（[下载 Windows 版并解压](https://opencv.org/releases/)）。
6. **CMake**：用于把 C++ 代码变成 `.exe` 程序的“包工头”。

> **💡 极其重要的一步**：
> 请将 OpenCV 的 `build/x64/vc16/bin` 目录，以及 TensorRT 的 `lib` 目录，**添加到 Windows 系统的“环境变量 -> Path”中**，否则程序运行时会报错“找不到 .dll 文件”！

---

## 📦 第二步：准备“食材” (准备模型与数据)

1. **克隆代码**：将本仓库下载到本地。

2. **准备视频**：在项目根目录新建 `data` 文件夹，放入一个测试视频（命名为 `input.mp4`）。

3. **炼制引擎**：
   - 将你训练好的 YOLOv8 模型导出为 `.onnx` 格式。
   
   - 打开终端，使用 TensorRT 自带的工具将其炼制为专属引擎（请将下方路径替换为你电脑里的实际路径）：
   
   - trtexec --onnx=你的模型.onnx --saveEngine=best.engine --fp16xxxxxxxxxx cmake --build build --config Releasebash
   
     
   
     
   
     

# **专业版**

## 🧠 核心架构与数据流转全景 (Under the Hood)

本项目不仅是一个能跑的黑盒，更是一个极其规范的 C++ 边缘部署教科书。整个系统分为四个核心车间（Pipeline），以下是数据从获取到渲染的完整生命周期：

### 1. 数据采集与预处理车间 (Pre-processing)
* **输入获取**：使用 OpenCV 的 `cv::VideoCapture` 捕获视频源，拿到的是最原始的 `cv::Mat` 矩阵（通常为 1920x1080，**BGR** 色彩排布，**HWC** 内存交织格式）。
* **形态转换 (Letterbox)**：AI 模型是个“强迫症”，它只吃 `640x640` 的正方形。我们不能直接强行拉伸（会导致安全帽变形），而是采用等比例缩放，并在边缘补上灰边/黑边（Padding）。
* **底层转换**：
  - 色彩重排：将 OpenCV 的 BGR 转换为模型需要的 RGB。
  - 内存展平：将像素交织的 HWC（高度x宽度x通道）排布，转换为 GPU 极其喜欢的连续平铺结构 CHW（通道x高度x宽度）。
  - 归一化：将 `0~255` 的像素值除以 `255.0`，映射到 `0.0~1.0` 之间。
* **显存搬运**：最后通过 `cudaMemcpy`，将整理好的极其纯净的浮点数组一把推入显卡 VRAM。

### 2. 黑盒计算引擎 (Inference)
* **核心库**：基于 NVIDIA TensorRT 的 C++ API (`nvinfer1::IExecutionContext`)。
* **执行**：调用 `enqueueV3` 方法。此时 CPU 释放控制权，GPU 内部的数千个 CUDA 核心瞬间开火，通过 YOLOv8m (Medium) 网络的数十层卷积，耗时仅需几毫秒。

### 3. 张量解析与后处理局 (Post-processing)
这是整个工程中最硬核的部分。模型吐出的不是现成的框，而是一坨高维张量（Tensor），维度为 `[1, 6, 8400]`。
* **8400**：代表 AI 撒在画面上的 8400 个候选锚框。
* **6**：代表每个框有 6 个属性，即 `(cx, cy, w, h, 置信度_没戴头盔, 置信度_戴了头盔)`。
* **数据提纯 (Thresholding)**：C++ 代码遍历这 8400 个框，剔除置信度低于阈值（如 `0.25`）的垃圾数据。
* **坐标反推 (Reverse Mapping)**：将模型输出的框坐标，减去前处理加上的 Padding，再除以缩放比例，精准映射回原始的 1920x1080 屏幕像素坐标。
* **非极大值抑制 (NMS)**：调用 OpenCV 的 `cv::dnn::NMSBoxes`，将重叠在同一个安全帽上的多个备选框，通过 IoU（交并比）计算，只保留最精准的那一个。

### 4. 视觉渲染与交付 (Rendering)
* 最后，基于提纯后的坐标集合，使用 `cv::rectangle` 绘制红绿边框。
* 调用 `std::chrono` 统计端到端耗时，换算为真实的 FPS，并通过 `cv::putText` 渲染至视频流的左上角，最终通过 `cv::imshow` 交付给用户。



# 🚑 避坑指南 (Q&A)

- **Q: 接入 RTSP 手机摄像头时，报错 `401 Unauthorized` 或者卡死超时怎么办？**
  - A**: 
    1. **401 报错**说明 App 有密码保护。请在网址中加入账号密码，格式为：`rtsp://账号:密码@IP:端口/live`。
    2. **超时卡死**通常是因为 UDP 协议被局域网拦截。请在手机 App 设置中关闭“强制 TCP”，允许 UDP 传输；或者在 PowerShell 中运行 `$env:OPENCV_FFMPEG_CAPTURE_OPTIONS="rtsp_transport;tcp"` 强行改用 TCP 通道。





# 🚀 **v2.0 架构升级**

彻底移除本地 GUI，采用 C++ 多线程 + httplib 构建后端，通过 SSE 协议向浏览器零延迟推送 Base64 视频流与 JSON 检测数据！









# 🚀 YOLOv8-Pose TensorRT SaaS 级实时监控平台 (V1.0)

本项目是一个基于 C++ TensorRT 高性能后端与 Web 前端交互的工业级 AI 视频分析原型系统。它不仅能实现极速的姿态识别，更赋予了 AI “理解空间”与“理解动作”的能力。

---

## 🌟 核心特性

- **⚡ 极致性能**：基于 TensorRT 8.x 硬件加速，在 RTX 4060 上实现多线程解耦的生产者-消费者模型，推理侧几乎零延迟。
- **📐 动态 ROI 电子围栏**：支持在 Web 界面使用鼠标实时绘制任意形状的多边形防区（射线法判定）。
- **🆘 行为分析 (SOS)**：基于 17 个骨架关键点的几何拓扑关系，实时检测“举手求救”等异常行为。
- **📸 异步抓拍系统**：发生违规或求救时，系统自动执行“视频流 + 算法层”图层合成，异步截取包含现场证据的高清 JPG 图片并持久化至本地磁盘。
- **🌐 全栈架构**：
  - **后端**：C++ 17, OpenCV, TensorRT, cpp-httplib (SSE 流式推送)。
  - **前端**：Canvas API, JavaScript 异步驱动, 响应式安防看板布局。

---

## 🏗️ 系统架构图



- **采集层 (CameraThread)**：负责 OpenCV 视频流解复用。
- **推理层 (InferenceThread)**：TensorRT 执行 Pose 估计，并进行业务逻辑判定。
- **分发层 (WebServer)**：通过 SSE (Server-Sent Events) 向前端毫秒级推送图像与 JSON 数据。
- **展示层 (Frontend)**：Canvas 实时渲染骨架、报警框，管理证据墙。

---

## 🛠️ 快速开始

### 1. 环境依赖
- Windows 10/11, Visual Studio 2022
- NVIDIA GPU (RTX 系列推荐) & CUDA 11.x/12.x
- TensorRT 8.x
- OpenCV 4.x

### 2. 编译
```powershell
mkdir build
cd build
cmake ..
cmake --build . --config Release
```



# 使用手机 IP 摄像头或 RTSP 流
.\Release\yolov8_trt_mvp.exe "rtsp://admin:密码@IP:8554/live" models/yolov8n-pose.engine