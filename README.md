# C++ TensorRT YOLOv8 极速部署模板 🚀

这是一个工业级的 YOLOv8 C++ 部署脚手架，基于 TensorRT 10.0 加速，支持动态解析模型输出、自动适应类别数。

## 🛠️ 环境依赖
* **OS**: Windows 10/11 (Linux 可无缝迁移)
* **CUDA & cuDNN**: CUDA 12.x 
* **TensorRT**: 10.0.0.6
* **OpenCV**: 4.x
* **CMake**: 3.20+

## 🚀 快速启动

### 1. 准备模型与数据
将导出的 `xxx.onnx` 放入 `models/` 目录，将测试视频放入 `data/` 目录。
利用 trtexec 转换为 engine (示例):
`trtexec --onnx=models/best.onnx --saveEngine=models/best.engine --fp16`

### 2. 编译工程
进入项目根目录，打开 PowerShell：
```bash
cmake --build build --config Release