#pragma once

#include <cstdint>
#include <cuda_runtime.h>

// CUDA 前处理参数：记录 letterbox 的缩放与填充，供后处理还原坐标。
struct LetterBoxInfo {
    float scale = 1.0f;
    float pad_x = 0.0f;
    float pad_y = 0.0f;
};

// 启动 CUDA Kernel：完成 letterbox + BGR2RGB + 归一化 + HWC 转 CHW。
void launch_preprocess_cuda(
    const uint8_t* src_bgr,
    int src_w,
    int src_h,
    float* dst_chw,
    int dst_w,
    int dst_h,
    const LetterBoxInfo& info,
    cudaStream_t stream);
