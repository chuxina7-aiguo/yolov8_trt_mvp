#include "preprocess_cuda.h"

#include <cmath>

__device__ float bilinear_sample(const uint8_t* src, int src_w, int src_h, float x, float y, int c) {
    if (x < 0.0f || y < 0.0f || x > static_cast<float>(src_w - 1) || y > static_cast<float>(src_h - 1)) {
        return 114.0f;
    }

    const int x0 = static_cast<int>(floorf(x));
    const int y0 = static_cast<int>(floorf(y));
    const int x1 = min(x0 + 1, src_w - 1);
    const int y1 = min(y0 + 1, src_h - 1);

    const float dx = x - static_cast<float>(x0);
    const float dy = y - static_cast<float>(y0);

    const int idx00 = (y0 * src_w + x0) * 3 + c;
    const int idx01 = (y0 * src_w + x1) * 3 + c;
    const int idx10 = (y1 * src_w + x0) * 3 + c;
    const int idx11 = (y1 * src_w + x1) * 3 + c;

    const float v00 = static_cast<float>(src[idx00]);
    const float v01 = static_cast<float>(src[idx01]);
    const float v10 = static_cast<float>(src[idx10]);
    const float v11 = static_cast<float>(src[idx11]);

    const float v0 = v00 + (v01 - v00) * dx;
    const float v1 = v10 + (v11 - v10) * dx;
    return v0 + (v1 - v0) * dy;
}

__global__ void preprocess_kernel(
    const uint8_t* src_bgr,
    int src_w,
    int src_h,
    float* dst_chw,
    int dst_w,
    int dst_h,
    float scale,
    float pad_x,
    float pad_y) {

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dst_w || y >= dst_h) {
        return;
    }

    const int area = dst_w * dst_h;
    const int out_idx = y * dst_w + x;

    // 逆变换到原图坐标，完成 letterbox 的采样映射。
    const float src_x = (static_cast<float>(x) - pad_x + 0.5f) / scale - 0.5f;
    const float src_y = (static_cast<float>(y) - pad_y + 0.5f) / scale - 0.5f;

    float b = 114.0f;
    float g = 114.0f;
    float r = 114.0f;

    if (src_x >= 0.0f && src_x <= static_cast<float>(src_w - 1) && src_y >= 0.0f && src_y <= static_cast<float>(src_h - 1)) {
        b = bilinear_sample(src_bgr, src_w, src_h, src_x, src_y, 0);
        g = bilinear_sample(src_bgr, src_w, src_h, src_x, src_y, 1);
        r = bilinear_sample(src_bgr, src_w, src_h, src_x, src_y, 2);
    }

    // 输出为 RGB 且归一化到 [0,1]，并按 CHW 布局写入。
    dst_chw[0 * area + out_idx] = r / 255.0f;
    dst_chw[1 * area + out_idx] = g / 255.0f;
    dst_chw[2 * area + out_idx] = b / 255.0f;
}

void launch_preprocess_cuda(
    const uint8_t* src_bgr,
    int src_w,
    int src_h,
    float* dst_chw,
    int dst_w,
    int dst_h,
    const LetterBoxInfo& info,
    cudaStream_t stream) {

    const dim3 block(16, 16);
    const dim3 grid((dst_w + block.x - 1) / block.x, (dst_h + block.y - 1) / block.y);

    preprocess_kernel<<<grid, block, 0, stream>>>(
        src_bgr, src_w, src_h, dst_chw, dst_w, dst_h, info.scale, info.pad_x, info.pad_y);
}
