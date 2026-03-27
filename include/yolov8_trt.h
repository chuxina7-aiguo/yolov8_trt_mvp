#pragma once

#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime.h>

#include <string>
#include <vector>

#include "preprocess_cuda.h"

struct Keypoint {
    float x = 0.0f;
    float y = 0.0f;
    float conf = 0.0f;
};

struct Detection {
    cv::Rect box;
    int class_id = -1;
    float score = 0.0f;
    std::vector<Keypoint> kpts;
};

class YoloV8TRT {
public:
    YoloV8TRT(const std::string& engine_path, float conf_thres = 0.1f, float nms_thres = 0.45f);
    ~YoloV8TRT();

    bool isReady() const;
    bool infer(const cv::Mat& frame, std::vector<Detection>& detections);
    void drawDetections(cv::Mat& frame, const std::vector<Detection>& detections) const;

private:
    class TrtLogger : public nvinfer1::ILogger {
    public:
        void log(Severity severity, const char* msg) noexcept override;
    };

    bool loadEngine(const std::string& engine_path);
    bool allocateBuffers();
    bool preprocess(const cv::Mat& frame, LetterBoxInfo& info);
    void postprocess(const std::vector<float>& output, const LetterBoxInfo& info, int img_w, int img_h,
                     std::vector<Detection>& detections) const;

    static float iou(const cv::Rect& a, const cv::Rect& b);
    static std::vector<Detection> nms(const std::vector<Detection>& candidates, float iou_thres);
    static const std::vector<std::string>& classNames();

private:
    TrtLogger logger_;

    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;

    std::string input_name_;
    std::string output_name_;
    int input_c_ = 3;
    int input_h_ = 640;
    int input_w_ = 640;

    int output_num_attrs_ = 7;   // 4 coords + 3 classes
    int output_num_boxes_ = 8400;

    std::vector<float> host_output_;

    float* d_input_ = nullptr;
    float* d_output_ = nullptr;
    uint8_t* d_src_bgr_ = nullptr;
    size_t d_src_bgr_bytes_ = 0;

    cudaStream_t stream_ = nullptr;

    float conf_thres_ = 0.1f;
    float nms_thres_ = 0.45f;
};
