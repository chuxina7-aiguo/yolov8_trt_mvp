#include "yolov8_trt.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <cstdio>

using namespace nvinfer1;

void YoloV8TRT::TrtLogger::log(Severity severity, const char* msg) noexcept {
    if (severity <= Severity::kWARNING) {
        std::cout << "[TensorRT] " << msg << std::endl;
    }
}

YoloV8TRT::YoloV8TRT(const std::string& engine_path, float conf_thres, float nms_thres)
    : conf_thres_(conf_thres), nms_thres_(nms_thres) {

    if (cudaStreamCreate(&stream_) != cudaSuccess) {
        std::cerr << "创建 CUDA stream 失败。" << std::endl;
        return;
    }

    if (!loadEngine(engine_path)) {
        std::cerr << "加载 TensorRT 引擎失败: " << engine_path << std::endl;
        return;
    }

    if (!allocateBuffers()) {
        std::cerr << "分配显存失败。" << std::endl;
        return;
    }
}

YoloV8TRT::~YoloV8TRT() {
    if (context_) delete context_;
    if (engine_) delete engine_;
    if (runtime_) delete runtime_;
    if (d_input_) cudaFree(d_input_);
    if (d_output_) cudaFree(d_output_);
    if (d_src_bgr_) cudaFree(d_src_bgr_);
    if (stream_) cudaStreamDestroy(stream_);
}

bool YoloV8TRT::isReady() const {
    return engine_ != nullptr && context_ != nullptr && d_input_ != nullptr && d_output_ != nullptr;
}

bool YoloV8TRT::loadEngine(const std::string& engine_path) {
    std::ifstream file(engine_path, std::ios::binary);
    if (!file) {
        std::cerr << "无法打开引擎文件: " << engine_path << std::endl;
        return false;
    }

    file.seekg(0, std::ios::end);
    const std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> engine_data(static_cast<size_t>(size));
    if (!file.read(engine_data.data(), size)) {
        std::cerr << "读取引擎文件失败。" << std::endl;
        return false;
    }

    IRuntime* runtime = createInferRuntime(logger_);
    if (!runtime) {
        std::cerr << "创建 TensorRT runtime 失败。" << std::endl;
        return false;
    }

    ICudaEngine* engine = runtime->deserializeCudaEngine(engine_data.data(), engine_data.size());
    if (!engine) {
        std::cerr << "反序列化引擎失败。" << std::endl;
        delete runtime;
        return false;
    }

    IExecutionContext* context = engine->createExecutionContext();
    if (!context) {
        std::cerr << "创建执行上下文失败。" << std::endl;
        delete engine;
        delete runtime;
        return false;
    }

    const int nb_io = engine->getNbIOTensors();
    if (nb_io != 2) {
        std::cerr << "MVP 版本仅支持 1 输入 + 1 输出模型。" << std::endl;
        delete context;
        delete engine;
        delete runtime;
        return false;
    }

    for (int i = 0; i < nb_io; ++i) {
        const char* name = engine->getIOTensorName(i);
        const auto mode = engine->getTensorIOMode(name);
        if (mode == TensorIOMode::kINPUT) {
            input_name_ = name;
        } else if (mode == TensorIOMode::kOUTPUT) {
            output_name_ = name;
        }
    }

    if (input_name_.empty() || output_name_.empty()) {
        std::cerr << "无法识别输入输出 binding。" << std::endl;
        delete context;
        delete engine;
        delete runtime;
        return false;
    }

    Dims in_dims = engine->getTensorShape(input_name_.c_str());
    if (in_dims.nbDims != 4) {
        std::cerr << "输入维度异常，期望 NCHW。" << std::endl;
        delete context;
        delete engine;
        delete runtime;
        return false;
    }

    // 若存在动态维度，默认设置为 1x3x640x640。
    if (in_dims.d[0] == -1 || in_dims.d[2] == -1 || in_dims.d[3] == -1) {
        Dims4 set_dims(1, 3, 640, 640);
        if (!context->setInputShape(input_name_.c_str(), set_dims)) {
            std::cerr << "设置动态输入维度失败。" << std::endl;
            delete context;
            delete engine;
            delete runtime;
            return false;
        }
        in_dims = context->getTensorShape(input_name_.c_str());
    }

    input_c_ = in_dims.d[1];
    input_h_ = in_dims.d[2];
    input_w_ = in_dims.d[3];

    Dims out_dims = context->getTensorShape(output_name_.c_str());
    if (out_dims.nbDims != 3) {
        std::cerr << "输出维度异常，期望 [batch, num_attrs, num_boxes]。" << std::endl;
        delete context;
        delete engine;
        delete runtime;
        return false;
    }

    output_num_attrs_ = out_dims.d[1];
    output_num_boxes_ = out_dims.d[2];

    if (output_num_attrs_ < 5) {
        std::cerr << "警告：输出属性数小于 5，当前值: " << output_num_attrs_ << std::endl;
    }

    runtime_ = runtime;
    engine_ = engine;
    context_ = context;
    return true;
}

bool YoloV8TRT::allocateBuffers() {
    const size_t input_count = static_cast<size_t>(input_c_ * input_h_ * input_w_);
    const size_t output_count = static_cast<size_t>(output_num_attrs_ * output_num_boxes_);

    if (cudaMalloc(&d_input_, input_count * sizeof(float)) != cudaSuccess) {
        return false;
    }
    if (cudaMalloc(&d_output_, output_count * sizeof(float)) != cudaSuccess) {
        return false;
    }

    if (!context_->setTensorAddress(input_name_.c_str(), d_input_)) {
        std::cerr << "绑定输入 Tensor 地址失败。" << std::endl;
        return false;
    }
    if (!context_->setTensorAddress(output_name_.c_str(), d_output_)) {
        std::cerr << "绑定输出 Tensor 地址失败。" << std::endl;
        return false;
    }

    host_output_.resize(output_count);
    return true;
}

bool YoloV8TRT::preprocess(const cv::Mat& frame, LetterBoxInfo& info) {
    const int src_w = frame.cols;
    const int src_h = frame.rows;

    if (src_w <= 0 || src_h <= 0) {
        return false;
    }

    const size_t bytes = static_cast<size_t>(src_w * src_h * 3);
    if (bytes != d_src_bgr_bytes_) {
        if (d_src_bgr_) {
            cudaFree(d_src_bgr_);
            d_src_bgr_ = nullptr;
        }
        if (cudaMalloc(&d_src_bgr_, bytes) != cudaSuccess) {
            std::cerr << "分配源图像显存失败。" << std::endl;
            return false;
        }
        d_src_bgr_bytes_ = bytes;
    }

    if (cudaMemcpyAsync(d_src_bgr_, frame.data, bytes, cudaMemcpyHostToDevice, stream_) != cudaSuccess) {
        std::cerr << "上传图像到显存失败。" << std::endl;
        return false;
    }

    // 计算 letterbox 参数，保证等比例缩放。
    const float scale = std::min(
        static_cast<float>(input_w_) / static_cast<float>(src_w),
        static_cast<float>(input_h_) / static_cast<float>(src_h));

    const float resized_w = static_cast<float>(src_w) * scale;
    const float resized_h = static_cast<float>(src_h) * scale;

    info.scale = scale;
    info.pad_x = (static_cast<float>(input_w_) - resized_w) * 0.5f;
    info.pad_y = (static_cast<float>(input_h_) - resized_h) * 0.5f;

    launch_preprocess_cuda(
        d_src_bgr_,
        src_w,
        src_h,
        d_input_,
        input_w_,
        input_h_,
        info,
        stream_);

    return cudaGetLastError() == cudaSuccess;
}

bool YoloV8TRT::infer(const cv::Mat& frame, std::vector<Detection>& detections) {
    detections.clear();
    if (!isReady() || frame.empty()) {
        return false;
    }

    LetterBoxInfo info;
    if (!preprocess(frame, info)) {
        std::cerr << "前处理失败。" << std::endl;
        return false;
    }

    if (!context_->enqueueV3(stream_)) {
        std::cerr << "TensorRT enqueueV3 失败。" << std::endl;
        return false;
    }

    const size_t out_bytes = host_output_.size() * sizeof(float);
    if (cudaMemcpyAsync(host_output_.data(), d_output_, out_bytes, cudaMemcpyDeviceToHost, stream_) != cudaSuccess) {
        std::cerr << "输出拷贝回主机失败。" << std::endl;
        return false;
    }

    if (cudaStreamSynchronize(stream_) != cudaSuccess) {
        std::cerr << "CUDA stream 同步失败。" << std::endl;
        return false;
    }

    postprocess(host_output_, info, frame.cols, frame.rows, detections);
    return true;
}

void YoloV8TRT::postprocess(
    const std::vector<float>& output,
    const LetterBoxInfo& info,
    int img_w,
    int img_h,
    std::vector<Detection>& detections) const {

    std::vector<Detection> candidates;
    candidates.reserve(static_cast<size_t>(output_num_boxes_));

    // yolov8n-pose 输出定义（已由 Netron 确认）：[1,56,8400]
    // [0:4] -> bbox(cx,cy,w,h), [4] -> objectness score,
    // [5:56] -> 17 * (x,y,conf) 关键点。
    constexpr int kObjScoreIndex = 4;
    constexpr int kKptStartIndex = 5;
    constexpr int kKptDim = 3;
    constexpr int kNumKeypoints = 17;

    if (output_num_attrs_ < kObjScoreIndex + 1) {
        std::cerr << "输出属性数量不足，无法解析 pose 输出。attrs=" << output_num_attrs_ << std::endl;
        detections.clear();
        return;
    }

    const bool is_box_major = false;

    const float* data = output.data();

    auto read_value = [&](int attr, int box_idx) -> float {
        if (is_box_major) {
            // [8400,56]：每个 box 连续存 56 个属性。
            return data[box_idx * output_num_attrs_ + attr];
        }
        // [56,8400]：每个属性连续存 8400 个 box。
        return data[attr * output_num_boxes_ + box_idx];
    };

    for (int i = 0; i < output_num_boxes_; ++i) {
        const float cx_n = read_value(0, i);
        const float cy_n = read_value(1, i);
        const float w_n = read_value(2, i);
        const float h_n = read_value(3, i);
        const float obj_score = read_value(kObjScoreIndex, i);

        if (obj_score < conf_thres_) {
            continue;
        }

        // 安全帽模型直接输出像素坐标 (0~640)，无需缩放。
        const float cx = cx_n;
        const float cy = cy_n;
        const float w = w_n;
        const float h = h_n;

        float raw_x1 = (cx - 0.5f * w - info.pad_x) / info.scale;
        float raw_y1 = (cy - 0.5f * h - info.pad_y) / info.scale;
        float raw_x2 = (cx + 0.5f * w - info.pad_x) / info.scale;
        float raw_y2 = (cy + 0.5f * h - info.pad_y) / info.scale;

        Detection det;
        det.class_id = 0;   // pose 输出无多类别分支，统一单类
        det.score = obj_score;
        det.kpts.reserve(kNumKeypoints);

        // 解析关键点通道（17 * [x,y,conf]，步长为 3），并映射回原图坐标后写入 det.kpts。
        for (int k = 0; k < kNumKeypoints; ++k) {
            const int base = kKptStartIndex + k * kKptDim;
            if (base + 2 >= output_num_attrs_) {
                break;
            }

            const float kx = read_value(base + 0, i);
            const float ky = read_value(base + 1, i);
            const float kc = read_value(base + 2, i);

            float mapped_kx = (kx - info.pad_x) / info.scale;
            float mapped_ky = (ky - info.pad_y) / info.scale;

            mapped_kx = std::max(0.0f, std::min(mapped_kx, static_cast<float>(img_w - 1)));
            mapped_ky = std::max(0.0f, std::min(mapped_ky, static_cast<float>(img_h - 1)));

            det.kpts.push_back({mapped_kx, mapped_ky, kc});
        }

        while (det.kpts.size() < static_cast<size_t>(kNumKeypoints)) {
            det.kpts.push_back({0.0f, 0.0f, 0.0f});
        }

        // 从 letterbox 坐标系映射回原图坐标系。
        float x1 = raw_x1;
        float y1 = raw_y1;
        float x2 = raw_x2;
        float y2 = raw_y2;

        x1 = std::max(0.0f, std::min(x1, static_cast<float>(img_w - 1)));
        y1 = std::max(0.0f, std::min(y1, static_cast<float>(img_h - 1)));
        x2 = std::max(0.0f, std::min(x2, static_cast<float>(img_w - 1)));
        y2 = std::max(0.0f, std::min(y2, static_cast<float>(img_h - 1)));

        const int left = static_cast<int>(std::round(x1));
        const int top = static_cast<int>(std::round(y1));
        const int right = static_cast<int>(std::round(x2));
        const int bottom = static_cast<int>(std::round(y2));

        if (right <= left || bottom <= top) {
            continue;
        }

        det.box = cv::Rect(left, top, right - left, bottom - top);
        candidates.push_back(det);
    }

    detections = nms(candidates, nms_thres_);
}

float YoloV8TRT::iou(const cv::Rect& a, const cv::Rect& b) {
    const int inter_x1 = std::max(a.x, b.x);
    const int inter_y1 = std::max(a.y, b.y);
    const int inter_x2 = std::min(a.x + a.width, b.x + b.width);
    const int inter_y2 = std::min(a.y + a.height, b.y + b.height);

    const int inter_w = std::max(0, inter_x2 - inter_x1);
    const int inter_h = std::max(0, inter_y2 - inter_y1);
    const int inter_area = inter_w * inter_h;

    const int union_area = a.area() + b.area() - inter_area;
    if (union_area <= 0) {
        return 0.0f;
    }
    return static_cast<float>(inter_area) / static_cast<float>(union_area);
}

std::vector<Detection> YoloV8TRT::nms(const std::vector<Detection>& candidates, float iou_thres) {
    if (candidates.empty()) {
        return {};
    }

    std::vector<int> order(candidates.size());
    std::iota(order.begin(), order.end(), 0);

    std::sort(order.begin(), order.end(), [&](int i, int j) {
        return candidates[i].score > candidates[j].score;
    });

    std::vector<Detection> kept;
    std::vector<bool> removed(candidates.size(), false);

    for (size_t m = 0; m < order.size(); ++m) {
        const int i = order[m];
        if (removed[i]) {
            continue;
        }

        kept.push_back(candidates[i]);

        for (size_t n = m + 1; n < order.size(); ++n) {
            const int j = order[n];
            if (removed[j]) {
                continue;
            }

            if (candidates[i].class_id != candidates[j].class_id) {
                continue;
            }

            if (iou(candidates[i].box, candidates[j].box) > iou_thres) {
                removed[j] = true;
            }
        }
    }

    return kept;
}

void YoloV8TRT::drawDetections(cv::Mat& frame, const std::vector<Detection>& detections) const {
    const auto& names = classNames();

    for (const auto& det : detections) {
        // 根据 class_id 选择颜色
        cv::Scalar box_color;
        if (det.class_id == 0) {
            // class_id 0: helmet (戴了安全帽) - 绿色
            box_color = cv::Scalar(0, 255, 0);
        } else if (det.class_id == 1) {
            // class_id 1: head (没戴安全帽) - 红色
            box_color = cv::Scalar(0, 0, 255);
        } else {
            // 其他类别 (person 等) - 蓝色
            box_color = cv::Scalar(255, 0, 0);
        }

        cv::rectangle(frame, det.box, box_color, 3, cv::LINE_8);

        std::string label;
        if (det.class_id >= 0 && det.class_id < static_cast<int>(names.size())) {
            label = names[det.class_id];
        } else {
            label = "cls_" + std::to_string(det.class_id);
        }
        label += " " + cv::format("%.2f", det.score);

        int baseline = 0;
        const cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        int y = std::max(det.box.y - text_size.height - 4, 0);

        cv::rectangle(frame,
                      cv::Point(det.box.x, y),
                      cv::Point(det.box.x + text_size.width + 4, y + text_size.height + baseline + 4),
                      box_color,
                      cv::FILLED);

        cv::putText(frame,
                    label,
                    cv::Point(det.box.x + 2, y + text_size.height + 1),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.5,
                    cv::Scalar(0, 0, 0),
                    1);
    }
}



const std::vector<std::string>& YoloV8TRT::classNames() {
    static const std::vector<std::string> names = {
        "helmet",    // 索引 0: 戴了安全帽的头
        "head",      // 索引 1: 没戴安全帽的头
        "person"     // 索引 2: person 类别
    };
    return names;
}
