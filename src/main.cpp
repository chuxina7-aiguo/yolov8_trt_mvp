#include "yolov8_trt.h"

#include <iostream>
#include <chrono>

int main(int argc, char** argv) {
    // 默认参数：直接读取 input.mp4，输出 output.mp4，加载 yolov8.engine。
    const std::string input_path = (argc > 1) ? argv[1] : "input.mp4";
    const std::string engine_path = (argc > 2) ? argv[2] : "yolov8.engine";
    const std::string output_path = (argc > 3) ? argv[3] : "output.mp4";

    cv::VideoCapture cap(input_path);
    if (!cap.isOpened()) {
        std::cerr << "无法打开输入视频: " << input_path << std::endl;
        return -1;
    }

    const int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    const int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    const double fps = cap.get(cv::CAP_PROP_FPS) > 1e-3 ? cap.get(cv::CAP_PROP_FPS) : 25.0;

    cv::VideoWriter writer;
    writer.open(output_path,
                cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                fps,
                cv::Size(width, height));

    if (!writer.isOpened()) {
        std::cerr << "无法创建输出视频: " << output_path << std::endl;
        return -2;
    }

    YoloV8TRT detector(engine_path, 0.1f, 0.45f);
    if (!detector.isReady()) {
        std::cerr << "检测器初始化失败，请检查 TensorRT engine 与 CUDA 环境。" << std::endl;
        return -3;
    }

    std::cout << "开始处理视频..." << std::endl;

    cv::Mat frame;
    int frame_id = 0;
    while (cap.read(frame)) {
        if (frame.empty()) {
            continue;
        }

        // 记录帧处理开始时间
        auto start_time = std::chrono::high_resolution_clock::now();

        std::vector<Detection> dets;
        if (!detector.infer(frame, dets)) {
            std::cerr << "第 " << frame_id << " 帧推理失败，已跳过。" << std::endl;
            ++frame_id;
            continue;
        }

        detector.drawDetections(frame, dets);

        // 记录帧处理结束时间，计算耗时（毫秒）
        auto end_time = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        double real_fps = 1000.0 / elapsed_ms;

        // 在画面左上角显示性能数据
        std::string perf_text = cv::format("FPS: %.0f | Time: %.1f ms", real_fps, elapsed_ms);
        cv::putText(frame,
                    perf_text,
                    cv::Point(30, 50),
                    cv::FONT_HERSHEY_SIMPLEX,
                    1.0,
                    cv::Scalar(0, 255, 255),  // 黄色
                    2);

        writer.write(frame);

        if ((frame_id + 1) % 30 == 0) {
            std::cout << "已处理帧数: " << (frame_id + 1) << " | FPS: " << real_fps << std::endl;
        }
        ++frame_id;
    }

    cap.release();
    writer.release();

    std::cout << "处理完成，总帧数: " << frame_id << std::endl;
    std::cout << "输出文件: " << output_path << std::endl;
    return 0;
}
