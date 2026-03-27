#include "yolov8_trt.h"

#include <iostream>
#include <chrono>
#include <string>
#include <cctype>

// 辅助函数：判断字符串是否全是数字
bool isNumericString(const std::string& str) {
    if (str.empty()) return false;
    return std::all_of(str.begin(), str.end(), [](unsigned char c) { return std::isdigit(c); });
}

int main(int argc, char** argv) {
    // 定义输入源和引擎路径
    std::string input_source = (argc > 1) ? argv[1] : "0";  // 默认摄像头 0
    const std::string engine_path = (argc > 2) ? argv[2] : "yolov8.engine";

    cv::VideoCapture cap;
    bool is_camera = isNumericString(input_source);

    // 智能打开输入源
    if (is_camera) {
        // 数字输入：打开摄像头
        int camera_index = std::stoi(input_source);
        cap.open(camera_index);
        if (!cap.isOpened()) {
            std::cerr << "无法打开摄像头设备: " << camera_index << std::endl;
            return -1;
        }

        // 设置摄像头的分辨率和帧率（可选，某些设备可能不支持）
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        cap.set(cv::CAP_PROP_FPS, 30);

        std::cout << "✓ 已打开摄像头设备: " << camera_index << std::endl;
    } else {
        // 字符串输入：打开视频文件
        cap.open(input_source);
        if (!cap.isOpened()) {
            std::cerr << "无法打开视频文件: " << input_source << std::endl;
            return -1;
        }

        double fps = cap.get(cv::CAP_PROP_FPS);
        int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
        std::cout << "✓ 已打开视频文件: " << input_source << std::endl;
        std::cout << "  总帧数: " << total_frames << " | 帧率: " << fps << " FPS" << std::endl;
    }

    YoloV8TRT detector(engine_path, 0.25f, 0.45f);
    if (!detector.isReady()) {
        std::cerr << "检测器初始化失败，请检查 TensorRT engine 与 CUDA 环境。" << std::endl;
        return -2;
    }

    std::cout << "✓ 检测器初始化成功，引擎: " << engine_path << std::endl;
    std::cout << "开始处理" << (is_camera ? "摄像头" : "视频文件") << "..." << std::endl;
    if (is_camera) {
        std::cout << "按 ESC 或 'q' 键退出。" << std::endl;
    } else {
        std::cout << "视频播放完成或按 ESC/'q' 键退出。" << std::endl;
    }

    cv::Mat frame;
    int frame_id = 0;
    double avg_fps = 0.0;
    int fps_count = 0;

    while (cap.read(frame)) {
        if (frame.empty()) {
            // 对于视频文件，读不到帧说明播放完毕
            if (!is_camera) {
                std::cout << "\n视频播放完成！" << std::endl;
            }
            break;
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

        // 累计 FPS 用于计算平均值
        avg_fps += real_fps;
        fps_count++;

        // 在画面左上角显示性能数据
        std::string perf_text = cv::format("FPS: %.0f | Time: %.1f ms", real_fps, elapsed_ms);
        cv::putText(frame,
                    perf_text,
                    cv::Point(30, 50),
                    cv::FONT_HERSHEY_SIMPLEX,
                    1.0,
                    cv::Scalar(0, 255, 255),  // 黄色
                    2);

        // 显示源类型信息
        std::string source_info = is_camera ? "Camera" : "Video";
        cv::putText(frame,
                    source_info,
                    cv::Point(30, 90),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.7,
                    cv::Scalar(0, 255, 255),
                    1);

        // 实时显示画面
        cv::imshow("Live Detection", frame);

        // 检测按键：ESC (27) 或 'q' 退出
        int key = cv::waitKey(1) & 0xFF;
        if (key == 27 || key == 'q') {
            std::cout << "\n用户按下退出键，停止检测。" << std::endl;
            break;
        }

        if ((frame_id + 1) % 30 == 0) {
            std::cout << "已处理帧数: " << (frame_id + 1) << " | FPS: " << real_fps << std::endl;
        }
        ++frame_id;
    }

    cap.release();
    cv::destroyAllWindows();

    // 计算并输出平均 FPS
    if (fps_count > 0) {
        double mean_fps = avg_fps / fps_count;
        std::cout << "\n══════════════════════════════════" << std::endl;
        std::cout << "处理完成！" << std::endl;
        std::cout << "总帧数: " << frame_id << std::endl;
        std::cout << "平均 FPS: " << mean_fps << std::endl;
        std::cout << "══════════════════════════════════" << std::endl;
    }

    return 0;
}
