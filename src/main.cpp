#include "yolov8_trt.h"

#include <iostream>
#include <chrono>
#include <string>
#include <cctype>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <queue>
#include <atomic>
#include <algorithm>
#include <fstream>
#include <direct.h>

#include "httplib.h"
#include "json.hpp"

using json = nlohmann::json;

// ============================================================================
// 【全局数据交换区】Thread-Safe Data Exchange Zone
// ============================================================================

struct FrameData {
    std::string base64_image;           // Base64 编码的 JPG 图片
    json detections;                    // 检测框的 JSON array
    double fps = 0.0;                   // 实时 FPS
    std::string source_info = "";       // 输入源信息
    long long timestamp = 0;            // 时间戳
};

FrameData g_latest_frame;               // 最新帧数据
std::mutex g_frame_mtx;                 // 帧数据互斥锁
std::condition_variable g_frame_cv;     // 帧数据条件变量

std::queue<cv::Mat> frame_queue;
std::mutex frame_mutex;
std::condition_variable frame_cv;

std::atomic<bool> g_inference_running(true);   // 推理线程运行标志
std::atomic<int> g_total_frames(0);            // 总帧计数

// ============================================================================
// 【辅助函数】
// ============================================================================

/// 判断字符串是否为数字
bool isNumericString(const std::string& str) {
    if (str.empty()) return false;
    return std::all_of(str.begin(), str.end(), [](unsigned char c) { return std::isdigit(c); });
}

/// Base64 编码函数（自实现，零依赖）
std::string base64_encode(const std::vector<uchar>& data) {
    static constexpr const char base64_chars[] = 
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    
    std::string result;
    result.reserve((data.size() + 2) / 3 * 4);
    
    for (size_t i = 0; i < data.size(); i += 3) {
        unsigned char a = data[i];
        unsigned char b = (i + 1 < data.size()) ? data[i + 1] : 0;
        unsigned char c = (i + 2 < data.size()) ? data[i + 2] : 0;
        
        result += base64_chars[(a >> 2) & 0x3F];
        result += base64_chars[((a & 0x03) << 4) | ((b >> 4) & 0x0F)];
        result += (i + 1 < data.size()) ? base64_chars[((b & 0x0F) << 2) | ((c >> 6) & 0x03)] : '=';
        result += (i + 2 < data.size()) ? base64_chars[c & 0x3F] : '=';
    }
    
    return result;
}

std::string base64_decode(const std::string& in) {
    static constexpr unsigned char kDecTable[256] = {
        64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,
        64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,
        64,64,64,64,64,64,64,64,64,64,64,62,64,64,64,63,
        52,53,54,55,56,57,58,59,60,61,64,64,64,64,64,64,
        64, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,
        15,16,17,18,19,20,21,22,23,24,25,64,64,64,64,64,
        64,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,
        41,42,43,44,45,46,47,48,49,50,51,64,64,64,64,64,
        64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,
        64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,
        64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,
        64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,
        64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,
        64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,
        64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,
        64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64
    };

    std::string out;
    out.reserve(in.size() * 3 / 4);

    int val = 0;
    int valb = -8;
    for (unsigned char c : in) {
        if (c == '=') {
            break;
        }
        const unsigned char d = kDecTable[c];
        if (d == 64) {
            continue;
        }
        val = (val << 6) + d;
        valb += 6;
        if (valb >= 0) {
            out.push_back(static_cast<char>((val >> valb) & 0xFF));
            valb -= 8;
        }
    }
    return out;
}

// ============================================================================
// 【拉流线程函数】Camera Producer Thread
// ============================================================================

void camera_thread_func(const std::string& url) {
    std::cout << "[CameraThread] 启动拉流线程..." << std::endl;

    cv::VideoCapture cap;
    if (isNumericString(url)) {
        int camera_index = std::stoi(url);
        cap.open(camera_index);
        if (cap.isOpened()) {
            cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
            cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
            cap.set(cv::CAP_PROP_FPS, 30);
        }
    } else {
        cap.open(url);
    }

    if (!cap.isOpened()) {
        std::cerr << "[CameraThread ERROR] 无法打开视频源: " << url << std::endl;
        g_inference_running = false;
        frame_cv.notify_all();
        return;
    }

    std::cout << "[CameraThread] ✓ 拉流已启动: " << url << std::endl;

    cv::Mat tmp;
    while (g_inference_running) {
        if (!cap.read(tmp)) {
            continue;
        }
        if (tmp.empty()) {
            continue;
        }

        {
            std::lock_guard<std::mutex> lock(frame_mutex);
            frame_queue.push(tmp.clone());
        }
        frame_cv.notify_one();
    }

    cap.release();
    std::cout << "[CameraThread] 拉流线程退出。" << std::endl;
}

// ============================================================================
// 【推理子线程函数】Inference Worker Thread
// ============================================================================

void inference_worker_thread(const std::string& input_source, const std::string& engine_path) {
    std::cout << "\n[InferenceThread] 启动推理线程..." << std::endl;

    const bool is_camera = isNumericString(input_source);

    // ========================================================================
    // 初始化 YoloV8TRT 检测器
    // ========================================================================
    YoloV8TRT detector(engine_path, 0.40f, 0.45f);  // 提高置信度阈值以过滤低分框
    if (!detector.isReady()) {
        std::cerr << "[InferenceThread ERROR] 检测器初始化失败！" << std::endl;
        g_inference_running = false;
        frame_cv.notify_all();
        return;
    }

    std::cout << "[InferenceThread] ✓ 检测器初始化成功，引擎: " << engine_path << std::endl;
    std::cout << "[InferenceThread] 开始实时推理循环...\n" << std::endl;

    // ========================================================================
    // 推理主循环
    // ========================================================================
    cv::Mat frame, resized_frame;
    int frame_id = 0;
    double avg_fps = 0.0;
    int fps_count = 0;

    while (g_inference_running) {
        auto start_time = std::chrono::steady_clock::now();

        {
            std::unique_lock<std::mutex> lock(frame_mutex);
            frame_cv.wait(lock, [] { return !frame_queue.empty() || !g_inference_running; });
            if (!g_inference_running) {
                break;
            }
            // 直接取最新帧并清空旧帧，彻底避免积压
            frame = std::move(frame_queue.back());
            std::queue<cv::Mat> empty;
            std::swap(frame_queue, empty);
        }

        if (frame.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
            continue;
        }

        // 记录推理耗时
        auto infer_start_time = std::chrono::high_resolution_clock::now();

        // ====================================================================
        // 推理
        // ====================================================================
        std::vector<Detection> dets;
        if (!detector.infer(frame, dets)) {
            std::cerr << "[InferenceThread] 第 " << frame_id << " 帧推理失败，已跳过。" << std::endl;
            ++frame_id;
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
            continue;
        }

        // ====================================================================
        // 绘制检测框
        // ====================================================================
        detector.drawDetections(frame, dets);

        // ====================================================================
        // 计算 FPS
        // ====================================================================
        auto infer_end_time = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(infer_end_time - infer_start_time).count();
        double real_fps = elapsed_ms > 0 ? 1000.0 / elapsed_ms : 0.0;
        avg_fps += real_fps;
        fps_count++;

        // ====================================================================
        // 显示性能数据到画面上（便于调试）
        // ====================================================================
        std::string perf_text = cv::format("FPS: %.1f | Dets: %zu", real_fps, dets.size());
        cv::putText(frame,
                    perf_text,
                    cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.8,
                    cv::Scalar(0, 255, 255),
                    2);

        // ====================================================================
        // 将图像缩小到 640x480（降低网络延迟）
        // ====================================================================
        cv::resize(frame, resized_frame, cv::Size(640, 480));

        // 高分辨率保护：编码前限制到不超过 1280x720（保持比例）
        if (resized_frame.cols > 1280 || resized_frame.rows > 720) {
            const double sx = 1280.0 / static_cast<double>(resized_frame.cols);
            const double sy = 720.0 / static_cast<double>(resized_frame.rows);
            const double scale = std::min(1.0, std::min(sx, sy));
            const int new_w = std::max(1, static_cast<int>(resized_frame.cols * scale));
            const int new_h = std::max(1, static_cast<int>(resized_frame.rows * scale));
            cv::resize(resized_frame, resized_frame, cv::Size(new_w, new_h));
        }

        // ====================================================================
        // 将图像压缩为 JPG
        // ====================================================================
        std::vector<uchar> jpg_buffer;
        std::vector<int> encode_params = {cv::IMWRITE_JPEG_QUALITY, 50};
        if (!cv::imencode(".jpg", resized_frame, jpg_buffer, encode_params)) {
            std::cerr << "[InferenceThread] JPG 编码失败！" << std::endl;
            ++frame_id;
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
            continue;
        }

        // ====================================================================
        // 编码为 Base64
        // ====================================================================
        std::string base64_image = base64_encode(jpg_buffer);

        // ====================================================================
        // 构建检测结果 JSON array
        // ====================================================================
        json detections_json = json::array();

        // 计算缩放比例（原图 vs 实际编码图）
        float scale_x = static_cast<float>(resized_frame.cols) / frame.cols;
        float scale_y = static_cast<float>(resized_frame.rows) / frame.rows;

        for (const auto& det : dets) {
            json det_obj;
            det_obj["class_id"] = det.class_id;
            det_obj["class_name"] = (det.class_id == 0) ? "helmet" : 
                                    (det.class_id == 1) ? "head" : "person";
            det_obj["score"] = det.score;
            
            // 映射到缩小后的分辨率
            det_obj["x"] = static_cast<int>(det.box.x * scale_x);
            det_obj["y"] = static_cast<int>(det.box.y * scale_y);
            det_obj["width"] = static_cast<int>(det.box.width * scale_x);
            det_obj["height"] = static_cast<int>(det.box.height * scale_y);
            json keypoints_json = json::array();
            for (const auto& kp : det.kpts) {
                json kp_obj;
                kp_obj["x"] = static_cast<int>(std::round(kp.x * scale_x));
                kp_obj["y"] = static_cast<int>(std::round(kp.y * scale_y));
                kp_obj["conf"] = kp.conf;
                keypoints_json.push_back(kp_obj);
            }
            det_obj["keypoints"] = keypoints_json;
            
            detections_json.push_back(det_obj);
        }

        // ====================================================================
        // 【线程安全地更新全局数据】
        // ====================================================================
        {
            std::lock_guard<std::mutex> lock(g_frame_mtx);
            g_latest_frame.base64_image = base64_image;
            g_latest_frame.detections = detections_json;
            g_latest_frame.fps = real_fps;
            g_latest_frame.source_info = is_camera ? "Camera-0" : input_source;
            g_latest_frame.timestamp = std::chrono::system_clock::now().time_since_epoch().count();
        }

        // 唤醒所有等待新帧的线程（Web 服务器的 SSE 推送线程）
        g_frame_cv.notify_all();

        // 定期输出诊断信息
        if ((frame_id + 1) % 30 == 0) {
            std::cout << "[InferenceThread] 帧 " << (frame_id + 1) 
                     << " | FPS: " << real_fps 
                     << " | JPG: " << jpg_buffer.size() << " bytes"
                     << " | Detections: " << dets.size() << std::endl;
        }

        ++frame_id;
        g_total_frames++;

        // 硬限流：每次循环总时长不低于 33ms
        auto end_time = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        if (elapsed < 33) {
            std::this_thread::sleep_for(std::chrono::milliseconds(33 - elapsed));
        }
    }

    g_inference_running = false;

    // 输出最终统计
    if (fps_count > 0) {
        double mean_fps = avg_fps / fps_count;
        std::cout << "\n[InferenceThread] ═══════════════════════════════════" << std::endl;
        std::cout << "[InferenceThread] 推理循环结束" << std::endl;
        std::cout << "[InferenceThread] 总帧数: " << frame_id << std::endl;
        std::cout << "[InferenceThread] 平均 FPS: " << mean_fps << std::endl;
        std::cout << "[InferenceThread] ═══════════════════════════════════\n" << std::endl;
    }
}

// ============================================================================
// 【主程序】Main - Web Server
// ============================================================================

int main(int argc, char** argv) {
    std::string input_source = (argc > 1) ? argv[1] : "0";
    const std::string engine_path = (argc > 2) ? argv[2] : "yolov8.engine";

    std::cout << "\n╔══════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║     YOLOv8 TensorRT 前后端分离 AI 实时流推送服务              ║" << std::endl;
    std::cout << "║     Web Server: http://localhost:8080                        ║" << std::endl;
    std::cout << "║     SSE Stream: http://localhost:8080/stream                 ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n" << std::endl;

    // ========================================================================
    // 启动拉流线程 + 推理线程
    // ========================================================================
    std::thread camera_thread(camera_thread_func, input_source);
    std::thread inference_thread(inference_worker_thread, input_source, engine_path);

    // 等待推理线程初始化完成
    std::this_thread::sleep_for(std::chrono::milliseconds(1500));

    // ========================================================================
    // 创建 HTTP 服务器
    // ========================================================================
    httplib::Server svr;

    // ========================================================================
    // 【路由 1】GET / - 返回完整的 HTML5 SPA
    // ========================================================================
    svr.Get("/", [](const httplib::Request&, httplib::Response& res) {
        const std::string html =
            R"(
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🎬 YOLOv8 TensorRT AI 实时检测</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        html, body {
            width: 100%;
            height: 100%;
            overflow: hidden;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 12px;
        }
        .container {
            background: white;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.35);
            max-width: 1360px;
            width: 100%;
            height: 92vh;
            overflow: auto;
            display: flex;
            flex-direction: column;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 24px;
            text-align: center;
            font-size: 26px;
            font-weight: bold;
            border-radius: 16px 16px 0 0;
        }
        .content {
            padding: 24px;
            flex: 1;
            overflow: hidden;
            display: flex;
            flex-direction: row;
            gap: 16px;
        }
        .left-panel {
            flex: 1;
            min-width: 0;
            overflow-y: auto;
            padding-right: 8px;
        }
        #history-panel {
            width: 300px;
            flex: 0 0 300px;
            border-radius: 10px;
            border: 1px solid #e7e7e7;
            background: #fafafa;
            padding: 12px;
            overflow-y: auto;
            box-shadow: inset 0 0 0 1px #ffffff;
        }
        #snapshot-list {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        .snapshot-item {
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid #f0c2c2;
            background: #fff;
            box-shadow: 0 4px 10px rgba(255, 0, 0, 0.08);
        }
        .snapshot-item img {
            width: 100%;
            display: block;
        }
        .snapshot-item p {
            margin: 0;
            padding: 8px;
            color: #b30000;
            font-size: 12px;
            font-weight: 600;
        }
        .video-section {
            margin-bottom: 20px;
        }
        .video-container {
            position: relative;
            width: 100%;
            background: #000;
            border-radius: 8px;
            overflow: hidden;
            aspect-ratio: 4/3;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        #video {
            width: 100%;
            height: 100%;
            object-fit: contain;
        }
        #overlay {
            position: absolute;
            top: 0;
            left: 0;
            cursor: crosshair;
        }
        .loading-spinner {
            width: 40px;
            height: 40px;
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 12px;
            margin-bottom: 20px;
        }
        .stat-card {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-radius: 8px;
            padding: 16px;
            text-align: center;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        }
        .stat-label {
            font-size: 12px;
            font-weight: 600;
            color: #555;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: #667eea;
            font-family: 'Courier New', monospace;
        }
        .status-badge {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #ff6b6b;
            margin-right: 8px;
            animation: pulse 1.5s ease-in-out infinite;
        }
        .status-badge.online {
            background: #51cf66;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .detections-section {
            margin-top: 20px;
        }
        .detections-title {
            font-size: 16px;
            font-weight: 600;
            color: #333;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
        }
        .detections-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 12px;
        }
        .detection-card {
            background: #f9f9f9;
            border-left: 4px solid #667eea;
            border-radius: 6px;
            padding: 12px;
            font-size: 13px;
            line-height: 1.6;
            transition: all 0.2s ease;
        }
        .detection-card:hover {
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2);
            transform: translateY(-2px);
        }
        .detection-card.helmet {
            border-left-color: #51cf66;
        }
        .detection-card.head {
            border-left-color: #ff6b6b;
        }
        .detection-card.person {
            border-left-color: #4dabf7;
        }
        .detection-class {
            font-weight: 700;
            font-size: 14px;
            color: #333;
            margin-bottom: 4px;
        }
        .detection-score {
            color: #999;
            font-size: 12px;
        }
        .error-box {
            background: #ffe3e3;
            border-left: 4px solid #ff6b6b;
            color: #c92a2a;
            padding: 12px;
            border-radius: 6px;
            margin-bottom: 16px;
            display: none;
        }
        .error-box.show {
            display: block;
        }
        @media (max-width: 768px) {
            .content {
                flex-direction: column;
                overflow-y: auto;
            }
            .left-panel {
                padding-right: 0;
            }
            #history-panel {
                width: 100%;
                flex: 0 0 auto;
                max-height: 260px;
            }
            .stats-grid {
                grid-template-columns: 1fr;
            }
            .detections-grid {
                grid-template-columns: 1fr;
            }
            .header {
                font-size: 20px;
                padding: 16px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">🎬 YOLOv8 TensorRT 实时检测系统</div>
        
        <div class="content">
            <div class="left-panel">
                <div class="error-box" id="error-box"></div>

                <div class="video-section">
                    <div class="video-container">
                        <img id="video" src="" alt="Video Stream">
                        <canvas id="overlay" style="display: none;"></canvas>
                        <div id="loading" class="loading-spinner"></div>
                    </div>
                </div>

                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-label">
                            <span class="status-badge" id="status-indicator"></span>连接状态
                        </div>
                        <div class="stat-value" id="status-text">离线</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">实时 FPS</div>
                        <div class="stat-value" id="fps-value">-- fps</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">检测框数</div>
                        <div class="stat-value" id="det-count">0</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">输入源</div>
                        <div class="stat-value" id="source-info" style="font-size: 14px;">--</div>
                    </div>
                </div>

                <div class="detections-section">
                    <div class="detections-title">📊 实时检测结果</div>
                    <div class="detections-grid" id="detections-grid">
                        <div style="grid-column: 1/-1; text-align: center; color: #999; padding: 20px;">
                            正在等待检测数据...
                        </div>
                    </div>
                </div>
            </div>

            <div id="history-panel">
                <div class="detections-title">📸 抓拍墙</div>
                <div id="snapshot-list"></div>
            </div>
        </div>
    </div>
)"
            R"(<script>
        const videoImg = document.getElementById('video');
        const overlay = document.getElementById('overlay');
        const overlayCtx = overlay.getContext('2d');
        const loading = document.getElementById('loading');
        const statusIndicator = document.getElementById('status-indicator');
        const statusText = document.getElementById('status-text');
        const fpsValue = document.getElementById('fps-value');
        const detCount = document.getElementById('det-count');
        const sourceInfo = document.getElementById('source-info');
        const detectionsGrid = document.getElementById('detections-grid');
        const snapshotList = document.getElementById('snapshot-list');
        const errorBox = document.getElementById('error-box');
        const canvas = overlay;

        const skeleton = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]];
        let customZone = [];
        let lastSnapshotTime = 0;
        const SNAPSHOT_COOLDOWN = 3000;

        function isPointInPolygon(point, vs) {
            let inside = false;
            for (let i = 0, j = vs.length - 1; i < vs.length; j = i++) {
                const xi = vs[i].x;
                const yi = vs[i].y;
                const xj = vs[j].x;
                const yj = vs[j].y;

                const intersect = ((yi > point.y) !== (yj > point.y))
                    && (point.x < (xj - xi) * (point.y - yi) / ((yj - yi) || 1e-9) + xi);
                if (intersect) inside = !inside;
            }
            return inside;
        }

        function showError(msg) {
            errorBox.textContent = msg;
            errorBox.classList.add('show');
            setTimeout(() => errorBox.classList.remove('show'), 5000);
        }

        overlay.addEventListener('click', (e) => {
            const x = Math.round(e.offsetX);
            const y = Math.round(e.offsetY);
            customZone.push({ x, y });
        });

        overlay.addEventListener('contextmenu', (e) => {
            e.preventDefault();
            customZone = [];
        });

        function connectStream() {
            console.log('[WebClient] 连接 SSE 流...');
            const eventSource = new EventSource('/stream');

            eventSource.onopen = () => {
                console.log('[WebClient] SSE 连接成功');
                loading.style.display = 'none';
                statusIndicator.classList.add('online');
                statusText.textContent = '在线';
                errorBox.classList.remove('show');
            };
)"
            R"(            eventSource.onmessage = (event) => {
                // 【后台标签页检查】如果页面在后台，直接丢弃数据，避免积压
                if (typeof document.hidden === 'boolean' && document.hidden) return;
                
                try {
                    const data = JSON.parse(event.data);
                    const dets = data.dets || data.detections || [];

                    videoImg.src = 'data:image/jpeg;base64,' + data.image;

                    const displayWidth = Math.max(1, Math.round(videoImg.clientWidth || 640));
                    const displayHeight = Math.max(1, Math.round(videoImg.clientHeight || 480));
                    if (overlay.width !== displayWidth || overlay.height !== displayHeight) {
                        overlay.width = displayWidth;
                        overlay.height = displayHeight;
                    }

                    overlay.style.display = 'block';
                    overlayCtx.clearRect(0, 0, overlay.width, overlay.height);

                    const scaleX = overlay.width / 640.0;
                    const scaleY = overlay.height / 480.0;

                    if (customZone.length > 0) {
                        overlayCtx.beginPath();
                        overlayCtx.moveTo(customZone[0].x, customZone[0].y);
                        for (let i = 1; i < customZone.length; i++) {
                            overlayCtx.lineTo(customZone[i].x, customZone[i].y);
                        }
                        if (customZone.length >= 3) {
                            overlayCtx.closePath();
                            overlayCtx.fillStyle = 'rgba(255, 0, 0, 0.3)';
                            overlayCtx.fill();
                        }
                        overlayCtx.strokeStyle = 'red';
                        overlayCtx.lineWidth = 2;
                        overlayCtx.setLineDash([]);
                        overlayCtx.stroke();

                        overlayCtx.fillStyle = 'red';
                        customZone.forEach((p) => {
                            overlayCtx.beginPath();
                            overlayCtx.arc(p.x, p.y, 4, 0, Math.PI * 2);
                            overlayCtx.fill();
                        });
                    }

                    dets.forEach(det => {
                        const x = Math.round(det.x * scaleX);
                        const y = Math.round(det.y * scaleY);
                        const w = Math.round(det.width * scaleX);
                        const h = Math.round(det.height * scaleY);
                        const keypoints = Array.isArray(det.keypoints) ? det.keypoints : [];

                        let isIntruding = false;
                        let isSos = false;
                        if (customZone.length >= 3) {
                            const leftAnkle = keypoints[15];
                            const rightAnkle = keypoints[16];

                            if (leftAnkle && leftAnkle.conf > 0) {
                                const leftPt = {
                                    x: Math.round(leftAnkle.x * scaleX),
                                    y: Math.round(leftAnkle.y * scaleY)
                                };
                                if (isPointInPolygon(leftPt, customZone)) {
                                    isIntruding = true;
                                }
                            }

                            if (!isIntruding && rightAnkle && rightAnkle.conf > 0) {
                                const rightPt = {
                                    x: Math.round(rightAnkle.x * scaleX),
                                    y: Math.round(rightAnkle.y * scaleY)
                                };
                                if (isPointInPolygon(rightPt, customZone)) {
                                    isIntruding = true;
                                }
                            }
                        }

                        const nose = keypoints[0];
                        const leftWrist = keypoints[9];
                        const rightWrist = keypoints[10];
                        if (nose && leftWrist && rightWrist
                            && nose.conf > 0.4
                            && leftWrist.conf > 0.4
                            && rightWrist.conf > 0.4
                            && leftWrist.y < nose.y
                            && rightWrist.y < nose.y) {
                            isSos = true;
                        }

                        if (isSos) {
                            overlayCtx.strokeStyle = 'orange';
                            overlayCtx.lineWidth = 3;
                            overlayCtx.setLineDash([]);
                            overlayCtx.strokeRect(x, y, w, h);

                            overlayCtx.fillStyle = 'orange';
                            overlayCtx.font = 'bold 28px Segoe UI';
                            overlayCtx.fillText('🆘 紧急求救！', x, Math.max(28, y - 10));
                        } else if (isIntruding) {
                            overlayCtx.strokeStyle = 'rgba(255, 0, 0, 1)';
                            overlayCtx.lineWidth = 3;
                            overlayCtx.setLineDash([]);
                            overlayCtx.strokeRect(x, y, w, h);

                            overlayCtx.fillStyle = 'rgba(255, 0, 0, 1)';
                            overlayCtx.font = 'bold 28px Segoe UI';
                            overlayCtx.fillText('⚠️ 危险区闯入！', x, Math.max(28, y - 10));
                        } else {
                            overlayCtx.strokeStyle = 'rgba(57, 255, 20, 1)';
                            overlayCtx.lineWidth = 2;
                            overlayCtx.setLineDash([8, 6]);
                            overlayCtx.strokeRect(x, y, w, h);
                        }

                        const now = Date.now();
                        if ((isIntruding || isSos) && (now - lastSnapshotTime > SNAPSHOT_COOLDOWN)) {
                            lastSnapshotTime = now;
                            // 【局部变量化】每次快照都创建新的 Image 对象，防止异步覆盖
                            const tempImg = new Image();
                            // 【先定义 onload】在赋值 src 前先设置加载完成回调
                            tempImg.onload = function() {
                                const snapCanvas = document.createElement('canvas');
                                snapCanvas.width = tempImg.width;
                                snapCanvas.height = tempImg.height;
                                const snapCtx = snapCanvas.getContext('2d');
                                // 1. 画视频底图
                                snapCtx.drawImage(tempImg, 0, 0);
                                // 2. 覆盖透明画板（拉伸到与底图同尺寸）
                                snapCtx.drawImage(canvas, 0, 0, tempImg.width, tempImg.height);

                                const finalDataUrl = snapCanvas.toDataURL('image/jpeg', 0.8);
                                fetch('/api/save_snapshot', { method: 'POST', body: finalDataUrl });

                                // 插入照片墙
                                const eventText = isSos ? '🆘 紧急求救' : '⚠️ 违规闯入';
                                const timeStr = new Date().toLocaleTimeString();
                                const card = document.createElement('div');
                                card.className = 'snapshot-item';
                                card.innerHTML = `<img src="${finalDataUrl}" style="width:100%;"><p>${eventText} - ${timeStr}</p>`;
                                document.getElementById('snapshot-list').prepend(card);
                            };
                            tempImg.src = 'data:image/jpeg;base64,' + data.image;
                        }

                        overlayCtx.strokeStyle = '#ff00ff';
                        overlayCtx.lineWidth = 2;
                        overlayCtx.setLineDash([]);
                        skeleton.forEach(([a, b]) => {
                            const p1 = keypoints[a];
                            const p2 = keypoints[b];
                            if (!p1 || !p2) return;
                            if ((p1.x === 0 && p1.y === 0) || (p2.x === 0 && p2.y === 0)) return;

                            const x1 = Math.round(p1.x * scaleX);
                            const y1 = Math.round(p1.y * scaleY);
                            const x2 = Math.round(p2.x * scaleX);
                            const y2 = Math.round(p2.y * scaleY);
                            overlayCtx.beginPath();
                            overlayCtx.moveTo(x1, y1);
                            overlayCtx.lineTo(x2, y2);
                            overlayCtx.stroke();
                        });

                        overlayCtx.fillStyle = '#39ff14';
                        keypoints.forEach((kp) => {
                            if (!kp) return;
                            if (typeof kp.conf === 'number' && kp.conf <= 0.4) return;
                            if (kp.x === 0 && kp.y === 0) return;

                            const kx = Math.round(kp.x * scaleX);
                            const ky = Math.round(kp.y * scaleY);
                            overlayCtx.beginPath();
                            overlayCtx.arc(kx, ky, 3, 0, Math.PI * 2);
                            overlayCtx.fill();
                        });
                    });

                    fpsValue.textContent = data.fps.toFixed(1) + ' fps';
                    detCount.textContent = dets.length;
                    sourceInfo.textContent = data.source_info || '--';

                    if (dets.length > 0) {
                        detectionsGrid.innerHTML = dets
                            .map(det => `
                                <div class="detection-card ${det.class_name}">
                                    <div class="detection-class">🏷️ ${det.class_name.toUpperCase()}</div>
                                    <div class="detection-score">
                                        置信度: ${(det.score * 100).toFixed(1)}%
                                    </div>
                                    <div style="font-size: 11px; color: #777; margin-top: 6px;">
                                        📍 位置: (${det.x}, ${det.y})<br>
                                        📐 大小: ${det.width}×${det.height}
                                    </div>
                                </div>
                            `)
                            .join('');
                    } else {
                        detectionsGrid.innerHTML = '<div style="grid-column: 1/-1; text-align: center; color: #999; padding: 20px;">暂无检测结果</div>';
                    }
                } catch (e) {
                    console.error('[WebClient] JSON 解析失败:', e);
                    showError('数据解析失败: ' + e.message);
                }
            };
)"
            R"(            eventSource.onerror = () => {
                console.error('[WebClient] SSE 连接错误');
                statusIndicator.classList.remove('online');
                statusText.textContent = '离线';
                loading.style.display = 'flex';
                eventSource.close();
                showError('连接断开，5 秒后重试...');
                setTimeout(connectStream, 5000);
            };

            window.addEventListener('beforeunload', () => eventSource.close());
        }

        window.addEventListener('load', connectStream);
    </script>
</body>
</html>
        )";

        res.set_content(html, "text/html; charset=utf-8");
    });

    // ========================================================================
    // 【路由 2】GET /stream - Server-Sent Events (SSE) 推送
    // ========================================================================
    svr.Get("/stream", [](const httplib::Request&, httplib::Response& res) {
        res.set_header("Content-Type", "text/event-stream; charset=utf-8");
        res.set_header("Cache-Control", "no-cache");
        res.set_header("Connection", "keep-alive");
        res.set_header("Access-Control-Allow-Origin", "*");

        res.set_chunked_content_provider("text/event-stream", [](size_t, httplib::DataSink& sink) {
            int heartbeat_count = 0;

            while (g_inference_running || heartbeat_count < 3) {
                std::unique_lock<std::mutex> lock(g_frame_mtx);

                // 等待新帧数据到达（超时 3 秒）
                bool have_data = g_frame_cv.wait_for(lock, std::chrono::seconds(3),
                    []() { return !g_latest_frame.base64_image.empty(); });

                if (!have_data) {
                    // 没有数据，发送心跳包保活
                    ++heartbeat_count;
                    std::string heartbeat = ": heartbeat\n\n";
                    if (!sink.write(heartbeat.data(), heartbeat.size())) {
                        return false;  // 客户端断开连接
                    }
                    continue;
                }

                heartbeat_count = 0;  // 重置心跳计数

                // 构建 SSE 格式的 JSON 数据
                json sse_payload;
                sse_payload["image"] = g_latest_frame.base64_image;
                sse_payload["fps"] = g_latest_frame.fps;
                sse_payload["detections"] = g_latest_frame.detections;
                sse_payload["source_info"] = g_latest_frame.source_info;
                sse_payload["timestamp"] = g_latest_frame.timestamp;

                std::string json_str = sse_payload.dump();
                std::string sse_message = "data: " + json_str + "\n\n";

                if (!sink.write(sse_message.data(), sse_message.size())) {
                    return false;  // 写入失败，客户端断开
                }
            }

            return true;
        });
    });

    svr.Post("/api/save_snapshot", [](const httplib::Request& req, httplib::Response& res) {
        if (req.body.empty()) {
            res.status = 400;
            res.set_content("empty request body", "text/plain; charset=utf-8");
            return;
        }

        std::string payload = req.body;
        const std::string prefix = "data:image/jpeg;base64,";
        if (payload.rfind(prefix, 0) == 0) {
            payload = payload.substr(prefix.size());
        }

        const std::string binary = base64_decode(payload);
        if (binary.empty()) {
            res.status = 400;
            res.set_content("base64 decode failed", "text/plain; charset=utf-8");
            return;
        }

        _mkdir("snapshots");
        const auto ts_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        const std::string filename = "snapshots/alert_" + std::to_string(ts_ms) + ".jpg";

        std::ofstream ofs(filename, std::ios::binary);
        if (!ofs) {
            res.status = 500;
            res.set_content("failed to open output file", "text/plain; charset=utf-8");
            return;
        }
        ofs.write(binary.data(), static_cast<std::streamsize>(binary.size()));
        ofs.close();

        if (!ofs.good()) {
            res.status = 500;
            res.set_content("failed to write snapshot", "text/plain; charset=utf-8");
            return;
        }

        res.set_content("saved: " + filename, "text/plain; charset=utf-8");
    });

    // ========================================================================
    // 启动 HTTP 服务器
    // ========================================================================
    std::cout << "[WebServer] 启动 HTTP 服务器..." << std::endl;
    std::cout << "[WebServer] 监听地址: http://0.0.0.0:8080" << std::endl;
    std::cout << "[WebServer] 前端地址: http://localhost:8080" << std::endl;
    std::cout << "[WebServer] SSE 流地址: http://localhost:8080/stream" << std::endl;
    std::cout << "[WebServer] 按 Ctrl+C 停止服务器\n" << std::endl;

    // 主线程在此阻塞，监听 HTTP 请求
    if (!svr.listen("0.0.0.0", 8080)) {
        std::cerr << "[WebServer ERROR] 服务器启动失败！" << std::endl;
        g_inference_running = false;
        frame_cv.notify_all();
        g_frame_cv.notify_all();
        camera_thread.join();
        inference_thread.join();
        return -1;
    }

    // ========================================================================
    // 清理和退出
    // ========================================================================
    std::cout << "\n[Main] HTTP 服务器已停止。" << std::endl;
    g_inference_running = false;
    frame_cv.notify_all();
    g_frame_cv.notify_all();

    std::cout << "[Main] 等待拉流线程退出..." << std::endl;
    camera_thread.join();
    std::cout << "[Main] 等待推理线程退出..." << std::endl;
    inference_thread.join();

    std::cout << "[Main] ✓ 所有线程已安全退出。程序结束。\n" << std::endl;
    return 0;
}
