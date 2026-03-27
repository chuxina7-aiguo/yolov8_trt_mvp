#include "yolov8_trt.h"

#include <iostream>
#include <chrono>
#include <string>
#include <cctype>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <atomic>
#include <algorithm>

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

// ============================================================================
// 【推理子线程函数】Inference Worker Thread
// ============================================================================

void inference_worker_thread(const std::string& input_source, const std::string& engine_path) {
    std::cout << "\n[InferenceThread] 启动推理线程..." << std::endl;

    // ========================================================================
    // 打开输入源（摄像头或视频文件）
    // ========================================================================
    cv::VideoCapture cap;
    bool is_camera = isNumericString(input_source);

    if (is_camera) {
        int camera_index = std::stoi(input_source);
        cap.open(camera_index);
        if (!cap.isOpened()) {
            std::cerr << "[InferenceThread ERROR] 无法打开摄像头设备: " << camera_index << std::endl;
            g_inference_running = false;
            return;
        }

        cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        cap.set(cv::CAP_PROP_FPS, 30);

        std::cout << "[InferenceThread] ✓ 已打开摄像头设备: " << camera_index << std::endl;
    } else {
        cap.open(input_source);
        if (!cap.isOpened()) {
            std::cerr << "[InferenceThread ERROR] 无法打开视频文件: " << input_source << std::endl;
            g_inference_running = false;
            return;
        }

        int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
        double fps = cap.get(cv::CAP_PROP_FPS);
        std::cout << "[InferenceThread] ✓ 已打开视频文件: " << input_source << std::endl;
        std::cout << "[InferenceThread]   帧数: " << total_frames << " | 帧率: " << fps << " FPS" << std::endl;
    }

    // ========================================================================
    // 初始化 YoloV8TRT 检测器
    // ========================================================================
    YoloV8TRT detector(engine_path, 0.40f, 0.45f);  // 提高置信度阈值以过滤低分框
    if (!detector.isReady()) {
        std::cerr << "[InferenceThread ERROR] 检测器初始化失败！" << std::endl;
        cap.release();
        g_inference_running = false;
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

    while (g_inference_running && cap.read(frame)) {
        if (frame.empty()) {
            std::cout << "[InferenceThread] 视频流结束。" << std::endl;
            break;
        }

        // 记录时间戳
        auto start_time = std::chrono::high_resolution_clock::now();

        // ====================================================================
        // 推理
        // ====================================================================
        std::vector<Detection> dets;
        if (!detector.infer(frame, dets)) {
            std::cerr << "[InferenceThread] 第 " << frame_id << " 帧推理失败，已跳过。" << std::endl;
            ++frame_id;
            continue;
        }

        // ====================================================================
        // 绘制检测框
        // ====================================================================
        detector.drawDetections(frame, dets);

        // ====================================================================
        // 计算 FPS
        // ====================================================================
        auto end_time = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
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

        // ====================================================================
        // 将图像压缩为 JPG
        // ====================================================================
        std::vector<uchar> jpg_buffer;
        if (!cv::imencode(".jpg", resized_frame, jpg_buffer, {cv::IMWRITE_JPEG_QUALITY, 80})) {
            std::cerr << "[InferenceThread] JPG 编码失败！" << std::endl;
            ++frame_id;
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

        // 计算缩放比例（原图 vs 缩小后的图）
        float scale_x = 640.0f / frame.cols;
        float scale_y = 480.0f / frame.rows;

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
    }

    cap.release();
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
    // 启动推理线程
    // ========================================================================
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
        std::string html = R"(
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
            max-width: 960px;
            width: 100%;
            max-height: 90vh;
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
            overflow-y: auto;
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
    </div>

    <script>
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
        const errorBox = document.getElementById('error-box');

        function showError(msg) {
            errorBox.textContent = msg;
            errorBox.classList.add('show');
            setTimeout(() => errorBox.classList.remove('show'), 5000);
        }

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

            eventSource.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    const dets = data.dets || data.detections || [];

                    // 更新视频画面
                    videoImg.src = 'data:image/jpeg;base64,' + data.image;

                    // 同步 canvas 尺寸并清空上一帧，避免框叠加
                    const displayWidth = Math.max(1, Math.round(videoImg.clientWidth || 640));
                    const displayHeight = Math.max(1, Math.round(videoImg.clientHeight || 480));
                    if (overlay.width !== displayWidth || overlay.height !== displayHeight) {
                        overlay.width = displayWidth;
                        overlay.height = displayHeight;
                    }
                    overlay.style.display = 'block';
                    overlayCtx.clearRect(0, 0, overlay.width, overlay.height);

                    // 将后端 640x480 坐标缩放到当前画布尺寸
                    const scaleX = overlay.width / 640.0;
                    const scaleY = overlay.height / 480.0;
                    dets.forEach(det => {
                        const x = Math.round(det.x * scaleX);
                        const y = Math.round(det.y * scaleY);
                        const w = Math.round(det.width * scaleX);
                        const h = Math.round(det.height * scaleY);

                        let color = '#4dabf7';
                        if (det.class_name === 'helmet' || det.class_id === 0) color = '#51cf66';
                        else if (det.class_name === 'head' || det.class_id === 1) color = '#ff6b6b';

                        overlayCtx.strokeStyle = color;
                        overlayCtx.lineWidth = 2;
                        overlayCtx.strokeRect(x, y, w, h);

                        const label = `${det.class_name || det.class_id} ${(det.score * 100).toFixed(1)}%`;
                        overlayCtx.font = '14px Segoe UI';
                        const textW = Math.ceil(overlayCtx.measureText(label).width);
                        const textX = x;
                        const textY = Math.max(16, y - 4);
                        overlayCtx.fillStyle = color;
                        overlayCtx.fillRect(textX, textY - 14, textW + 8, 16);
                        overlayCtx.fillStyle = '#ffffff';
                        overlayCtx.fillText(label, textX + 4, textY - 2);
                    });

                    // 更新性能指标
                    fpsValue.textContent = data.fps.toFixed(1) + ' fps';
                    detCount.textContent = dets.length;
                    sourceInfo.textContent = data.source_info || '--';

                    // 更新检测结果卡片
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

            eventSource.onerror = () => {
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

        // 页面加载完毕后连接流
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
        inference_thread.join();
        return -1;
    }

    // ========================================================================
    // 清理和退出
    // ========================================================================
    std::cout << "\n[Main] HTTP 服务器已停止。" << std::endl;
    g_inference_running = false;
    g_frame_cv.notify_all();

    std::cout << "[Main] 等待推理线程退出..." << std::endl;
    inference_thread.join();

    std::cout << "[Main] ✓ 所有线程已安全退出。程序结束。\n" << std::endl;
    return 0;
}
