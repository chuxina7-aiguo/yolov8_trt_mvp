cmake_minimum_required(VERSION 3.18)
project(yolov8_trt_mvp LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 89)

if(MSVC)
    # 仅对 CXX 传递 /utf-8；CUDA 通过 -Xcompiler 转发给宿主编译器，避免 nvcc 误判参数。
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/utf-8>")
    add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>")
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)

# 可通过环境变量 TENSORRT_DIR 指定 TensorRT 根目录。
# Windows 示例: set TENSORRT_DIR=C:/TensorRT-10.0.1.6
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS
        $ENV{TENSORRT_DIR}
    PATH_SUFFIXES include)

find_library(TENSORRT_NVINFER nvinfer
    HINTS
        $ENV{TENSORRT_DIR}
    PATH_SUFFIXES lib lib64)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_NVINFER)
    message(FATAL_ERROR "未找到 TensorRT，请设置 TENSORRT_DIR，保证包含 include/NvInfer.h 和 lib/nvinfer 库。")
endif()

add_executable(${PROJECT_NAME}
    src/main.cpp
    src/yolov8_trt.cpp
    src/preprocess_cuda.cu
)

target_include_directories(${PROJECT_NAME} PRIVATE
    include
    ${OpenCV_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIR}
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    ${OpenCV_LIBS}
    ${TENSORRT_NVINFER}
    CUDA::cudart
    ws2_32
)

# 某些环境下需要显式启用可分离编译。
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

message(STATUS "OpenCV include: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "TensorRT include: ${TENSORRT_INCLUDE_DIR}")
message(STATUS "TensorRT nvinfer: ${TENSORRT_NVINFER}")
