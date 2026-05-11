// engine/pipeline.h
#pragma once
#include <memory>
#include <vector>
#include <thread>
#include <atomic>
#include <opencv2/opencv.hpp>
#include "bounded_queue.h"
#include "thread_pool.h"
#include "adapter_interface.h"

// 流水线任务包
struct InferenceTask {
    int frame_id;
    cv::Mat original_frame;                 // 原始帧（用于最终绘制）
    uint8_t* input_buf = nullptr;           // 指向预处理后的数据
    int input_size = 0;
    std::shared_ptr<IModelAdapter> adapter; // 负责推理的适配器实例（持有内存）
};

class Pipeline {
public:
    // 构造函数：传入基准适配器（未初始化）、模型路径、线程数、NPU 掩码、CPU 绑定等
    Pipeline(std::shared_ptr<IModelAdapter> base_adapter,
             const std::string& model_path,
             int num_infer_threads,
             const std::vector<int>& npu_cores,
             const std::vector<int>& cpu_cores = {4,5,6},
             int pre_queue_capacity = 2,
             int post_queue_capacity = 2);
    ~Pipeline();

    void Run();   // 启动流水线，阻塞直到停止
    void Stop();  // 请求停止

private:
    void PreprocessLoop();
    void InferenceLoop(int thread_id);
    void PostprocessLoop();

    // 绘制结果（使用正点原子 utils 或 OpenCV）
    void DrawResult(cv::Mat& frame, const std::string& result_json);

    // 配置
    std::string model_path_;
    int num_infer_threads_;
    std::vector<int> npu_cores_;
    std::vector<int> cpu_cores_;

    // 队列
    BoundedQueue<InferenceTask> pre_queue_;
    BoundedQueue<InferenceTask> post_queue_;

    // 线程池和单独线程
    std::unique_ptr<ThreadPool> infer_pool_;
    std::thread pre_thread_, post_thread_;
    std::atomic<bool> stop_{false};

    // 为每个推理线程保留独立的适配器实例
    std::vector<std::shared_ptr<IModelAdapter>> adapters_;

    // 视频捕获设备
    cv::VideoCapture capture_;
};