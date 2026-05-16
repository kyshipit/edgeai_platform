// engine/pipeline.h
#pragma once
#include <memory>
#include <vector>
#include <thread>
#include <atomic>
#include <opencv2/opencv.hpp> // 完整的 OpenCV 功能
#include "bounded_queue.h"
#include "thread_pool.h"
#include "adapter_interface.h"

// 流水线任务包。每个任务包含原始帧和对应的预处理结果指针，
// 以及用于推理/后处理的模型适配器实例。
struct InferenceTask {
    int frame_id;
    cv::Mat original_frame;                 
    uint8_t* input_buf = nullptr;           
    int input_size = 0;
    std::shared_ptr<void> model_output;
    std::shared_ptr<IModelAdapter> adapter; 
};

class Pipeline {
public:
    Pipeline(std::shared_ptr<IModelAdapter> base_adapter,
             const std::string& model_path,
             int num_infer_threads,
             const std::vector<int>& npu_cores,
             const std::string& input_source = "0");
    ~Pipeline();

    void Run();   // 启动流水线，阻塞直到停止
    void Stop();  // 请求停止

private:
    void PreprocessLoop();
    void InferenceLoop(int thread_id);
    void PostprocessLoop();
    void DrawResult(cv::Mat& frame, const std::string& result_json);

    std::string input_source_;
    std::string model_path_;
    int num_infer_threads_;
    std::vector<int> npu_cores_;

    // 每个推理线程使用独立队列，避免竞争和绑定对应的模型实例
    std::vector<std::unique_ptr<BoundedQueue<InferenceTask>>> infer_queues_;
    BoundedQueue<InferenceTask> post_queue_;

    std::unique_ptr<ThreadPool> infer_pool_;
    std::thread pre_thread_, post_thread_;
    std::atomic<bool> stop_{false};
    std::atomic<uint64_t> processed_frames_{0};
    std::chrono::steady_clock::time_point start_time_;

    std::vector<std::shared_ptr<IModelAdapter>> adapters_;
    cv::VideoCapture capture_;
};