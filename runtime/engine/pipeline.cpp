/*
设计思路:这是框架最高层的调度逻辑，它串联起摄像头、预处理、推理线程、后处理和显示。

1.预处理线程：读摄像头帧，调用适配器的 Preprocess，将结果封装成任务放入 pre_queue。
2.推理线程池：每个线程持有独立的适配器实例（通过克隆或工厂），从 pre_queue 取任务，执行 Inference，再将任务放入 post_queue。
3.后处理/显示线程：从 post_queue 取任务，调用适配器的 Postprocess，结果叠加到帧上显示。

Clone()：在 IModelAdapter 中加入 virtual std::shared_ptr<IModelAdapter> Clone() = 0;，
    每个推理线程需要独立的 NPU context，因此需要深拷贝适配器状态（主要是重新 rknn_init 并复制参数）。
*/


// engine/pipeline.cpp
#include "pipeline.h"
#include <iostream>
#include <chrono>

// 为了使用正点原子的绘图函数（可选）
#include "../utils/image_drawing.h"  // 画框、写字
#include "../utils/image_utils.h"   // 图像格式转换（如需）

Pipeline::Pipeline(std::shared_ptr<IModelAdapter> base_adapter,
                   const std::string& model_path,
                   int num_infer_threads,
                   const std::vector<int>& npu_cores,
                   const std::vector<int>& cpu_cores,
                   int pre_queue_capacity,
                   int post_queue_capacity)
    : model_path_(model_path),
      num_infer_threads_(num_infer_threads),
      npu_cores_(npu_cores),
      cpu_cores_(cpu_cores),
      pre_queue_(pre_queue_capacity),
      post_queue_(post_queue_capacity) {

    // 1. 为每个推理线程创建独立的适配器实例（通过 Clone + Init）
    for (int i = 0; i < num_infer_threads_; ++i) {
        auto clone = base_adapter->Clone();
        if (!clone) {
            throw std::runtime_error("Adapter Clone() returned nullptr");
        }
        int core_mask = (i < npu_cores_.size()) ? npu_cores_[i] : 0x1;
        int ret = clone->Init(model_path_, core_mask);
        if (ret != 0) {
            throw std::runtime_error("Failed to init adapter clone");
        }
        adapters_.push_back(std::move(clone));
    }

    // 2. 创建推理线程池（绑定 CPU 大核）
    infer_pool_ = std::make_unique<ThreadPool>(num_infer_threads_, cpu_cores_);

    // 3. 打开摄像头（默认 /dev/video0，后续可从配置读取）
    capture_.open(0);
    if (!capture_.isOpened()) {
        throw std::runtime_error("Cannot open camera");
    }
}

Pipeline::~Pipeline() {
    Stop();
}

void Pipeline::Run() {
    // 启动预处理线程
    pre_thread_ = std::thread(&Pipeline::PreprocessLoop, this);
    // 为每个推理线程向线程池提交永久工作循环
    for (int i = 0; i < num_infer_threads_; ++i) {
        infer_pool_->Enqueue([this, i]() { InferenceLoop(i); });
    }
    // 启动后处理/显示线程
    post_thread_ = std::thread(&Pipeline::PostprocessLoop, this);

    // 等待所有线程结束（这里简单等待两个线程 join，推理线程在 stop 后会自动返回）
    if (pre_thread_.joinable()) pre_thread_.join();
    if (post_thread_.joinable()) post_thread_.join();
}

void Pipeline::Stop() {
    stop_ = true;
    // 唤醒可能阻塞的队列操作（Push/Pop 会检查 stop_ 或自然超时）
    // 强制 push 一个终止任务
    InferenceTask stop_task;
    stop_task.frame_id = -1;
    pre_queue_.Push(stop_task);  // 让预处理循环退出
}

void Pipeline::PreprocessLoop() {
    int frame_id = 0;
    while (!stop_) {
        cv::Mat frame;
        capture_ >> frame;
        if (frame.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        // 使用第一个适配器做预处理（预处理不涉及 NPU 上下文，可复用）
        int size = 0;
        uint8_t* buf = adapters_[0]->Preprocess(frame, size);
        if (!buf) continue;

        InferenceTask task;
        task.frame_id = frame_id++;
        task.original_frame = frame.clone();
        task.input_buf = buf;
        task.input_size = size;
        task.adapter = adapters_[0];  // 保留引用，确保内存有效

        pre_queue_.Push(std::move(task));
    }
    // 发送结束标记
    InferenceTask stop_task;
    stop_task.frame_id = -1;
    pre_queue_.Push(stop_task);
}

void Pipeline::InferenceLoop(int thread_id) {
    auto& local_adapter = adapters_[thread_id];
    while (true) {
        InferenceTask task = pre_queue_.Pop();
        if (task.frame_id == -1) { // 结束标记
            // 把标记放回，让其他推理线程也能退出（仅一份标记）
            // 注意：这里需要确保每个线程都收到一次结束标记。
            // 简单做法：使用原子计数器或每个线程 pop 到 -1 后直接退出，
            // 但需保证队列中有足够多标记。我们可以在 Stop 时 push 多个标记。
            break;
        }

        // 执行推理（使用本地适配器实例）
        int ret = local_adapter->Inference();
        if (ret != 0) {
            std::cerr << "Inference failed on thread " << thread_id << std::endl;
            continue;
        }

        // 将任务推到后处理队列
        post_queue_.Push(std::move(task));
    }
}

void Pipeline::PostprocessLoop() {
    while (true) {
        InferenceTask task = post_queue_.Pop();
        if (task.frame_id == -1) break;

        std::string result = task.adapter->Postprocess();
        DrawResult(task.original_frame, result);

        // 显示图像
        cv::imshow("EdgeAI Platform", task.original_frame);
        if (cv::waitKey(1) == 27) { // ESC 键退出
            Stop();
        }
    }
}

void Pipeline::DrawResult(cv::Mat& frame, const std::string& result_json) {
    // 简单示例：在左上角打印文字
    cv::putText(frame, result_json, cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    // 如果需要检测框绘制，可调用 utils/image_drawing 中的函数解析 JSON
}