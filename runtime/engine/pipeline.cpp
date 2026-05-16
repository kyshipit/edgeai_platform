// engine/pipeline.cpp
#include "pipeline.h"
#include <iostream>
#include <chrono>
#include <cctype>
#include <algorithm>
#include <sstream>

Pipeline::Pipeline(std::shared_ptr<IModelAdapter> base_adapter,
                   const std::string& model_path,
                   int num_infer_threads,
                   const std::vector<int>& npu_cores,
                   const std::string& input_source)
        : model_path_(model_path),
            num_infer_threads_(num_infer_threads),
            npu_cores_(npu_cores),
            input_source_(input_source),
            post_queue_(2) {

    // 1. 克隆并初始化适配器实例。每个推理线程使用独立的模型适配器实例，
    //    这样可以避免不同线程在 RKNN 上下文和输入缓冲之间产生竞争。
    for (int i = 0; i < num_infer_threads_; ++i) {
        auto clone = base_adapter->Clone();
        if (!clone) throw std::runtime_error("Adapter Clone() returned nullptr");
        int core_mask = (i < npu_cores_.size()) ? npu_cores_[i] : 0x1;
        if (clone->Init(model_path_, core_mask) != 0) {
            throw std::runtime_error("Failed to init adapter clone");
        }
        adapters_.push_back(std::move(clone));
    }

    // 2. 创建推理线程池
    infer_pool_.reset(new ThreadPool(num_infer_threads_));

    // 2.1 初始化每个推理线程的队列
    infer_queues_.reserve(num_infer_threads_);
    for (int i = 0; i < num_infer_threads_; ++i) {
        infer_queues_.push_back(std::unique_ptr<BoundedQueue<InferenceTask>>(new BoundedQueue<InferenceTask>(2)));
    }

    // 3. 打开输入源（摄像头或文件）。数字字符串会被解释为摄像头索引。
    bool opened = false;
    if (!input_source_.empty() && std::all_of(input_source_.begin(), input_source_.end(), [](char c){ return std::isdigit(static_cast<unsigned char>(c)); })) {
        opened = capture_.open(std::stoi(input_source_));
    } else {
        opened = capture_.open(input_source_);
    }
    if (!opened) {
        throw std::runtime_error("Cannot open input source: " + input_source_);
    }
    // initialize FPS counters
    processed_frames_ = 0;
    start_time_ = std::chrono::steady_clock::now();
}

Pipeline::~Pipeline() {
    printf("Pipeline::~Pipeline: entering destructor\n");
    Stop();
}

void Pipeline::Run() {
    // 启动一个预处理线程、多个推理线程，以及一个后处理线程。
    pre_thread_ = std::thread(&Pipeline::PreprocessLoop, this);
    for (int i = 0; i < num_infer_threads_; ++i) {
        infer_pool_->Enqueue([this, i]() { InferenceLoop(i); });
    }
    post_thread_ = std::thread(&Pipeline::PostprocessLoop, this);

    // 等待后处理线程先退出，再等待预处理线程退出。
    if (post_thread_.joinable()) post_thread_.join();
    printf("Pipeline::Run: post_thread joined\n");
    printf("Pipeline::Run: destroying OpenCV windows\n");
    cv::destroyAllWindows();
    printf("Pipeline::Run: destroyAllWindows done\n");
    if (pre_thread_.joinable()) pre_thread_.join();
}

void Pipeline::Stop() {
    if (stop_) {
        printf("Pipeline::Stop: already requested, returning\n");
        return;
    }
    stop_ = true;
    printf("Pipeline::Stop: stop requested\n");

    // 释放摄像头，避免 PreprocessLoop 阻塞在 capture_ >> frame。
    if (capture_.isOpened()) {
        capture_.release();
        printf("Pipeline::Stop: capture released\n");
    }
    for (int i = 0; i < num_infer_threads_; ++i) {
        InferenceTask quit_task;
        quit_task.frame_id = -1;
        quit_task.input_buf = nullptr;
        quit_task.input_size = 0;
        quit_task.adapter.reset();
        infer_queues_[i]->Push(quit_task);
        printf("Pipeline::Stop: pushed quit_task to infer_queue %d\n", i);
    }
    {
        InferenceTask quit_task;
        quit_task.frame_id = -1;
        quit_task.input_buf = nullptr;
        quit_task.input_size = 0;
        quit_task.adapter.reset();
        post_queue_.Push(quit_task);
        printf("Pipeline::Stop: pushed quit_task to post_queue\n");
        printf("Pipeline::Stop: sentinels pushed\n");
    }
}

void Pipeline::PreprocessLoop() {
    int frame_id = 0;
    while (!stop_) {
        cv::Mat frame;
        capture_ >> frame;
        if (frame.empty()) continue;

        int size = 0;
        int adapter_index = frame_id % num_infer_threads_;
        auto adapter = adapters_[adapter_index];
        uint8_t* buf = adapter->Preprocess(frame, size);
        if (buf == nullptr || size <= 0) {
            printf("Pipeline: Preprocess failed for frame %d, skipping inference\n", frame_id);
            continue;
        }
        InferenceTask task;
        task.frame_id = frame_id++;
        task.original_frame = frame.clone();
        task.input_buf = buf;
        task.input_size = size;
        task.adapter = adapter;
        // Preprocess 完成后，把任务投递到对应线程的推理队列。
        // 这里采用 round-robin 绑定模型实例，以保证每个适配器使用独立 NPU 上下文。
        infer_queues_[adapter_index]->Push(task);
    }
}

void Pipeline::InferenceLoop(int thread_id) {
    while (true) {
        InferenceTask task = infer_queues_[thread_id]->Pop();
        if (task.frame_id == -1) break;
        if (task.adapter) {
            int ret = task.adapter->Inference(task.model_output);
            if (ret != 0) {
                printf("Pipeline::InferenceLoop: adapter inference failed for frame %d ret=%d\n", task.frame_id, ret);
                task.model_output.reset();
            }
        }
        post_queue_.Push(std::move(task));
    }
}

void Pipeline::PostprocessLoop() {
    printf("Pipeline::PostprocessLoop: entered loop\n");
    while (true) {
        InferenceTask task = post_queue_.Pop();
        if (task.frame_id == -1) break;
        // 后处理线程从队列取出推理完成的任务，调用模型适配器的 Postprocess
        // 将结果画到原始帧上，并显示到窗口中。
        std::string result = task.adapter->Postprocess(task.model_output);
        DrawResult(task.original_frame, result);
        cv::imshow("EdgeAI Platform", task.original_frame);
        if (cv::waitKey(1) == 27) {
            printf("Pipeline::PostprocessLoop: ESC pressed, requesting stop\n");
            Stop();
        }
        // update FPS counters and report periodically
        uint64_t count = ++processed_frames_;
        if (count % 100 == 0) {
            auto now = std::chrono::steady_clock::now();
            double elapsed_s = std::chrono::duration_cast<std::chrono::duration<double>>(now - start_time_).count();
            double fps = (elapsed_s > 0.0) ? (double)count / elapsed_s : 0.0;
            printf("Pipeline: processed %lu frames, avg FPS=%.2f\n", (unsigned long)count, fps);
        }
    }
    printf("Pipeline::PostprocessLoop: exiting loop\n");
}

void Pipeline::DrawResult(cv::Mat& frame, const std::string& result_json) {
    // DrawResult 解析适配器返回的结果文本格式，将每个检测框绘制到帧上。
    // 约定格式：label x1 y1 x2 y2 score
    if (result_json.empty()) {
        return;
    }

    std::istringstream ss(result_json);
    std::string line;
    while (std::getline(ss, line)) {
        if (line.empty()) {
            continue;
        }

        std::istringstream line_stream(line);
        std::string label;
        int x1, y1, x2, y2;
        float score;
        if (!(line_stream >> label >> x1 >> y1 >> x2 >> y2 >> score)) {
            continue;
        }

        x1 = std::max(0, std::min(x1, frame.cols - 1));
        y1 = std::max(0, std::min(y1, frame.rows - 1));
        x2 = std::max(0, std::min(x2, frame.cols - 1));
        y2 = std::max(0, std::min(y2, frame.rows - 1));

        cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0), 2);

        std::ostringstream text_stream;
        text_stream << label << " " << static_cast<int>(score * 100) << "%";
        std::string text = text_stream.str();

        int baseline = 0;
        cv::Size text_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        cv::Point text_origin(x1, std::max(y1 - 5, text_size.height + 5));
        cv::rectangle(frame,
                      cv::Point(text_origin.x, text_origin.y - text_size.height - baseline),
                      cv::Point(text_origin.x + text_size.width, text_origin.y + 2),
                      cv::Scalar(0, 0, 0),
                      cv::FILLED);
        cv::putText(frame, text, text_origin, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    }
}