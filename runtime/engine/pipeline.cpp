/*
 * engine/pipeline.cpp — 调度内核实现
 */
#include "pipeline.h"
#include "platform/logging.h"
#include <iostream>
#include <chrono>
#include <cctype>
#include <cerrno>
#include <sys/select.h>
#include <unistd.h>

// 合并本帧各槽 GetAdapterSignals（OR），供 UpdateAfterFrame 去抖。
AdapterSignals Pipeline::MergeSlotSignals(const std::vector<SlotInferenceResult>& slot_results) {
    AdapterSignals merged;
    for (const auto& r : slot_results) {
        merged.person_present = merged.person_present || r.signals.person_present;
        merged.face_detected = merged.face_detected || r.signals.face_detected;
        if (r.signals.avg_brightness > 0.0f) {
            merged.avg_brightness = r.signals.avg_brightness;
        }
        if (!r.signals.scene_label.empty()) {
            merged.scene_label = r.signals.scene_label;
        }
    }
    return merged;
}

// 对当前 enabled 槽顺序 Preprocess→Inference→Postprocess。
bool Pipeline::RunEnabledSlots(ModelCoordinator& coordinator, const cv::Mat& frame,
                               std::vector<SlotInferenceResult>& slot_results,
                               AdapterSignals& merged_signals) {
    slot_results.clear();
    auto slots = coordinator.GetEnabledSlotAdapters();
    if (slots.empty()) {
        return false;
    }
    for (const auto& entry : slots) {
        if (!entry.second) {
            continue;
        }
        int input_size = 0;
        if (entry.second->Preprocess(frame, input_size) == nullptr || input_size <= 0) {
            continue;
        }
        std::shared_ptr<void> model_output;
        if (entry.second->Inference(model_output) != 0) {
            model_output.reset();
        }
        SlotInferenceResult one;
        one.slot = entry.first;
        one.result_json = entry.second->Postprocess(model_output);
        one.signals = entry.second->GetAdapterSignals();
        slot_results.push_back(std::move(one));
    }
    merged_signals = MergeSlotSignals(slot_results);
    return !slot_results.empty();
}

// 打开相机、Init 协调器默认 yolo 槽（Init 内锁外 EnableSlot）、创建 infer 队列与线程池。
Pipeline::Pipeline(ModelCoordinator& coordinator,
                 CameraSource& camera,
                 FrameTransform& frame_transform,
                 ResultOverlay& overlay,
                 IDisplaySink& display,
                 std::shared_ptr<IModelAdapter> base_adapter,
                 const std::string& model_path,
                 int num_infer_threads,
                 const std::vector<int>& npu_cores,
                 bool single_thread)
    : coordinator_(coordinator),
      camera_(camera),
      frame_transform_(frame_transform),
      overlay_(overlay),
      display_(display),
      model_path_(model_path),
      num_infer_threads_(num_infer_threads),
      npu_cores_(npu_cores),
      post_queue_(4),
      single_thread_(single_thread) {

    if (!camera_.Open()) {
        LogFatal("Pipeline: cannot open camera");
        throw std::runtime_error("Cannot open input source");
    }

    if (!coordinator_.Init("yolo", base_adapter, model_path, npu_cores, num_infer_threads)) {
        camera_.Release();
        throw std::runtime_error(
            "ModelCoordinator failed to init adapters. Check model.yolo.path: " + model_path);
    }

    infer_pool_.reset(new ThreadPool(num_infer_threads_));
    infer_queues_.reserve(num_infer_threads_);
    for (int i = 0; i < num_infer_threads_; ++i) {
        infer_queues_.push_back(
            std::unique_ptr<BoundedQueue<InferenceTask>>(new BoundedQueue<InferenceTask>(2)));
    }

    processed_frames_ = 0;
    start_time_ = std::chrono::steady_clock::now();
}

// 析构时 Stop、join 预处理/推理 future，释放相机。
Pipeline::~Pipeline() {
    if (!stop_.load()) {
        Stop();
    }
    JoinWorkerThreads();
    camera_.Release();
}

void Pipeline::RegisterFactory(const std::string& name,
                               std::function<std::shared_ptr<IModelAdapter>()> factory,
                               const std::string& model_path) {
    // 工厂注册仅保存定义，不立即初始化，真正启用时由 ModelCoordinator 懒加载。
    coordinator_.RegisterFactory(name, std::move(factory), model_path);
}

void Pipeline::SetSwitchDebounceThresholds(int present_threshold, int absent_threshold) {
    // 透传给协调器，统一管理 person/idle 切换阈值。
    coordinator_.SetSwitchDebounceThresholds(present_threshold, absent_threshold);
}

void Pipeline::SetExternalStopFlag(std::atomic<bool>* flag) {
    // 允许 main 的信号处理通过外部标志触发优雅停止。
    external_stop_ = flag;
}

// 排空后处理队列，便于 Stop 时投递 quit 哨兵。
void Pipeline::DrainPostQueue() {
    // 无需处理内容，目标是清空积压以确保 quit 哨兵可成功入队。
    InferenceTask discarded;
    while (post_queue_.TryPop(discarded, 0)) {
    }
}

// 排空各推理队列中的积压帧，避免 quit_task 因队列满无法入队。
void Pipeline::DrainInferQueues() {
    // 清空每个推理队列，避免 stop 时因为队列满导致退出信号丢失。
    InferenceTask discarded;
    for (auto& q : infer_queues_) {
        while (q->TryPop(discarded, 0)) {
        }
    }
}

// 本管道 stop_ 或 main 注入的 g_stop_requested 任一为真则各循环退出。
bool Pipeline::ShouldStop() const {
    return stop_.load() || (external_stop_ != nullptr && external_stop_->load());
}

// 等待预处理与推理 worker 结束，再关闭显示。
void Pipeline::JoinWorkerThreads() {
    // 关闭顺序：先 join 线程，再 shutdown 显示，避免窗口线程竞争。
    if (pre_thread_.joinable()) {
        pre_thread_.join();
    }
    for (auto& fut : infer_futures_) {
        if (fut.valid()) {
            fut.wait();
        }
    }
    infer_futures_.clear();
    display_.Shutdown();
}

// 排空队列后向 infer/post 投递 frame_id=-1，通知各线程退出。
void Pipeline::PushQuitTasksBestEffort() {
    DrainPostQueue();
    DrainInferQueues();
    InferenceTask quit_task;
    quit_task.frame_id = -1;
    const int kTimeoutMs = 500;
    // 向每个推理队列都投递一个 quit，保证所有 worker 都能观察到退出事件。
    for (int i = 0; i < num_infer_threads_; ++i) {
        if (!infer_queues_[i]->TryPush(quit_task, kTimeoutMs)) {
            LogWarn("Pipeline::Stop: infer_queue %d quit_task not delivered (queue full)", i);
        }
    }
    if (!post_queue_.TryPush(quit_task, kTimeoutMs)) {
        LogWarn("Pipeline::Stop: post_queue quit_task not delivered (queue full)");
    }
}

// 置 stop、释放相机、排空队列后投递 quit 哨兵（frame_id=-1）。
void Pipeline::Stop() {
    if (stop_.exchange(true)) {
        return;
    }
    LogInfo("Pipeline::Stop: stop requested");
    coordinator_.GetLlmGreeting().AbortActiveGeneration();
    camera_.Release();
    PushQuitTasksBestEffort();
}

// 主入口：单线程直连显示，或多线程 pre→infer→主线程 post/显示。
void Pipeline::Run() {
    LogInfo("Pipeline::Run: begin (infer_threads=%d)", num_infer_threads_);
    coordinator_.GetLlmGreeting().LogStartupHint();
    if (!::isatty(STDIN_FILENO)) {
        LogWarn("Pipeline: stdin is not a TTY, YOU> input may be unavailable");
    } else {
        LogDebug("Pipeline: stdin is a TTY");
    }
    display_.Prepare();
    LogInfo("Pipeline::Run: display prepared");

    // 单线程模式用于调试；生产默认走 pre/infer/post 解耦流水线。
    if (single_thread_) {
        RunSingleThreaded();
    } else {
        pre_thread_ = std::thread(&Pipeline::PreprocessLoop, this);
        infer_futures_.reserve(static_cast<size_t>(num_infer_threads_));
        for (int i = 0; i < num_infer_threads_; ++i) {
            infer_futures_.push_back(
                infer_pool_->Enqueue([this, i]() { InferenceLoop(i); }));
        }
        RunPostprocessOnMainThread();
        JoinWorkerThreads();
    }
    camera_.Release();
    LogInfo("Pipeline::Run: exited normally");
}

// 单线程模式：读帧→推理→ProcessDisplayTask，无队列。
void Pipeline::RunSingleThreaded() {
    int frame_id = 0;
    while (!ShouldStop()) {
        PollTerminalPromptInput();
        cv::Mat frame;
        if (!camera_.ReadFrame(frame, &stop_)) {
            if (ShouldStop() || !camera_.IsOpened()) {
                break;
            }
            continue;
        }
        frame_transform_.Apply(frame);
        if (!frame_transform_.Validate(frame, frame_id)) {
            continue;
        }
        if (!frame.isContinuous()) {
            frame = frame.clone();
        }

        InferenceTask task;
        task.frame_id = frame_id++;
        task.original_frame = frame.clone();
        if (!RunEnabledSlots(coordinator_, frame, task.slot_results, task.merged_signals)) {
            PumpDisplayWhenIdle();
            continue;
        }
        if (!ProcessDisplayTask(task)) {
            break;
        }
    }
}

// 主线程：UpdateAfterFrame 改槽、多层 overlay、显示；返回 false 表示收到 quit。
bool Pipeline::ProcessDisplayTask(InferenceTask& task) {
    if (task.frame_id == -1) {
        return false;
    }
    if (!frame_transform_.Validate(task.original_frame, task.frame_id)) {
        return true;
    }

    // 先更新状态机再绘制，确保当前帧上的 badge 与门控状态一致。
    coordinator_.UpdateAfterFrame(task.merged_signals, task.original_frame);

    const std::string badge = coordinator_.GetEnabledSlotsBadge();
    if (badge != last_display_badge_) {
        LogDebug("Pipeline: enabled slots -> %s (frame_id=%d)", badge.c_str(), task.frame_id);
        last_display_badge_ = badge;
    }

    // scrfd 在场时抑制 yolo person 框，避免双层框造成视觉噪声。
    const bool suppress_yolo_person = coordinator_.ShouldSuppressYoloPersonDraw();
    for (const auto& layer : task.slot_results) {
        const bool suppress = (layer.slot == "yolo") && suppress_yolo_person;
        overlay_.Apply(task.original_frame, layer.result_json, suppress);
    }
    overlay_.DrawModelBadge(task.original_frame, badge);

    cv::Mat display = task.original_frame.isContinuous() ? task.original_frame : task.original_frame.clone();
    display_.Show(display);
    const int key = display_.PollKey(1);
    if (key == 27) {
        Stop();
    }
    PollTerminalPromptInput();

    const uint64_t count = ++processed_frames_;
    if (count == 1) {
        LogInfo("Pipeline: first frame displayed id=%d", task.frame_id);
    }
    return true;
}

// post 队列空时泵送 stdin/OpenCV，避免无帧时窗口与 ESC 失效。
void Pipeline::PumpDisplayWhenIdle() {
    const int key = display_.PollKey(1);
    if (key == 27) {
        Stop();
        return;
    }
    PollTerminalPromptInput();
}

void Pipeline::PollTerminalPromptInput() {
    // 非阻塞轮询 stdin：主循环每帧调用，不影响视觉推理吞吐。
    const int input_fd = STDIN_FILENO;
    fd_set rfds;
    FD_ZERO(&rfds);
    FD_SET(input_fd, &rfds);
    timeval tv;
    tv.tv_sec = 0;
    tv.tv_usec = 0;

    const int ret = select(input_fd + 1, &rfds, nullptr, nullptr, &tv);
    if (ret <= 0 || !FD_ISSET(input_fd, &rfds)) {
        return;
    }

    char buf[256];
    const ssize_t n = ::read(input_fd, buf, sizeof(buf));
    if (n <= 0) {
        if (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK) {
            LogWarn("Pipeline: prompt input read failed errno=%d", errno);
        }
        return;
    }

    // 行缓冲协议：读取任意字节流，遇到 CR/LF 才提交整行。
    for (ssize_t i = 0; i < n; ++i) {
        const char ch = buf[i];
        if (ch == '\r' || ch == '\n') {
            std::string line = terminal_input_buffer_;
            terminal_input_buffer_.clear();
            while (!line.empty() && std::isspace(static_cast<unsigned char>(line.back()))) {
                line.pop_back();
            }
            size_t start = 0;
            while (start < line.size() &&
                   std::isspace(static_cast<unsigned char>(line[start]))) {
                ++start;
            }
            if (start > 0) {
                line.erase(0, start);
            }
            if (line.empty()) {
                continue;
            }

            // 先回显 YOU>，再尝试提交门控，失败会打印 rejected 调试日志。
            LogUser("%s", line.c_str());
            const bool accepted = coordinator_.GetLlmGreeting().SubmitUserPrompt(line);
            if (accepted) {
                LogDebug("Pipeline: terminal prompt submitted (%zu chars)", line.size());
            } else {
                LogDebug("Pipeline: terminal prompt rejected");
            }
            continue;
        }

        // 串口常见退格字符（BS/DEL）
        if (ch == '\b' || static_cast<unsigned char>(ch) == 0x7f) {
            if (!terminal_input_buffer_.empty()) {
                terminal_input_buffer_.pop_back();
            }
            continue;
        }

        const unsigned char uch = static_cast<unsigned char>(ch);
        // 允许 UTF-8 多字节字符（>=0x80）进入输入缓冲，避免中文被错误丢弃。
        if (ch == '\t' || uch >= 0x20) {
            terminal_input_buffer_.push_back(ch);
        }
    }
}

// 预处理线程：读帧变换后 push 到 infer_queue[0]，满则丢帧。
void Pipeline::PreprocessLoop() {
    int frame_id = 0;
    const int kPushTimeoutMs = 100;
    while (!ShouldStop()) {
        if (!camera_.IsOpened()) {
            break;
        }
        cv::Mat frame;
        if (!camera_.ReadFrame(frame, &stop_)) {
            if (ShouldStop() || !camera_.IsOpened()) {
                break;
            }
            continue;
        }
        frame_transform_.Apply(frame);
        if (!frame_transform_.Validate(frame, frame_id)) {
            continue;
        }
        if (!frame.isContinuous()) {
            frame = frame.clone();
        }

        // 当前实现将预处理产物送入 infer_queue[0]，由对应 worker 消费。
        const int queue_index = 0;
        InferenceTask task;
        task.frame_id = frame_id++;
        task.original_frame = frame.clone();
        if (!infer_queues_[queue_index]->TryPush(std::move(task), kPushTimeoutMs)) {
            if (ShouldStop()) {
                break;
            }
            continue;
        }
    }
}

// 推理线程：RunEnabledSlots 后 push post_queue；Stop 且 post 满时退出。
void Pipeline::InferenceLoop(int thread_id) {
    const int kPopTimeoutMs = 200;
    const int kPushTimeoutMs = 100;
    while (!ShouldStop()) {
        InferenceTask task;
        // 每个线程固定消费自己的队列，减少锁竞争与任务窃取抖动。
        if (!infer_queues_[thread_id]->TryPop(task, kPopTimeoutMs)) {
            continue;
        }
        if (task.frame_id == -1) {
            break;
        }
        if (!frame_transform_.Validate(task.original_frame, task.frame_id)) {
            continue;
        }
        if (!RunEnabledSlots(coordinator_, task.original_frame, task.slot_results, task.merged_signals)) {
            continue;
        }
        if (!post_queue_.TryPush(std::move(task), kPushTimeoutMs)) {
            if (ShouldStop()) {
                break;
            }
            continue;
        }
    }
}

// 主线程消费 post_queue_，调用 ProcessDisplayTask 显示。
void Pipeline::RunPostprocessOnMainThread() {
    LogInfo("Pipeline::RunPostprocessOnMainThread: entered (main thread display)");
    const int kPopTimeoutMs = 200;
    while (!ShouldStop()) {
        PollTerminalPromptInput();
        InferenceTask task;
        if (!post_queue_.TryPop(task, kPopTimeoutMs)) {
            PumpDisplayWhenIdle();
            continue;
        }
        if (!ProcessDisplayTask(task)) {
            break;
        }
    }
    LogInfo("Pipeline::RunPostprocessOnMainThread: exiting");
}
