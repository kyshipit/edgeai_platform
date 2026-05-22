/*
 * engine/pipeline.cpp — 调度内核实现
 */
#include "pipeline.h"
#include "platform/logging.h"
#include <iostream>
#include <chrono>
#include <cctype>
#include <sys/select.h>
#include <unistd.h>

// 按方案 A 合并本帧各槽 GetAdapterSignals（OR），供 UpdateAfterFrame 去抖。
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

void Pipeline::RegisterModel(const std::string& name, std::shared_ptr<IModelAdapter> adapter,
                             const std::string& model_path) {
    coordinator_.RegisterModel(name, adapter, model_path);
}

void Pipeline::RegisterFactory(const std::string& name,
                               std::function<std::shared_ptr<IModelAdapter>()> factory,
                               const std::string& model_path) {
    coordinator_.RegisterFactory(name, std::move(factory), model_path);
}

void Pipeline::SetSwitchDebounceThresholds(int present_threshold, int absent_threshold) {
    coordinator_.SetSwitchDebounceThresholds(present_threshold, absent_threshold);
}

void Pipeline::SetExternalStopFlag(std::atomic<bool>* flag) {
    external_stop_ = flag;
}

// 排空后处理队列，便于 Stop 时投递 quit 哨兵。
void Pipeline::DrainPostQueue() {
    InferenceTask discarded;
    while (post_queue_.TryPop(discarded, 0)) {
    }
}

// 排空各推理队列中的积压帧，避免 quit_task 因队列满无法入队。
void Pipeline::DrainInferQueues() {
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
    camera_.Release();
    PushQuitTasksBestEffort();
}

// 主入口：单线程直连显示，或多线程 pre→infer→主线程 post/显示。
void Pipeline::Run() {
    LogInfo("Pipeline::Run: begin (infer_threads=%d)", num_infer_threads_);
    LogInfo("Pipeline: terminal input enabled, type prompt in terminal and press Enter to talk");
    display_.Prepare();
    LogInfo("Pipeline::Run: display prepared");

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

    coordinator_.UpdateAfterFrame(task.merged_signals, task.original_frame);

    const std::string badge = coordinator_.GetEnabledSlotsBadge();
    if (badge != last_display_badge_) {
        LogInfo("Pipeline: enabled slots -> %s (frame_id=%d)", badge.c_str(), task.frame_id);
        last_display_badge_ = badge;
    }

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
    if (badge.find("yolo") != std::string::npos && (count % 60 == 0)) {
        size_t det_chars = 0;
        for (const auto& layer : task.slot_results) {
            if (layer.slot == "yolo") {
                det_chars = layer.result_json.size();
                break;
            }
        }
        LogInfo("Pipeline: frame=%d person_present=%d yolo_det_bytes=%zu slots=%s",
                task.frame_id, task.merged_signals.person_present ? 1 : 0, det_chars, badge.c_str());
    }
    if (count % 100 == 0) {
        auto now = std::chrono::steady_clock::now();
        double elapsed_s = std::chrono::duration_cast<std::chrono::duration<double>>(now - start_time_).count();
        double fps = (elapsed_s > 0.0) ? (double)count / elapsed_s : 0.0;
        LogInfo("Pipeline: processed %lu frames, slots=%s, FPS=%.2f",
                (unsigned long)count, coordinator_.GetEnabledSlotsBadge().c_str(), fps);
    }
    return true;
}

void Pipeline::PollTerminalPromptInput() {
    fd_set rfds;
    FD_ZERO(&rfds);
    FD_SET(STDIN_FILENO, &rfds);
    timeval tv;
    tv.tv_sec = 0;
    tv.tv_usec = 0;

    const int ret = select(STDIN_FILENO + 1, &rfds, nullptr, nullptr, &tv);
    if (ret <= 0 || !FD_ISSET(STDIN_FILENO, &rfds)) {
        return;
    }

    char ch = '\0';
    const ssize_t n = ::read(STDIN_FILENO, &ch, 1);
    if (n <= 0) {
        return;
    }

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
            return;
        }

        const bool accepted = coordinator_.GetLlmGreeting().SubmitUserPrompt(line);
        if (accepted) {
            LogInfo("Pipeline: terminal prompt submitted (%zu chars)", line.size());
        } else {
            LogInfo("Pipeline: terminal prompt rejected (gate closed)");
        }
        return;
    }

    // 串口常见退格字符（BS/DEL）
    if (ch == '\b' || static_cast<unsigned char>(ch) == 0x7f) {
        if (!terminal_input_buffer_.empty()) {
            terminal_input_buffer_.pop_back();
        }
        return;
    }

    if (std::isprint(static_cast<unsigned char>(ch)) != 0 || ch == '\t') {
        terminal_input_buffer_.push_back(ch);
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
            continue;
        }
        if (!ProcessDisplayTask(task)) {
            break;
        }
    }
    LogInfo("Pipeline::RunPostprocessOnMainThread: exiting");
}
