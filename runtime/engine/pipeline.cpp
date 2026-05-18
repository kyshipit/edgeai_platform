/*
 * engine/pipeline.cpp — 调度内核实现
 */
#include "pipeline.h"
#include "platform/logging.h"
#include <iostream>
#include <chrono>

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
            "ModelCoordinator failed to init adapters. Check model.path: " + model_path);
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

void Pipeline::SetSwitchDebounceThresholds(int present_threshold, int absent_threshold,
                                           int text_absent_threshold) {
    coordinator_.SetSwitchDebounceThresholds(present_threshold, absent_threshold, text_absent_threshold);
}

void Pipeline::SetToPpocrMode(const std::string& mode) {
    coordinator_.SetToPpocrMode(mode);
}

void Pipeline::SetPpocrBackPolicy(bool auto_back_to_yolo, int min_ppocr_frames, bool back_on_no_text) {
    coordinator_.SetPpocrBackPolicy(auto_back_to_yolo, min_ppocr_frames, back_on_no_text);
}

void Pipeline::SetExternalStopFlag(std::atomic<bool>* flag) {
    external_stop_ = flag;
}

void Pipeline::SetOcrLogIntervalFrames(int interval) {
    ocr_log_interval_frames_ = interval > 0 ? interval : 30;
}

void Pipeline::DrainPostQueue() {
    InferenceTask discarded;
    while (post_queue_.TryPop(discarded, 0)) {
    }
}

bool Pipeline::ShouldStop() const {
    return stop_.load() || (external_stop_ != nullptr && external_stop_->load());
}

void Pipeline::JoinWorkerThreads() {
    if (pre_thread_.joinable()) {
        pre_thread_.join();
    }
    display_.Shutdown();
}

void Pipeline::PushQuitTasksBestEffort() {
    DrainPostQueue();
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

void Pipeline::Stop() {
    if (stop_.exchange(true)) {
        return;
    }
    LogInfo("Pipeline::Stop: stop requested");
    camera_.Release();
    PushQuitTasksBestEffort();
}

void Pipeline::Run() {
    LogInfo("Pipeline::Run: begin (infer_threads=%d)", num_infer_threads_);
    display_.Prepare();
    LogInfo("Pipeline::Run: display prepared");

    if (single_thread_) {
        RunSingleThreaded();
    } else {
        pre_thread_ = std::thread(&Pipeline::PreprocessLoop, this);
        for (int i = 0; i < num_infer_threads_; ++i) {
            infer_pool_->Enqueue([this, i]() { InferenceLoop(i); });
        }
        RunPostprocessOnMainThread();
        JoinWorkerThreads();
    }
    camera_.Release();
    LogInfo("Pipeline::Run: exited normally");
}

void Pipeline::RunSingleThreaded() {
    int frame_id = 0;
    while (!ShouldStop()) {
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

        auto adapter = coordinator_.GetAdapterForThread(0);
        if (!adapter) {
            continue;
        }

        int size = 0;
        InferenceTask task;
        task.frame_id = frame_id++;
        task.original_frame = frame.clone();
        task.adapter = adapter;
        if (adapter->Preprocess(frame, size) == nullptr || size <= 0) {
            continue;
        }

        std::shared_ptr<void> model_output;
        if (adapter->Inference(model_output) != 0) {
            model_output.reset();
        }
        task.result_json = adapter->Postprocess(model_output);
        task.signals = adapter->GetAdapterSignals();
        if (!ProcessDisplayTask(task)) {
            break;
        }
    }
}

bool Pipeline::ProcessDisplayTask(InferenceTask& task) {
    if (task.frame_id == -1) {
        return false;
    }
    if (!frame_transform_.Validate(task.original_frame, task.frame_id)) {
        return true;
    }

    coordinator_.UpdateAfterFrame(task.signals, task.original_frame);

    const std::string model_name = coordinator_.GetCurrentModelName();
    if (model_name != last_display_model_) {
        LogInfo("Pipeline: display model -> %s (frame_id=%d)", model_name.c_str(), task.frame_id);
        last_display_model_ = model_name;
        frames_since_ocr_log_ = ocr_log_interval_frames_;
    }
    if (model_name == "ppocr") {
        ++frames_since_ocr_log_;
        if (frames_since_ocr_log_ >= ocr_log_interval_frames_) {
            frames_since_ocr_log_ = 0;
            ResultOverlay::LogOcrResultsToTerminal(task.result_json);
        }
    }

    overlay_.Apply(task.original_frame, task.result_json);
    overlay_.DrawModelBadge(task.original_frame, model_name);

    cv::Mat display = task.original_frame.isContinuous() ? task.original_frame : task.original_frame.clone();
    display_.Show(display);
    if (display_.PollKey(1) == 27) {
        Stop();
    }

    const uint64_t count = ++processed_frames_;
    if (count == 1) {
        LogInfo("Pipeline: first frame displayed id=%d", task.frame_id);
    }
    if (model_name == "yolo" && (count % 60 == 0)) {
        LogInfo("Pipeline: yolo frame=%d person_present=%d det_lines=%zu",
                task.frame_id, task.signals.person_present ? 1 : 0, task.result_json.size());
    }
    if (count % 100 == 0) {
        auto now = std::chrono::steady_clock::now();
        double elapsed_s = std::chrono::duration_cast<std::chrono::duration<double>>(now - start_time_).count();
        double fps = (elapsed_s > 0.0) ? (double)count / elapsed_s : 0.0;
        LogInfo("Pipeline: processed %lu frames, model=%s, FPS=%.2f",
                (unsigned long)count, coordinator_.GetCurrentModelName().c_str(), fps);
    }
    return true;
}

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

        const int adapter_index = frame_id % num_infer_threads_;
        auto adapter = coordinator_.GetAdapterForThread(adapter_index);
        if (!adapter) {
            continue;
        }
        InferenceTask task;
        task.frame_id = frame_id++;
        task.original_frame = frame.clone();
        task.adapter = adapter;
        if (!infer_queues_[adapter_index]->TryPush(std::move(task), kPushTimeoutMs)) {
            if (ShouldStop()) {
                break;
            }
            continue;
        }
    }
}

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
        if (!task.adapter) {
            continue;
        }
        if (!frame_transform_.Validate(task.original_frame, task.frame_id)) {
            continue;
        }
        int input_size = 0;
        if (task.adapter->Preprocess(task.original_frame, input_size) == nullptr || input_size <= 0) {
            continue;
        }
        std::shared_ptr<void> model_output;
        if (task.adapter->Inference(model_output) != 0) {
            model_output.reset();
        }
        task.result_json = task.adapter->Postprocess(model_output);
        task.signals = task.adapter->GetAdapterSignals();
        if (!post_queue_.TryPush(std::move(task), kPushTimeoutMs)) {
            if (ShouldStop()) {
                break;
            }
        }
    }
}

void Pipeline::RunPostprocessOnMainThread() {
    LogInfo("Pipeline::RunPostprocessOnMainThread: entered (main thread display)");
    const int kPopTimeoutMs = 200;
    while (!ShouldStop()) {
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
