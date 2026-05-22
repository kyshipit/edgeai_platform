/*
 * engine/pipeline.h
 *
 * 【engine 层】调度内核：队列、线程池、任务流转。
 */
#pragma once
#include <atomic>
#include <chrono>
#include <future>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <opencv2/opencv.hpp>

#include "bounded_queue.h"
#include "thread_pool.h"
#include "adapter_interface.h"
#include "platform/adapter_signals.h"
#include "platform/model_coordinator.h"
#include "io/camera_source.h"
#include "io/frame_transform.h"
#include "viz/display_sink.h"
#include "viz/result_overlay.h"

struct SlotInferenceResult {
    std::string slot;
    std::string result_json;
    AdapterSignals signals;
};

struct InferenceTask {
    int frame_id = 0;
    cv::Mat original_frame;
    std::vector<SlotInferenceResult> slot_results;
    AdapterSignals merged_signals;
};

class Pipeline {
public:
    Pipeline(ModelCoordinator& coordinator,
             CameraSource& camera,
             FrameTransform& frame_transform,
             ResultOverlay& overlay,
             IDisplaySink& display,
             std::shared_ptr<IModelAdapter> base_adapter,
             const std::string& model_path,
             int num_infer_threads,
             const std::vector<int>& npu_cores,
             bool single_thread = false);
    ~Pipeline();

    void Run();
    void Stop();

    void RegisterModel(const std::string& name, std::shared_ptr<IModelAdapter> adapter, const std::string& model_path);
    void RegisterFactory(const std::string& name,
                         std::function<std::shared_ptr<IModelAdapter>()> factory,
                         const std::string& model_path);
    void SetSwitchDebounceThresholds(int present_threshold, int absent_threshold);
    void SetExternalStopFlag(std::atomic<bool>* flag);

private:
    bool ShouldStop() const;
    void JoinWorkerThreads();
    void PreprocessLoop();
    void InferenceLoop(int thread_id);
    void RunSingleThreaded();
    void RunPostprocessOnMainThread();
    bool ProcessDisplayTask(InferenceTask& task);
    void DrainPostQueue();
    void DrainInferQueues();
    void PushQuitTasksBestEffort();
    void PollTerminalPromptInput();

    ModelCoordinator& coordinator_;
    CameraSource& camera_;
    FrameTransform& frame_transform_;
    ResultOverlay& overlay_;
    IDisplaySink& display_;

    std::string model_path_;
    int num_infer_threads_;
    std::vector<int> npu_cores_;

    std::vector<std::unique_ptr<BoundedQueue<InferenceTask>>> infer_queues_;
    BoundedQueue<InferenceTask> post_queue_;

    std::unique_ptr<ThreadPool> infer_pool_;
    std::vector<std::future<void>> infer_futures_;
    std::thread pre_thread_;
    std::atomic<bool> stop_{false};
    std::atomic<bool>* external_stop_ = nullptr;
    std::atomic<uint64_t> processed_frames_{0};
    bool single_thread_ = false;
    std::chrono::steady_clock::time_point start_time_;
    std::string last_display_badge_;
    std::string terminal_input_buffer_;

    static AdapterSignals MergeSlotSignals(const std::vector<SlotInferenceResult>& slot_results);
    static bool RunEnabledSlots(ModelCoordinator& coordinator, const cv::Mat& frame,
                                std::vector<SlotInferenceResult>& slot_results,
                                AdapterSignals& merged_signals);
};
