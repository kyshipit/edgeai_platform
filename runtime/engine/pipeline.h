/*
 * engine/pipeline.h
 *
 * 【engine 层】调度内核：队列、线程池、任务流转；不含摄像头/画框/热切换策略实现。
 */
#pragma once
#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
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

struct InferenceTask {
    int frame_id = 0;
    cv::Mat original_frame;
    std::shared_ptr<IModelAdapter> adapter;
    std::string result_json;
    AdapterSignals signals;
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
    void SetSwitchDebounceThresholds(int present_threshold, int absent_threshold,
                                     int text_absent_threshold = 30);
    void SetToPpocrMode(const std::string& mode);
    void SetPpocrBackPolicy(bool auto_back_to_yolo, int min_ppocr_frames, bool back_on_no_text);
    void SetExternalStopFlag(std::atomic<bool>* flag);
    void SetOcrLogIntervalFrames(int interval);

private:
    bool ShouldStop() const;
    void JoinWorkerThreads();
    void PreprocessLoop();
    void InferenceLoop(int thread_id);
    void RunSingleThreaded();
    void RunPostprocessOnMainThread();
    bool ProcessDisplayTask(InferenceTask& task);
    void DrainPostQueue();
    void PushQuitTasksBestEffort();

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
    std::thread pre_thread_;
    std::atomic<bool> stop_{false};
    std::atomic<bool>* external_stop_ = nullptr;
    std::atomic<uint64_t> processed_frames_{0};
    bool single_thread_ = false;
    std::chrono::steady_clock::time_point start_time_;
    int ocr_log_interval_frames_ = 30;
    std::string last_display_model_;
    int frames_since_ocr_log_ = 0;
};
