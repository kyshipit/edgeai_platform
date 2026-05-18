/*
 * platform/model_coordinator.h
 *
 * 模型协调器：模型池、懒加载、去抖、异步切换。
 * 切换语义（见 docs/设计要点.md）：
 *   yolo 哨兵 → person 稳定 → scrfd；文字区域稳定 → ppocr
 *   scrfd 无人脸 / person 离开 → yolo；ppocr 无字 → yolo
 */
#pragma once

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <future>
#include <functional>
#include <opencv2/opencv.hpp>

#include "engine/adapter_interface.h"
#include "shared_state.h"
#include "adapter_signals.h"

class ModelCoordinator {
public:
    ModelCoordinator();

    bool Init(const std::string& default_model_name,
              std::shared_ptr<IModelAdapter> default_adapter,
              const std::string& model_path,
              const std::vector<int>& npu_cores,
              int num_infer_threads);

    bool HasActiveAdapters() const;

    void RegisterModel(const std::string& name, std::shared_ptr<IModelAdapter> adapter);
    void RegisterModel(const std::string& name, std::shared_ptr<IModelAdapter> adapter, const std::string& model_path);
    bool RequestSwitchModelAsync(const std::string& name);
    void CheckPendingSwitch();
    void RegisterFactory(const std::string& name,
                         std::function<std::shared_ptr<IModelAdapter>()> factory,
                         const std::string& model_path);

    std::shared_ptr<IModelAdapter> GetAdapterForThread(int thread_id);
    SharedState& GetSharedState();
    std::string GetCurrentModelName() const;
    void SetSwitchDebounceThresholds(int present_threshold, int absent_threshold,
                                     int text_absent_threshold = 30);
    // to_ppocr: off | text（需 text_region_present）| frames（联调，满 N 帧即切）
    void SetToPpocrMode(const std::string& mode);
    void SetPpocrBackPolicy(bool auto_back_to_yolo, int min_ppocr_frames, bool back_on_no_text);

    void UpdateAfterFrame(const AdapterSignals& signals, const cv::Mat& frame);

private:
    struct ModelEntry {
        std::shared_ptr<IModelAdapter> prototype;
        std::string model_path;
    };

    bool EnsureActiveAdapters();
    bool TryCreateAdaptersForModel(const std::string& name, std::vector<std::shared_ptr<IModelAdapter>>& out_adapters);
    bool SwitchModel(const std::string& name);
    void MergeSignals(const AdapterSignals& signals);
    bool IsModelAvailableUnlocked(const std::string& name) const;
    bool EnsureModelInPoolUnlocked(const std::string& name);

    std::unordered_map<std::string, ModelEntry> model_pool_;
    std::unordered_map<std::string, std::function<std::shared_ptr<IModelAdapter>()>> factory_map_;
    std::unordered_map<std::string, std::string> factory_path_map_;
    std::vector<std::shared_ptr<IModelAdapter>> active_adapters_;
    std::future<bool> pending_switch_future_;
    std::string pending_switch_name_;
    std::string current_model_name_;
    std::string model_path_;
    std::vector<int> npu_cores_;
    int num_infer_threads_ = 0;

    SharedState shared_state_;
    AdapterSignals last_signals_;
    int person_present_count_ = 0;
    int person_absent_count_ = 0;
    int text_region_present_count_ = 0;
    int face_absent_count_ = 0;
    int present_threshold_ = 15;
    int absent_threshold_ = 30;
    int text_absent_count_ = 0;
    int text_absent_threshold_ = 30;
    std::string to_ppocr_mode_ = "text";
    int ppocr_lab_frame_count_ = 0;
    bool auto_back_to_yolo_ = true;
    bool back_to_yolo_on_no_text_ = true;
    int min_ppocr_frames_ = 90;
    int ppocr_frame_count_ = 0;
    mutable std::mutex mutex_;
};
