/*
 * platform/model_coordinator.h
 *
 * 按需多槽：yolo / scrfd + llm 逻辑槽（无 OCR）。
 */
#pragma once

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include <functional>
#include <opencv2/opencv.hpp>

#include "engine/adapter_interface.h"
#include "shared_state.h"
#include "adapter_signals.h"
#include "llm_greeting.h"

enum class CoordinatorScene { Idle, Person };

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
    void RegisterModel(const std::string& name, std::shared_ptr<IModelAdapter> adapter,
                       const std::string& model_path);
    void RegisterFactory(const std::string& name,
                         std::function<std::shared_ptr<IModelAdapter>()> factory,
                         const std::string& model_path);

    void SetSlotOptions(bool yolo_always_on);
    void SetSceneDwellFrames(int frames);

    bool WarmupSlot(const std::string& name);

    std::vector<std::pair<std::string, std::shared_ptr<IModelAdapter>>> GetEnabledSlotAdapters() const;
    std::string GetEnabledSlotsBadge() const;
    std::string GetCurrentModelName() const;
    std::string GetCurrentScene() const;
    bool ShouldSuppressYoloPersonDraw() const;

    SharedState& GetSharedState();
    LlmGreeting& GetLlmGreeting();

    void SetSwitchDebounceThresholds(int present_threshold, int absent_threshold);

    void UpdateAfterFrame(const AdapterSignals& signals, const cv::Mat& frame);

private:
    struct ModelEntry {
        std::shared_ptr<IModelAdapter> prototype;
        std::string model_path;
    };

    struct SlotRuntime {
        std::shared_ptr<IModelAdapter> adapter;
    };

    struct SlotPlan {
        CoordinatorScene scene = CoordinatorScene::Idle;
        bool want_yolo = false;
        bool want_scrfd = false;
    };

    bool IsModelAvailableUnlocked(const std::string& name) const;
    bool EnsureModelInPoolUnlocked(const std::string& name);
    bool EnableSlot(const std::string& name);
    void DisableSlot(const std::string& name);
    SlotPlan BuildSlotPlanUnlocked();
    void ApplySlotPlan(const SlotPlan& plan);
    void MergeSignalsUnlocked(const AdapterSignals& signals);
    CoordinatorScene ComputeSceneUnlocked() const;
    static const char* SceneName(CoordinatorScene scene);
    int CoreMaskForSlot(const std::string& slot) const;
    static std::vector<std::string> OrderedSlotNames();

    std::unordered_map<std::string, ModelEntry> model_pool_;
    std::unordered_map<std::string, std::function<std::shared_ptr<IModelAdapter>()>> factory_map_;
    std::unordered_map<std::string, std::string> factory_path_map_;
    std::unordered_map<std::string, SlotRuntime> slot_runtimes_;
    std::unordered_map<std::string, SlotRuntime> warm_runtimes_;
    std::unordered_set<std::string> enabled_slots_;

    std::vector<int> npu_cores_;
    int num_infer_threads_ = 1;
    bool yolo_always_on_ = true;
    CoordinatorScene current_scene_ = CoordinatorScene::Idle;
    CoordinatorScene applied_scene_ = CoordinatorScene::Idle;
    CoordinatorScene pending_scene_ = CoordinatorScene::Idle;
    int scene_dwell_count_ = 0;
    int scene_dwell_frames_ = 5;
    std::string last_logged_scene_;

    SharedState shared_state_;
    AdapterSignals last_signals_;
    LlmGreeting llm_greeting_;

    int person_present_count_ = 0;
    int person_absent_count_ = 0;
    int present_threshold_ = 15;
    int absent_threshold_ = 30;

    mutable std::mutex mutex_;
};
