/*
 * platform/model_coordinator.cpp
 *
 * 模型协调器实现：模型池、factory 懒加载、去抖切换、异步 Clone+Init。
 */
#include "model_coordinator.h"
#include "logging.h"
#include <iostream>
#include <future>
#include <chrono>

namespace {

// default.yaml 中 npu_cores: [0,1,2] 表示核索引；RKNN 需要 1/2/4 掩码。
int ToRknnCoreMask(int core_cfg) {
    if (core_cfg >= 1 && core_cfg <= 4 && (core_cfg & (core_cfg - 1)) == 0) {
        return core_cfg;
    }
    if (core_cfg >= 0 && core_cfg <= 2) {
        return 1 << core_cfg;
    }
    return 1;
}

}  // namespace

ModelCoordinator::ModelCoordinator() {
}

bool ModelCoordinator::Init(const std::string& default_model_name,
                          std::shared_ptr<IModelAdapter> default_adapter,
                          const std::string& model_path,
                          const std::vector<int>& npu_cores,
                          int num_infer_threads) {
    std::lock_guard<std::mutex> lock(mutex_);
    current_model_name_ = default_model_name;
    model_path_ = model_path;
    npu_cores_ = npu_cores;
    num_infer_threads_ = num_infer_threads;

    if (default_adapter) {
        model_pool_[default_model_name] = {default_adapter, model_path};
    }
    LogInfo("ModelCoordinator: initialized default model '%s' path='%s' threads=%d",
            default_model_name.c_str(), model_path.c_str(), num_infer_threads);
    if (!EnsureActiveAdapters()) {
        LogError("ModelCoordinator: failed to create active adapters for '%s', check model path '%s' and NPU cores",
                 default_model_name.c_str(), model_path.c_str());
        return false;
    }
    return true;
}

bool ModelCoordinator::HasActiveAdapters() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return !active_adapters_.empty();
}

void ModelCoordinator::RegisterModel(const std::string& name, std::shared_ptr<IModelAdapter> adapter) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (adapter) {
        model_pool_[name] = {adapter, model_path_};
    }
}

void ModelCoordinator::RegisterModel(const std::string& name, std::shared_ptr<IModelAdapter> adapter, const std::string& model_path) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (adapter) {
        model_pool_[name] = {adapter, model_path};
    }
}

// 仅登记工厂与路径，不创建实例；首次切换时由 EnsureModelInPoolUnlocked 调用。
void ModelCoordinator::RegisterFactory(const std::string& name,
                                       std::function<std::shared_ptr<IModelAdapter>()> factory,
                                       const std::string& model_path) {
    std::lock_guard<std::mutex> lock(mutex_);
    factory_map_[name] = std::move(factory);
    factory_path_map_[name] = model_path;
    LogInfo("ModelCoordinator: registered factory for '%s' path='%s'", name.c_str(), model_path.c_str());
}

bool ModelCoordinator::IsModelAvailableUnlocked(const std::string& name) const {
    return model_pool_.count(name) > 0 || factory_map_.count(name) > 0;
}

// factory 懒加载：若池中无 prototype，则调用 factory() 创建并写入 model_pool_。
bool ModelCoordinator::EnsureModelInPoolUnlocked(const std::string& name) {
    auto pool_it = model_pool_.find(name);
    if (pool_it != model_pool_.end()) {
        return pool_it->second.prototype != nullptr;
    }
    auto factory_it = factory_map_.find(name);
    if (factory_it == factory_map_.end()) {
        return false;
    }
    auto adapter = factory_it->second();
    if (!adapter) {
        LogWarn("ModelCoordinator: factory for '%s' returned nullptr", name.c_str());
        return false;
    }
    std::string path = factory_path_map_[name];
    model_pool_[name] = {adapter, path};
    LogInfo("ModelCoordinator: lazy-loaded prototype for '%s' path='%s'", name.c_str(), path.c_str());
    return true;
}

// 从 prototype Clone 出 num_infer_threads_ 份，分别 Init 到不同 NPU 核，供多线程推理。
bool ModelCoordinator::TryCreateAdaptersForModel(const std::string& name, std::vector<std::shared_ptr<IModelAdapter>>& out_adapters) {
    std::shared_ptr<IModelAdapter> prototype;
    std::string model_path;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = model_pool_.find(name);
        if (it == model_pool_.end() || !it->second.prototype) {
            return false;
        }
        prototype = it->second.prototype;
        model_path = it->second.model_path;
    }

    out_adapters.clear();
    for (int i = 0; i < num_infer_threads_; ++i) {
        auto clone = prototype->Clone();
        if (!clone) {
            LogWarn("ModelCoordinator: Clone() returned nullptr for model %s", name.c_str());
            return false;
        }
        int core_cfg = (i < static_cast<int>(npu_cores_.size())) ? npu_cores_[i] : 0;
        int core_mask = ToRknnCoreMask(core_cfg);
        if (clone->Init(model_path, core_mask) != 0) {
            LogWarn("ModelCoordinator: failed to init clone %d for model %s core_mask=0x%x path=%s",
                    i, name.c_str(), core_mask, model_path.c_str());
            return false;
        }
        out_adapters.push_back(std::move(clone));
    }
    return !out_adapters.empty();
}

std::shared_ptr<IModelAdapter> ModelCoordinator::GetAdapterForThread(int thread_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (active_adapters_.empty()) {
        LogWarn("ModelCoordinator: no active adapters (current_model='%s')",
                current_model_name_.c_str());
        return nullptr;
    }
    int index = thread_id % active_adapters_.size();
    return active_adapters_[index];
}

SharedState& ModelCoordinator::GetSharedState() {
    return shared_state_;
}

std::string ModelCoordinator::GetCurrentModelName() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return current_model_name_;
}

void ModelCoordinator::MergeSignals(const AdapterSignals& signals) {
    if (current_model_name_ == "yolo") {
        shared_state_.person_present = signals.person_present;
        shared_state_.text_region_present = signals.text_region_present;
    }
    if (current_model_name_ == "scrfd") {
        shared_state_.face_detected = signals.face_detected;
    }
    if (current_model_name_ == "ppocr") {
        shared_state_.text_detected = signals.text_detected;
    }
    if (signals.avg_brightness > 0.0f) {
        shared_state_.avg_brightness = signals.avg_brightness;
    }
    if (!signals.scene_label.empty()) {
        shared_state_.scene_label = signals.scene_label;
    }
    last_signals_ = signals;
}

// 每帧决策入口：合并信号 → 去抖计数 → 在锁外发起异步切换（避免死锁）。
void ModelCoordinator::UpdateAfterFrame(const AdapterSignals& signals, const cv::Mat& frame) {
    (void)frame;
    CheckPendingSwitch();

    std::string switch_target;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        MergeSignals(signals);

        if (current_model_name_ == "yolo") {
            if (signals.person_present) {
                person_present_count_++;
                person_absent_count_ = 0;
            } else {
                person_absent_count_++;
                person_present_count_ = 0;
            }

            if (person_present_count_ >= present_threshold_ &&
                IsModelAvailableUnlocked("scrfd")) {
                LogInfo("ModelCoordinator: person debounced %d/%d -> request scrfd",
                        person_present_count_, present_threshold_);
                switch_target = "scrfd";
            } else if (to_ppocr_mode_ == "text") {
                if (signals.text_region_present) {
                    text_region_present_count_++;
                } else {
                    text_region_present_count_ = 0;
                }
                if (text_region_present_count_ >= present_threshold_ &&
                    IsModelAvailableUnlocked("ppocr")) {
                    switch_target = "ppocr";
                }
            } else if (to_ppocr_mode_ == "frames") {
                ppocr_lab_frame_count_++;
                if (ppocr_lab_frame_count_ >= present_threshold_ &&
                    IsModelAvailableUnlocked("ppocr")) {
                    switch_target = "ppocr";
                }
            }
        } else if (current_model_name_ == "scrfd") {
            if (signals.face_detected) {
                face_absent_count_ = 0;
            } else {
                face_absent_count_++;
            }
            if (face_absent_count_ >= absent_threshold_ &&
                IsModelAvailableUnlocked("yolo")) {
                switch_target = "yolo";
            }
        } else if (current_model_name_ == "ppocr") {
            ppocr_frame_count_++;
            if (signals.text_detected) {
                text_absent_count_ = 0;
            } else {
                text_absent_count_++;
            }
            if (auto_back_to_yolo_ && back_to_yolo_on_no_text_ &&
                ppocr_frame_count_ >= min_ppocr_frames_ &&
                text_absent_count_ >= text_absent_threshold_ &&
                IsModelAvailableUnlocked("yolo")) {
                switch_target = "yolo";
            }
        }
    }

    if (!switch_target.empty() && RequestSwitchModelAsync(switch_target)) {
        LogInfo("ModelCoordinator: requested async switch to %s", switch_target.c_str());
    }
}

// 轮询异步切换 future：成功则已 swap active_adapters_；失败则保留原模型。
void ModelCoordinator::CheckPendingSwitch() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!pending_switch_future_.valid()) {
        return;
    }
    auto status = pending_switch_future_.wait_for(std::chrono::milliseconds(0));
    if (status != std::future_status::ready) {
        return;
    }
    bool success = false;
    try {
        success = pending_switch_future_.get();
    } catch (const std::exception& e) {
        LogWarn("ModelCoordinator: pending model switch threw exception: %s", e.what());
    } catch (...) {
        LogWarn("ModelCoordinator: pending model switch threw unknown exception");
    }
    if (success) {
        LogInfo("ModelCoordinator: async switch to %s succeeded", pending_switch_name_.c_str());
        if (pending_switch_name_ == "ppocr") {
            ppocr_frame_count_ = 0;
            text_absent_count_ = 0;
        } else if (pending_switch_name_ == "scrfd") {
            face_absent_count_ = 0;
            person_present_count_ = 0;
        } else if (pending_switch_name_ == "yolo") {
            ppocr_frame_count_ = 0;
            ppocr_lab_frame_count_ = 0;
            person_present_count_ = 0;
            person_absent_count_ = 0;
            text_region_present_count_ = 0;
            face_absent_count_ = 0;
        }
    } else {
        LogWarn("ModelCoordinator: async switch to %s failed, retaining %s", pending_switch_name_.c_str(), current_model_name_.c_str());
    }
    pending_switch_name_.clear();
    pending_switch_future_ = std::future<bool>();
}

// 异步切换：后台 EnsureModelInPool → TryCreateAdaptersForModel → swap active_adapters_。
// 失败时保留 current_model_name_ 与原有 active_adapters_。
bool ModelCoordinator::RequestSwitchModelAsync(const std::string& name) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (name == current_model_name_) {
            return false;
        }
        if (pending_switch_future_.valid()) {
            auto status = pending_switch_future_.wait_for(std::chrono::milliseconds(0));
            if (status == std::future_status::timeout) {
                LogInfo("ModelCoordinator: switch to %s already pending", name.c_str());
                return false;
            }
        }
        if (!IsModelAvailableUnlocked(name)) {
            LogWarn("ModelCoordinator: model not available (no pool entry or factory): %s", name.c_str());
            return false;
        }
        pending_switch_name_ = name;
    }

    pending_switch_future_ = std::async(std::launch::async, [this, name]() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!EnsureModelInPoolUnlocked(name)) {
                LogWarn("ModelCoordinator: lazy load failed for '%s'", name.c_str());
                return false;
            }
        }
        std::vector<std::shared_ptr<IModelAdapter>> new_adapters;
        if (!TryCreateAdaptersForModel(name, new_adapters)) {
            return false;
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            active_adapters_.swap(new_adapters);
            current_model_name_ = name;
        }
        return true;
    });
    return true;
}

void ModelCoordinator::SetToPpocrMode(const std::string& mode) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (mode == "off" || mode == "text" || mode == "frames") {
        to_ppocr_mode_ = mode;
    } else {
        to_ppocr_mode_ = "text";
        LogWarn("ModelCoordinator: unknown to_ppocr mode '%s', use text", mode.c_str());
    }
    ppocr_lab_frame_count_ = 0;
    text_region_present_count_ = 0;
    LogInfo("ModelCoordinator: to_ppocr mode='%s' (person -> scrfd, not ppocr)", to_ppocr_mode_.c_str());
}

void ModelCoordinator::SetPpocrBackPolicy(bool auto_back_to_yolo, int min_ppocr_frames, bool back_on_no_text) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto_back_to_yolo_ = auto_back_to_yolo;
    min_ppocr_frames_ = min_ppocr_frames > 0 ? min_ppocr_frames : 90;
    back_to_yolo_on_no_text_ = back_on_no_text;
    LogInfo("ModelCoordinator: ppocr auto_back=%d min_stay=%d back_on_no_text=%d text_absent_th=%d",
            auto_back_to_yolo_ ? 1 : 0, min_ppocr_frames_, back_to_yolo_on_no_text_ ? 1 : 0,
            text_absent_threshold_);
}

void ModelCoordinator::SetSwitchDebounceThresholds(int present_threshold, int absent_threshold,
                                                   int text_absent_threshold) {
    std::lock_guard<std::mutex> lock(mutex_);
    present_threshold_ = present_threshold;
    absent_threshold_ = absent_threshold;
    text_absent_threshold_ = text_absent_threshold;
    LogInfo("ModelCoordinator: debounce present=%d absent=%d text_absent=%d",
            present_threshold_, absent_threshold_, text_absent_threshold_);
}

bool ModelCoordinator::EnsureActiveAdapters() {
    if (current_model_name_.empty()) {
        return false;
    }
    auto it = model_pool_.find(current_model_name_);
    if (it == model_pool_.end() || !it->second.prototype) {
        return false;
    }

    std::vector<std::shared_ptr<IModelAdapter>> new_active;
    new_active.reserve(static_cast<size_t>(num_infer_threads_));
    for (int i = 0; i < num_infer_threads_; ++i) {
        auto clone = it->second.prototype->Clone();
        if (!clone) {
            LogError("ModelCoordinator: Clone() returned nullptr for model %s", current_model_name_.c_str());
            active_adapters_.clear();
            return false;
        }
        int core_cfg = (i < static_cast<int>(npu_cores_.size())) ? npu_cores_[i] : 0;
        int core_mask = ToRknnCoreMask(core_cfg);
        if (clone->Init(it->second.model_path, core_mask) != 0) {
            LogError("ModelCoordinator: failed to init clone %d for model %s core_mask=0x%x path=%s",
                     i, current_model_name_.c_str(), core_mask, it->second.model_path.c_str());
            active_adapters_.clear();
            return false;
        }
        new_active.push_back(std::move(clone));
    }
    active_adapters_.swap(new_active);
    LogInfo("ModelCoordinator: %zu active adapter(s) ready for '%s'",
            active_adapters_.size(), current_model_name_.c_str());
    return !active_adapters_.empty();
}

bool ModelCoordinator::SwitchModel(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (current_model_name_ == name) {
        return true;
    }
    if (!IsModelAvailableUnlocked(name)) {
        std::cerr << "ModelCoordinator: model not available: " << name << std::endl;
        return false;
    }
    if (!EnsureModelInPoolUnlocked(name)) {
        std::cerr << "ModelCoordinator: failed to lazy-load model " << name << std::endl;
        return false;
    }
    std::string old_name = current_model_name_;
    auto old_adapters = active_adapters_;
    current_model_name_ = name;
    bool ok = EnsureActiveAdapters();
    if (!ok) {
        std::cerr << "ModelCoordinator: failed to switch to model " << name << std::endl;
        current_model_name_ = old_name;
        active_adapters_.swap(old_adapters);
        return false;
    }
    return true;
}
