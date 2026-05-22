/*
 * platform/model_coordinator.cpp — yolo + scrfd 多槽
 */
#include "model_coordinator.h"
#include "logging.h"
#include <sstream>

namespace {

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

ModelCoordinator::ModelCoordinator() = default;

std::vector<std::string> ModelCoordinator::OrderedSlotNames() {
    return {"yolo", "scrfd"};
}

bool ModelCoordinator::Init(const std::string& default_model_name,
                            std::shared_ptr<IModelAdapter> default_adapter,
                            const std::string& model_path,
                            const std::vector<int>& npu_cores,
                            int num_infer_threads) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        npu_cores_ = npu_cores;
        num_infer_threads_ = num_infer_threads > 0 ? num_infer_threads : 1;
        enabled_slots_.clear();
        slot_runtimes_.clear();
        warm_runtimes_.clear();

        if (default_adapter) {
            model_pool_[default_model_name] = {default_adapter, model_path};
        }
        LogInfo("ModelCoordinator: init default='%s' path='%s'",
                default_model_name.c_str(), model_path.c_str());
    }

    if (!EnableSlot(default_model_name)) {
        LogError("ModelCoordinator: failed to enable default slot '%s'", default_model_name.c_str());
        return false;
    }
    return true;
}

bool ModelCoordinator::HasActiveAdapters() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return !enabled_slots_.empty();
}

void ModelCoordinator::RegisterModel(const std::string& name, std::shared_ptr<IModelAdapter> adapter) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (adapter && !model_pool_.count(name)) {
        model_pool_[name] = {adapter, factory_path_map_.count(name) ? factory_path_map_[name] : ""};
    }
}

void ModelCoordinator::RegisterModel(const std::string& name, std::shared_ptr<IModelAdapter> adapter,
                                     const std::string& model_path) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (adapter) {
        model_pool_[name] = {adapter, model_path};
        factory_path_map_[name] = model_path;
    }
}

void ModelCoordinator::RegisterFactory(const std::string& name,
                                       std::function<std::shared_ptr<IModelAdapter>()> factory,
                                       const std::string& model_path) {
    std::lock_guard<std::mutex> lock(mutex_);
    factory_map_[name] = std::move(factory);
    factory_path_map_[name] = model_path;
    LogInfo("ModelCoordinator: registered factory '%s' path='%s'", name.c_str(), model_path.c_str());
}

void ModelCoordinator::SetSlotOptions(bool yolo_always_on) {
    std::lock_guard<std::mutex> lock(mutex_);
    yolo_always_on_ = yolo_always_on;
}

void ModelCoordinator::SetSceneDwellFrames(int frames) {
    std::lock_guard<std::mutex> lock(mutex_);
    scene_dwell_frames_ = frames > 0 ? frames : 1;
}

bool ModelCoordinator::WarmupSlot(const std::string& name) {
    if (!EnableSlot(name)) {
        return false;
    }
    DisableSlot(name);
    LogInfo("ModelCoordinator: warmup slot '%s' (in warm pool)", name.c_str());
    return true;
}

bool ModelCoordinator::IsModelAvailableUnlocked(const std::string& name) const {
    return model_pool_.count(name) > 0 || factory_map_.count(name) > 0;
}

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
        return false;
    }
    std::string path = factory_path_map_[name];
    model_pool_[name] = {adapter, path};
    LogInfo("ModelCoordinator: lazy-loaded prototype for '%s'", name.c_str());
    return true;
}

int ModelCoordinator::CoreMaskForSlot(const std::string& slot) const {
    int idx = (slot == "scrfd") ? 1 : 0;
    if (idx < static_cast<int>(npu_cores_.size())) {
        return ToRknnCoreMask(npu_cores_[idx]);
    }
    return ToRknnCoreMask(0);
}

bool ModelCoordinator::EnableSlot(const std::string& name) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (enabled_slots_.count(name) && slot_runtimes_.count(name)) {
            return true;
        }
        auto warm_it = warm_runtimes_.find(name);
        if (warm_it != warm_runtimes_.end() && warm_it->second.adapter) {
            slot_runtimes_[name] = std::move(warm_it->second);
            warm_runtimes_.erase(warm_it);
            enabled_slots_.insert(name);
            LogInfo("ModelCoordinator: enabled slot '%s' (from warm pool)", name.c_str());
            return true;
        }
    }

    std::string model_path;
    std::shared_ptr<IModelAdapter> prototype;
    int core_mask = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!EnsureModelInPoolUnlocked(name)) {
            LogWarn("ModelCoordinator: cannot enable slot '%s' (no prototype)", name.c_str());
            return false;
        }
        auto& entry = model_pool_[name];
        prototype = entry.prototype;
        model_path = entry.model_path;
        core_mask = CoreMaskForSlot(name);
    }

    auto clone = prototype ? prototype->Clone() : nullptr;
    if (!clone) {
        return false;
    }
    if (clone->Init(model_path, core_mask) != 0) {
        LogWarn("ModelCoordinator: Init failed for slot '%s' path=%s core=0x%x",
                name.c_str(), model_path.c_str(), core_mask);
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (enabled_slots_.count(name)) {
            return true;
        }
        slot_runtimes_[name] = {std::move(clone)};
        enabled_slots_.insert(name);
        LogInfo("ModelCoordinator: enabled slot '%s' core_mask=0x%x", name.c_str(), core_mask);
    }
    return true;
}

void ModelCoordinator::DisableSlot(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!enabled_slots_.count(name)) {
        return;
    }
    auto it = slot_runtimes_.find(name);
    if (it != slot_runtimes_.end()) {
        warm_runtimes_[name] = std::move(it->second);
        slot_runtimes_.erase(it);
    }
    enabled_slots_.erase(name);
    LogInfo("ModelCoordinator: disabled slot '%s' (kept warm)", name.c_str());
}

std::vector<std::pair<std::string, std::shared_ptr<IModelAdapter>>>
ModelCoordinator::GetEnabledSlotAdapters() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::pair<std::string, std::shared_ptr<IModelAdapter>>> out;
    for (const auto& name : OrderedSlotNames()) {
        if (!enabled_slots_.count(name)) {
            continue;
        }
        auto it = slot_runtimes_.find(name);
        if (it != slot_runtimes_.end() && it->second.adapter) {
            out.emplace_back(name, it->second.adapter);
        }
    }
    return out;
}

std::string ModelCoordinator::GetEnabledSlotsBadge() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::ostringstream oss;
    bool first = true;
    for (const auto& name : OrderedSlotNames()) {
        if (!enabled_slots_.count(name)) {
            continue;
        }
        if (!first) {
            oss << "+";
        }
        oss << name;
        first = false;
    }
    return first ? "none" : oss.str();
}

std::string ModelCoordinator::GetCurrentModelName() const {
    return GetEnabledSlotsBadge();
}

std::string ModelCoordinator::GetCurrentScene() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return SceneName(current_scene_);
}

bool ModelCoordinator::ShouldSuppressYoloPersonDraw() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return enabled_slots_.count("yolo") > 0 && enabled_slots_.count("scrfd") > 0;
}

const char* ModelCoordinator::SceneName(CoordinatorScene scene) {
    switch (scene) {
        case CoordinatorScene::Person:
            return "person";
        case CoordinatorScene::Idle:
        default:
            return "idle";
    }
}

CoordinatorScene ModelCoordinator::ComputeSceneUnlocked() const {
    if (person_present_count_ >= present_threshold_) {
        return CoordinatorScene::Person;
    }
    return CoordinatorScene::Idle;
}

SharedState& ModelCoordinator::GetSharedState() {
    return shared_state_;
}

LlmGreeting& ModelCoordinator::GetLlmGreeting() {
    return llm_greeting_;
}

void ModelCoordinator::SetSwitchDebounceThresholds(int present_threshold, int absent_threshold) {
    std::lock_guard<std::mutex> lock(mutex_);
    present_threshold_ = present_threshold;
    absent_threshold_ = absent_threshold;
}

void ModelCoordinator::MergeSignalsUnlocked(const AdapterSignals& signals) {
    if (signals.person_present) {
        shared_state_.person_present = true;
    }
    if (signals.face_detected) {
        shared_state_.face_detected = true;
    }
    if (signals.avg_brightness > 0.0f) {
        shared_state_.avg_brightness = signals.avg_brightness;
    }
    if (!signals.scene_label.empty()) {
        shared_state_.scene_label = signals.scene_label;
    }
    last_signals_ = signals;
}

ModelCoordinator::SlotPlan ModelCoordinator::BuildSlotPlanUnlocked() {
    SlotPlan plan;
    const CoordinatorScene computed = ComputeSceneUnlocked();
    plan.scene = (scene_dwell_count_ >= scene_dwell_frames_) ? computed : applied_scene_;

    switch (plan.scene) {
        case CoordinatorScene::Person:
            plan.want_scrfd = true;
            plan.want_yolo = true;
            break;
        case CoordinatorScene::Idle:
        default:
            plan.want_yolo = yolo_always_on_;
            break;
    }
    return plan;
}

void ModelCoordinator::ApplySlotPlan(const SlotPlan& plan) {
    auto slot_available = [this](const std::string& name) {
        std::lock_guard<std::mutex> lock(mutex_);
        return IsModelAvailableUnlocked(name);
    };

    if (plan.want_yolo && slot_available("yolo")) {
        EnableSlot("yolo");
    } else {
        DisableSlot("yolo");
    }

    if (plan.want_scrfd && slot_available("scrfd")) {
        EnableSlot("scrfd");
    } else {
        DisableSlot("scrfd");
    }
}

void ModelCoordinator::UpdateAfterFrame(const AdapterSignals& signals, const cv::Mat& frame) {
    (void)frame;
    SlotPlan plan;
    {
        std::lock_guard<std::mutex> lock(mutex_);

        shared_state_.person_present = false;
        shared_state_.face_detected = false;

        MergeSignalsUnlocked(signals);

        if (signals.person_present) {
            person_present_count_++;
            person_absent_count_ = 0;
        } else {
            person_absent_count_++;
            if (person_absent_count_ >= absent_threshold_) {
                person_present_count_ = 0;
            }
        }

        const CoordinatorScene computed = ComputeSceneUnlocked();
        current_scene_ = computed;
        shared_state_.scene_label = SceneName(computed);
        if (computed != pending_scene_) {
            pending_scene_ = computed;
            scene_dwell_count_ = 0;
        } else {
            scene_dwell_count_++;
        }
        if (scene_dwell_count_ >= scene_dwell_frames_) {
            applied_scene_ = computed;
        }

        plan = BuildSlotPlanUnlocked();

        const std::string scene_str = SceneName(computed);
        if (scene_str != last_logged_scene_) {
            LogInfo("ModelCoordinator: scene -> %s (applied=%s dwell=%d/%d)",
                    scene_str.c_str(), SceneName(applied_scene_), scene_dwell_count_,
                    scene_dwell_frames_);
            last_logged_scene_ = scene_str;
        }
    }

    ApplySlotPlan(plan);

    {
        std::lock_guard<std::mutex> lock(mutex_);
        const bool scrfd_on = enabled_slots_.count("scrfd") > 0;
        llm_greeting_.Update(signals, scrfd_on);
        llm_greeting_.PollDeferred();
    }
}
