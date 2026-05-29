/*
 * platform/llm_greeting.cpp
 */
#include "llm_greeting.h"

#include <cstdio>

#include "adapters/llm/llm_worker.h"
#include "adapters/tts/tts_worker.h"
#include "logging.h"

namespace {
// 简单字符串替换工具：用于日志转义，不参与业务语义。
void ReplaceAll(std::string& s, const std::string& from, const std::string& to) {
    if (from.empty()) {
        return;
    }
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
}

std::string EscapeForMachineLog(std::string s) {
    ReplaceAll(s, "\\", "\\\\");
    ReplaceAll(s, "\n", "\\n");
    ReplaceAll(s, "\r", "");
    ReplaceAll(s, "|", "\\|");
    return s;
}
}  // namespace

// 重置一次会话门控状态；通常在流程重启或模块重置时调用。
void LlmGreeting::Reset() {
    face_stable_count_ = 0;
    face_absent_count_ = 0;
    prompt_gate_open_ = false;
    gate_reject_notified_ = false;
    dialogue_unavailable_notified_ = false;
    auto_prompt_sent_for_visit_ = false;
    had_face_leave_since_boot_ = false;
    scrfd_was_active_ = false;
    state_ = SessionState::Locked;
    grace_deadline_ = std::chrono::steady_clock::time_point{};
    last_activity_ts_ = std::chrono::steady_clock::now();
    if (worker_) {
        worker_->ClearPendingPrompts();
    }
}

// 绑定 LlmWorker，并注册文本输出回调（统一走 SetBannerLine）。
void LlmGreeting::SetLlmWorker(LlmWorker* worker) {
    worker_ = worker;
    if (worker_) {
        worker_->SetBannerCallback([this](const std::string& line, LlmPromptSource src, bool is_final) {
            SetBannerLine(line, src, is_final);
        });
    }
}

// 绑定 TTS；问候 is_final 时可选播报（与 AI> 同源）。
void LlmGreeting::SetTtsWorker(TtsWorker* tts, bool skip_static_greeting) {
    tts_ = tts;
    skip_static_greeting_ = skip_static_greeting;
}

// 设置自动问候语（严格按配置覆盖，空串表示关闭自动问候）。
void LlmGreeting::SetAutoGreetingText(const std::string& text) {
    auto_greeting_text_ = text;
}

// 设置稳定人脸触发帧阈值（必须 >0）。
void LlmGreeting::SetTriggerThreshold(int frames) {
    if (frames > 0) {
        trigger_threshold_ = frames;
    }
}

// 设置人脸缺失关门阈值（必须 >0）。
void LlmGreeting::SetFaceAbsentThreshold(int frames) {
    if (frames > 0) {
        face_absent_threshold_ = frames;
    }
}

// 设置 Grace 超时时长（必须 >0）。
void LlmGreeting::SetGraceTimeoutMs(int timeout_ms) {
    if (timeout_ms > 0) {
        grace_timeout_ms_ = timeout_ms;
    }
}

// 设置会话空闲超时时长（必须 >0）。
void LlmGreeting::SetIdleTimeoutMs(int timeout_ms) {
    if (timeout_ms > 0) {
        idle_timeout_ms_ = timeout_ms;
    }
}

// 设置是否在 SCRFD 激活时触发 LLM 预加载。
void LlmGreeting::SetPreloadOnScrfd(bool preload) {
    preload_on_scrfd_ = preload;
}

const char* LlmGreeting::StateName(SessionState state) {
    switch (state) {
        case SessionState::Locked:
            return "Locked";
        case SessionState::Arming:
            return "Arming";
        case SessionState::Active:
            return "Active";
        case SessionState::Grace:
            return "Grace";
    }
    return "Unknown";
}

// 将枚举来源转成日志可读字符串。
const char* LlmGreeting::SourceName(LlmPromptSource src) {
    switch (src) {
        case LlmPromptSource::FaceAppear:
            return "FaceAppear";
        case LlmPromptSource::FaceReenter:
            return "FaceReenter";
        case LlmPromptSource::Microphone:
            return "Microphone";
        case LlmPromptSource::Button:
            return "Button";
        case LlmPromptSource::Command:
            return "Command";
    }
    return "Unknown";
}

// 统一输出会话文本：AI> 人类可读 + LLM_OUT 机器可解析。
void LlmGreeting::SetBannerLine(const std::string& line, LlmPromptSource src, bool is_final) {
    std::lock_guard<std::mutex> lock(ai_stream_mutex_);
    if (!line.empty()) {
        last_activity_ts_ = std::chrono::steady_clock::now();
        ai_stream_text_ += line;
        if (!ai_stream_open_) {
            std::fprintf(stdout, "AI> %s", line.c_str());
            ai_stream_open_ = true;
            ai_stream_src_ = src;
        } else if (ai_stream_src_ == src) {
            std::fprintf(stdout, "%s", line.c_str());
        } else {
            std::fprintf(stdout, "\nAI> %s", line.c_str());
            ai_stream_src_ = src;
            ai_stream_text_.clear();
            ai_stream_text_ = line;
        }
        std::fflush(stdout);
    }
    if (is_final && ai_stream_open_) {
        std::fprintf(stdout, "\n");
        std::fflush(stdout);
        if (!ai_stream_text_.empty()) {
            LogDebug("LLM_OUT|src=%s|text=%s", SourceName(ai_stream_src_),
                     EscapeForMachineLog(ai_stream_text_).c_str());
        }
        if (tts_ && !skip_static_greeting_ && !ai_stream_text_.empty()) {
            tts_->PlayText(ai_stream_text_);
        }
        ai_stream_text_.clear();
        ai_stream_open_ = false;
    }
}

// Pipeline 停止时中断 RKLLM，避免 Shutdown 卡在 JoinInferThread。
void LlmGreeting::AbortActiveGeneration() {
    if (worker_) {
        worker_->RequestAbortGeneration();
    } else if (tts_) {
        tts_->Cancel();
    }
}

// 每帧轮询 worker 的延迟任务（初始化状态/回调收尾/排队提交）。
void LlmGreeting::PollDeferred() {
    if (worker_) {
        worker_->PollDeferred();
    }
    if (tts_) {
        tts_->PollInitState();
    }
    TryOpenDialogueIfReady();
}

// Pipeline 启动时打印一条与 LLM 状态一致的 SYS 提示。
void LlmGreeting::LogStartupHint() {
    PollDeferred();
    if (!worker_) {
        return;
    }
    if (worker_->IsLoadFailed()) {
        return;
    }
    if (worker_->IsReady()) {
        LogSystem("输入通道已就绪，人脸稳定后可对话");
        return;
    }
    LogSystem("对话模型加载中，请稍候");
}

// 模型刚就绪且人脸已稳定时打开输入门控并补发静态问候。
void LlmGreeting::TryOpenDialogueIfReady() {
    if (!worker_ || !worker_->IsReady()) {
        return;
    }
    if (state_ != SessionState::Active && state_ != SessionState::Grace) {
        return;
    }
    if (face_stable_count_ < trigger_threshold_) {
        return;
    }
    if (!prompt_gate_open_) {
        prompt_gate_open_ = true;
        gate_reject_notified_ = false;
        dialogue_unavailable_notified_ = false;
    }
    AdapterSignals signals;
    signals.face_detected = true;
    TryAutoPromptOnStableFace(signals);
}

// 按配置尝试预加载模型；仅在未 ready、未 initializing 且未永久失败时触发。
void LlmGreeting::TryPreload() {
    if (!worker_ || !preload_on_scrfd_) {
        return;
    }
    if (worker_->IsLoadFailed()) {
        return;
    }
    if (!worker_->IsReady() && !worker_->IsInitializing()) {
        worker_->RequestInitializeAsync();
        LogInfo("LlmGreeting: request async llm init (preload)");
    }
}

// 当人脸稳定且对话模型就绪时自动打一条欢迎语（每次“到访”仅一次）。
void LlmGreeting::TryAutoPromptOnStableFace(const AdapterSignals& signals) {
    if (!worker_ || !worker_->IsReady()) {
        return;
    }
    if (!signals.face_detected || face_stable_count_ < trigger_threshold_) {
        return;
    }
    if (auto_prompt_sent_for_visit_) {
        return;
    }
    if (auto_greeting_text_.empty()) {
        return;
    }
    // 有正在进行的流式回复时不插入自动问候，避免与用户对话输出相互打断。
    if (worker_->IsBusy()) {
        return;
    }

    const bool reenter = had_face_leave_since_boot_;
    const LlmPromptSource src =
        reenter ? LlmPromptSource::FaceReenter : LlmPromptSource::FaceAppear;
    SetBannerLine(auto_greeting_text_, src);
    auto_prompt_sent_for_visit_ = true;
    LogInfo("LlmGreeting: auto greeting emitted (%s)", reenter ? "reenter" : "appear");
}

// 接收用户文本：门控关闭或模型不可用时拒绝，并做一次性提示防刷屏。
bool LlmGreeting::SubmitUserPrompt(const std::string& text) {
    if (!worker_ || text.empty()) {
        return false;
    }
    if (worker_->IsLoadFailed()) {
        if (!dialogue_unavailable_notified_) {
            LogSystem("对话不可用（模型未加载）");
            dialogue_unavailable_notified_ = true;
        }
        return false;
    }
    if (!prompt_gate_open_) {
        if (!gate_reject_notified_) {
            LogSystem("当前未检测到稳定人脸，暂不接收对话输入");
            LogDebug("LlmGreeting: reject input gate_open=0 stable=%d absent=%d scrfd_active=%d",
                     face_stable_count_, face_absent_count_, scrfd_was_active_ ? 1 : 0);
            gate_reject_notified_ = true;
        }
        return false;
    }
    const bool accepted = worker_->SubmitPrompt(text, LlmPromptSource::Microphone, prompt_gate_open_);
    if (accepted) {
        last_activity_ts_ = std::chrono::steady_clock::now();
    }
    return accepted;
}

// 每帧更新门控状态：处理 scrfd 开关、人脸稳定/缺失计数、自动问候与队列丢弃策略。
void LlmGreeting::Update(const AdapterSignals& signals, bool scrfd_slot_active) {
    const auto now = std::chrono::steady_clock::now();
    if (!scrfd_slot_active) {
        if (scrfd_was_active_) {
            LogDebug("LlmGreeting: scrfd off");
        }
        scrfd_was_active_ = false;
    } else if (!scrfd_was_active_) {
        scrfd_was_active_ = true;
        LogDebug("LlmGreeting: scrfd on");
        TryPreload();
    }

    // 空闲超时：Active/Grace 长时间无交互时自动锁定，防止永久敞开。
    if ((state_ == SessionState::Active || state_ == SessionState::Grace) &&
        idle_timeout_ms_ > 0 &&
        std::chrono::duration_cast<std::chrono::milliseconds>(now - last_activity_ts_).count() >=
            idle_timeout_ms_) {
        const SessionState prev = state_;
        state_ = SessionState::Locked;
        prompt_gate_open_ = false;
        gate_reject_notified_ = false;
        auto_prompt_sent_for_visit_ = false;
        had_face_leave_since_boot_ = true;
        face_stable_count_ = 0;
        face_absent_count_ = 0;
        if (worker_) {
            worker_->DropQueuedPrompts();
        }
        LogDebug("LlmGreeting: state %s -> %s (idle timeout)",
                 StateName(prev), StateName(state_));
        return;
    }

    if (signals.face_detected) {
        face_stable_count_++;
        face_absent_count_ = 0;
    } else {
        face_stable_count_ = 0;
        face_absent_count_++;
    }

    if (worker_ && scrfd_slot_active && !worker_->IsReady() && !worker_->IsLoadFailed()) {
        worker_->RequestInitializeAsync();
    }

    if (state_ == SessionState::Locked && scrfd_slot_active && signals.face_detected) {
        const SessionState prev = state_;
        state_ = SessionState::Arming;
        LogDebug("LlmGreeting: state %s -> %s", StateName(prev), StateName(state_));
    }

    if (state_ == SessionState::Arming) {
        if (!scrfd_slot_active) {
            const SessionState prev = state_;
            state_ = SessionState::Locked;
            LogDebug("LlmGreeting: state %s -> %s (scrfd off)",
                     StateName(prev), StateName(state_));
            return;
        }
        if (face_absent_count_ >= face_absent_threshold_) {
            const SessionState prev = state_;
            state_ = SessionState::Locked;
            LogDebug("LlmGreeting: state %s -> %s (face unstable)",
                     StateName(prev), StateName(state_));
            return;
        }
        if (face_stable_count_ >= trigger_threshold_) {
            const SessionState prev = state_;
            state_ = SessionState::Active;
            prompt_gate_open_ = worker_ && worker_->IsReady();
            gate_reject_notified_ = false;
            dialogue_unavailable_notified_ = false;
            auto_prompt_sent_for_visit_ = false;
            last_activity_ts_ = now;
            LogDebug("LlmGreeting: state %s -> %s", StateName(prev), StateName(state_));
            TryAutoPromptOnStableFace(signals);
        }
        return;
    }

    if (state_ == SessionState::Active) {
        if (face_absent_count_ >= face_absent_threshold_) {
            const SessionState prev = state_;
            state_ = SessionState::Grace;
            grace_deadline_ = now + std::chrono::milliseconds(grace_timeout_ms_);
            LogDebug("LlmGreeting: state %s -> %s (face absent)",
                     StateName(prev), StateName(state_));
            return;
        }
        return;
    }

    if (state_ == SessionState::Grace) {
        if (face_stable_count_ >= trigger_threshold_) {
            const SessionState prev = state_;
            state_ = SessionState::Active;
            prompt_gate_open_ = worker_ && worker_->IsReady();
            face_absent_count_ = 0;
            last_activity_ts_ = now;
            LogDebug("LlmGreeting: state %s -> %s (face recovered)",
                     StateName(prev), StateName(state_));
            TryAutoPromptOnStableFace(signals);
            return;
        }
        if (grace_deadline_ != std::chrono::steady_clock::time_point{} && now >= grace_deadline_) {
            const SessionState prev = state_;
            state_ = SessionState::Locked;
            prompt_gate_open_ = false;
            gate_reject_notified_ = false;
            auto_prompt_sent_for_visit_ = false;
            had_face_leave_since_boot_ = true;
            face_stable_count_ = 0;
            face_absent_count_ = 0;
            if (worker_) {
                worker_->DropQueuedPrompts();
            }
            LogDebug("LlmGreeting: state %s -> %s (grace timeout)",
                     StateName(prev), StateName(state_));
        }
    }
}
