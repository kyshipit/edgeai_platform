/*
 * platform/llm_greeting.cpp
 */
#include "llm_greeting.h"

#include "adapters/llm/llm_worker.h"
#include "logging.h"

namespace {
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

void EraseThinkBlocks(std::string& s) {
    const std::string open_tag = "<think>";
    const std::string close_tag = "</think>";
    while (true) {
        const size_t start = s.find(open_tag);
        if (start == std::string::npos) {
            break;
        }
        const size_t end = s.find(close_tag, start + open_tag.size());
        if (end == std::string::npos) {
            s.erase(start);
            break;
        }
        s.erase(start, end + close_tag.size() - start);
    }
    ReplaceAll(s, close_tag, "");
}

std::string EscapeForMachineLog(std::string s) {
    ReplaceAll(s, "\\", "\\\\");
    ReplaceAll(s, "\n", "\\n");
    ReplaceAll(s, "\r", "");
    ReplaceAll(s, "|", "\\|");
    return s;
}

std::string NormalizeLlmText(std::string s) {
    EraseThinkBlocks(s);
    ReplaceAll(s, "<|User|>", " ");
    ReplaceAll(s, "<|Assistant|>", " ");
    ReplaceAll(s, "\r", "");
    while (!s.empty() && (s.front() == ' ' || s.front() == '\n' || s.front() == '\t')) {
        s.erase(s.begin());
    }
    while (!s.empty() && (s.back() == ' ' || s.back() == '\n' || s.back() == '\t')) {
        s.pop_back();
    }
    return s;
}
}  // namespace

void LlmGreeting::Reset() {
    face_stable_count_ = 0;
    face_absent_count_ = 0;
    prompt_gate_open_ = false;
    auto_prompt_sent_for_visit_ = false;
    had_face_leave_since_boot_ = false;
    scrfd_was_active_ = false;
    banner_line_.clear();
    if (worker_) {
        worker_->ClearPendingPrompts();
    }
}

void LlmGreeting::SetLlmWorker(LlmWorker* worker) {
    worker_ = worker;
    if (worker_) {
        worker_->SetBannerCallback(
            [this](const std::string& line, LlmPromptSource src) { SetBannerLine(line, src); });
    }
}

void LlmGreeting::SetGreetingPrompt(const std::string& prompt) {
    if (!prompt.empty()) {
        greeting_prompt_ = prompt;
    }
}

void LlmGreeting::SetReenterPrompt(const std::string& prompt) {
    if (!prompt.empty()) {
        reenter_prompt_ = prompt;
    }
}

void LlmGreeting::SetAutoGreetingText(const std::string& text) {
    if (!text.empty()) {
        auto_greeting_text_ = text;
    }
}

void LlmGreeting::SetTriggerThreshold(int frames) {
    if (frames > 0) {
        trigger_threshold_ = frames;
    }
}

void LlmGreeting::SetFaceAbsentThreshold(int frames) {
    if (frames > 0) {
        face_absent_threshold_ = frames;
    }
}

void LlmGreeting::SetPreloadOnScrfd(bool preload) {
    preload_on_scrfd_ = preload;
}

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

void LlmGreeting::SetBannerLine(const std::string& line, LlmPromptSource src) {
    // 当前阶段仅终端输出，后续可在此处转接语音按键/播报链路。
    banner_line_.clear();
    const std::string normalized = NormalizeLlmText(line);
    if (!normalized.empty()) {
        LogInfo("LLM_OUT|src=%s|text=%s", SourceName(src), EscapeForMachineLog(normalized).c_str());
    }
}

void LlmGreeting::PollDeferred() {
    if (worker_) {
        worker_->PollDeferred();
    }
}

void LlmGreeting::TryPreload() {
    if (!worker_ || !preload_on_scrfd_) {
        return;
    }
    if (!worker_->IsReady() && !worker_->IsInitializing()) {
        worker_->RequestInitializeAsync();
        LogInfo("LlmGreeting: request async llm init (preload)");
    }
}

void LlmGreeting::TryAutoPromptOnStableFace(const AdapterSignals& signals) {
    if (!signals.face_detected || face_stable_count_ < trigger_threshold_) {
        return;
    }
    if (auto_prompt_sent_for_visit_) {
        return;
    }

    prompt_gate_open_ = true;
    const bool reenter = had_face_leave_since_boot_;
    const LlmPromptSource src =
        reenter ? LlmPromptSource::FaceReenter : LlmPromptSource::FaceAppear;
    SetBannerLine(auto_greeting_text_, src);
    auto_prompt_sent_for_visit_ = true;
    LogInfo("LlmGreeting: auto greeting emitted (%s)", reenter ? "reenter" : "appear");
}

bool LlmGreeting::SubmitUserPrompt(const std::string& text) {
    if (!worker_ || text.empty()) {
        return false;
    }
    if (!prompt_gate_open_) {
        LogInfo("LlmGreeting: user prompt rejected (gate closed)");
        return false;
    }
    return worker_->SubmitPrompt(text, LlmPromptSource::Microphone, prompt_gate_open_);
}

void LlmGreeting::Update(const AdapterSignals& signals, bool scrfd_slot_active) {
    if (!scrfd_slot_active) {
        if (scrfd_was_active_) {
            LogInfo("LlmGreeting: scrfd off -> gate closed, clear pending");
        }
        scrfd_was_active_ = false;
        face_stable_count_ = 0;
        face_absent_count_ = 0;
        prompt_gate_open_ = false;
        if (worker_) {
            worker_->ClearPendingPrompts();
        }
        return;
    }

    if (!scrfd_was_active_) {
        scrfd_was_active_ = true;
        LogInfo("LlmGreeting: scrfd on");
        TryPreload();
    }

    if (signals.face_detected) {
        face_stable_count_++;
        face_absent_count_ = 0;
        prompt_gate_open_ = true;
        if (worker_ && !worker_->IsReady()) {
            worker_->RequestInitializeAsync();
        }
        TryAutoPromptOnStableFace(signals);
    } else {
        face_stable_count_ = 0;
        face_absent_count_++;
        if (face_absent_count_ >= face_absent_threshold_) {
            if (prompt_gate_open_) {
                LogInfo("LlmGreeting: face lost (%d frames) -> gate closed, no new prompts",
                        face_absent_count_);
            }
            prompt_gate_open_ = false;
            auto_prompt_sent_for_visit_ = false;
            had_face_leave_since_boot_ = true;
            if (worker_) {
                worker_->ClearPendingPrompts();
            }
        }
    }
}
