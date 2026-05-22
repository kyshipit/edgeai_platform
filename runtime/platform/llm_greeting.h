/*
 * platform/llm_greeting.h — 人脸门控：出现/再现各一句，脸离关闭 Prompt 门（不卸载 RKLLM）。
 */
#pragma once

#include <string>
#include "adapter_signals.h"
#include "adapters/llm/llm_worker.h"

class LlmGreeting {
public:
    void Reset();
    void SetLlmWorker(LlmWorker* worker);
    void SetGreetingPrompt(const std::string& prompt);
    void SetReenterPrompt(const std::string& prompt);
    void SetAutoGreetingText(const std::string& text);
    void SetTriggerThreshold(int frames);
    void SetFaceAbsentThreshold(int frames);
    void SetPreloadOnScrfd(bool preload);

    void Update(const AdapterSignals& signals, bool scrfd_slot_active);
    void PollDeferred();

    // 麦克风/按键等：gate 关闭时拒绝。
    bool SubmitUserPrompt(const std::string& text);

    void SetBannerLine(const std::string& line, LlmPromptSource src);
    const std::string& GetBannerLine() const { return banner_line_; }
    bool HasBanner() const { return !banner_line_.empty(); }
    bool IsPromptGateOpen() const { return prompt_gate_open_; }

private:
    void TryPreload();
    void TryAutoPromptOnStableFace(const AdapterSignals& signals);
    static const char* SourceName(LlmPromptSource src);

    LlmWorker* worker_ = nullptr;
    std::string greeting_prompt_ =
        "请用一句简短、自然的中文向镜头前的人问好，不要超过二十个字。";
    std::string reenter_prompt_ = "欢迎回来，请用一句简短中文向镜头前的人问好。";
    std::string auto_greeting_text_ = "您好，我是deepseek, 有什么需要可以直接和我对话";
    int face_stable_count_ = 0;
    int face_absent_count_ = 0;
    int trigger_threshold_ = 5;
    int face_absent_threshold_ = 10;
    bool preload_on_scrfd_ = true;
    bool prompt_gate_open_ = false;
    bool auto_prompt_sent_for_visit_ = false;
    bool had_face_leave_since_boot_ = false;
    bool scrfd_was_active_ = false;
    std::string banner_line_;
};
