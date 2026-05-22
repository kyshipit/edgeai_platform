/*
 * adapters/llm/llm_worker.h — RKLLM 常驻适配器：InitOnce + 按需 SubmitPrompt。
 */
#pragma once

#include <functional>
#include <future>
#include <mutex>
#include <string>

#include "rkllm_session.h"

enum class LlmPromptSource {
    FaceAppear,
    FaceReenter,
    Microphone,
    Button,
    Command,
};

class LlmWorker {
public:
    using BannerCallback = std::function<void(const std::string& line, LlmPromptSource src)>;

    LlmWorker();
    ~LlmWorker();

    void Configure(const std::string& model_path, int max_new_tokens, int max_context_len);
    bool EnsureInitialized();
    void RequestInitializeAsync();
    void SetBannerCallback(BannerCallback cb);

    // gate_open=false 时不提交；忙时可缓存一句待 gate 恢复后发送。
    bool SubmitPrompt(const std::string& user_text, LlmPromptSource src, bool gate_open);
    void ClearPendingPrompts();

    // 主线程每帧调用：处理 RKLLM 回调里排队的下一句（禁止在回调内 rkllm_run_async）。
    void PollDeferred();

    void Shutdown();

    bool IsReady() const;
    bool IsInitializing() const;
    bool IsBusy() const;

private:
    static void ChunkTrampoline(const char* text_chunk, LLMCallState state, void* user_data);
    void OnLlmChunk(const char* text_chunk, LLMCallState state);
    bool RunPromptNow(const std::string& user_text, LlmPromptSource src);
    static const char* SourceName(LlmPromptSource src);
    bool IsReadyUnlocked() const;
    bool IsInitializingUnlocked() const;
    void PollInitState();

    enum class InitState {
        Uninitialized,
        Initializing,
        Ready,
        Failed
    };

    RkllmSession session_;
    BannerCallback banner_cb_;
    mutable std::mutex mutex_;
    std::string model_path_;
    int max_new_tokens_ = 64;
    int max_context_len_ = 4096;
    bool configured_ = false;
    InitState init_state_ = InitState::Uninitialized;
    std::future<int> init_future_;
    std::string pending_text_;
    bool infer_busy_ = false;
    bool has_pending_ = false;
    std::string pending_prompt_;
    LlmPromptSource pending_src_ = LlmPromptSource::FaceAppear;
    bool deferred_run_ = false;
    std::string deferred_prompt_;
    LlmPromptSource deferred_src_ = LlmPromptSource::FaceAppear;
    bool banner_pending_ = false;
    std::string pending_banner_;
    LlmPromptSource banner_src_ = LlmPromptSource::FaceAppear;
    LlmPromptSource current_src_ = LlmPromptSource::FaceAppear;
};
