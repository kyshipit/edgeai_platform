/*
 * adapters/llm/rkllm_session.h — RKLLM 会话封装（init / async run / abort / destroy）。
 */
#pragma once

#include <cstdint>
#include <string>

#include "rkllm.h"

// librkllmrt 回调线程调用的 C 函数指针；禁止在回调内再调 rkllm_*。
typedef void (*RkllmChunkFn)(const char* text_chunk, LLMCallState state, void* user_data);

class RkllmSession {
public:
    RkllmSession();
    ~RkllmSession();

    RkllmSession(const RkllmSession&) = delete;
    RkllmSession& operator=(const RkllmSession&) = delete;

    // 加载 .rkllm；注册 StaticCallback；chunk_fn 在推理回调里被调用。
    int Init(const std::string& model_path, int max_new_tokens, int max_context_len,
             RkllmChunkFn chunk_fn, void* user_data);
    // 按 <|User|>…<|Assistant|> 拼接后异步生成；prompt 保存在成员 buffer。
    int RunPromptAsync(const std::string& user_text);
    int Abort();
    // Abort 后等待推理结束再 destroy，避免与回调并发。
    void Shutdown();
    bool IsInitialized() const { return handle_ != nullptr; }
    bool IsRunning() const;

private:
    static void StaticCallback(RKLLMResult* result, void* userdata, LLMCallState state);

    static constexpr uint32_t kMagic = 0x524B4C4Du;

    LLMHandle handle_ = nullptr;
    RkllmChunkFn chunk_fn_ = nullptr;
    void* chunk_user_data_ = nullptr;
    uint32_t magic_ = kMagic;
    std::string prompt_buffer_;
    // 异步推理入参必须在 run_async 返回后仍有效，避免库侧延迟访问栈对象。
    RKLLMInput async_input_;
    RKLLMInferParam async_infer_param_;
};
