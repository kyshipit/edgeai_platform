/*
 * adapters/llm/rkllm_session.cpp
 */
#include "rkllm_session.h"

#include <chrono>
#include <cstring>
#include <pthread.h>
#include <thread>

#include "platform/logging.h"

namespace {
constexpr const char* kPromptPrefix = "<|User|>";
constexpr const char* kPromptPostfix = "<|Assistant|>";

RkllmSession* g_active_session = nullptr;

// 将 RKLLM 回调状态转为可读字符串，便于 SIGSEGV 现场日志。
const char* LlmCallStateName(LLMCallState state) {
    switch (state) {
        case RKLLM_RUN_NORMAL:
            return "NORMAL";
        case RKLLM_RUN_WAITING:
            return "WAITING";
        case RKLLM_RUN_FINISH:
            return "FINISH";
        case RKLLM_RUN_ERROR:
            return "ERROR";
        case RKLLM_RUN_GET_LAST_HIDDEN_LAYER:
            return "HIDDEN";
        default:
            return "UNKNOWN";
    }
}
}  // namespace

// 默认构造；句柄为空。
RkllmSession::RkllmSession() {
    std::memset(&async_input_, 0, sizeof(async_input_));
    std::memset(&async_infer_param_, 0, sizeof(async_infer_param_));
}

// 析构时释放 RKLLM 句柄。
RkllmSession::~RkllmSession() {
    Shutdown();
}

// 初始化 RKLLM：设置默认参数、注册库回调、保存上层 chunk_fn。
int RkllmSession::Init(const std::string& model_path, int max_new_tokens, int max_context_len,
                         RkllmChunkFn chunk_fn, void* user_data) {
    LogInfo("LLM_DBG RkllmSession::Init enter this=%p path=%s max_new=%d max_ctx=%d chunk_fn=%p user=%p",
            static_cast<void*>(this), model_path.c_str(), max_new_tokens, max_context_len,
            reinterpret_cast<void*>(chunk_fn), user_data);
    Shutdown();
    chunk_fn_ = chunk_fn;
    chunk_user_data_ = user_data;
    magic_ = kMagic;
    g_active_session = this;

    RKLLMParam param = rkllm_createDefaultParam();
    param.model_path = model_path.c_str();
    param.top_k = 1;
    param.top_p = 0.95f;
    param.temperature = 0.8f;
    param.repeat_penalty = 1.1f;
    param.frequency_penalty = 0.0f;
    param.presence_penalty = 0.0f;
    param.max_new_tokens = max_new_tokens;
    param.max_context_len = max_context_len;
    param.skip_special_token = true;
    param.is_async = true;
    param.extend_param.base_domain_id = 0;

    LogInfo("LLM_DBG RkllmSession::Init calling rkllm_init pthread=%p", reinterpret_cast<void*>(pthread_self()));
    const int ret = rkllm_init(&handle_, &param, &RkllmSession::StaticCallback);
    if (ret != 0) {
        LogWarn("LLM_DBG RkllmSession::Init rkllm_init failed ret=%d", ret);
        g_active_session = nullptr;
        handle_ = nullptr;
        chunk_fn_ = nullptr;
        chunk_user_data_ = nullptr;
    } else {
        LogInfo("LLM_DBG RkllmSession::Init ok handle=%p g_active=%p", handle_, static_cast<void*>(g_active_session));
    }
    return ret;
}

// 拼接对话模板后调用 rkllm_run_async；userdata 传 this 供 StaticCallback 识别会话。
int RkllmSession::RunPromptAsync(const std::string& user_text) {
    LogInfo("LLM_DBG RkllmSession::RunPromptAsync enter this=%p handle=%p user_len=%zu pthread=%p",
            static_cast<void*>(this), handle_, user_text.size(),
            reinterpret_cast<void*>(pthread_self()));
    if (!handle_ || !chunk_fn_) {
        LogWarn("LLM_DBG RkllmSession::RunPromptAsync abort handle=%p chunk_fn=%p",
                handle_, reinterpret_cast<void*>(chunk_fn_));
        return -1;
    }
    prompt_buffer_ = std::string(kPromptPrefix) + user_text + kPromptPostfix;

    std::memset(&async_input_, 0, sizeof(async_input_));
    async_input_.input_type = RKLLM_INPUT_PROMPT;
    async_input_.prompt_input = prompt_buffer_.c_str();

    std::memset(&async_infer_param_, 0, sizeof(async_infer_param_));
    async_infer_param_.mode = RKLLM_INFER_GENERATE;

    LogInfo("LLM_DBG RkllmSession::RunPromptAsync prompt_len=%zu prompt_ptr=%p prefix=%s postfix=%s",
            prompt_buffer_.size(), static_cast<const void*>(prompt_buffer_.c_str()), kPromptPrefix,
            kPromptPostfix);
    const int ret = rkllm_run_async(handle_, &async_input_, &async_infer_param_, this);
    LogInfo("LLM_DBG RkllmSession::RunPromptAsync rkllm_run_async ret=%d is_running=%d",
            ret, IsRunning() ? 1 : 0);
    return ret;
}

// 中止当前推理任务。
int RkllmSession::Abort() {
    LogInfo("LLM_DBG RkllmSession::Abort handle=%p pthread=%p", handle_,
            reinterpret_cast<void*>(pthread_self()));
    if (!handle_) {
        return -1;
    }
    const int ret = rkllm_abort(handle_);
    LogInfo("LLM_DBG RkllmSession::Abort ret=%d", ret);
    return ret;
}

// 等待推理结束并销毁句柄，清空全局 active 指针。
void RkllmSession::Shutdown() {
    LogInfo("LLM_DBG RkllmSession::Shutdown enter this=%p handle=%p pthread=%p",
            static_cast<void*>(this), handle_, reinterpret_cast<void*>(pthread_self()));
    if (g_active_session == this) {
        g_active_session = nullptr;
    }
    if (!handle_) {
        chunk_fn_ = nullptr;
        chunk_user_data_ = nullptr;
        return;
    }
    Abort();
    for (int i = 0; i < 300; ++i) {
        if (!IsRunning()) {
            LogInfo("LLM_DBG RkllmSession::Shutdown wait done i=%d", i);
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    LLMHandle tmp = handle_;
    handle_ = nullptr;
    LogInfo("LLM_DBG RkllmSession::Shutdown calling rkllm_destroy tmp=%p", tmp);
    rkllm_destroy(tmp);
    chunk_fn_ = nullptr;
    chunk_user_data_ = nullptr;
    std::memset(&async_input_, 0, sizeof(async_input_));
    std::memset(&async_infer_param_, 0, sizeof(async_infer_param_));
    LogInfo("LLM_DBG RkllmSession::Shutdown done");
}

// rkllm_is_running 返回 0 表示仍在运行。
bool RkllmSession::IsRunning() const {
    if (!handle_) {
        return false;
    }
    return rkllm_is_running(handle_) == 0;
}

// librkllmrt 注册的静态回调：解析 userdata/magic 后转发 chunk_fn_。
void RkllmSession::StaticCallback(RKLLMResult* result, void* userdata, LLMCallState state) {
    RkllmSession* self = static_cast<RkllmSession*>(userdata);
    if (!self || self->magic_ != kMagic) {
        LogWarn("LLM_DBG StaticCallback fallback g_active (self=%p magic_ok=%d)",
                static_cast<void*>(self), (self && self->magic_ == kMagic) ? 1 : 0);
        self = g_active_session;
    }
    if (!self || self->magic_ != kMagic || !self->chunk_fn_) {
        LogWarn("LLM_DBG StaticCallback drop self=%p chunk_fn=%p", static_cast<void*>(self),
                self ? reinterpret_cast<void*>(self->chunk_fn_) : nullptr);
        return;
    }
    const char* chunk = (result && result->text) ? result->text : "";
    self->chunk_fn_(chunk, state, self->chunk_user_data_);
    if (state == RKLLM_RUN_FINISH || state == RKLLM_RUN_ERROR) {
        LogInfo("LLM_DBG StaticCallback state=%s", LlmCallStateName(state));
    }
}
