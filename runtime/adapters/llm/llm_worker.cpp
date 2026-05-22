/*
 * adapters/llm/llm_worker.cpp
 */
#include "llm_worker.h"

#include <chrono>
#include "platform/logging.h"

LlmWorker::LlmWorker() = default;

LlmWorker::~LlmWorker() {
    Shutdown();
}

void LlmWorker::Configure(const std::string& model_path, int max_new_tokens, int max_context_len) {
    std::lock_guard<std::mutex> lock(mutex_);
    model_path_ = model_path;
    max_new_tokens_ = max_new_tokens;
    max_context_len_ = max_context_len;
    configured_ = true;
    if (init_state_ == InitState::Failed) {
        init_state_ = InitState::Uninitialized;
    }
}

void LlmWorker::ChunkTrampoline(const char* text_chunk, LLMCallState state, void* user_data) {
    if (!user_data) {
        return;
    }
    static_cast<LlmWorker*>(user_data)->OnLlmChunk(text_chunk, state);
}

bool LlmWorker::EnsureInitialized() {
    PollInitState();
    if (IsReady()) {
        return true;
    }
    RequestInitializeAsync();
    return false;
}

void LlmWorker::RequestInitializeAsync() {
    std::string path;
    int max_new = 0;
    int max_ctx = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!configured_) {
            LogWarn("LlmWorker: async init skipped (not configured)");
            return;
        }
        if (IsReadyUnlocked() || IsInitializingUnlocked()) {
            return;
        }
        path = model_path_;
        max_new = max_new_tokens_;
        max_ctx = max_context_len_;
        init_state_ = InitState::Initializing;
    }

    LogInfo("LlmWorker: async InitOnce start %s ...", path.c_str());
    init_future_ = std::async(std::launch::async, [this, path, max_new, max_ctx]() {
        return session_.Init(path, max_new, max_ctx, &LlmWorker::ChunkTrampoline, this);
    });
}

void LlmWorker::SetBannerCallback(BannerCallback cb) {
    std::lock_guard<std::mutex> lock(mutex_);
    banner_cb_ = std::move(cb);
}

bool LlmWorker::IsBusy() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return infer_busy_;
}

bool LlmWorker::IsReady() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return IsReadyUnlocked();
}

bool LlmWorker::IsInitializing() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return IsInitializingUnlocked();
}

bool LlmWorker::IsReadyUnlocked() const {
    return init_state_ == InitState::Ready;
}

bool LlmWorker::IsInitializingUnlocked() const {
    return init_state_ == InitState::Initializing;
}

void LlmWorker::ClearPendingPrompts() {
    std::lock_guard<std::mutex> lock(mutex_);
    has_pending_ = false;
    pending_prompt_.clear();
    deferred_run_ = false;
    deferred_prompt_.clear();
    banner_pending_ = false;
    pending_banner_.clear();
    banner_src_ = LlmPromptSource::FaceAppear;
}

void LlmWorker::PollDeferred() {
    PollInitState();

    BannerCallback cb;
    std::string banner;
    LlmPromptSource banner_src = LlmPromptSource::FaceAppear;
    std::string prompt;
    LlmPromptSource src = LlmPromptSource::FaceAppear;
    bool run_deferred = false;

    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (banner_pending_) {
            banner = pending_banner_;
            pending_banner_.clear();
            banner_pending_ = false;
            banner_src = banner_src_;
            cb = banner_cb_;
        }
        if (deferred_run_) {
            deferred_run_ = false;
            prompt = deferred_prompt_;
            src = deferred_src_;
            deferred_prompt_.clear();
            run_deferred = !prompt.empty();
        }
        if (!run_deferred && has_pending_ && IsReadyUnlocked() && !infer_busy_) {
            prompt = pending_prompt_;
            src = pending_src_;
            pending_prompt_.clear();
            has_pending_ = false;
            run_deferred = !prompt.empty();
        }
    }

    if (cb && !banner.empty()) {
        cb(banner, banner_src);
        LogInfo("LlmWorker: turn done (%zu chars)", banner.size());
    }
    if (run_deferred) {
        RunPromptNow(prompt, src);
    }
}

const char* LlmWorker::SourceName(LlmPromptSource src) {
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

void LlmWorker::PollInitState() {
    std::future<int> done_future;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (init_state_ != InitState::Initializing || !init_future_.valid()) {
            return;
        }
        if (init_future_.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready) {
            return;
        }
        done_future = std::move(init_future_);
    }

    const int ret = done_future.get();

    std::lock_guard<std::mutex> lock(mutex_);
    if (ret == 0) {
        init_state_ = InitState::Ready;
        LogInfo("LlmWorker: rkllm_init ok (async, model stays loaded until process exit)");
    } else {
        init_state_ = InitState::Failed;
        LogWarn("LlmWorker: rkllm_init failed (async ret=%d)", ret);
    }
}

bool LlmWorker::RunPromptNow(const std::string& user_text, LlmPromptSource src) {
    if (!EnsureInitialized()) {
        LogInfo("LlmWorker: submit deferred until async init ready src=%s", SourceName(src));
        return false;
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (infer_busy_) {
            return false;
        }
        infer_busy_ = true;
        pending_text_.clear();
        current_src_ = src;
    }
    LogInfo("LlmWorker: SubmitPrompt src=%s", SourceName(src));
    const int ret = session_.RunPromptAsync(user_text);
    if (ret != 0) {
        std::lock_guard<std::mutex> lock(mutex_);
        infer_busy_ = false;
        LogWarn("LlmWorker: rkllm_run_async failed (%d)", ret);
        return false;
    }
    LogInfo("LlmWorker: inference started");
    return true;
}

bool LlmWorker::SubmitPrompt(const std::string& user_text, LlmPromptSource src, bool gate_open) {
    if (!gate_open) {
        return false;
    }
    if (user_text.empty()) {
        return false;
    }
    if (RunPromptNow(user_text, src)) {
        return true;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (infer_busy_ || IsInitializingUnlocked() || !IsReadyUnlocked()) {
        has_pending_ = true;
        pending_prompt_ = user_text;
        pending_src_ = src;
        if (infer_busy_) {
            LogInfo("LlmWorker: queued one prompt (busy) src=%s", SourceName(src));
        } else {
            LogInfo("LlmWorker: queued one prompt (init pending) src=%s", SourceName(src));
        }
    }
    return false;
}

void LlmWorker::Shutdown() {
    {
        std::future<int> pending_init;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (init_future_.valid()) {
                pending_init = std::move(init_future_);
            }
        }
        if (pending_init.valid()) {
            pending_init.wait();
        }
    }

    session_.Shutdown();
    std::lock_guard<std::mutex> lock(mutex_);
    init_state_ = InitState::Uninitialized;
    infer_busy_ = false;
    pending_text_.clear();
    has_pending_ = false;
    pending_prompt_.clear();
    deferred_run_ = false;
    deferred_prompt_.clear();
    banner_pending_ = false;
    pending_banner_.clear();
}

void LlmWorker::OnLlmChunk(const char* text_chunk, LLMCallState state) {
    std::string queued_prompt;
    LlmPromptSource queued_src = LlmPromptSource::FaceAppear;
    bool has_queued = false;

    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (state == RKLLM_RUN_NORMAL && text_chunk && text_chunk[0] != '\0') {
            pending_text_ += text_chunk;
        } else if (state == RKLLM_RUN_FINISH) {
            if (!pending_text_.empty()) {
                pending_banner_ = pending_text_;
                banner_pending_ = true;
                banner_src_ = current_src_;
            }
            pending_text_.clear();
            infer_busy_ = false;
            if (has_pending_) {
                queued_prompt = pending_prompt_;
                queued_src = pending_src_;
                has_queued = true;
                has_pending_ = false;
                pending_prompt_.clear();
            }
        } else if (state == RKLLM_RUN_ERROR) {
            LogWarn("LlmWorker: RKLLM_RUN_ERROR");
            pending_text_.clear();
            infer_busy_ = false;
            has_pending_ = false;
            pending_prompt_.clear();
        }
    }

    if (has_queued && !queued_prompt.empty()) {
        std::lock_guard<std::mutex> lock(mutex_);
        deferred_run_ = true;
        deferred_prompt_ = std::move(queued_prompt);
        deferred_src_ = queued_src;
    }
}
