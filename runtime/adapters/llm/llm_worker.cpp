/*
 * adapters/llm/llm_worker.cpp
 */
#include "llm_worker.h"

#include <chrono>
#include <cstdio>
#include <sys/stat.h>

#include "adapters/tts/tts_worker.h"
#include "platform/logging.h"

namespace {

// 启动预检：缺失或非普通文件时不进入 rkllm_init，避免占 NPU 并误导后续 YOLO Init 日志。
bool LlmModelFileReadable(const std::string& path) {
    if (path.empty()) {
        return false;
    }
    struct stat st;
    if (stat(path.c_str(), &st) != 0) {
        return false;
    }
    return S_ISREG(st.st_mode);
}

}  // namespace

LlmWorker::LlmWorker() = default;

// 析构时确保会话和异步初始化都被安全回收。
LlmWorker::~LlmWorker() {
    Shutdown();
}

// 写入模型参数；若上次初始化失败，允许状态回到可重试。
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

// C 回调跳板：转发到实例方法，避免在 C API 里直接捕获 C++ 对象。
void LlmWorker::ChunkTrampoline(const char* text_chunk, LLMCallState state, void* user_data) {
    if (!user_data) {
        return;
    }
    static_cast<LlmWorker*>(user_data)->OnLlmChunk(text_chunk, state);
}

// 轻量确保初始化：ready 直接成功，否则触发异步初始化请求。
bool LlmWorker::EnsureInitialized() {
    PollInitState();
    if (IsReady()) {
        return true;
    }
    RequestInitializeAsync();
    return false;
}

// 请求异步初始化（幂等）：未配置/已 ready/正在 init/已失败时均直接返回；缺文件不调 rkllm_init。
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
        if (init_state_ == InitState::Failed) {
            return;
        }
        path = model_path_;
        max_new = max_new_tokens_;
        max_ctx = max_context_len_;
    }

    if (!LlmModelFileReadable(path)) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (IsReadyUnlocked() || IsInitializingUnlocked()) {
            return;
        }
        init_state_ = InitState::Failed;
        LogWarn("LlmWorker: model file lost and failed to load, skip rkllm_init path=%s",
                path.c_str());
        LogSystem("仅视觉模式（对话模型未加载）");
        return;
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (IsReadyUnlocked() || IsInitializingUnlocked()) {
            return;
        }
        init_state_ = InitState::Initializing;
    }

    LogInfo("LlmWorker: async InitOnce start %s ...", path.c_str());
    LogSystem("正在加载模型，请稍候...");
    init_future_ = std::async(std::launch::async, [this, path, max_new, max_ctx]() {
        return session_.Init(path, max_new, max_ctx, &LlmWorker::ChunkTrampoline, this);
    });
}

// 设置回合结束文本回调（通常由 LlmGreeting 注入）。
void LlmWorker::SetBannerCallback(BannerCallback cb) {
    std::lock_guard<std::mutex> lock(mutex_);
    banner_cb_ = std::move(cb);
}

// 绑定 TTS 播报器（可为空）。
void LlmWorker::SetTtsWorker(TtsWorker* tts) {
    std::lock_guard<std::mutex> lock(mutex_);
    tts_ = tts;
}

// 是否在本轮 rkllm 结束后播报累积正文。
void LlmWorker::SetTtsEnabled(bool enabled) {
    std::lock_guard<std::mutex> lock(mutex_);
    tts_enabled_ = enabled;
}

// 查询当前是否正在生成。
bool LlmWorker::IsBusy() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return infer_busy_;
}

// 查询模型是否初始化就绪。
bool LlmWorker::IsReady() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return IsReadyUnlocked();
}

// 查询是否处于异步初始化中。
bool LlmWorker::IsInitializing() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return IsInitializingUnlocked();
}

// 查询是否已判定加载失败（缺文件或 rkllm_init 失败，本进程内不重试）。
bool LlmWorker::IsLoadFailed() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return IsLoadFailedUnlocked();
}

// 锁内 ready 判定。
bool LlmWorker::IsReadyUnlocked() const {
    return init_state_ == InitState::Ready;
}

// 锁内 initializing 判定。
bool LlmWorker::IsInitializingUnlocked() const {
    return init_state_ == InitState::Initializing;
}

// 锁内 failed 判定。
bool LlmWorker::IsLoadFailedUnlocked() const {
    return init_state_ == InitState::Failed;
}

// 清空所有待处理内容（包括当前已聚合但未发布的 banner）。
void LlmWorker::ClearPendingPrompts() {
    std::lock_guard<std::mutex> lock(mutex_);
    has_pending_ = false;
    pending_prompt_.clear();
    deferred_run_ = false;
    deferred_prompt_.clear();
    pending_text_.clear();
    banner_pending_ = false;
    pending_banner_.clear();
    streamed_chars_ = 0;
    banner_src_ = LlmPromptSource::FaceAppear;
}

// 仅丢弃排队输入，不打断当前正在输出的这一轮。
void LlmWorker::DropQueuedPrompts() {
    std::lock_guard<std::mutex> lock(mutex_);
    has_pending_ = false;
    pending_prompt_.clear();
    deferred_run_ = false;
    deferred_prompt_.clear();
    // 保留 pending_text_/banner_pending_，确保正在生成的一句仍可正常收尾输出。
}

// 主线程轮询：发布已完成 banner，并在可运行时投递下一条 deferred/pending 输入。
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
        // 回调线程只负责堆积结果，真正发布统一在主线程完成，避免并发乱序。
        if (banner_pending_) {
            banner = pending_banner_;
            pending_banner_.clear();
            banner_pending_ = false;
            banner_src = banner_src_;
            cb = banner_cb_;
        }
        // 优先运行回调阶段安排的 deferred（FINISH 后衔接的下一句）。
        if (deferred_run_) {
            deferred_run_ = false;
            prompt = deferred_prompt_;
            src = deferred_src_;
            deferred_prompt_.clear();
            run_deferred = !prompt.empty();
        }
        // 其次运行普通排队输入（等待 init 或 busy 解除后触发）。
        if (!run_deferred && has_pending_ && IsReadyUnlocked() && !infer_busy_) {
            prompt = pending_prompt_;
            src = pending_src_;
            pending_prompt_.clear();
            has_pending_ = false;
            run_deferred = !prompt.empty();
        }
    }

    if (cb && !banner.empty()) {
        cb(banner, banner_src, true);
        LogInfo("LlmWorker: turn done (%zu chars)", banner.size());
    }
    if (run_deferred) {
        RunPromptNow(prompt, src);
    }
}

// 源标识字符串化，便于日志定位来源链路。
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

// 轮询异步初始化 future，并在状态转移时输出系统提示。
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
        LogSystem("对话模型已就绪，人脸稳定后可输入");
    } else {
        init_state_ = InitState::Failed;
        LogWarn("LlmWorker: rkllm_init failed (async ret=%d)", ret);
        LogSystem("仅视觉模式（对话模型未加载）");
    }
}

// 等待推理线程结束，Shutdown 前必须 join 避免与 rkllm_destroy 并发。
void LlmWorker::JoinInferThread() {
    if (infer_thread_.joinable()) {
        infer_thread_.join();
    }
}

// 立即尝试发送一条输入：在独立线程上 rkllm_run，主循环不被阻塞。
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
        streamed_chars_ = 0;
        current_src_ = src;
    }

    JoinInferThread();
    infer_thread_ = std::thread([this, user_text]() {
        std::fprintf(stdout, "AI> ");
        std::fflush(stdout);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            reply_accumulator_.clear();
            if (tts_enabled_) {
                session_.SetReplyAccumulator(&reply_accumulator_);
            } else {
                session_.SetReplyAccumulator(nullptr);
            }
        }
        const int ret = session_.RunPromptSync(user_text);
        std::string reply;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            session_.SetReplyAccumulator(nullptr);
            reply = std::move(reply_accumulator_);
            infer_busy_ = false;
        }
        if (ret != 0) {
            LogWarn("LlmWorker: rkllm_run failed (%d)", ret);
        } else if (tts_ && tts_enabled_ && !reply.empty()) {
            tts_->PlayText(reply);
        }
    });
    return true;
}

// 门控入口：允许时直接跑，不允许或忙时走“只保留一条”排队策略。
bool LlmWorker::SubmitPrompt(const std::string& user_text, LlmPromptSource src, bool gate_open) {
    if (!gate_open) {
        return false;
    }
    if (user_text.empty()) {
        return false;
    }
    if (tts_) {
        tts_->Cancel();
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (IsLoadFailedUnlocked()) {
            return false;
        }
    }
    if (RunPromptNow(user_text, src)) {
        return true;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (infer_busy_ || IsInitializingUnlocked() || !IsReadyUnlocked()) {
        // 当前策略：只保留最后一条待处理输入，后来的会覆盖更早的一条。
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

// 请求 rkllm_abort，使推理线程尽快从 RunPromptSync 返回。
void LlmWorker::RequestAbortGeneration() {
    session_.Abort();
    if (tts_) {
        tts_->Cancel();
    }
}

// 关闭 worker：先 abort 再 join 推理线程，最后 destroy 会话。
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

    session_.Abort();
    JoinInferThread();
    session_.Shutdown();
    std::lock_guard<std::mutex> lock(mutex_);
    init_state_ = InitState::Uninitialized;
    infer_busy_ = false;
    pending_text_.clear();
    streamed_chars_ = 0;
    has_pending_ = false;
    pending_prompt_.clear();
    deferred_run_ = false;
    deferred_prompt_.clear();
    banner_pending_ = false;
    pending_banner_.clear();
}

// RKLLM 回调处理：NORMAL 已在 StaticCallback 直出 stdout；此处只处理 FINISH/ERROR 状态。
void LlmWorker::OnLlmChunk(const char* text_chunk, LLMCallState state) {
    (void)text_chunk;
    std::string queued_prompt;
    LlmPromptSource queued_src = LlmPromptSource::FaceAppear;
    bool has_queued = false;

    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (state == RKLLM_RUN_FINISH) {
            pending_text_.clear();
            streamed_chars_ = 0;
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
            streamed_chars_ = 0;
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
