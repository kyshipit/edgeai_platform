# TTS 与 MeloTTS 集成说明

> 实现代码：`runtime/adapters/tts/`；配置：`runtime/config/default.yaml` → `model.llm.tts`（需 `model.llm.enabled: true`）。

## 1. 概述

在 RK3588 Edge AI Runtime（`edgeai_platform_app`）上，将终端 **`AI>` 助手话术** 合成为语音并播放（**已集成** MeloTTS）：

- **静态自动问候**（yaml `auto_greeting_text`，经 `SetBannerLine` 上屏；**仅 LLM `IsReady()`** 时输出，缺 `.rkllm` 不播）
- **`YOU>` 后 RKLLM 整轮回复**（`reply_accumulator` 与 stdout 同源，**FINISH 后**整段播报）

不播报：`SYS>`、用户输入、视觉检测日志等。

技术选型：板端 **MeloTTS `ZH_MIX_EN`**（encoder/decoder RKNN + `lexicon.txt` / `tokens.txt`）。已弃用 `mms_tts` 英文音素方案。TTS 与 RKLLM 同为旁路，**不进**视觉 Pipeline / `IModelCoordinator`。

适配器文件与路径速览见 [适配器说明.md](适配器说明.md) § TTS。

## 2. 模块

| 文件 | 职责 |
|------|------|
| `tts_worker.*` | 异步队列、最新优先、`Cancel`/`Shutdown` |
| `melotts_session.*` | 加载 RKNN/词表、分句、`SynthesizeText` |
| `melotts_engine.*` | encoder/decoder RKNN 推理 |
| `melotts_process.*` | encoder 与 decoder 间 duration/attention |
| `tts_text_sanitizer.*` | 去 markdown、合并空白、长度截断 |
| `audio_player.*` | `save_audio` → `/tmp/edgeai_tts.wav` → `gst-play-1.0` |
| `lexicon.hpp` / `split.hpp` | 词表与分句 |

模型路径：仓库根 **`./model/`**（与 yolo/scrfd 同级）。音频落盘：`runtime/utils/audio_utils.c`（CMake 目标 `audioutils`）。链接 `libsndfile.a` 非 PIC 时，可执行文件使用 `-no-pie`（见 `runtime/CMakeLists.txt`）。

## 3. 数据流

```text
人脸稳定 → LlmGreeting::SetBannerLine(问候) → TtsWorker::PlayText
YOU> → rkllm_run → AI> 流式 + reply_accumulator → FINISH → TtsWorker::PlayText
TtsWorker → MeloTtsSession → RKNN → /tmp/edgeai_tts.wav → gst-play-1.0 -q --no-interactive
```

集成点：

- `LlmGreeting::SetTtsWorker`：`SetBannerLine(is_final)` 时播问候（可 `skip_static_greeting`）
- `LlmWorker` + `RkllmSession::SetReplyAccumulator`：NORMAL 累积，FINISH 后 `PlayText`
- `SubmitPrompt` / `Abort` / `Stop`：`TtsWorker::Cancel` 打断播放并清空待播

## 4. 配置

字段说明见 [`config/default.yaml`](../runtime/config/default.yaml) 中 `model.llm.tts` 注释。常用项：

| 字段 | 说明 |
|------|------|
| `enabled` | 总开关 |
| `skip_static_greeting` | `true` 仅播 RKLLM 回复 |
| `encoder_path` / `decoder_path` | RKNN 路径 |
| `lexicon_path` / `tokens_path` | 词表 |
| `language` | 如 `ZH_MIX_EN` |
| `speak_id` / `speed` / `disable_bert` | 推理参数 |
| `preload_on_startup` | 启动即异步加载，减首句延迟 |
| `max_speak_chars` | `0` 不截断 |

## 5. 已知限制

- **v1 batch**：RKLLM **FINISH 后**才整段合成，答完到出声有延迟；v2 可按句边生成边播（未实现）。
- **抢占**：新 `YOU>` 或 `Cancel` 会停播并只保留最新一轮文本。
- **清洗**：`TtsTextSanitizer` 去掉常见 markdown 符号。
- **资源**：TTS 在独立线程；与视觉/LLM 争 NPU 时延迟可能上升。

## 6. 验收

1. 默认配置：人脸稳定后有问候 **文字与语音**。
2. `YOU>` 对话：流式 `AI>`，**轮次结束后**播放该轮回复。
3. `skip_static_greeting: true`：仅文字问候，问答仍播。
4. `model.llm.tts.enabled: false`：全程无 TTS。

板端需已安装 `gst-play-1.0`。

## 7. 相关文档

- [适配器说明.md](适配器说明.md) § TTS
- [LLM与ModelCoordinator集成.md](LLM与ModelCoordinator集成.md)
- [接续开发说明.md](接续开发说明.md)

---

*实现说明文档；若与代码不一致，以代码为准。*
