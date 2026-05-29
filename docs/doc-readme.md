# Runtime 开发说明

板端开发与排障入口。

| 入口 | 内容 |
|------|------|
| 仓库根 [README.md](../README.md) | 项目总览、架构、快速上手 |
| [系统架构与运行逻辑.md](系统架构与运行逻辑.md) | **平台**框架、模块加载顺序、槽关系、设计取舍（**推荐阅读**） |
| [接续开发说明.md](接续开发说明.md) | 目录、多槽、配置、编译（板端开发必读） |
| [仓库与文档说明.md](仓库与文档说明.md) | 唯一主仓路径；勿在 Cursor worktree 副本编辑 |

**唯一仓库路径：** `/home/ky/work/edgeai_platform`

## 能力摘要

- **可执行文件**：`edgeai_platform_app`（`runtime/` 下 `./build-linux.sh`）
- **视觉槽**：YOLOv5（哨兵）、SCRFD（人脸）；`ModelCoordinator` 去抖切换
- **LLM**（`model.llm.enabled`）：自动问候为 yaml 静态文案 + `SetBannerLine`（不经 RKLLM）；终端 `YOU>` → `SubmitPrompt` → 独立线程 `rkllm_run`
- **TTS**（`model.llm.tts.enabled`）：播报 `AI>` 问候与 RKLLM 整轮回复（MeloTTS + `gst-play-1.0`）
- **流水线**：预处理 + 推理线程池 + 主线程显示；终端 `SYS>`/`YOU>`/`AI>` 走 stdout
- **配置**：`config/default.yaml` 为唯一默认来源

**已移除：** PPOCR / document 场景。根目录 `deploy/` 不参与板端构建。`runtime/3rdparty`、`runtime/utils` 为正点原子/RK 上游，勿随意修改。

细节见 [系统架构与运行逻辑.md](系统架构与运行逻辑.md)、[接续开发说明.md](接续开发说明.md)、[LLM与ModelCoordinator集成.md](LLM与ModelCoordinator集成.md)。

## 常见问题

**YOLO 0 框，日志 `output num: 9`**

- `model.yolo.path` 误指 `scrfd.rknn`；YOLO 应为 3 路，9 路是 SCRFD → [YOLO与SCRFD问题排查记录.md](YOLO与SCRFD问题排查记录.md)

**SCRFD 满屏乱框**

- 9 路输出索引与后处理布局不符 → 同上专文 §4.4

**缺 `.rkllm` 却报 YOLO Init 失败**

- 多为 LLM 预加载与视觉争 NPU；`LlmWorker` 应对缺失文件 stat 预检。见 [系统架构与运行逻辑.md](系统架构与运行逻辑.md) §3、§5

**Ctrl+C 不退出**

- 队列或 LLM 推理线程阻塞；查 `TryPush`、哨兵帧 → [错误修复调试说明.md](错误修复调试说明.md)、排查记录 §4.5

**退出 SIGSEGV**

- OpenCV / 摄像头 / 线程释放顺序竞态 → [错误修复调试说明.md](错误修复调试说明.md)

**LLM 回答截断**

- 查 `max_new_tokens`；R1 类模型 thinking 段占 token → [LLM与ModelCoordinator集成.md](LLM与ModelCoordinator集成.md)

**有 YOU> 无 AI> 或极慢**

- 查 `model.llm.enabled`、`rkllm_init`；与 YOLO/SCRFD 争 NPU 时延迟上升

**无预览窗口**

- `input.show_window: false` 或无 GUI 时 headless 降级

## 文档索引

| 文档 | 说明 |
|------|------|
| [系统架构与运行逻辑.md](系统架构与运行逻辑.md) | 平台框架、加载顺序、参考应用场景 |
| [接续开发说明.md](接续开发说明.md) | 目录、多槽、配置、编译 |
| [适配器说明.md](适配器说明.md) | YOLO / SCRFD / LLM / TTS 适配器速览 |
| [LLM与ModelCoordinator集成.md](LLM与ModelCoordinator集成.md) | RKLLM、门控、终端 UX |
| [TTS与MeloTTS集成说明.md](TTS与MeloTTS集成说明.md) | MeloTTS 配置与验收 |
| [YOLO与SCRFD问题排查记录.md](YOLO与SCRFD问题排查记录.md) | 路径与拓扑排障 |
| [错误修复调试说明.md](错误修复调试说明.md) | 崩溃与退出 |
| [模型演进与待办.md](模型演进与待办.md) | 演进路线与 backlog（非验收） |
| [仓库与文档说明.md](仓库与文档说明.md) | 主仓与 worktree |

## License

MIT License
