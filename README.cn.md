🌐 语言: **中文** | [English](README.md)

# RK3588 Edge AI Inference Platform

基于 Rockchip RK3588 NPU 的端侧 AI 演示平台（正点原子开发板）。摄像头取流 → NPU 视觉推理 → 人脸稳定后本地大模型对话，**全部在板端完成**。

主程序 **edgeai_platform_app**，由 `runtime/config/default.yaml` 驱动；运行需 `runtime/` 编译产物与 `model/` 下 RKNN / RKLLM 权重。

## 链路

有人靠近 → YOLO 检人体 → SCRFD 检人脸 → 问候 / 终端对话，屏幕同步画框预览。

多模型按场景自动启停；LLM 在独立线程运行，不占每帧视觉 Preprocess→Inference→Postprocess。

## 流程

1. **启动**：读 YAML，加载 YOLO；启用 LLM 时后台预加载 RKLLM。
2. **每帧**：采集 → 推理当前视觉槽 → 主线程画框显示。
3. **有人**：启用 SCRFD；双槽同开时优先显示人脸框。
4. **有人脸**：开门控、自动问候；`YOU>` 输入，`AI>` 流式回复。
5. **人脸离开**：拒新输入，当前句说完；模型保持加载。
6. **退出**：ESC / Ctrl+C，释放摄像头、线程、LLM。

终端：`SYS>` / `YOU>` / `AI>` → stdout；`[INFO]` 等 → stderr。

## 架构

![EdgeAI 架构](assets/architecture.svg)

实线：视频帧与推理结果。虚线：YAML 配置、人体/人脸检测信号。LLM 仅在门控打开时于 `adapters/llm/` 独立线程生成。

| 层 | 目录 | 职责 |
|----|------|------|
| 入口 | `runtime/app/` | 读 YAML，启动 Pipeline 与 ModelCoordinator |
| 采集 / 显示 | `runtime/capture/` `runtime/display/` | 读帧旋转、画框、OpenCV 窗口 |
| 流水线 | `runtime/engine/` | 预处理 → 推理 → 主线程显示 |
| 策略 | `runtime/platform/` | 场景切换、人脸门控、自动问候 |
| 模型 | `runtime/adapters/` | yolo / scrfd / llm 插件，按槽启停 |

## 目录

```text
edgeai_platform/
├── model/          # yolov5.rknn、scrfd.rknn、.rkllm
├── assets/         # 架构图等
├── runtime/
│   ├── app/ engine/ platform/ capture/ display/
│   ├── adapters/yolo|scrfd|llm/
│   └── config/default.yaml
└── deploy/         # PC 侧转换，不参与板端运行
```

## 上手

环境：正点原子 RK3588，`/opt/atk-dlrk3588-toolchain`，模型放入 `model/`。

```bash
cd runtime && ./build-linux.sh
cd install/rk3588_linux_aarch64/rknn_edgeai_platform
./edgeai_platform_app config/default.yaml
```

按板端改 `config/default.yaml` 中的摄像头、模型路径、`model.llm.enabled`。

## License

MIT License
