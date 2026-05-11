# RK3588 General Edge AI Inference Platform

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Rockchip_RK3588](https://img.shields.io/badge/SoC-Rockchip_RK3588-1B8A6D)
![AI](https://img.shields.io/badge/AI-On--Device_6TOPS-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)

## Overview

A general-purpose, extensible edge AI inference platform built on the Rockchip RK3588 NPU (6 TOPS). By decoupling model-specific pre/post-processing from the core inference engine through a plugin-based adapter architecture, the platform supports dynamic switching between multiple model types — including CNN-based object detection and Transformer-based image classification — without modifying the main application code.

**Keywords:** RK3588, RKNN, INT8 Quantization, C++ Multithreading, ViT, YOLO, Edge AI

## Features

- **Plugin-based Adapter Architecture** — Core inference engine separated from model-specific adapters via a stable abstract interface. Adding a new model requires only implementing a new adapter, leaving the main framework unchanged.

- **Multi-Model Support** — Validated with both CNN (YOLOv5/v8) and Transformer (ViT) architectures, proving the platform's capability to deploy structurally different model families on the same NPU.

- **Full-Stack INT8 Quantization Pipeline** — PyTorch → ONNX → .rknn conversion with hybrid INT8 quantization, including calibration dataset management and accuracy-vs-speed benchmarking.

- **High-Performance C++ Multithreading Inference** — Thread-pool-based concurrent NPU scheduling maximizing tri-core NPU utilization. Video capture, NPU inference, and result rendering run in independent pipeline stages.

- **Industrial-Grade Multi-Stream Processing** — Multi-channel RTSP stream ingestion with hardware-accelerated decoding (FFmpeg + RKMPP + RGA), mimicking real-world video surveillance and industrial inspection workloads.

- **Comprehensive Performance Evaluation** — Quantization precision comparison reports, end-to-end latency measurement, and NPU/CPU/memory utilization profiling.

## Hardware & Software Baseline

| Component               | Specification / Version                     |
|-------------------------|---------------------------------------------|
| Development Board       | ALIENTEK-RK3588                             |
| Linux Kernel            | 6.1                                         |
| NPU Driver (RKNPU2)     | 0.9.8                                       |
| PC-Side Toolkit         | RKNN-Toolkit2 ≥ 2.0.0                       |
| Board-Side Runtime      | librknn_api.so                              |
| Build System            | CMake ≥ 3.16, GCC ≥ 9 (cross-compilation)   |
| Video Processing        | OpenCV, RGA, FFmpeg + RKMPP                 |

## Architecture

![System Architecture](./assets/architecture.svg)

## Repository Structure

```
edgeai_platform/
├── model_zoo/                  # Model conversion, quantization (PC-side)
│   ├── output/                 # onnx and rknn end files
│   ├── convert/                # Model conversion scripts (PyTorch → ONNX → RKNN)
│   ├── data/                   # INT8 quantization calibration dataset
│   └── requirements.txt        # Python dependencies
├── runtime/                    # On-device C++ inference framework (cross-compile)
│   ├── CMakeLists.txt          # Cross-compilation build config
│   ├── main.cpp                # Application entry point
│   ├── engine/                 # Core inference engine (thread pool, scheduler, pipeline)
│   ├── adapters/               # Model-specific adapters (YOLO, ViT, ResNet)
│   ├── preprocess/             # Image preprocessing (RGA hardware acceleration)
│   ├── config/                 # YAML runtime configuration files
│   └── utils/                  # Shared utilities (logger, timer, config parser)
├── assets/                     # Static assets (benchmark charts, screenshots, architecture diagrams)
├── tests/                      # Unit tests and integration tests
├── tools/                      # Helper scripts (board info, model deployment)
├── CHANGELOG.md
└── README.md
```

## Skill Coverage

| Skill Category | Specific Technique | Implementation |
|----------------|---------------------|----------------|
| Model Format Conversion | PyTorch → ONNX → .rknn | model_conversion/ scripts |
| Inference Framework | RKNN Runtime API (C++) | inference/engine/ |
| Model Compression | INT8 / Hybrid Quantization | Calibration set + comparison report |
| CNN Deployment | YOLOv5/v8 object detection | adapters/yolo_adapter.cpp |
| Transformer Deployment | ViT image classification | adapters/vit_adapter.cpp |
| C++ Multithreading | Thread pool + pipeline parallelism | engine/rknn_pool.hpp |
| Hardware Acceleration | RGA preprocessing, NPU offloading | utils/ RGA integration |
| Operator Fusion | Conv+BN+ReLU auto-merge | Conversion documentation |
| Multi-Stream Video | RTSP ingest + MPP hardware decode | config/multi_stream.yaml |
| Model Training & Evaluation | PyTorch fine-tuning + mAP/Top-1 | Training scripts & logs |

## Performance Benchmarks

Measured on ALIENTEK-RK3588 (Linux 6.1, driver 0.9.8) with single-stream RGA.
preprocessing pipeline:

| Model | Input Size | Quantization | NPU FPS | Model Size | Accuracy |
|--------|-------------|---------------|----------|-------------|-----------|
| YOLOv8n | 640×640 | INT8 | ≈40 FPS | ≈6 MB | mAP@0.5 ≈ baseline−2% |
| YOLOv8s | 640×640 | INT8 | 18–28 FPS | ≈11 MB | mAP@0.5 ≈ baseline−2% |
| MobileNetV2 | 224×224 | INT8 | >60 FPS | ≈3 MB | Top-1 ≈ baseline−1% |
| ViT-Tiny | 224×224 | INT8 hybrid | >15 FPS | ≈5 MB | Top-1 ≈ baseline−1.5% |

Multi-stream stress test (reference data):

| Streams | Resolution | NPU Load | CPU Load | Per-Stream FPS |
|----------|------------|-----------|----------|----------------|
| 4 | 1080P | ~70% | ~50% | ≥20 FPS |
| 6 | 1080P | ~90% | ~65% | ≥15 FPS |

## Demo Scenarios
- **YOLO Real-Time Detection** — Live camera feed with detection bounding boxes and real-time FPS overlay.

- **Dynamic Model Switching** — Switch from detection to classification by modifying config file, no recompilation required.

- **INT8 Quantization Comparison** — Display comparison chart of speed, size, and accuracy before/after quantization.

- **Multi-Stream RTSP Inference (Optional)** — 4–6 concurrent RTSP streams with independent inference results.

## Credits
This project builds upon the following community resources:

- airockchip/rknn_model_zoo

- leafqycc/rknn-cpp-Multithreading

- ZIFENG278/RK3588-Multi-Stream-YOLOv8-Detection

- BXC_VideoAnalyzer_v4

- ALIENTEK-RK3588 development board documentation and examples

## License
MIT License