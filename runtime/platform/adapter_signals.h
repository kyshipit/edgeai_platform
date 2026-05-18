/*
 * platform/adapter_signals.h
 *
 * 各适配器在 GetAdapterSignals() 中填写；ModelCoordinator 按 ownership 合并进 SharedState。
 */
#pragma once

#include <string>

struct AdapterSignals {
    bool person_present = false;       // YOLO：画面中是否有人
    bool face_detected = false;        // SCRFD：是否检测到人脸（需 C++ ScrfdAdapter）
    bool text_region_present = false;  // 文字区域出现（YOLO 自定义类或专用检测；标准 COCO 无此类）
    bool text_detected = false;        // PPOCR：是否识别到有效文本
    float avg_brightness = 0.0f;
    std::string scene_label;
};
