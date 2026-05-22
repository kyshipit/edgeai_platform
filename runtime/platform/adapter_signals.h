/*
 * platform/adapter_signals.h
 *
 * 各适配器在 GetAdapterSignals() 中填写；ModelCoordinator 合并进 SharedState。
 */
#pragma once

#include <string>

struct AdapterSignals {
    bool person_present = false;
    bool face_detected = false;
    float avg_brightness = 0.0f;
    std::string scene_label;
};
