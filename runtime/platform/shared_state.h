/*
 * platform/shared_state.h
 *
 * 跨帧、跨模型合并后的决策状态（去抖后由 ModelCoordinator 读写）。
 */
#pragma once

#include <string>

struct SharedState {
    bool person_present = false;
    bool face_detected = false;
    bool text_region_present = false;
    bool text_detected = false;
    float avg_brightness = 0.0f;
    std::string scene_label;
};
