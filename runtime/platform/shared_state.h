/*
 * platform/shared_state.h
 *
 * 跨帧合并后的决策状态（去抖后由 ModelCoordinator 读写）。
 */
#pragma once

#include <string>

struct SharedState {
    bool person_present = false;
    bool face_detected = false;
    float avg_brightness = 0.0f;
    std::string scene_label;
};
