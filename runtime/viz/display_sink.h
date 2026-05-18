/*
 * viz/display_sink.h
 * 显示输出抽象（imshow / headless）。
 */
#pragma once

#include <memory>
#include <opencv2/opencv.hpp>

#include "display_layout.h"

class IDisplaySink {
public:
    virtual ~IDisplaySink() = default;
    virtual void Prepare() = 0;
    virtual void Show(const cv::Mat& frame) = 0;
    virtual int PollKey(int delay_ms) = 0;
    virtual void Shutdown() = 0;
};

std::unique_ptr<IDisplaySink> CreateOpenCVDisplaySink(const DisplayWindowConfig& cfg,
                                                      const char* window_name = "EdgeAI Platform");
