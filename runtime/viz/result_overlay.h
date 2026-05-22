/*
 * viz/result_overlay.h
 * 将适配器 Postprocess 行文本绘制到帧上。
 */
#pragma once

#include <opencv2/opencv.hpp>
#include <string>

class ResultOverlay {
public:
    void Apply(cv::Mat& frame, const std::string& result_json,
               bool suppress_yolo_person = false) const;
    void DrawModelBadge(cv::Mat& frame, const std::string& model_name) const;
    void DrawGreetingBanner(cv::Mat& frame, const std::string& text) const;
};
