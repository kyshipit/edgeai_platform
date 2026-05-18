/*
 * io/frame_transform.h
 * 帧旋转与合法性校验（与采集设备无关的图像变换）。
 */
#pragma once

#include <opencv2/opencv.hpp>
#include <string>

class FrameTransform {
public:
    explicit FrameTransform(const std::string& rotate = "ccw90");

    void Apply(cv::Mat& frame) const;
    bool Validate(const cv::Mat& frame, int frame_id) const;

private:
    int rotate_mode_ = 1;  // 0=none 1=ccw90 2=cw90 3=180
};

int ParseInputRotateMode(const std::string& rotate);
