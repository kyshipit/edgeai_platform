/*
 * io/camera_source.h
 * V4L2 / OpenCV 摄像头采集（打开、读帧、释放）。
 */
#pragma once

#include <atomic>
#include <opencv2/opencv.hpp>
#include <string>

class CameraSource {
public:
    CameraSource(const std::string& source, int width = 0, int height = 0);

    bool Open();
    void Release();
    bool IsOpened() const;
    bool ReadFrame(cv::Mat& frame, const std::atomic<bool>* stop_flag);

private:
    std::string source_;
    int width_ = 0;
    int height_ = 0;
    cv::VideoCapture capture_;
};
