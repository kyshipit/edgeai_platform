/*
 * engine/adapter_interface.h
 *
 * 【engine 层】模型插件统一协议（与具体 RKNN/YOLO 无关）。
 */
#pragma once
#include <memory>
#include <string>
#include <opencv2/opencv.hpp>
#include "platform/adapter_signals.h"

class IModelAdapter {
public:
    virtual ~IModelAdapter() = default;

    virtual int Init(const std::string& model_path, int npu_core_mask) = 0;
    virtual uint8_t* Preprocess(const cv::Mat& frame, int& out_size) = 0;
    virtual int Inference(std::shared_ptr<void>& model_output) = 0;
    virtual std::string Postprocess(const std::shared_ptr<void>& model_output) = 0;
    virtual std::shared_ptr<IModelAdapter> Clone() const = 0;
    virtual AdapterSignals GetAdapterSignals() const = 0;
};
