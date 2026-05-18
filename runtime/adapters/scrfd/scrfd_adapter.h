/*
 * adapters/scrfd/scrfd_adapter.h — SCRFD 人脸检测 RKNN 适配器
 */
#pragma once

#include <memory>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#include "engine/adapter_interface.h"
#include "scrfd_context.h"
#include "scrfd_postprocess.h"

class ScrfdAdapter : public IModelAdapter {
public:
    ScrfdAdapter();
    ~ScrfdAdapter() override;

    int Init(const std::string& model_path, int npu_core_mask) override;
    uint8_t* Preprocess(const cv::Mat& frame, int& out_size) override;
    int Inference(std::shared_ptr<void>& model_output) override;
    std::string Postprocess(const std::shared_ptr<void>& model_output) override;
    std::shared_ptr<IModelAdapter> Clone() const override;
    AdapterSignals GetAdapterSignals() const override;

    void SetThresholds(float conf_threshold, float nms_threshold);

    const std::vector<ScrfdFaceBox>& GetLastFaces() const;

private:
    struct RknnOutputHolder {
        std::vector<rknn_output> outputs;
        rknn_context ctx = 0;
    };

    scrfd_app_context_t app_ctx_;
    ScrfdLetterbox letterbox_;
    std::vector<ScrfdFaceBox> last_faces_;
    std::vector<uint8_t> input_buf_;
    float conf_threshold_ = 0.5f;
    float nms_threshold_ = 0.5f;
    bool initialized_ = false;
};
