/*
 * adapters/ppocr/ppocr_adapter.h
 *
 * PPOCR 两阶段（det+rec）模型插件，薄封装 ppocr_system 例程 API。
 */
#pragma once

#include <memory>
#include <string>
#include <opencv2/opencv.hpp>
#include "engine/adapter_interface.h"
#include "platform/adapter_signals.h"
#include "ppocr_system.h"

class PPOCRAdapter : public IModelAdapter {
public:
    PPOCRAdapter(const std::string& det_path, const std::string& rec_path);
    void SetDetThresholds(float det_threshold, float box_threshold, float rec_score_threshold);
    void SetMinInferWidth(int min_width);
    ~PPOCRAdapter() override;

    int Init(const std::string& model_path, int npu_core_mask) override;
    uint8_t* Preprocess(const cv::Mat& frame, int& out_size) override;
    int Inference(std::shared_ptr<void>& model_output) override;
    std::string Postprocess(const std::shared_ptr<void>& model_output) override;
    std::shared_ptr<IModelAdapter> Clone() const override;
    AdapterSignals GetAdapterSignals() const override;

    const ppocr_text_recog_array_result_t& GetLastResults() const;

private:
    std::string det_path_;
    std::string rec_path_;
    ppocr_system_app_context sys_ctx_;
    ppocr_det_postprocess_params det_params_;
    ppocr_text_recog_array_result_t last_results_;
    image_buffer_t src_image_;
    cv::Mat frame_rgb_;
    cv::Mat frame_infer_;
    float infer_inv_scale_ = 1.0f;
    float rec_score_threshold_ = 0.35f;
    int min_infer_width_ = 960;
    int last_raw_result_count_ = 0;
    int infer_log_counter_ = 0;
    bool initialized_ = false;
};
