/*
 * adapters/ppocr/ppocr_adapter.cpp
 */
#include "ppocr_adapter.h"
#include "rknn_api.h"
#include "image_utils.h"
#include "platform/logging.h"
#include <cstring>
#include <sstream>
#include <algorithm>

namespace {
constexpr float kDefaultDetThreshold = 0.2f;
constexpr float kDefaultBoxThreshold = 0.45f;
constexpr float kDefaultRecScore = 0.35f;
constexpr float kDbUnclipRatio = 1.5f;
}  // namespace

PPOCRAdapter::PPOCRAdapter(const std::string& det_path, const std::string& rec_path)
    : det_path_(det_path), rec_path_(rec_path) {
    memset(&sys_ctx_, 0, sizeof(sys_ctx_));
    memset(&last_results_, 0, sizeof(last_results_));
    memset(&src_image_, 0, sizeof(src_image_));
    det_params_.threshold = kDefaultDetThreshold;
    det_params_.box_threshold = kDefaultBoxThreshold;
    det_params_.use_dilate = false;
    det_params_.db_score_mode = const_cast<char*>("slow");
    det_params_.db_box_type = const_cast<char*>("poly");
    det_params_.db_unclip_ratio = kDbUnclipRatio;
    rec_score_threshold_ = kDefaultRecScore;
}

PPOCRAdapter::~PPOCRAdapter() {
    if (initialized_) {
        release_ppocr_model(&sys_ctx_.det_context);
        release_ppocr_model(&sys_ctx_.rec_context);
        initialized_ = false;
    }
}

void PPOCRAdapter::SetDetThresholds(float det_threshold, float box_threshold, float rec_score_threshold) {
    det_params_.threshold = det_threshold;
    det_params_.box_threshold = box_threshold;
    rec_score_threshold_ = rec_score_threshold;
}

void PPOCRAdapter::SetMinInferWidth(int min_width) {
    min_infer_width_ = min_width > 0 ? min_width : 960;
}

int PPOCRAdapter::Init(const std::string& model_path, int npu_core_mask) {
    (void)model_path;
    if (initialized_) {
        return 0;
    }
    int ret = init_ppocr_model(det_path_.c_str(), &sys_ctx_.det_context);
    if (ret != 0) {
        return ret;
    }
    ret = init_ppocr_model(rec_path_.c_str(), &sys_ctx_.rec_context);
    if (ret != 0) {
        release_ppocr_model(&sys_ctx_.det_context);
        return ret;
    }
    rknn_set_core_mask(sys_ctx_.det_context.rknn_ctx, static_cast<rknn_core_mask>(npu_core_mask));
    rknn_set_core_mask(sys_ctx_.rec_context.rknn_ctx, static_cast<rknn_core_mask>(npu_core_mask));
    initialized_ = true;
    LogInfo("PPOCRAdapter: init ok det_th=%.2f box_th=%.2f rec_th=%.2f min_infer_w=%d",
            det_params_.threshold, det_params_.box_threshold, rec_score_threshold_, min_infer_width_);
    return 0;
}

uint8_t* PPOCRAdapter::Preprocess(const cv::Mat& frame, int& out_size) {
    if (!initialized_ || frame.empty()) {
        out_size = 0;
        return nullptr;
    }
    if (frame.channels() == 3) {
        cv::cvtColor(frame, frame_rgb_, cv::COLOR_BGR2RGB);
    } else if (frame.channels() == 4) {
        cv::cvtColor(frame, frame_rgb_, cv::COLOR_BGRA2RGB);
    } else {
        frame_rgb_ = frame.clone();
    }

    infer_inv_scale_ = 1.0f;
    const cv::Mat* infer_src = &frame_rgb_;
    if (frame_rgb_.cols < min_infer_width_) {
        const float scale = static_cast<float>(min_infer_width_) / static_cast<float>(frame_rgb_.cols);
        cv::resize(frame_rgb_, frame_infer_, cv::Size(), scale, scale, cv::INTER_LINEAR);
        infer_src = &frame_infer_;
        infer_inv_scale_ = 1.0f / scale;
    } else {
        frame_infer_.release();
    }

    src_image_.height = infer_src->rows;
    src_image_.width = infer_src->cols;
    src_image_.width_stride = static_cast<int>(infer_src->step[0]);
    src_image_.virt_addr = infer_src->data;
    src_image_.format = IMAGE_FORMAT_RGB888;
    src_image_.size = static_cast<int>(infer_src->total() * infer_src->elemSize());
    out_size = src_image_.size;
    return src_image_.virt_addr;
}

int PPOCRAdapter::Inference(std::shared_ptr<void>& model_output) {
    if (!initialized_) {
        return -1;
    }
    memset(&last_results_, 0, sizeof(last_results_));
    int ret = inference_ppocr_system_model(&sys_ctx_, &src_image_, &det_params_, &last_results_);
    last_raw_result_count_ = last_results_.count;
    if (ret != 0) {
        LogWarn("PPOCRAdapter: inference failed ret=%d", ret);
    } else if (last_raw_result_count_ == 0) {
        if (++infer_log_counter_ % 45 == 0) {
            LogInfo("PPOCRAdapter: 0 text regions frame=%dx%d infer_scale=%.2f",
                    src_image_.width, src_image_.height, infer_inv_scale_ > 0 ? (1.0f / infer_inv_scale_) : 1.0f);
        }
    } else if (infer_log_counter_ % 45 == 0) {
        LogInfo("PPOCRAdapter: %d text region(s) frame=%dx%d", last_raw_result_count_,
                src_image_.width, src_image_.height);
    }
    model_output.reset();
    return ret;
}

std::string PPOCRAdapter::Postprocess(const std::shared_ptr<void>& model_output) {
    (void)model_output;
    std::ostringstream out;
    const float inv = infer_inv_scale_;
    auto scale_coord = [inv](int v) { return static_cast<int>(v * inv + 0.5f); };

    for (int i = 0; i < last_results_.count; ++i) {
        const auto& item = last_results_.text_result[i];
        if (item.text.score < rec_score_threshold_) {
            continue;
        }
        const auto& b = item.box;
        out << "OCR "
            << scale_coord(b.left_top.x) << " " << scale_coord(b.left_top.y) << " "
            << scale_coord(b.right_top.x) << " " << scale_coord(b.right_top.y) << " "
            << scale_coord(b.right_bottom.x) << " " << scale_coord(b.right_bottom.y) << " "
            << scale_coord(b.left_bottom.x) << " " << scale_coord(b.left_bottom.y) << " "
            << item.text.score << " |" << item.text.str << "\n";
    }
    return out.str();
}

std::shared_ptr<IModelAdapter> PPOCRAdapter::Clone() const {
    auto copy = std::make_shared<PPOCRAdapter>(det_path_, rec_path_);
    copy->SetDetThresholds(det_params_.threshold, det_params_.box_threshold, rec_score_threshold_);
    copy->SetMinInferWidth(min_infer_width_);
    return copy;
}

AdapterSignals PPOCRAdapter::GetAdapterSignals() const {
    AdapterSignals signals;
    for (int i = 0; i < last_results_.count; ++i) {
        if (last_results_.text_result[i].text.score >= rec_score_threshold_ &&
            last_results_.text_result[i].text.str_size > 0) {
            signals.text_detected = true;
            break;
        }
    }
    return signals;
}

const ppocr_text_recog_array_result_t& PPOCRAdapter::GetLastResults() const {
    return last_results_;
}
