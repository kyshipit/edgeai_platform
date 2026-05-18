/*
 * adapters/scrfd/scrfd_adapter.cpp
 */
#include "scrfd_adapter.h"

#include "file_utils.h"
#include "platform/adapter_signals.h"
#include "platform/logging.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>

namespace {

constexpr int kInputW = 640;
constexpr int kInputH = 640;

static int InitScrfdModel(const std::string& model_path, scrfd_app_context_t* app_ctx) {
    char* model = nullptr;
    const int model_len = read_data_from_file(model_path.c_str(), &model);
    if (model == nullptr) {
        LogError("ScrfdAdapter: load model failed: %s", model_path.c_str());
        return -1;
    }

    rknn_context ctx = 0;
    int ret = rknn_init(&ctx, model, model_len, 0, nullptr);
    free(model);
    if (ret < 0) {
        LogError("ScrfdAdapter: rknn_init failed ret=%d", ret);
        return -1;
    }

    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC) {
        rknn_destroy(ctx);
        return -1;
    }

    app_ctx->input_attrs = static_cast<rknn_tensor_attr*>(
        malloc(sizeof(rknn_tensor_attr) * io_num.n_input));
    app_ctx->output_attrs = static_cast<rknn_tensor_attr*>(
        malloc(sizeof(rknn_tensor_attr) * io_num.n_output));
    memset(app_ctx->input_attrs, 0, sizeof(rknn_tensor_attr) * io_num.n_input);
    memset(app_ctx->output_attrs, 0, sizeof(rknn_tensor_attr) * io_num.n_output);

    for (uint32_t i = 0; i < io_num.n_input; ++i) {
        app_ctx->input_attrs[i].index = i;
        rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &app_ctx->input_attrs[i], sizeof(rknn_tensor_attr));
    }
    for (uint32_t i = 0; i < io_num.n_output; ++i) {
        app_ctx->output_attrs[i].index = i;
        rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &app_ctx->output_attrs[i], sizeof(rknn_tensor_attr));
    }

    app_ctx->rknn_ctx = ctx;
    app_ctx->io_num = io_num;

    const rknn_tensor_attr& in0 = app_ctx->input_attrs[0];
    if (in0.fmt == RKNN_TENSOR_NCHW) {
        app_ctx->model_channel = in0.dims[1];
        app_ctx->model_height = in0.dims[2];
        app_ctx->model_width = in0.dims[3];
    } else {
        app_ctx->model_height = in0.dims[1];
        app_ctx->model_width = in0.dims[2];
        app_ctx->model_channel = in0.dims[3];
    }

    LogInfo("ScrfdAdapter: model loaded outputs=%u input=%dx%d", io_num.n_output,
            app_ctx->model_width, app_ctx->model_height);
    return 0;
}

static void ReleaseScrfdModel(scrfd_app_context_t* app_ctx) {
    if (app_ctx->input_attrs) {
        free(app_ctx->input_attrs);
        app_ctx->input_attrs = nullptr;
    }
    if (app_ctx->output_attrs) {
        free(app_ctx->output_attrs);
        app_ctx->output_attrs = nullptr;
    }
    if (app_ctx->rknn_ctx) {
        rknn_destroy(app_ctx->rknn_ctx);
        app_ctx->rknn_ctx = 0;
    }
}

static cv::Mat LetterboxResize(const cv::Mat& src, ScrfdLetterbox& meta) {
    meta.orig_w = src.cols;
    meta.orig_h = src.rows;
    meta.pad_w = 0;
    meta.pad_h = 0;
    meta.new_w = kInputW;
    meta.new_h = kInputH;

    cv::Mat out;
    if (src.empty()) {
        return out;
    }

    const float hw_scale = static_cast<float>(src.rows) / static_cast<float>(src.cols);
    if (src.rows != src.cols) {
        if (hw_scale > 1.f) {
            meta.new_h = kInputH;
            meta.new_w = static_cast<int>(kInputW / hw_scale);
            cv::resize(src, out, cv::Size(meta.new_w, meta.new_h), 0, 0, cv::INTER_AREA);
            meta.pad_w = (kInputW - meta.new_w) / 2;
            cv::copyMakeBorder(out, out, 0, 0, meta.pad_w, kInputW - meta.new_w - meta.pad_w,
                               cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
        } else {
            meta.new_w = kInputW;
            meta.new_h = static_cast<int>(kInputH * hw_scale) + 1;
            cv::resize(src, out, cv::Size(meta.new_w, meta.new_h), 0, 0, cv::INTER_AREA);
            meta.pad_h = (kInputH - meta.new_h) / 2;
            cv::copyMakeBorder(out, out, meta.pad_h, kInputH - meta.new_h - meta.pad_h, 0, 0,
                               cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
        }
    } else {
        cv::resize(src, out, cv::Size(kInputW, kInputH), 0, 0, cv::INTER_AREA);
    }
    return out;
}

}  // namespace

ScrfdAdapter::ScrfdAdapter() {
    memset(&app_ctx_, 0, sizeof(app_ctx_));
}

ScrfdAdapter::~ScrfdAdapter() {
    if (initialized_) {
        ReleaseScrfdModel(&app_ctx_);
        initialized_ = false;
    }
}

void ScrfdAdapter::SetThresholds(float conf_threshold, float nms_threshold) {
    conf_threshold_ = conf_threshold;
    nms_threshold_ = nms_threshold;
}

int ScrfdAdapter::Init(const std::string& model_path, int npu_core_mask) {
    if (initialized_) {
        return 0;
    }
    if (InitScrfdModel(model_path, &app_ctx_) != 0) {
        return -1;
    }
    rknn_set_core_mask(app_ctx_.rknn_ctx, static_cast<rknn_core_mask>(npu_core_mask));
    input_buf_.resize(static_cast<size_t>(kInputW * kInputH * 3));
    initialized_ = true;
    return 0;
}

uint8_t* ScrfdAdapter::Preprocess(const cv::Mat& frame, int& out_size) {
    if (!initialized_ || frame.empty()) {
        out_size = 0;
        return nullptr;
    }
    cv::Mat letterboxed = LetterboxResize(frame, letterbox_);
    if (letterboxed.empty() || !letterboxed.isContinuous()) {
        letterboxed = letterboxed.clone();
    }
    const size_t bytes = letterboxed.total() * letterboxed.elemSize();
    if (input_buf_.size() < bytes) {
        input_buf_.resize(bytes);
    }
    memcpy(input_buf_.data(), letterboxed.data, bytes);
    out_size = static_cast<int>(bytes);
    return input_buf_.data();
}

int ScrfdAdapter::Inference(std::shared_ptr<void>& model_output) {
    if (!initialized_) {
        return -1;
    }

    rknn_input input;
    memset(&input, 0, sizeof(input));
    input.index = 0;
    input.type = RKNN_TENSOR_UINT8;
    input.fmt = RKNN_TENSOR_NHWC;
    input.size = kInputW * kInputH * 3;
    input.buf = input_buf_.data();

    int ret = rknn_inputs_set(app_ctx_.rknn_ctx, 1, &input);
    if (ret < 0) {
        return ret;
    }
    ret = rknn_run(app_ctx_.rknn_ctx, nullptr);
    if (ret < 0) {
        return ret;
    }

    auto holder = std::make_shared<RknnOutputHolder>();
    holder->ctx = app_ctx_.rknn_ctx;
    holder->outputs.resize(app_ctx_.io_num.n_output);
    for (uint32_t i = 0; i < app_ctx_.io_num.n_output; ++i) {
        holder->outputs[i].index = i;
        holder->outputs[i].want_float = 1;
    }
    ret = rknn_outputs_get(app_ctx_.rknn_ctx, app_ctx_.io_num.n_output, holder->outputs.data(), nullptr);
    if (ret < 0) {
        return ret;
    }
    model_output = holder;
    return 0;
}

std::string ScrfdAdapter::Postprocess(const std::shared_ptr<void>& model_output) {
    last_faces_.clear();
    auto holder = std::static_pointer_cast<RknnOutputHolder>(model_output);
    if (!holder || holder->outputs.empty()) {
        return std::string();
    }

    scrfd_postprocess(&app_ctx_, holder->outputs.data(), letterbox_, conf_threshold_, nms_threshold_,
                      last_faces_);

    if (app_ctx_.rknn_ctx) {
        rknn_outputs_release(app_ctx_.rknn_ctx, app_ctx_.io_num.n_output, holder->outputs.data());
    }

    std::ostringstream out;
    for (const auto& f : last_faces_) {
        const int x1 = static_cast<int>(f.x1);
        const int y1 = static_cast<int>(f.y1);
        const int x2 = static_cast<int>(f.x2);
        const int y2 = static_cast<int>(f.y2);
        out << "face " << x1 << " " << y1 << " " << x2 << " " << y2 << " " << f.score << "\n";
    }
    return out.str();
}

std::shared_ptr<IModelAdapter> ScrfdAdapter::Clone() const {
    auto copy = std::make_shared<ScrfdAdapter>();
    copy->SetThresholds(conf_threshold_, nms_threshold_);
    return copy;
}

AdapterSignals ScrfdAdapter::GetAdapterSignals() const {
    AdapterSignals signals;
    signals.face_detected = !last_faces_.empty();
    return signals;
}

const std::vector<ScrfdFaceBox>& ScrfdAdapter::GetLastFaces() const {
    return last_faces_;
}
