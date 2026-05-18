/*
 * adapters/scrfd/scrfd_postprocess.cpp
 */
#include "scrfd_postprocess.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace {

constexpr int kInputSize = 640;
constexpr int kFmc = 3;
constexpr int kStrides[3] = {8, 16, 32};
constexpr int kNumAnchors = 2;

static void Distance2Bbox(const float* points_xy, int num_points,
                          const float* distance, int distance_cols,
                          std::vector<float>& boxes_xyxy) {
    boxes_xyxy.resize(static_cast<size_t>(num_points) * 4);
    for (int i = 0; i < num_points; ++i) {
        const float px = points_xy[i * 2];
        const float py = points_xy[i * 2 + 1];
        boxes_xyxy[i * 4 + 0] = px - distance[i * distance_cols + 0];
        boxes_xyxy[i * 4 + 1] = py - distance[i * distance_cols + 1];
        boxes_xyxy[i * 4 + 2] = px + distance[i * distance_cols + 2];
        boxes_xyxy[i * 4 + 3] = py + distance[i * distance_cols + 3];
    }
}

static int TensorElements(const rknn_tensor_attr& attr) {
    int n = 1;
    for (int i = 0; i < attr.n_dims; ++i) {
        n *= attr.dims[i];
    }
    return n;
}

static int TensorHeight(const rknn_tensor_attr& attr) {
    if (attr.n_dims >= 4) {
        return attr.dims[2];
    }
    if (attr.n_dims >= 3) {
        return attr.dims[1];
    }
    return 1;
}

static int TensorWidth(const rknn_tensor_attr& attr) {
    if (attr.n_dims >= 4) {
        return attr.dims[3];
    }
    if (attr.n_dims >= 3) {
        return attr.dims[2];
    }
    return 1;
}

static int FindOutputByName(const scrfd_app_context_t* ctx, const char* prefix, int stride) {
    if (!ctx || !prefix) {
        return -1;
    }
    char expected[32] = {0};
    snprintf(expected, sizeof(expected), "%s_%d", prefix, stride);
    for (uint32_t i = 0; i < ctx->io_num.n_output; ++i) {
        const char* name = ctx->output_attrs[i].name;
        if (name != nullptr && strcmp(name, expected) == 0) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

static bool ResolveScrfdHeadOutputs(const scrfd_app_context_t* ctx,
                                    int scale_idx,
                                    int* out_score,
                                    int* out_bbox,
                                    int* out_kps) {
    if (!ctx || !out_score || !out_bbox || !out_kps) {
        return false;
    }
    const int stride = kStrides[scale_idx];
    const int by_name_score = FindOutputByName(ctx, "score", stride);
    const int by_name_bbox = FindOutputByName(ctx, "bbox", stride);
    const int by_name_kps = FindOutputByName(ctx, "kps", stride);
    if (by_name_score >= 0 && by_name_bbox >= 0 && by_name_kps >= 0) {
        *out_score = by_name_score;
        *out_bbox = by_name_bbox;
        *out_kps = by_name_kps;
        return true;
    }

    // 兼容按类型分组布局: [score_8,score_16,score_32,bbox_8,...,kps_32]
    if (ctx->io_num.n_output >= 9) {
        *out_score = scale_idx;
        *out_bbox = scale_idx + 3;
        *out_kps = scale_idx + 6;
        return true;
    }

    // 兼容旧交错布局: [score_8,bbox_8,kps_8,score_16,...]
    *out_score = scale_idx * 3;
    *out_bbox = scale_idx * 3 + 1;
    *out_kps = scale_idx * 3 + 2;
    return (*out_kps < static_cast<int>(ctx->io_num.n_output));
}

}  // namespace

int scrfd_postprocess(scrfd_app_context_t* ctx,
                      const rknn_output* outputs,
                      const ScrfdLetterbox& letterbox,
                      float conf_threshold,
                      float nms_threshold,
                      std::vector<ScrfdFaceBox>& faces_out) {
    faces_out.clear();
    if (!ctx || !outputs) {
        return -1;
    }

    std::vector<float> all_scores;
    std::vector<cv::Rect2f> all_boxes;
    std::vector<std::vector<cv::Point2f>> all_kps;

    for (int idx = 0; idx < kFmc; ++idx) {
        const int stride = kStrides[idx];
        int out_score = -1;
        int out_bbox = -1;
        int out_kps = -1;
        if (!ResolveScrfdHeadOutputs(ctx, idx, &out_score, &out_bbox, &out_kps)) {
            return -1;
        }

        const float* scores = reinterpret_cast<const float*>(outputs[out_score].buf);
        const float* bbox_preds = reinterpret_cast<const float*>(outputs[out_bbox].buf);
        const float* kps_preds = reinterpret_cast<const float*>(outputs[out_kps].buf);
        if (!scores || !bbox_preds || !kps_preds) {
            continue;
        }

        const rknn_tensor_attr& score_attr = ctx->output_attrs[out_score];
        const int score_elems = TensorElements(score_attr);
        const int grid_h = kInputSize / stride;
        const int grid_w = kInputSize / stride;
        const int num_grid = grid_h * grid_w;
        const int num_points = num_grid * kNumAnchors;

        if (score_elems < num_points) {
            continue;
        }

        std::vector<float> anchor_xy(static_cast<size_t>(num_points) * 2);
        int p = 0;
        for (int y = 0; y < grid_h; ++y) {
            for (int x = 0; x < grid_w; ++x) {
                const float cx = static_cast<float>(x * stride);
                const float cy = static_cast<float>(y * stride);
                for (int a = 0; a < kNumAnchors; ++a) {
                    anchor_xy[p * 2] = cx;
                    anchor_xy[p * 2 + 1] = cy;
                    ++p;
                }
            }
        }

        std::vector<float> bbox_scaled(static_cast<size_t>(num_points) * 4);
        const int bbox_count = TensorElements(ctx->output_attrs[out_bbox]);
        for (int i = 0; i < num_points * 4 && i < bbox_count; ++i) {
            bbox_scaled[i] = bbox_preds[i] * static_cast<float>(stride);
        }

        std::vector<float> boxes_xyxy;
        Distance2Bbox(anchor_xy.data(), num_points, bbox_scaled.data(), 4, boxes_xyxy);

        const int kps_dim = 10;
        for (int i = 0; i < num_points; ++i) {
            if (scores[i] < conf_threshold) {
                continue;
            }

            float x1 = boxes_xyxy[i * 4 + 0];
            float y1 = boxes_xyxy[i * 4 + 1];
            float x2 = boxes_xyxy[i * 4 + 2];
            float y2 = boxes_xyxy[i * 4 + 3];

            for (int k = 0; k < kps_dim; k += 2) {
                const float px = anchor_xy[i * 2] + kps_preds[i * kps_dim + k] * static_cast<float>(stride);
                const float py = anchor_xy[i * 2 + 1] + kps_preds[i * kps_dim + k + 1] * static_cast<float>(stride);
                (void)px;
                (void)py;
            }

            std::vector<cv::Point2f> kps(5);
            for (int k = 0; k < 5; ++k) {
                kps[k].x = anchor_xy[i * 2] + kps_preds[i * kps_dim + k * 2] * static_cast<float>(stride);
                kps[k].y = anchor_xy[i * 2 + 1] + kps_preds[i * kps_dim + k * 2 + 1] * static_cast<float>(stride);
            }

            all_scores.push_back(scores[i]);
            all_boxes.emplace_back(x1, y1, x2 - x1, y2 - y1);
            all_kps.push_back(std::move(kps));
        }

        (void)TensorHeight(score_attr);
        (void)TensorWidth(score_attr);
    }

    if (all_scores.empty()) {
        return 0;
    }

    const float ratio_h = letterbox.orig_h > 0 && letterbox.new_h > 0
                            ? static_cast<float>(letterbox.orig_h) / static_cast<float>(letterbox.new_h)
                            : 1.f;
    const float ratio_w = letterbox.orig_w > 0 && letterbox.new_w > 0
                            ? static_cast<float>(letterbox.orig_w) / static_cast<float>(letterbox.new_w)
                            : 1.f;

    std::vector<cv::Rect> nms_boxes;
    std::vector<float> nms_scores;
    for (size_t i = 0; i < all_boxes.size(); ++i) {
        float x1 = (all_boxes[i].x - letterbox.pad_w) * ratio_w;
        float y1 = (all_boxes[i].y - letterbox.pad_h) * ratio_h;
        float w = all_boxes[i].width * ratio_w;
        float h = all_boxes[i].height * ratio_h;
        nms_boxes.emplace_back(static_cast<int>(x1), static_cast<int>(y1),
                               static_cast<int>(w), static_cast<int>(h));
        nms_scores.push_back(all_scores[i]);
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(nms_boxes, nms_scores, conf_threshold, nms_threshold, indices);

    for (int idx : indices) {
        if (idx < 0 || idx >= static_cast<int>(all_boxes.size())) {
            continue;
        }
        ScrfdFaceBox face;
        const cv::Rect& r = nms_boxes[idx];
        face.x1 = static_cast<float>(r.x);
        face.y1 = static_cast<float>(r.y);
        face.x2 = static_cast<float>(r.x + r.width);
        face.y2 = static_cast<float>(r.y + r.height);
        face.score = nms_scores[idx];
        for (int k = 0; k < 5; ++k) {
            face.kps[k].x = (all_kps[idx][k].x - letterbox.pad_w) * ratio_w;
            face.kps[k].y = (all_kps[idx][k].y - letterbox.pad_h) * ratio_h;
        }
        faces_out.push_back(face);
    }

    return 0;
}
