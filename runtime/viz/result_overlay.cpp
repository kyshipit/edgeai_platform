#include "result_overlay.h"
#include "platform/logging.h"

#include <sstream>

namespace {

float FontScaleForFrame(int rows) {
    if (rows >= 1600) {
        return 2.0f;
    }
    if (rows >= 900) {
        return 1.5f;
    }
    return 1.0f;
}

int LineThicknessForFrame(int rows) {
    if (rows >= 1600) {
        return 4;
    }
    if (rows >= 900) {
        return 3;
    }
    return 2;
}

}  // namespace

void ResultOverlay::DrawModelBadge(cv::Mat& frame, const std::string& model_name) const {
    if (frame.empty() || model_name.empty()) {
        return;
    }
    const float scale = FontScaleForFrame(frame.rows);
    const int thick = LineThicknessForFrame(frame.rows);
    std::string badge = "MODEL: " + model_name;
    int baseline = 0;
    cv::Size ts = cv::getTextSize(badge, cv::FONT_HERSHEY_SIMPLEX, scale, thick, &baseline);
    const int pad = 16;
    const int y0 = pad + ts.height;
    cv::rectangle(frame, cv::Point(pad, pad), cv::Point(pad + ts.width + pad * 2, y0 + baseline + pad),
                  cv::Scalar(0, 0, 0), cv::FILLED);
    cv::Scalar color = cv::Scalar(0, 255, 0);
    if (model_name.find("scrfd") != std::string::npos) {
        color = cv::Scalar(255, 0, 255);
    }
    cv::putText(frame, badge, cv::Point(pad + 4, y0), cv::FONT_HERSHEY_SIMPLEX, scale, color, thick);
}

void ResultOverlay::DrawGreetingBanner(cv::Mat& frame, const std::string& text) const {
    if (frame.empty() || text.empty()) {
        return;
    }
    const float scale = FontScaleForFrame(frame.rows) * 0.9f;
    const int thick = LineThicknessForFrame(frame.rows);
    int baseline = 0;
    cv::Size ts = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, scale, thick, &baseline);
    const int pad = 16;
    const int y = frame.rows - pad - baseline;
    cv::rectangle(frame, cv::Point(pad, y - ts.height - pad),
                  cv::Point(pad + ts.width + pad * 2, y + baseline + pad / 2), cv::Scalar(0, 0, 0),
                  cv::FILLED);
    cv::putText(frame, text, cv::Point(pad + 4, y), cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(255, 200, 0),
                thick);
}

void ResultOverlay::Apply(cv::Mat& frame, const std::string& result_json,
                          bool suppress_yolo_person) const {
    if (frame.empty() || frame.data == nullptr) {
        LogError("ResultOverlay: invalid frame buffer");
        return;
    }
    if (result_json.empty()) {
        return;
    }

    const float text_scale = FontScaleForFrame(frame.rows);
    const int box_thick = LineThicknessForFrame(frame.rows);

    std::istringstream ss(result_json);
    std::string line;
    while (std::getline(ss, line)) {
        if (line.empty()) {
            continue;
        }

        std::istringstream line_stream(line);
        std::string label;
        int x1, y1, x2, y2;
        float score;
        if (!(line_stream >> label >> x1 >> y1 >> x2 >> y2 >> score)) {
            continue;
        }

        if (suppress_yolo_person && label == "person") {
            continue;
        }

        x1 = std::max(0, std::min(x1, frame.cols - 1));
        y1 = std::max(0, std::min(y1, frame.rows - 1));
        x2 = std::max(0, std::min(x2, frame.cols - 1));
        y2 = std::max(0, std::min(y2, frame.rows - 1));

        const bool is_face = (label == "face");
        const cv::Scalar box_color = is_face ? cv::Scalar(255, 0, 255) : cv::Scalar(255, 0, 0);
        cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), box_color, box_thick);

        if (is_face) {
            const int kps_radius = std::max(3, box_thick + 1);
            const cv::Scalar kps_colors[5] = {
                cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 255),
                cv::Scalar(255, 128, 0), cv::Scalar(255, 128, 0)};
            for (int k = 0; k < 5; ++k) {
                int kx = 0;
                int ky = 0;
                if (!(line_stream >> kx >> ky)) {
                    break;
                }
                kx = std::max(0, std::min(kx, frame.cols - 1));
                ky = std::max(0, std::min(ky, frame.rows - 1));
                cv::circle(frame, cv::Point(kx, ky), kps_radius, kps_colors[k], cv::FILLED);
            }
        }

        std::ostringstream text_stream;
        text_stream << label << " " << static_cast<int>(score * 100) << "%";
        std::string text = text_stream.str();

        int baseline = 0;
        cv::Size text_size =
            cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, text_scale, box_thick, &baseline);
        cv::Point text_origin(x1, std::max(y1 - 5, text_size.height + 5));
        cv::rectangle(frame,
                      cv::Point(text_origin.x, text_origin.y - text_size.height - baseline),
                      cv::Point(text_origin.x + text_size.width, text_origin.y + 2),
                      cv::Scalar(0, 0, 0),
                      cv::FILLED);
        cv::putText(frame, text, text_origin, cv::FONT_HERSHEY_SIMPLEX, text_scale, cv::Scalar(0, 255, 0),
                    box_thick);
    }
}
