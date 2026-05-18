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
    cv::Scalar color = (model_name == "ppocr") ? cv::Scalar(0, 255, 255) : cv::Scalar(0, 255, 0);
    cv::putText(frame, badge, cv::Point(pad + 4, y0), cv::FONT_HERSHEY_SIMPLEX, scale, color, thick);
}

void ResultOverlay::LogOcrResultsToTerminal(const std::string& result_json) {
    if (result_json.empty()) {
        LogInfo("[OCR] (no text detected this frame)");
        return;
    }
    std::istringstream ss(result_json);
    std::string line;
    int count = 0;
    while (std::getline(ss, line)) {
        if (line.compare(0, 4, "OCR ") != 0) {
            continue;
        }
        auto bar = line.find(" |");
        std::string text = (bar != std::string::npos && bar + 2 < line.size()) ? line.substr(bar + 2) : "";
        float score = 0.f;
        std::istringstream ls(line);
        std::string label;
        int coords[8];
        if (!(ls >> label >> coords[0] >> coords[1] >> coords[2] >> coords[3] >> coords[4] >>
              coords[5] >> coords[6] >> coords[7] >> score)) {
            continue;
        }
        ++count;
        LogInfo("[OCR #%d] score=%.2f text=\"%s\"", count, score, text.c_str());
    }
    if (count == 0) {
        LogInfo("[OCR] (no text above rec threshold; check lighting/focus or lower model.ppocr.rec_score_threshold)");
    }
}

void ResultOverlay::Apply(cv::Mat& frame, const std::string& result_json) const {
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
        if (line.compare(0, 4, "OCR ") == 0) {
            int x0, y0, x1, y1, x2, y2, x3, y3;
            float score = 0.f;
            line_stream >> label >> x0 >> y0 >> x1 >> y1 >> x2 >> y2 >> x3 >> y3 >> score;
            if (line_stream.fail()) {
                continue;
            }
            std::string text;
            auto bar = line.find(" |");
            if (bar != std::string::npos && bar + 2 < line.size()) {
                text = line.substr(bar + 2);
            }
            auto clamp_pt = [&frame](int x, int y) {
                return cv::Point(std::max(0, std::min(x, frame.cols - 1)),
                                 std::max(0, std::min(y, frame.rows - 1)));
            };
            cv::Point p0 = clamp_pt(x0, y0);
            cv::Point p1 = clamp_pt(x1, y1);
            cv::Point p2 = clamp_pt(x2, y2);
            cv::Point p3 = clamp_pt(x3, y3);
            cv::line(frame, p0, p1, cv::Scalar(255, 128, 0), box_thick);
            cv::line(frame, p1, p2, cv::Scalar(255, 128, 0), box_thick);
            cv::line(frame, p2, p3, cv::Scalar(255, 128, 0), box_thick);
            cv::line(frame, p3, p0, cv::Scalar(255, 128, 0), box_thick);
            if (!text.empty()) {
                cv::putText(frame, text, p0, cv::FONT_HERSHEY_SIMPLEX, text_scale, cv::Scalar(0, 255, 255),
                            box_thick);
            }
            continue;
        }

        int x1, y1, x2, y2;
        float score;
        if (!(line_stream >> label >> x1 >> y1 >> x2 >> y2 >> score)) {
            continue;
        }

        x1 = std::max(0, std::min(x1, frame.cols - 1));
        y1 = std::max(0, std::min(y1, frame.rows - 1));
        x2 = std::max(0, std::min(x2, frame.cols - 1));
        y2 = std::max(0, std::min(y2, frame.rows - 1));

        const bool is_face = (label == "face");
        const cv::Scalar box_color = is_face ? cv::Scalar(255, 0, 255) : cv::Scalar(255, 0, 0);
        cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), box_color, box_thick);

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
