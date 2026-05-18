#include "display_sink.h"
#include "platform/logging.h"
#include <algorithm>
#include <cstdlib>

namespace {

bool HasGuiDisplay() {
    const char* display = std::getenv("DISPLAY");
    if (display != nullptr && display[0] != '\0') {
        return true;
    }
    const char* wayland = std::getenv("WAYLAND_DISPLAY");
    return wayland != nullptr && wayland[0] != '\0';
}

class OpenCVDisplaySink : public IDisplaySink {
public:
    OpenCVDisplaySink(const DisplayWindowConfig& cfg, const char* window_name)
        : cfg_(cfg), window_name_(window_name) {}

    void Prepare() override {
        if (!cfg_.enabled) {
            return;
        }
        if (!HasGuiDisplay()) {
            LogWarn("OpenCVDisplaySink: no DISPLAY/WAYLAND_DISPLAY, window disabled");
            cfg_.enabled = false;
            return;
        }
        LogInfo("OpenCVDisplaySink: GUI ok, screen=%dx%d max_ratio=%.2f",
                cfg_.screen_width, cfg_.screen_height, cfg_.max_screen_ratio);
    }

    void Show(const cv::Mat& frame) override {
        if (!cfg_.enabled || frame.empty()) {
            return;
        }
        if (!window_created_) {
            SetupWindow(frame.cols, frame.rows);
        }

        cv::Mat scaled;
        if (frame.cols == window_w_ && frame.rows == window_h_) {
            scaled = frame;
        } else {
            cv::resize(frame, scaled, cv::Size(window_w_, window_h_));
        }
        cv::imshow(window_name_, scaled);
    }

    int PollKey(int delay_ms) override {
        if (!cfg_.enabled) {
            return -1;
        }
        return cv::waitKey(delay_ms);
    }

    void Shutdown() override {
        if (cfg_.enabled && window_created_) {
            cv::destroyWindow(window_name_);
            window_created_ = false;
        }
    }

private:
    void SetupWindow(int frame_w, int frame_h) {
        if (cfg_.fullscreen) {
            window_w_ = cfg_.screen_width;
            window_h_ = cfg_.screen_height;
            window_x_ = 0;
            window_y_ = 0;
        } else {
            const int reserve = std::max(0, cfg_.title_bar_reserve_px);
            const int max_w = std::max(1, static_cast<int>(cfg_.screen_width * cfg_.max_screen_ratio));
            const int max_h = std::max(1, static_cast<int>(cfg_.screen_height * cfg_.max_screen_ratio) - reserve);
            const double scale = std::min(static_cast<double>(max_w) / frame_w,
                                          static_cast<double>(max_h) / frame_h);
            window_w_ = std::max(1, static_cast<int>(frame_w * scale));
            window_h_ = std::max(1, static_cast<int>(frame_h * scale));
            window_x_ = (cfg_.screen_width - window_w_) / 2;
            window_y_ = reserve;
        }

        cv::namedWindow(window_name_, cv::WINDOW_NORMAL);
        cv::resizeWindow(window_name_, window_w_, window_h_);
        cv::moveWindow(window_name_, window_x_, window_y_);
        if (cfg_.fullscreen) {
            cv::setWindowProperty(window_name_, cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
        }
        window_created_ = true;
        LogInfo("OpenCVDisplaySink: window '%s' size=%dx%d pos=(%d,%d) fullscreen=%d frame=%dx%d",
                window_name_, window_w_, window_h_, window_x_, window_y_, cfg_.fullscreen ? 1 : 0,
                frame_w, frame_h);
    }

    DisplayWindowConfig cfg_;
    const char* window_name_;
    bool window_created_ = false;
    int window_w_ = 0;
    int window_h_ = 0;
    int window_x_ = 0;
    int window_y_ = 0;
};

class NullDisplaySink : public IDisplaySink {
public:
    void Prepare() override {}
    void Show(const cv::Mat&) override {}
    int PollKey(int) override { return -1; }
    void Shutdown() override {}
};

}  // namespace

std::unique_ptr<IDisplaySink> CreateOpenCVDisplaySink(const DisplayWindowConfig& cfg,
                                                    const char* window_name) {
    if (!cfg.enabled) {
        return std::unique_ptr<IDisplaySink>(new NullDisplaySink());
    }
    return std::unique_ptr<IDisplaySink>(new OpenCVDisplaySink(cfg, window_name));
}
