#include "camera_source.h"
#include "platform/logging.h"

#include <algorithm>
#include <cctype>
#include <climits>
#include <cstdlib>
#include <thread>
#include <unistd.h>

namespace {

bool ResolveVideoDevice(const std::string& src, std::string& resolved_path, int& device_index) {
    device_index = -1;
    resolved_path = src;
    char buf[PATH_MAX];
    if (!src.empty() && ::realpath(src.c_str(), buf) != nullptr) {
        resolved_path = buf;
    }
    if (resolved_path.size() <= 10 || resolved_path.compare(0, 10, "/dev/video") != 0) {
        return false;
    }
    const std::string suffix = resolved_path.substr(10);
    if (suffix.empty()) {
        return false;
    }
    for (char c : suffix) {
        if (!std::isdigit(static_cast<unsigned char>(c))) {
            return false;
        }
    }
    device_index = std::stoi(suffix);
    return true;
}

}  // namespace

CameraSource::CameraSource(const std::string& source, int width, int height)
    : source_(source), width_(width), height_(height) {}

bool CameraSource::Open() {
    const int kMaxRetry = 8;
    const int kRetryDelayMs = 150;

    auto try_open_once = [this]() -> bool {
        bool opened = false;
        const bool all_digit = !source_.empty() &&
            std::all_of(source_.begin(), source_.end(),
                        [](char c) { return std::isdigit(static_cast<unsigned char>(c)); });
        std::string resolved_path;
        int device_index = -1;
        const bool has_video_index = ResolveVideoDevice(source_, resolved_path, device_index);

        if (all_digit) {
            const int idx = std::stoi(source_);
            opened = capture_.open(idx, cv::CAP_V4L2);
            if (!opened) {
                opened = capture_.open(idx);
            }
        } else if (has_video_index && device_index >= 0) {
            opened = capture_.open(device_index, cv::CAP_V4L2);
            if (!opened) {
                opened = capture_.open(device_index);
            }
            if (!opened) {
                opened = capture_.open(resolved_path, cv::CAP_V4L2);
            }
            if (!opened) {
                opened = capture_.open(resolved_path);
            }
            if (!opened) {
                opened = capture_.open(source_);
            }
        } else if (source_.compare(0, 10, "/dev/video") == 0) {
            opened = capture_.open(source_, cv::CAP_V4L2);
            if (!opened) {
                opened = capture_.open(source_);
            }
        } else {
            opened = capture_.open(source_);
        }

        if (!opened || !capture_.isOpened()) {
            return false;
        }
        if (has_video_index) {
            LogInfo("CameraSource: opened index %d (path %s)", device_index, resolved_path.c_str());
        } else {
            LogInfo("CameraSource: opened '%s'", source_.c_str());
        }
        if (width_ > 0) {
            capture_.set(cv::CAP_PROP_FRAME_WIDTH, width_);
        }
        if (height_ > 0) {
            capture_.set(cv::CAP_PROP_FRAME_HEIGHT, height_);
        }
        const double actual_w = capture_.get(cv::CAP_PROP_FRAME_WIDTH);
        const double actual_h = capture_.get(cv::CAP_PROP_FRAME_HEIGHT);
        LogInfo("CameraSource: capture resolution %.0fx%.0f", actual_w, actual_h);
        return true;
    };

    for (int attempt = 0; attempt < kMaxRetry; ++attempt) {
        if (try_open_once()) {
            if (attempt > 0) {
                LogInfo("CameraSource: opened '%s' after %d retries", source_.c_str(), attempt);
            }
            return true;
        }
        Release();
        if (attempt + 1 < kMaxRetry) {
            std::this_thread::sleep_for(std::chrono::milliseconds(kRetryDelayMs));
        }
    }
    return false;
}

void CameraSource::Release() {
    if (capture_.isOpened()) {
        cv::Mat drain;
        for (int i = 0; i < 4; ++i) {
            if (!capture_.grab()) {
                break;
            }
        }
        capture_.release();
    }
    capture_ = cv::VideoCapture();
}

bool CameraSource::IsOpened() const {
    return capture_.isOpened();
}

bool CameraSource::ReadFrame(cv::Mat& frame, const std::atomic<bool>* stop_flag) {
    if (stop_flag && stop_flag->load()) {
        return false;
    }
    if (!capture_.isOpened()) {
        return false;
    }
    if (!capture_.grab()) {
        return false;
    }
    if (stop_flag && stop_flag->load()) {
        return false;
    }
    if (!capture_.isOpened()) {
        return false;
    }
    capture_.retrieve(frame);
    return !frame.empty();
}
