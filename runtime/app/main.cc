/*
 * app/main.cc — 进程入口：读配置、注册适配器、组装各模块并运行 Pipeline。
 */
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <atomic>
#include <algorithm>

#include "engine/pipeline.h"
#include "platform/logging.h"
#include "platform/model_coordinator.h"
#include "io/camera_source.h"
#include "io/frame_transform.h"
#include "viz/display_sink.h"
#include "viz/display_layout.h"
#include "viz/result_overlay.h"
#include "adapters/yolo/yolo_adapter.h"
#include "adapters/scrfd/scrfd_adapter.h"
#include "adapters/llm/llm_worker.h"
#include "utils/config_parser.h"

#include <signal.h>
#include <execinfo.h>
#include <unistd.h>

static std::atomic<bool> g_stop_requested{false};
static std::atomic<int> g_sigint_count{0};

static void segv_handler(int sig) {
    void* bt[20];
    int bt_size = backtrace(bt, 20);
    LogFatal("signal %d, backtrace:", sig);
    fprintf(stderr, "Fatal signal %d, backtrace:\n", sig);
    backtrace_symbols_fd(bt, bt_size, STDERR_FILENO);
    fsync(STDERR_FILENO);
    _exit(128 + sig);
}

static void stop_handler(int sig) {
    (void)sig;
    const int n = g_sigint_count.fetch_add(1) + 1;
    g_stop_requested.store(true);
    fprintf(stderr, "\nSignal received, request stop (workers will exit)...\n");
    if (n >= 2) {
        fprintf(stderr, "Second interrupt, forcing exit.\n");
        _exit(130);
    }
}

int main(int argc, char** argv) {
    std::string config_path = "config/default.yaml";
    if (argc == 2) {
        config_path = argv[1];
    }

    ConfigParser cfg;
    if (!cfg.LoadFromFile(config_path)) {
        std::cerr << "Failed to load config: " << config_path << std::endl;
        return -1;
    }

    std::string model_type = cfg.GetString("model.type", "yolo");
    std::string yolo_model_path = cfg.GetString("model.yolo.path", "./model/yolov5.rknn");
    std::string scrfd_model_path = cfg.GetString("model.scrfd.path", "./model/scrfd.rknn");
    float scrfd_conf_th = static_cast<float>(cfg.GetInt("model.scrfd.conf_threshold_percent", 50)) / 100.0f;
    float scrfd_nms_th = static_cast<float>(cfg.GetInt("model.scrfd.nms_threshold_percent", 50)) / 100.0f;
    int infer_threads = cfg.GetInt("system.infer_threads", 1);
    bool yolo_always_on = cfg.GetInt("system.slots.yolo_always_on", 1) != 0;
    int scene_dwell_frames = cfg.GetInt("system.slots.scene_dwell_frames", 5);
    int switch_present_threshold = cfg.GetInt("system.switch.present_threshold", 15);
    int switch_absent_threshold = cfg.GetInt("system.switch.absent_threshold", 30);
    bool single_thread = cfg.GetInt("system.switch.single_thread", 0) != 0;
    std::vector<int> npu_cores = cfg.GetIntArray("system.npu_cores");
    if (npu_cores.empty()) {
        npu_cores = {0, 1};
    }
    std::string input_source = cfg.GetString("input.source", "/dev/video0");
    int input_width = cfg.GetInt("input.width", 0);
    int input_height = cfg.GetInt("input.height", 0);
    std::string input_rotate = cfg.GetString("input.rotate", "ccw90");
    int yolo_person_threshold_percent = cfg.GetInt("model.yolo.person_threshold_percent", 35);
    bool show_window = cfg.GetInt("input.show_window", 1) != 0;
    int display_screen_w = cfg.GetInt("input.display.screen_width", 1080);
    int display_screen_h = cfg.GetInt("input.display.screen_height", 1920);
    int display_max_ratio_percent = cfg.GetInt("input.display.max_screen_ratio_percent", 85);
    bool display_fullscreen = cfg.GetInt("input.display.fullscreen", 0) != 0;
    int display_title_reserve = cfg.GetInt("input.display.title_bar_reserve_px", 56);
    bool llm_enabled = cfg.GetBool("model.llm.enabled", false);
    LogInfo("Main: config %s model.llm.enabled=%s", config_path.c_str(),
            llm_enabled ? "true" : "false");
    std::string llm_model_path = cfg.GetString("model.llm.path", "./model/deepseek.rkllm");
    int llm_max_new_tokens = cfg.GetInt("model.llm.max_new_tokens", 64);
    int llm_max_context_len = cfg.GetInt("model.llm.max_context_len", 4096);
    int llm_face_stable_frames = cfg.GetInt("model.llm.face_stable_frames", 5);
    int llm_face_absent_frames = cfg.GetInt("model.llm.face_absent_frames", 10);
    bool llm_preload_on_scrfd = cfg.GetBool("model.llm.preload_on_scrfd", true);
    bool llm_preload_on_startup = cfg.GetBool("model.llm.preload_on_startup", true);
    std::string llm_greeting_prompt = cfg.GetString(
        "model.llm.greeting_prompt",
        "请用一句简短、自然的中文向镜头前的人问好，不要超过二十个字。");
    std::string llm_reenter_prompt = cfg.GetString(
        "model.llm.reenter_prompt",
        "欢迎回来，请用一句简短中文向镜头前的人问好。");
    std::string llm_auto_greeting_text = cfg.GetString(
        "model.llm.auto_greeting_text",
        "您好，我是deepseek, 有什么需要可以直接和我对话");

    std::shared_ptr<IModelAdapter> base_adapter;
    if (model_type == "yolo") {
        auto yolo = std::make_shared<YoloAdapter>();
        yolo->SetPersonScoreThreshold(yolo_person_threshold_percent / 100.0f);
        base_adapter = yolo;
    } else {
        std::cerr << "Unsupported model type: " << model_type << std::endl;
        return -1;
    }

    try {
        setvbuf(stdout, NULL, _IONBF, 0);
        setvbuf(stderr, NULL, _IONBF, 0);
        signal(SIGSEGV, segv_handler);
        signal(SIGINT, stop_handler);
        signal(SIGTERM, stop_handler);

        ModelCoordinator coordinator;
        coordinator.SetSlotOptions(yolo_always_on);
        coordinator.SetSceneDwellFrames(scene_dwell_frames);

        std::shared_ptr<LlmWorker> llm_worker;
        coordinator.GetLlmGreeting().SetTriggerThreshold(llm_face_stable_frames);
        coordinator.GetLlmGreeting().SetFaceAbsentThreshold(llm_face_absent_frames);
        coordinator.GetLlmGreeting().SetPreloadOnScrfd(llm_preload_on_scrfd);
        coordinator.GetLlmGreeting().SetGreetingPrompt(llm_greeting_prompt);
        coordinator.GetLlmGreeting().SetReenterPrompt(llm_reenter_prompt);
        coordinator.GetLlmGreeting().SetAutoGreetingText(llm_auto_greeting_text);
        if (llm_enabled) {
            llm_worker = std::make_shared<LlmWorker>();
            llm_worker->Configure(llm_model_path, llm_max_new_tokens, llm_max_context_len);
            coordinator.GetLlmGreeting().SetLlmWorker(llm_worker.get());
            if (llm_preload_on_startup) {
                llm_worker->RequestInitializeAsync();
            }
            LogInfo("Main: LLM configured path=%s preload_on_startup=%d preload_on_scrfd=%d",
                    llm_model_path.c_str(), llm_preload_on_startup ? 1 : 0,
                    llm_preload_on_scrfd ? 1 : 0);
        } else {
            LogInfo("Main: LLM disabled (model.llm.enabled=false)");
        }

        CameraSource camera(input_source, input_width, input_height);
        FrameTransform frame_transform(input_rotate);
        ResultOverlay overlay;
        DisplayWindowConfig display_cfg;
        display_cfg.enabled = show_window;
        display_cfg.screen_width = display_screen_w;
        display_cfg.screen_height = display_screen_h;
        display_cfg.max_screen_ratio =
            std::max(10, std::min(100, display_max_ratio_percent)) / 100.0f;
        display_cfg.fullscreen = display_fullscreen;
        display_cfg.title_bar_reserve_px = display_title_reserve;
        std::unique_ptr<IDisplaySink> display = CreateOpenCVDisplaySink(display_cfg);

        Pipeline pipeline(coordinator, camera, frame_transform, overlay, *display,
                          base_adapter, yolo_model_path, infer_threads, npu_cores, single_thread);
        pipeline.SetExternalStopFlag(&g_stop_requested);

        pipeline.RegisterFactory("scrfd",
                                 [scrfd_conf_th, scrfd_nms_th]() {
                                     auto adapter = std::make_shared<ScrfdAdapter>();
                                     adapter->SetThresholds(scrfd_conf_th, scrfd_nms_th);
                                     return adapter;
                                 },
                                 scrfd_model_path);
        pipeline.SetSwitchDebounceThresholds(switch_present_threshold, switch_absent_threshold);

        LogInfo("Main: warming up scrfd slot...");
        coordinator.WarmupSlot("scrfd");

        LogInfo("Main: yolo=%s scrfd=%s infer_threads=%d llm=%s",
                yolo_model_path.c_str(), scrfd_model_path.c_str(), infer_threads,
                llm_enabled ? "on" : "off");

        pipeline.Run();
        if (llm_worker) {
            llm_worker->Shutdown();
        }
    } catch (const std::exception& e) {
        LogError("Pipeline failed: %s", e.what());
        std::cerr << "Pipeline failed: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
