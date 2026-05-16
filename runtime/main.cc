/*-------------------------------------------
                Includes
/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "engine/pipeline.h"
#include "adapters/yolo_adapter.h"
#include "utils/config_parser.h"

// 程序入口，它从 YAML 配置加载模型类型、模型路径、推理线程数、NPU 核心集合和输入源，
// 然后创建统一的流水线 Pipeline，将摄像头帧交给模型适配器并显示结果。

#include <signal.h>
#include <execinfo.h>
#include <unistd.h>

static void segv_handler(int sig) {
    void *bt[20];
    int bt_size = backtrace(bt, 20);
    fprintf(stderr, "Fatal signal %d, backtrace:\n", sig);
    backtrace_symbols_fd(bt, bt_size, STDERR_FILENO);
    // ensure logs flushed, then exit
    fsync(STDERR_FILENO);
    _exit(128 + sig);
}
/*-------------------------------------------
                  Main Function
-------------------------------------------*/
int main(int argc, char **argv)
{
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
    std::string model_path = cfg.GetString("model.path", "./model/yolov5.rknn");
    int infer_threads = cfg.GetInt("system.infer_threads", 3);
    std::vector<int> npu_cores = cfg.GetIntArray("system.npu_cores");
    if (npu_cores.empty()) {
        npu_cores = {0, 1, 2};
    }
    std::string input_source = cfg.GetString("input.source", "/dev/video0");

    std::shared_ptr<IModelAdapter> base_adapter;
    if (model_type == "yolo") {
        base_adapter = std::make_shared<YoloAdapter>();
    } else {
        std::cerr << "Unsupported model type: " << model_type << std::endl;
        return -1;
    }

    try {
            // Disable stdout/stderr buffering to avoid losing logs on crash.
            setvbuf(stdout, NULL, _IONBF, 0);
            setvbuf(stderr, NULL, _IONBF, 0);
            // install segfault handler to print backtrace immediately
            signal(SIGSEGV, segv_handler);

        Pipeline pipeline(base_adapter, model_path, infer_threads, npu_cores, input_source);
        pipeline.Run();
    } catch (const std::exception &e) {
        std::cerr << "Pipeline failed: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}

