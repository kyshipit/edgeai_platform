
/*
main.cpp 对模型一无所知，只通过配置和接口操作，新增模型只需增加一个 else if（或更优雅的工厂注册表，但初期保持简单）。
流水线启动后，预处理、推理、后处理自动并行，充分利用三核 NPU。

// 1. 加载 YAML 配置
// 2. 根据配置创建对应的 IModelAdapter (工厂模式)
// 3. 打开摄像头
// 4. 初始化引擎 (线程池、流水线)
// 5. 主循环：读帧 -> 提交到流水线 -> 获取结果 -> 显示   ????
*/

#include "utils/config_parser.h"   // 新增的极简解析器
#include "engine/adapter_interface.h"
#include "engine/pipeline.h"
#include "adapters/yolo_adapter.h"
#include "adapters/resnet_adapter.h"
// ...

int main(int argc, char* argv[]) {
    std::string config_path = "config/default.yaml";
    if (argc > 1) config_path = argv[1];

    ConfigParser cfg;
    if (!cfg.LoadFromFile(config_path)) {
        fprintf(stderr, "Failed to load config: %s\n", config_path.c_str());
        return -1;
    }

    std::string model_type = cfg.GetString("model.type");
    std::string model_path = cfg.GetString("model.path");
    int infer_threads = cfg.GetInt("system.infer_threads", 3);

    // 工厂创建基准适配器
    std::shared_ptr<IModelAdapter> base_adapter;
    if (model_type == "yolo") {
        base_adapter = std::make_shared<YoloAdapter>();
    } else if (model_type == "resnet") {
        base_adapter = std::make_shared<ResNetAdapter>();
    } else {
        fprintf(stderr, "Unknown model type: %s\n", model_type.c_str());
        return -1;
    }

    // 配置 NPU 核心掩码（从配置文件或默认）
    std::vector<int> npu_masks = {0x1, 0x2, 0x4}; // Core0,1,2
    // 也可以从 cfg 读取

    // 启动流水线
    Pipeline pipeline(base_adapter, infer_threads, npu_masks);
    pipeline.Run(); // 阻塞直到退出

    return 0;
}
