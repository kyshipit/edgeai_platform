/*
这是引擎与所有模型交互的唯一协议。设计要点：
纯虚接口，不含任何 NPU 或模型相关头文件。
返回值语义清晰：Init 返回 0 成功，Preprocess 返回预处理后的数据指针和大小。
结果统一为 std::string（内部可以是 JSON），方便打印和传输。

Preprocess 返回原始指针而不返回 shared_ptr，是因为内存实际由适配器的 input_mem_ 持有，外部只需使用，无需管理生命周期。
Inference() 无参数，因为预处理结果已经通过内部桥梁（input_buf_）传递，简化了流水线任务打包 ???
*/


// engine/adapter_interface.h (header-only（纯虚接口）)
#pragma once
#include <string>
#include <memory>
#include <opencv2/opencv.hpp>

class IModelAdapter {
public:
    virtual ~IModelAdapter() = default;

    // 初始化：加载模型、初始化 RKNN 上下文，并绑定指定的 NPU 核心掩码。
    // 这里的核心掩码是按照硬件 NPU 核心映射生成，便于在多核环境中固定模型到指定核。
    virtual int Init(const std::string& model_path, int npu_core_mask) = 0;

    // 预处理：将 cv::Mat 原始帧转换为模型输入格式。
    // 返回值是预处理后的数据指针和数据长度，内存由适配器内部持有，调用方不负责释放。
    virtual uint8_t* Preprocess(const cv::Mat& frame, int& out_size) = 0;

    // 执行一次推理。推理结果的 opaque 数据将由流水线任务持有，
    // 保证后处理阶段与推理阶段的输出生命周期分离。
    virtual int Inference(std::shared_ptr<void>& model_output) = 0;

    // 后处理：返回推理结果字符串（可理解为 JSON /行文本格式）。
    // model_output 包含推理阶段生成的输出资源，后处理阶段应使用该资源直到结果生成。
    virtual std::string Postprocess(const std::shared_ptr<void>& model_output) = 0;

    // Clone：用于创建与当前适配器逻辑一致的独立实例。
    // 在多线程模式下，Pipeline 会为每个推理线程创建独立的适配器，从而保证每个线程拥有独立 RKNN 上下文。
    virtual std::shared_ptr<IModelAdapter> Clone() const = 0;
};