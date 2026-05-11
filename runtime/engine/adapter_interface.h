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

    // 初始化：加载模型，绑定 NPU 核心 (mask: 0x1/0x2/0x4)
    virtual int Init(const std::string& model_path, int npu_core_mask) = 0;

    // 预处理：返回预处理后的数据指针和大小（内存由适配器管理）
    // 输入cv::Mat帧，输出预处理后的数据指针和字节大小
    // 内存由适配器管理，调用方不应释放
    virtual uint8_t* Preprocess(const cv::Mat& frame, int& out_size) = 0;

    // 执行一次推理（假设预处理数据已就绪）
    virtual int Inference() = 0;

    // 后处理：返回 JSON 结果，内部释放输出 buffer
    virtual std::string Postprocess() = 0;

    // 新增：深拷贝当前适配器（用于多线程独立上下文）
    virtual std::shared_ptr<IModelAdapter> Clone() const = 0;
};