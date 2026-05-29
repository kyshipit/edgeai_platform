/*
 * adapters/yolo_adapter.h
 *
 * YOLOv5 RKNN 插件；见 yolo_adapter.cpp 与 docs/适配器说明.md  
 */
#pragma once

#include <memory>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#include "engine/adapter_interface.h"
#include "yolo_postprocess.h"
#include "common.h"
#include "image_utils.h"

class YoloAdapter : public IModelAdapter {
public:
	// 构造：初始化内部上下文与缓冲状态。
	YoloAdapter();
	// 析构：释放 RKNN 与后处理资源。
	~YoloAdapter() override;

	// Init: 加载模型文件，初始化 RKNN 上下文，绑定指定 NPU 核心。
	int Init(const std::string& model_path, int npu_core_mask) override;
	// Preprocess: 将摄像头帧转换为模型输入，返回内部输入缓冲指针。
	uint8_t* Preprocess(const cv::Mat& frame, int& out_size) override;
	// Inference: 使用内部输入缓冲执行 RKNN 推理。
	int Inference(std::shared_ptr<void>& model_output) override;
	// Postprocess: 解析 RKNN 输出并生成标准检测结果字符串。
	std::string Postprocess(const std::shared_ptr<void>& model_output) override;
	// Clone: 生成当前适配器的独立副本，用于多线程推理。
	std::shared_ptr<IModelAdapter> Clone() const override;
	AdapterSignals GetAdapterSignals() const override;

	// 返回最近一帧后处理结果（调试/扩展用途）。
	const object_detect_result_list& GetLastResults() const;

	// 设置 person_present 判定阈值。
	void SetPersonScoreThreshold(float threshold);

private:
	float person_score_threshold_ = 0.35f;
	// RKNN 上下文和输入/输出信息
	rknn_app_context_t app_ctx_;
	// 用于存储预处理后图像的目标缓冲
	image_buffer_t dst_img_;
	// letterbox 填充参数，用于后处理坐标反变换
	letterbox_t letter_box_;
	// 保存后处理结果
	object_detect_result_list od_results_;
	// 模型输入缓冲
	std::vector<uint8_t> input_buf_;
	// RKNN 输入输出描述
	std::vector<rknn_input> inputs_;
};
