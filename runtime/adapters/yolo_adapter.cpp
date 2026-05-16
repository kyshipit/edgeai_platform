#include "yolo_adapter.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "file_utils.h"

static void dump_tensor_attr(rknn_tensor_attr *attr)
{
	printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
		   "zp=%d, scale=%f\n",
		   attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
		   attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
		   get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

static int init_rknn_model(const std::string &model_path, rknn_app_context_t *app_ctx)
{
	int ret;
	char *model = NULL;
	int model_len = read_data_from_file(model_path.c_str(), &model);
	if (model == NULL) {
		printf("load_model fail!\n");
		return -1;
	}

	rknn_context ctx = 0;
	ret = rknn_init(&ctx, model, model_len, 0, NULL);
	free(model);
	if (ret < 0) {
		printf("rknn_init fail! ret=%d\n", ret);
		return -1;
	}

	rknn_input_output_num io_num;
	ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
	if (ret != RKNN_SUCC) {
		printf("rknn_query fail! ret=%d\n", ret);
		rknn_destroy(ctx);
		return -1;
	}

	rknn_tensor_attr input_attrs[io_num.n_input];
	memset(input_attrs, 0, sizeof(input_attrs));
	for (int i = 0; i < io_num.n_input; ++i) {
		input_attrs[i].index = i;
		ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &input_attrs[i], sizeof(rknn_tensor_attr));
		if (ret != RKNN_SUCC) {
			printf("rknn_query fail! ret=%d\n", ret);
			rknn_destroy(ctx);
			return -1;
		}
		dump_tensor_attr(&input_attrs[i]);
	}

	rknn_tensor_attr output_attrs[io_num.n_output];
	memset(output_attrs, 0, sizeof(output_attrs));
	for (int i = 0; i < io_num.n_output; ++i) {
		output_attrs[i].index = i;
		ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &output_attrs[i], sizeof(rknn_tensor_attr));
		if (ret != RKNN_SUCC) {
			printf("rknn_query fail! ret=%d\n", ret);
			rknn_destroy(ctx);
			return -1;
		}
		dump_tensor_attr(&output_attrs[i]);
	}

	app_ctx->rknn_ctx = ctx;
	app_ctx->io_num = io_num;
	app_ctx->input_attrs = (rknn_tensor_attr *)malloc(io_num.n_input * sizeof(rknn_tensor_attr));
	memcpy(app_ctx->input_attrs, input_attrs, io_num.n_input * sizeof(rknn_tensor_attr));
	app_ctx->output_attrs = (rknn_tensor_attr *)malloc(io_num.n_output * sizeof(rknn_tensor_attr));
	memcpy(app_ctx->output_attrs, output_attrs, io_num.n_output * sizeof(rknn_tensor_attr));

	if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
		app_ctx->model_channel = input_attrs[0].dims[1];
		app_ctx->model_height = input_attrs[0].dims[2];
		app_ctx->model_width = input_attrs[0].dims[3];
	} else {
		app_ctx->model_height = input_attrs[0].dims[1];
		app_ctx->model_width = input_attrs[0].dims[2];
		app_ctx->model_channel = input_attrs[0].dims[3];
	}

	// 如果输入是量化模型，后处理和输出类型应按量化方式处理。
	app_ctx->is_quant = (input_attrs[0].qnt_type != RKNN_TENSOR_QNT_NONE);
	return 0;
}

static int release_rknn_model(rknn_app_context_t *app_ctx)
{
	if (app_ctx->input_attrs != NULL) {
		free(app_ctx->input_attrs);
		app_ctx->input_attrs = NULL;
	}
	if (app_ctx->output_attrs != NULL) {
		free(app_ctx->output_attrs);
		app_ctx->output_attrs = NULL;
	}
	if (app_ctx->rknn_ctx != 0) {
		rknn_destroy(app_ctx->rknn_ctx);
		app_ctx->rknn_ctx = 0;
	}
	return 0;
}

struct RknnOutputHolder {
	std::vector<rknn_output> outputs;
	rknn_context ctx = 0;
};
		// 初始化 RKNN 模型，加载模型文件并设置上下文信息。

YoloAdapter::YoloAdapter() {
	memset(&app_ctx_, 0, sizeof(app_ctx_));
	memset(&dst_img_, 0, sizeof(dst_img_));
	memset(&letter_box_, 0, sizeof(letter_box_));
		// 释放 RKNN 模型资源并清理上下文信息。
	memset(&od_results_, 0, sizeof(od_results_));
}

YoloAdapter::~YoloAdapter() {
	printf("YoloAdapter::~YoloAdapter: destroying adapter %p\n", this);
	release_rknn_model(&app_ctx_);
}

int YoloAdapter::Init(const std::string& model_path, int npu_core_mask) {
	int ret = init_rknn_model(model_path, &app_ctx_);
	if (ret != 0) {
		printf("YoloAdapter init_yolov5_model fail! ret=%d model_path=%s\n", ret, model_path.c_str());
		return ret;
	}

	int mask_ret = rknn_set_core_mask(app_ctx_.rknn_ctx, (rknn_core_mask)npu_core_mask);
	if (mask_ret != RKNN_SUCC) {
		printf("Warning: rknn_set_core_mask fail ret=%d\n", mask_ret);
	}

	return 0;
}

uint8_t* YoloAdapter::Preprocess(const cv::Mat& frame, int& out_size) {
	cv::Mat rotated_frame;
	cv::rotate(frame, rotated_frame, cv::ROTATE_90_COUNTERCLOCKWISE);

	image_buffer_t src_image;
	src_image.height = rotated_frame.rows;
	src_image.width = rotated_frame.cols;
	src_image.width_stride = rotated_frame.step[0];
	src_image.virt_addr = rotated_frame.data;
	src_image.format = IMAGE_FORMAT_RGB888;
	src_image.size = rotated_frame.total() * rotated_frame.elemSize();

	dst_img_.width = app_ctx_.model_width;
	dst_img_.height = app_ctx_.model_height;
	dst_img_.format = IMAGE_FORMAT_RGB888;
		// 初始化 YoloAdapter，设置 NPU 核心掩码。
	dst_img_.size = get_image_size(&dst_img_);
	input_buf_.resize(dst_img_.size);
	dst_img_.virt_addr = input_buf_.data();
	if (dst_img_.virt_addr == NULL) {
		printf("YoloAdapter Preprocess: ERROR dst_img_.virt_addr is NULL\n");
	}
	dst_img_.width_stride = dst_img_.width * 3;
		// 预处理负责：
		// 1) 旋转摄像头帧到正确方向；
		// 2) 计算模型输入分辨率和内存；
		// 3) 使用 letterbox 变换将原始图像缩放并填充到模型输入尺寸；
		// 4) 返回可以直接传给 RKNN 的输入缓冲指针与大小。

	int ret = convert_image_with_letterbox(&src_image, &dst_img_, &letter_box_, 114);
	if (ret < 0) {
		printf("YoloAdapter convert_image_with_letterbox fail! ret=%d\n", ret);
		out_size = 0;
		return nullptr;
	}

	out_size = dst_img_.size;
	return input_buf_.data();
}

int YoloAdapter::Inference(std::shared_ptr<void>& model_output) {
	if (app_ctx_.rknn_ctx == 0) {
		return -1;
	}

	inputs_.assign(app_ctx_.io_num.n_input, {});

	inputs_[0].index = 0;
	inputs_[0].type = RKNN_TENSOR_UINT8;
	inputs_[0].fmt = RKNN_TENSOR_NHWC;
	inputs_[0].size = app_ctx_.model_width * app_ctx_.model_height * app_ctx_.model_channel;
	inputs_[0].buf = input_buf_.data();

	int ret = rknn_inputs_set(app_ctx_.rknn_ctx, app_ctx_.io_num.n_input, inputs_.data());
	if (ret < 0) {
		printf("YoloAdapter rknn_inputs_set fail! ret=%d\n", ret);
		return ret;
	}
		// Inference 将预处理后的输入绑定到 RKNN，并执行推理。
		// 这里不携带额外参数，因为输入缓冲已保存在适配器内部。

	ret = rknn_run(app_ctx_.rknn_ctx, nullptr);
	if (ret < 0) {
		printf("YoloAdapter rknn_run fail! ret=%d\n", ret);
		return ret;
	}

	std::shared_ptr<RknnOutputHolder> holder = std::make_shared<RknnOutputHolder>();
	holder->outputs.assign(app_ctx_.io_num.n_output, {});
	for (int i = 0; i < app_ctx_.io_num.n_output; ++i) {
		holder->outputs[i].buf = NULL;
		holder->outputs[i].index = i;
		holder->outputs[i].want_float = (!app_ctx_.is_quant);
	}

	ret = rknn_outputs_get(app_ctx_.rknn_ctx, app_ctx_.io_num.n_output, holder->outputs.data(), NULL);
	if (ret < 0) {
		printf("YoloAdapter rknn_outputs_get fail! ret=%d\n", ret);
		return ret;
	}
	holder->ctx = app_ctx_.rknn_ctx;
	model_output = std::static_pointer_cast<void>(holder);

	return 0;
}

std::string YoloAdapter::Postprocess(const std::shared_ptr<void>& model_output) {
	memset(&od_results_, 0x00, sizeof(od_results_));
	// minimal debug: ensure outputs appear valid; avoid noisy per-frame logging
	auto holder = std::static_pointer_cast<RknnOutputHolder>(model_output);
	if (!holder) {
		printf("YoloAdapter Postprocess: ERROR model_output is null\n");
		return std::string();
	}
	for (int i = 0; i < (int)holder->outputs.size(); ++i) {
		if (holder->outputs[i].buf == NULL) {
			printf("YoloAdapter Postprocess: WARNING output[%d].buf is NULL\n", i);
		}
	}
	int ret = post_process(&app_ctx_, holder->outputs.data(), &letter_box_, BOX_THRESH, NMS_THRESH, &od_results_);
	if (ret < 0) {
		printf("YoloAdapter post_process fail! ret=%d\n", ret);
	}

	std::string result;
	for (int i = 0; i < od_results_.count; ++i) {
		auto &det = od_results_.results[i];
		char buffer[256];
		snprintf(buffer, sizeof(buffer), "%s %d %d %d %d %.4f\n",
				 coco_cls_to_name(det.cls_id), det.box.left, det.box.top,
				 det.box.right, det.box.bottom, det.prop);
		result += buffer;
	}

	if (app_ctx_.io_num.n_output > 0) {
		rknn_outputs_release(app_ctx_.rknn_ctx, app_ctx_.io_num.n_output, holder->outputs.data());
		// Postprocess 将 RKNN 输出转为标准结果文本，每一行包括 label + bbox + score。
		// 该字符串由 Pipeline::DrawResult 解析并画框显示。
	}

	return result;
}

std::shared_ptr<IModelAdapter> YoloAdapter::Clone() const {
	return std::make_shared<YoloAdapter>();
}

const object_detect_result_list& YoloAdapter::GetLastResults() const {
	return od_results_;
}
