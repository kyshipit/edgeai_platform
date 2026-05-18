/*
 * yolo_adapter.cpp
 *
 * 与 runtime/cpp/yolov5.cc + main.cc 推理链路对齐，封装为 IModelAdapter。
 * 预处理/推理/后处理行为以 cpp 例程为准；平台额外提供 person_present 信号（SCRFD 切换）。
 */

#include "yolo_adapter.h"

#include <math.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include <mutex>

#include "file_utils.h"
#include "platform/adapter_signals.h"
#include "platform/logging.h"

/* 打印模型真实路径和文件大小，避免相对路径导致日志误判 */
static void log_model_file_info(const std::string &model_path)
{
	char resolved_path[PATH_MAX] = {0};
	if (realpath(model_path.c_str(), resolved_path) != nullptr) {
		printf("YoloAdapter: model realpath=%s\n", resolved_path);
	} else {
		printf("YoloAdapter: model realpath unresolved, use=%s\n", model_path.c_str());
	}

	struct stat st;
	if (stat(model_path.c_str(), &st) == 0) {
		printf("YoloAdapter: model size=%lld bytes\n", static_cast<long long>(st.st_size));
	}
}

/* 打印 RKNN API/Driver 版本，便于对比不同程序的运行时差异 */
static void log_rknn_sdk_version(rknn_context ctx)
{
	rknn_sdk_version ver;
	memset(&ver, 0, sizeof(ver));
	if (rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &ver, sizeof(ver)) == RKNN_SUCC) {
		printf("YoloAdapter: rknn api=%s drv=%s\n", ver.api_version, ver.drv_version);
	}
}

/* 正点原子 YOLOv5：3 路融合头 output[i] 的 dims[1]==255 */
static int validate_yolov5_model_io(uint32_t n_output, rknn_tensor_attr *output_attrs, const char *model_path)
{
	if (n_output != 3) {
		printf("YoloAdapter: unsupported RKNN outputs=%u (need 3 fused heads), path=%s\n",
		       n_output, model_path);
		for (uint32_t i = 0; i < n_output && i < 12; ++i) {
			rknn_tensor_attr &a = output_attrs[i];
			printf("  out[%u] name=%s dims=[%d,%d,%d,%d] type=%s\n", i, a.name, a.dims[0],
			       a.dims[1], a.dims[2], a.dims[3], get_type_string(a.type));
		}
		return -1;
	}
	if (output_attrs[0].dims[1] != 255) {
		printf("YoloAdapter: out0 dims[1]=%d (need 255), name=%s path=%s\n",
		       output_attrs[0].dims[1], output_attrs[0].name, model_path);
		return -1;
	}
	return 0;
}

/* 打印 RKNN 张量属性（与 cpp/yolov5.cc dump_tensor_attr 一致） */
static void dump_tensor_attr(rknn_tensor_attr *attr)
{
	printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
	       "zp=%d, scale=%f\n",
	       attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
	       attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
	       get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

/* 加载 RKNN、query 输入输出属性（对应 cpp init_yolov5_model） */
static int init_yolov5_model(const std::string &model_path, rknn_app_context_t *app_ctx)
{
	int ret;
	log_model_file_info(model_path);
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
	log_rknn_sdk_version(ctx);

	rknn_input_output_num io_num;
	ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
	if (ret != RKNN_SUCC) {
		printf("rknn_query fail! ret=%d\n", ret);
		rknn_destroy(ctx);
		return -1;
	}
	printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

	printf("input tensors:\n");
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

	printf("output tensors:\n");
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

	if (validate_yolov5_model_io(io_num.n_output, output_attrs, model_path.c_str()) != 0) {
		rknn_destroy(ctx);
		return -1;
	}

	app_ctx->rknn_ctx = ctx;

	if (output_attrs[0].qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC &&
	    output_attrs[0].type != RKNN_TENSOR_FLOAT16) {
		app_ctx->is_quant = true;
	} else {
		app_ctx->is_quant = false;
	}

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
	printf("model input height=%d, width=%d, channel=%d\n",
	       app_ctx->model_height, app_ctx->model_width, app_ctx->model_channel);
	LogInfo("YoloAdapter: in=%dx%d out=%u is_quant=%d label=./model/coco_80_labels_list.txt",
	        app_ctx->model_width, app_ctx->model_height, io_num.n_output, app_ctx->is_quant ? 1 : 0);
	return 0;
}

/* 释放 RKNN 与属性（对应 cpp release_yolov5_model） */
static int release_yolov5_model(rknn_app_context_t *app_ctx)
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

/* 跨 Inference/Postprocess 持有 rknn_output，Postprocess 末尾 release */
struct RknnOutputHolder {
	std::vector<rknn_output> outputs;
	rknn_context ctx = 0;
};

YoloAdapter::YoloAdapter()
{
	memset(&app_ctx_, 0, sizeof(app_ctx_));
	memset(&dst_img_, 0, sizeof(dst_img_));
	memset(&letter_box_, 0, sizeof(letter_box_));
	memset(&od_results_, 0, sizeof(od_results_));
}

YoloAdapter::~YoloAdapter()
{
	release_yolov5_model(&app_ctx_);
}

/* 加载模型、初始化标签、绑定 NPU 核 */
int YoloAdapter::Init(const std::string &model_path, int npu_core_mask)
{
	static std::once_flag postprocess_init_flag;
	std::call_once(postprocess_init_flag, []() {
		if (init_post_process() != 0) {
			printf("YoloAdapter: init_post_process failed (check ./model/coco_80_labels_list.txt)\n");
		}
	});

	int ret = init_yolov5_model(model_path, &app_ctx_);
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

/*
 * letterbox 预处理（对应 cpp main.cc：BGR Mat 作 RGB888，不 cvtColor）。
 * 旋转由 Pipeline FrameTransform 完成。
 */
uint8_t *YoloAdapter::Preprocess(const cv::Mat &frame, int &out_size)
{
	if (frame.empty() || frame.data == nullptr || frame.channels() != 3) {
		LogError("YoloAdapter::Preprocess: invalid frame (%dx%d ch=%d)",
		         frame.cols, frame.rows, frame.channels());
		out_size = 0;
		return nullptr;
	}
	if (!frame.isContinuous()) {
		LogError("YoloAdapter::Preprocess: frame must be continuous");
		out_size = 0;
		return nullptr;
	}

	image_buffer_t src_image;
	memset(&src_image, 0, sizeof(src_image));
	src_image.height = frame.rows;
	src_image.width = frame.cols;
	src_image.width_stride = frame.step[0];
	src_image.virt_addr = frame.data;
	src_image.format = IMAGE_FORMAT_RGB888;
	src_image.size = static_cast<int>(frame.total() * frame.elemSize());

	dst_img_.width = app_ctx_.model_width;
	dst_img_.height = app_ctx_.model_height;
	dst_img_.format = IMAGE_FORMAT_RGB888;
	dst_img_.size = get_image_size(&dst_img_);
	input_buf_.resize(dst_img_.size);
	dst_img_.virt_addr = input_buf_.data();
	dst_img_.width_stride = dst_img_.width * 3;

	const int bg_color = 114;
	int ret = convert_image_with_letterbox(&src_image, &dst_img_, &letter_box_, bg_color);
	if (ret < 0) {
		printf("YoloAdapter convert_image_with_letterbox fail! ret=%d\n", ret);
		out_size = 0;
		return nullptr;
	}

	out_size = dst_img_.size;
	return input_buf_.data();
}

/* rknn_inputs_set → rknn_run → rknn_outputs_get（对应 cpp main 推理段） */
int YoloAdapter::Inference(std::shared_ptr<void> &model_output)
{
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

	ret = rknn_run(app_ctx_.rknn_ctx, nullptr);
	if (ret < 0) {
		printf("YoloAdapter rknn_run fail! ret=%d\n", ret);
		return ret;
	}

	auto holder = std::make_shared<RknnOutputHolder>();
	holder->outputs.assign(app_ctx_.io_num.n_output, {});
	for (int i = 0; i < app_ctx_.io_num.n_output; ++i) {
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

/* post_process + 格式化为 overlay 行文本 */
std::string YoloAdapter::Postprocess(const std::shared_ptr<void> &model_output)
{
	memset(&od_results_, 0, sizeof(od_results_));

	auto holder = std::static_pointer_cast<RknnOutputHolder>(model_output);
	if (!holder) {
		return std::string();
	}

	const int ret = post_process(&app_ctx_, holder->outputs.data(), &letter_box_, BOX_THRESH, NMS_THRESH,
	                             &od_results_);
	if (ret < 0) {
		printf("YoloAdapter post_process fail! ret=%d\n", ret);
	}

	std::string result;
	for (int i = 0; i < od_results_.count; ++i) {
		const auto &det = od_results_.results[i];
		char buffer[256];
		snprintf(buffer, sizeof(buffer), "%s %d %d %d %d %.4f\n", coco_cls_to_name(det.cls_id),
		         det.box.left, det.box.top, det.box.right, det.box.bottom, det.prop);
		result += buffer;
	}

	if (app_ctx_.io_num.n_output > 0) {
		rknn_outputs_release(app_ctx_.rknn_ctx, app_ctx_.io_num.n_output, holder->outputs.data());
	}
	return result;
}

/* 克隆空实例，Init 由 ModelCoordinator 对各线程分别调用 */
std::shared_ptr<IModelAdapter> YoloAdapter::Clone() const
{
	auto copy = std::make_shared<YoloAdapter>();
	copy->SetPersonScoreThreshold(person_score_threshold_);
	return copy;
}

const object_detect_result_list &YoloAdapter::GetLastResults() const
{
	return od_results_;
}

void YoloAdapter::SetPersonScoreThreshold(float threshold)
{
	person_score_threshold_ = threshold;
}

/* person 类（cls_id==0 或标签名）→ person_present，供 SCRFD 去抖切换 */
AdapterSignals YoloAdapter::GetAdapterSignals() const
{
	AdapterSignals signals;
	for (int i = 0; i < od_results_.count; ++i) {
		const auto &det = od_results_.results[i];
		if (det.prop < person_score_threshold_) {
			continue;
		}
		if (det.cls_id == 0) {
			signals.person_present = true;
			break;
		}
		const char *label = coco_cls_to_name(det.cls_id);
		if (label && strcmp(label, "person") == 0) {
			signals.person_present = true;
			break;
		}
	}
	return signals;
}
