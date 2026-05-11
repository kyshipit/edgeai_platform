#include "engine/adapter_interface.h"
#include "yolov5.h"  // 或者直接把 yolov5 的代码合并进来

class YoloAdapter : public IModelAdapter {
private:
    rknn_context ctx_;
    rknn_tensor_mem* input_mem_;
    // 原来 YOLOv5 类的成员变量搬过来
    std::vector<float> anchors_;
    std::vector<std::string> labels_;
    int input_w_, input_h_;
    int npu_core_mask_;

public:
    int Init(const std::string& model_path, int npu_core_mask) override {
        // = 原来 yolov5::init() 的代码
        npu_core_mask_ = npu_core_mask;
        // 1. rknn_init
        // 2. rknn_set_core_mask
        // 3. rknn_query 获取输入输出属性
        // 4. rknn_create_mem 分配 input/output mem
        // 5. 加载 anchors_yolov5.txt 和 coco_80_labels_list.txt
    }

    uint8_t* Preprocess(const cv::Mat& frame, int& size) override {
        // = 原来 main.cc 里摄像头帧的预处理代码
        // 使用 RGA 进行 letterbox resize + BGR→RGB
        // 结果写入 input_mem_->virt_addr
        size = input_attr_.size_with_stride;
        return (uint8_t*)input_mem_->virt_addr;
    }

    int Inference() override {
        // = 原来 yolov5::detect() 中的 rknn_inputs_set + rknn_run 部分
        rknn_input inputs[1];
        inputs[0].buf = input_mem_->virt_addr;
        rknn_inputs_set(ctx_, 1, inputs);
        rknn_run(ctx_, nullptr);
        // 获取输出到内部 output_mem_
        rknn_outputs_get(ctx_, output_count_, outputs_, nullptr);
    }

    std::string Postprocess() override {
        // = 原来 postprocess.cc 的代码
        // 解码 + NMS，返回 JSON 字符串
        std::vector<Detection> dets;
        decode_outputs(outputs_, dets);  // 从 postprocess.cc 来
        nms(dets);
        return detections_to_json(dets);
    }
};