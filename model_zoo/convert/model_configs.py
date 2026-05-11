"""
模型配置字典 MODEL_CONFIGS。

每个键为模型名称（如 "yolov5s"），值为一个字典，包含该模型的所有配置。
当前只保留 yolov5s，这个模型使用 3 个检测头输出（对应 YOLOv5 的 P3/P4/P5 层）。
ONNX 是从 YOLOv5 官方仓库导出的 3 输出版本，不是 Ultralytics 的单输出版本。

路径说明：
    onnx 文件  → ./output/onnx/         （原始 ONNX 模型）
    rknn 文件  → ./output/rknn/         （转换后的 RKNN 模型）
    calib_txt  → ./data/calib_*.txt      （INT8 量化校准图片路径列表）
    test_txt   → ./data/test_*.txt       （测试图片路径列表）
"""

MODEL_CONFIGS = {
    # ============================================================
    # YOLOv5s — 目标检测（3 输出 raw conv）
    #
    # ONNX 来源：
    #   YOLOv5 官方仓库中修改 Detect.forward 后导出，去掉 YOLOv5 原版 export.py 中
    #   的 box decode + NMS 后处理，让 ONNX 直接输出 3 个 raw conv 特征图。
    #   这样 RKNN 转换时可以自行处理 decode。
    #
    # 输入：
    #   BGR uint8 → LetterBox（等比例缩放 + 灰边填充）→ /255.0 → [0,1] NCHW float
    #
    # 输入名称 "images"，对应 ONNX 的输入张量名，不能随意改，要与 ONNX 文件一致。
    #
    # 输出：
    #   output0 [1, 3, 80, 80, 85]  ← P3/8 层，stride=8，检测小物体
    #   output1 [1, 3, 40, 40, 85]  ← P4/16 层，stride=16，检测中物体
    #   output2 [1, 3, 20, 20, 85]  ← P5/32 层，stride=32，检测大物体
    #
    #   85 维的组成：cx, cy, w, h, obj_conf, cls_scores(80)
    #   （cx, cy, w, h 是未 decode 的 raw 值，需要后处理 decode）
    #
    # RKNN 配置：
    #   mean_values=0, std_values=255
    #   含义是：RKNN 内部做 (pixel - mean) / std = (pixel - 0) / 255 = pixel / 255.0
    #   这样 RKNN 输出与 ONNX 输出对齐（两者都对输入做了 /255.0 归一化）。
    # ============================================================
    "yolov5s": {
        "onnx_path": "./output/onnx/yolov5s.onnx",      # 输入的 ONNX 模型文件路径
        "fp16_path": "./output/rknn/yolov5s_fp16.rknn",  # 转换后的 FP16 RKNN 模型保存路径
        "int8_path": "./output/rknn/yolov5s_int8.rknn",  # 转换后的 INT8 RKNN 模型保存路径
        "input_size": (640, 640),                         # 模型输入尺寸 (width, height)
        "input_name": "images",                           # ONNX 输入张量名称，必须与 ONNX 文件匹配
        "output_names": ["output0", "output1", "output2"], # ONNX 输出张量名称列表
        "output_num": 3,                                  # 输出数量，用于验证 ONNX 输出是否正确
        "mean_values": [[0, 0, 0]],                       # RKNN 归一化均值，每个通道一个值
        "std_values": [[255, 255, 255]],                  # RKNN 归一化标准差，注意 255 等价于 /255.0
        "calib_txt": "./data/calib_coco.txt",             # INT8 量化校准图片的路径列表
        "test_txt": "./data/test_coco.txt",               # 测试图片的路径列表，用于验证精度
        "task": "detect",                                 # 任务类型（detect=目标检测，也可用于分类/分割）
        "stride": 32,                                     # 模型最大下采样倍数，LetterBox 时用来对齐尺寸
        "quantized_dtype": "w8a8",                        # 量化数据类型：w8a8=权重和激活都量化为 INT8
        "quantized_algorithm": "kl_divergence",            # 量化算法：KL 散度，也可以是 "min_max" 等
        "optimization_level": 2,                           # RKNN 优化等级（0~3，越高优化越多但可能损失精度）
        "hybrid_cos_thresh": 0.99,                         # 混合量化余弦相似度阈值，低于此值的层使用 FP16
        "hybrid_euc_thresh": 1.0,                          # 混合量化欧氏距离阈值
        "fp16_min_avg_cosine": 0.995,                      # FP16 对比 ONNX 的最小平均余弦相似度
        "int8_min_avg_cosine": 0.95,                       # INT8 对比 ONNX 的最小平均余弦相似度
    },
}