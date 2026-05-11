"""
将 ONNX 模型转换为 FP16 精度的 RKNN 模型，并与 ONNXRuntime 的输出做余弦相似度验证。

这个过程在 RKNN Toolkit 中的流程是：
  1. RKNN()：创建 RKNN 对象，相当于初始化一个空的模型容器
  2. config()：配置 RKNN 的归一化参数、目标平台等
  3. load_onnx()：把 ONNX 模型加载到 RKNN 对象中
  4. build()：RKNN 内部做模型编译和优化（ONNX → RKNN 的格式转换）
  5. export_rknn()：把编译好的模型保存为 .rknn 文件
  6. init_runtime()：在 NPU 或模拟器上初始化运行环境
  7. inference()：用 RKNN 模型做推理
  8. release()：释放 RKNN 资源

FP16 精度验证：
  用相同的输入图片分别跑 ONNX 和 RKNN，算出两者输出的余弦相似度。
  FP16 只把模型权重的精度从 float32 降低到 float16，理论上精度损失很小。
  阈值 fp16_min_avg_cosine 默认 0.995，如果低于这个值说明转换有问题。
"""

import cv2
import numpy as np
import onnxruntime as ort
from rknn.api import RKNN
from scipy.spatial.distance import cosine

from .preproc_utils import yolo_onnx_nchw_float01, yolo_rknn_nchw_float255


def convert_fp16_and_validate(cfg: dict) -> None:
    """
    执行 FP16 RKNN 转换 + 精度验证。

    步骤：
      1. 从 cfg 中读取 ONNX 路径、RKNN 保存路径、测试图片列表等配置
      2. 用 RKNN Toolkit 完成 ONNX → FP16 RKNN 的转换并保存
      3. 加载转换好的 RKNN 模型，初始化运行时
      4. 用测试图片跑 ONNX 和 RKNN 推理
      5. 计算每个输出的余弦相似度，取平均
      6. 如果平均余弦 < 阈值，则报错

    参数：
        cfg: 模型配置字典，在 model_configs.py 中定义
    """
    onnx_path = cfg["onnx_path"]
    rknn_path = cfg["fp16_path"]
    test_txt = cfg["test_txt"]
    input_size = cfg["input_size"]
    input_name = cfg["input_name"]
    mean_values = cfg["mean_values"]
    std_values = cfg["std_values"]

    # step 1: 创建 RKNN 对象，verbose=True 会打印详细的转换日志
    rknn = RKNN(verbose=True)

    # step 2: 配置 RKNN
    # mean_values=[[0,0,0]], std_values=[[255,255,255]]：
    #   RKNN 内部做 (pixel - mean) / std = pixel / 255.0
    #   这样 RKNN 内部做完归一化后，和 ONNX 的输入一致
    # target_platform="rk3588"：目标芯片是 RK3588
    rknn.config(mean_values=mean_values, std_values=std_values, target_platform="rk3588")

    # step 3: 加载 ONNX 模型
    ret = rknn.load_onnx(model=onnx_path)
    assert ret == 0, "Load ONNX failed"

    # step 4: 编译构建 RKNN 模型
    # do_quantization=False：不量化，保持 FP16 精度
    ret = rknn.build(do_quantization=False)
    assert ret == 0, "Build FP16 failed"

    # step 5: 导出 .rknn 文件
    ret = rknn.export_rknn(rknn_path)
    assert ret == 0, "Export failed"

    # step 6: 初始化运行时（连接 NPU 驱动或模拟器）
    ret = rknn.init_runtime()
    assert ret == 0, "Init runtime failed"

    # ===== 精度验证 =====
    print(f"\n[INFO] FP16 validation: {onnx_path}")

    # 加载 ONNX 模型作为参考基准
    ort_sess = ort.InferenceSession(onnx_path)
    actual_output_names = [o.name for o in ort_sess.get_outputs()]
    print(f"  ONNX outputs: {actual_output_names} ({len(actual_output_names)} total)")

    # 读取测试图片路径列表（最多 100 张，从 test_txt 文件中读取）
    with open(test_txt, encoding="utf-8") as f:
        img_paths = [ln.strip() for ln in f if ln.strip()][:100]

    stride = int(cfg.get("stride", 32))
    sims: list[float] = []  # 保存每张图片的平均余弦相似度

    for idx, path in enumerate(img_paths, start=1):
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            continue

        # 两张图片分别做预处理：
        #   img_onnx: /255.0 归一化到 [0,1]  → 给 ONNX 用
        #   img_rknn: 保持 [0,255] 不归一化   → 给 RKNN 用（RKNN 内部自己做归一化）
        img_onnx = yolo_onnx_nchw_float01(img_bgr, input_size, stride)
        img_rknn = yolo_rknn_nchw_float255(img_bgr, input_size, stride)

        # ONNX 推理
        onnx_outs = ort_sess.run(None, {input_name: img_onnx})
        # RKNN 推理，data_format="nchw" 表示输入数据已经是 NCHW 格式
        rknn_outs = rknn.inference(inputs=[img_rknn], data_format="nchw")

        # 计算每个输出的余弦相似度
        # 余弦相似度 = 1 - cosine_distance
        # 值越接近 1，说明两个输出越一致
        per_sim = [1 - cosine(o.flatten(), r.flatten()) for o, r in zip(onnx_outs, rknn_outs)]
        sims.append(float(np.mean(per_sim)))

        if idx % 10 == 0:
            print(f"  [{idx:3d}/100] current avg cosine: {np.mean(sims):.6f}")

    assert sims, "[ERROR] No valid test images, check test_txt"

    # 计算所有测试图片的平均余弦相似度
    avg_sim = float(np.mean(sims))
    min_cos = float(cfg.get("fp16_min_avg_cosine", 0.995))
    status = "[SUCCESS]" if avg_sim >= min_cos else "[FAIL]"
    print(f"\nFP16 vs ONNX avg cosine: {avg_sim:.6f} (threshold: {min_cos:.6f}) {status}")
    if avg_sim < min_cos:
        print(f"  [WARN] Cosine {avg_sim:.6f} below threshold {min_cos:.6f}")
        print(f"  Adjust fp16_min_avg_cosine in model_config if needed")
    assert avg_sim >= min_cos, (
        f"[FAIL] FP16 avg cosine {avg_sim:.6f} < threshold {min_cos}"
    )
    print("  [SUCCESS] FP16 validation passed")

    # 释放 RKNN 资源
    rknn.release()