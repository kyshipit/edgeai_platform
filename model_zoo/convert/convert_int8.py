"""
将 ONNX 模型转换为 INT8 量化的 RKNN 模型，并与 ONNXRuntime 的输出做余弦相似度验证。

与 FP16 转换的流程基本一致，主要区别在于 build 阶段：
  - FP16：do_quantization=False，不做量化
  - INT8：do_quantization=True，使用校准集做量化

量化（Quantization）的含义：
  把模型权重和激活值从 float32 压缩到 int8，可以减少模型体积和加速推理，
  但会引入精度损失。精度损失的程度取决于量化算法和校准数据。

量化参数说明（在 model_configs.py 中定义）：
  - quantized_dtype: "w8a8" 表示权重（weight）和激活（activation）都量化到 8-bit
  - quantized_algorithm: "kl_divergence" 使用 KL 散度选择最佳量化阈值
  - optimization_level: 优化等级，越高 RKNN 做的优化越多
  - auto_hybrid_cos_thresh: 混合量化阈值，余弦相似度低于此值的层保留为 FP16
  - auto_hybrid_euc_thresh: 混合量化欧氏距离阈值
"""

import cv2
import numpy as np
import onnxruntime as ort
from rknn.api import RKNN
from scipy.spatial.distance import cosine

from .preproc_utils import yolo_onnx_nchw_float01, yolo_rknn_nchw_float255


def convert_int8_and_validate(cfg: dict) -> None:
    """
    执行 INT8 RKNN 转换（量化）+ 精度验证。

    步骤：
      1. 从 cfg 中读取 ONNX 路径、RKNN 保存路径、校准集路径、测试集路径等配置
      2. 配置 RKNN 的量化参数（量化算法、优化等级等）
      3. 加载 ONNX → build（使用校准集做量化）→ 导出 .rknn
      4. 初始化运行时
      5. 用测试图片跑 ONNX 和 RKNN 推理，计算余弦相似度
      6. 如果平均余弦 < 阈值，报错

    参数：
        cfg: 模型配置字典，在 model_configs.py 中定义
    """
    onnx_path = cfg["onnx_path"]
    rknn_path = cfg["int8_path"]
    calib_txt = cfg["calib_txt"]
    test_txt = cfg["test_txt"]
    input_size = cfg["input_size"]
    input_name = cfg["input_name"]
    mean_values = cfg["mean_values"]
    std_values = cfg["std_values"]

    # 创建 RKNN 对象，verbose=True 打印详细日志
    rknn = RKNN(verbose=True)

    # 配置 RKNN，这里比 FP16 多了量化相关的参数
    rknn.config(
        mean_values=mean_values,
        std_values=std_values,
        target_platform="rk3588",
        quantized_dtype=cfg.get("quantized_dtype", "w8a8"),         # 权重和激活都量化到 8-bit
        quantized_algorithm=cfg.get("quantized_algorithm", "kl_divergence"),  # KL 散度校准
        quantized_method="channel",                                          # 按通道量化
        optimization_level=int(cfg.get("optimization_level", 2)),            # 优化等级
        auto_hybrid_cos_thresh=cfg.get("hybrid_cos_thresh", 0.99),          # 混合量化阈值
        auto_hybrid_euc_thresh=cfg.get("hybrid_euc_thresh", 1.0),
    )

    # 加载 ONNX 模型
    ret = rknn.load_onnx(model=onnx_path)
    assert ret == 0, "Load ONNX failed"

    # build 阶段做 INT8 量化
    # do_quantization=True：开启量化
    # dataset=calib_txt：校准集图片路径列表
    #   校准集用于统计每层激活值的分布，从而找到最优的量化参数
    #   校准集不需要很多图片（几十到几百张即可），但需要覆盖模型会遇到的典型场景
    ret = rknn.build(do_quantization=True, dataset=calib_txt)
    assert ret == 0, "Build INT8 failed"

    # 导出 INT8 RKNN 模型
    ret = rknn.export_rknn(rknn_path)
    assert ret == 0, "Export INT8 failed"

    # 初始化运行时
    ret = rknn.init_runtime()
    assert ret == 0, "Init runtime failed"

    # ===== 精度验证 =====
    print(f"\n[INFO] INT8 validation: {onnx_path}")
    ort_sess = ort.InferenceSession(onnx_path)

    # 读取测试图片路径列表（最多 100 张）
    with open(test_txt, encoding="utf-8") as f:
        img_paths = [ln.strip() for ln in f if ln.strip()][:100]

    stride = int(cfg.get("stride", 32))
    sims: list[float] = []

    for idx, path in enumerate(img_paths, start=1):
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            continue

        img_onnx = yolo_onnx_nchw_float01(img_bgr, input_size, stride)
        img_rknn = yolo_rknn_nchw_float255(img_bgr, input_size, stride)

        onnx_outs = ort_sess.run(None, {input_name: img_onnx})
        rknn_outs = rknn.inference(inputs=[img_rknn], data_format="nchw")

        # 计算余弦相似度
        per_sim = [1 - cosine(o.flatten(), r.flatten()) for o, r in zip(onnx_outs, rknn_outs)]
        sims.append(float(np.mean(per_sim)))

        if idx % 10 == 0:
            print(f"  [{idx:3d}/100] current avg cosine: {np.mean(sims):.6f}")

    assert sims, "[ERROR] No valid test images, check test_txt"

    avg_sim = float(np.mean(sims))
    min_cos = float(cfg.get("int8_min_avg_cosine", 0.95))
    status = "[SUCCESS]" if avg_sim >= min_cos else "[FAIL]"
    print(f"\nINT8 vs ONNX avg cosine: {avg_sim:.6f} (threshold: {min_cos:.6f}) {status}")
    if avg_sim < min_cos:
        print(f"  [WARN] Cosine {avg_sim:.6f} below threshold {min_cos:.6f}")
        print(f"  Adjust int8_min_avg_cosine in model_config if needed")
    assert avg_sim >= min_cos, (
        f"[FAIL] INT8 avg cosine {avg_sim:.6f} < threshold {min_cos}"
    )
    print("  [SUCCESS] INT8 validation passed")

    rknn.release()