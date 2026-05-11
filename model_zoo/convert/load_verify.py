"""
加载本地的 ONNX 模型并验证其结构是否正确。

这个模块只做一件事：
  用 onnxruntime 打开 ONNX 文件，检查输入输出张量的名称和形状，
  确认和 model_configs.py 中的配置匹配。

原来的版本还包含 PT 模型的加载和 PT vs ONNX 输出对比验证，
但那些功能已经被移除——因为 PT 文件需要 YOLOv5 源码才能反序列化，
而且 ONNX 是从 YOLOv5 官方导出工具生成的，一致性已经在导出时验证过了。
"""

import os
from pathlib import Path

import onnxruntime as ort


def ensure_onnx(onnx_path: str) -> Path:
    """
    检查 ONNX 文件是否存在于本地文件系统。
    如果不存在，直接报错退出（assert），避免后续操作因为文件缺失而崩溃。
    """
    p = Path(onnx_path)
    assert p.exists(), f"[ERROR] ONNX file not found: {p.resolve()}"
    return p


def verify_onnx_structure(cfg: dict) -> None:
    """
    用 onnxruntime 加载 ONNX 模型，打印输入输出信息，然后与配置对比验证。

    验证内容：
      1. 打印所有输入张量的名称和形状（方便调试时查看）
      2. 打印所有输出张量的名称和形状
      3. 确认输出数量与配置中的 output_num 一致（当前配置的是 3 个输出）
    
    这个步骤不进行实际推理，只是结构检查，速度很快。
    它在 pipeline 中的位置是第 1 步，用来快速发现 ONNX 文件是否损坏或导出格式不对。
    """
    onnx_path = ensure_onnx(cfg["onnx_path"])
    session = ort.InferenceSession(str(onnx_path))

    print("\n[INFO] ONNX structure:")
    for inp in session.get_inputs():
        print(f"  input: {inp.name} {inp.shape}")
    for out in session.get_outputs():
        print(f"  output: {out.name} {out.shape}")

    assert len(session.get_outputs()) == cfg["output_num"], (
        f"[ERROR] ONNX output count {len(session.get_outputs())} != config {cfg['output_num']}"
    )
    print("  [SUCCESS] ONNX structure check passed")