"""
YOLOv5s 完整 pipeline：验证 ONNX 结构 → FP16 RKNN 转换 → INT8 RKNN 转换 → 检测效果验证。

前置条件：
    ./output/onnx/yolov5s.onnx   - 3 输出 ONNX 模型（已从 YOLOv5 官方仓库导出好）
    ./data/calib_coco.txt        - INT8 量化校准集的图片路径列表（每行一个完整路径）
    ./data/test_coco.txt         - 测试集图片路径列表（每行一个完整路径）

流程概述（共 4 步）：
    第 1 步：验证 ONNX 结构
        - 用 onnxruntime 加载 ONNX 文件
        - 打印输入输出的名称和形状
        - 确认输出数量 = 3（对应 P3/P4/P5 三个检测头）
        - 这一步只检查结构，不推理，速度很快

    第 2 步：FP16 RKNN 转换
        - 使用 RKNN Toolkit 加载 ONNX
        - 编译为 FP16 精度的 RKNN 模型
        - 导出 .rknn 文件到 ./output/rknn/
        - 用测试集图片对比 ONNX 和 RKNN 的输出，计算余弦相似度
        - 阈值 0.995，低于此值则报错

    第 3 步：INT8 RKNN 转换
        - 与 FP16 类似，但 build 时使用校准集做 INT8 量化
        - 量化参数在 model_configs.py 中配置
        - 阈值 0.95（INT8 精度损失比 FP16 大，所以阈值更低）

    第 4 步：检测效果验证
        - 用 ONNX 模型对测试图片做完整推理
        - decode（raw conv → 边界框）→ 置信度过滤 → NMS → 画框
        - 结果图片保存到 ./output/detections/ 目录
        - 这个步骤用来直观地检查模型的检测效果

注意：
    - 第 2、3 步需要 RKNN Toolkit 和 NPU 驱动（或模拟器）支持
    - 第 1、4 步只需要 onnxruntime，不需要 RKNN 环境
    - 所有路径等配置都在 model_configs.py 中统一管理

灵活运行：
    通过 --steps 参数指定要运行的步骤，例如：
        python run_all.py --steps 1          # 只跑第 1 步
        python run_all.py --steps 2,4        # 只跑第 2 步和第 4 步
        python run_all.py --steps 1-3        # 跑第 1~3 步
        python run_all.py                    # 不传参数则跑所有 4 步
"""

import argparse
import glob
import os
from convert.model_configs import MODEL_CONFIGS
from convert.load_verify import verify_onnx_structure


def _parse_steps(steps_str: str) -> set[int]:
    """
    解析 --steps 参数。
    支持格式：
        "1"        → {1}
        "1,3,4"    → {1,3,4}
        "1-3"      → {1,2,3}
        "1,3-4"    → {1,3,4}
    """
    steps = set()
    for part in steps_str.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            steps.update(range(int(a), int(b) + 1))
        else:
            steps.add(int(part))
    return steps


def _cleanup():
    """清理 RKNN 编译产生的 check* 中间文件。"""
    for f in glob.glob("check*.onnx"):
        os.remove(f)


def main():
    parser = argparse.ArgumentParser(description="YOLOv5s pipeline")
    parser.add_argument(
        "--steps", type=str, default=None,
        help="要运行的步骤，如 '1'、'2,4'、'1-3'。不传则跑全部。"
    )
    args = parser.parse_args()

    model_name = "yolov5s"
    assert model_name in MODEL_CONFIGS, f"Unknown model: {model_name}"
    cfg = MODEL_CONFIGS[model_name]

    # 确定要跑的步骤
    if args.steps:
        run_steps = _parse_steps(args.steps)
    else:
        run_steps = {1, 2, 3, 4}

    print("=" * 60)
    print("  YOLOv5s Pipeline Start")
    print(f"  Running steps: {sorted(run_steps)}")
    print("=" * 60)

    # 第 1 步：验证 ONNX 结构
    if 1 in run_steps:
        print("\n[STEP 1] Verify ONNX structure")
        verify_onnx_structure(cfg)

    # 第 2 步：FP16 RKNN 转换 + 精度验证
    if 2 in run_steps:
        print(f"\n{'='*60}")
        print("  [STEP 2] FP16 RKNN Conversion")
        print(f"{'='*60}")
        from convert.convert_fp16 import convert_fp16_and_validate
        convert_fp16_and_validate(cfg)

    # 第 3 步：INT8 RKNN 转换 + 精度验证
    if 3 in run_steps:
        print(f"\n{'='*60}")
        print("  [STEP 3] INT8 RKNN Conversion")
        print(f"{'='*60}")
        from convert.convert_int8 import convert_int8_and_validate
        convert_int8_and_validate(cfg)

    # 第 4 步：检测效果验证（decode + NMS + 画框）
    if 4 in run_steps:
        print(f"\n{'='*60}")
        print("  [STEP 4] Detection Verification (decode + NMS + draw)")
        print(f"{'='*60}")
        from convert.decode_yolo import verify_detection
        verify_detection(cfg)

    _cleanup()
    print(f"\n{'='*60}")
    print("  [SUCCESS] YOLOv5s Pipeline Complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()