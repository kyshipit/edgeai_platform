"""
YOLO 图像预处理工具函数。

这两个函数做的事情基本相同——对输入图片做 LetterBox 缩放填充，
然后调整维度顺序（HWC → CHW → NCHW）给模型推理。
区别在于归一化程度不同，对应两种推理路径：
  - ONNX 路径：做完 /255.0 归一化到 [0,1]（因为 ONNX 模型内部不做归一化）
  - RKNN 路径：保持 [0,255]（因为 RKNN 芯片内部会根据 mean/std 自己做归一化）

为什么需要两个路径？
  因为 RKNN 的 config 里配了 mean_values=0, std_values=255，
  RKNN 内部会自动做 (pixel - 0) / 255 的归一化。
  如果我们在预处理时也做 /255.0，就等于归一化了两次，输出就不对了。
"""

import cv2
import numpy as np


def yolo_onnx_nchw_float01(img_bgr: np.ndarray, input_size: tuple, stride: int) -> np.ndarray:
    """
    ONNX 推理路径的预处理。

    处理流程：
      1. LetterBox：等比例缩放原图到模型输入尺寸，不足的部分用 114（灰）填充
      2. 归一化：pixel / 255.0 将值从 [0,255] 映射到 [0,1]
      3. 维度变换：HWC (H, W, 3) → CHW (3, H, W)
                 推理时需要 CHW 顺序，因为 ONNX 各层卷积算子都按 CHW 约定工作
      4. 添加 batch 维度：CHW → NCHW (1, 3, H, W)

    参数：
        img_bgr:    OpenCV 读取的 BGR 图像，形状 (H, W, 3)
        input_size: 模型输入尺寸 (width, height)，如 (640, 640)
        stride:     模型的最大下采样步长，YOLOv5s 是 32
                    缩放到新尺寸时需要对齐到 stride 的整数倍
                    这样可以避免在特征图上出现"半个格子"的问题

    返回：
        形状 (1, 3, H, W)，dtype float32，数值范围 [0, 1]
    """
    h, w = img_bgr.shape[:2]
    target_w, target_h = input_size

    # 计算缩放比例：取宽和高中较小的缩放比例，确保原始图像完全放入输入框内
    scale = min(target_w / w, target_h / h)

    # 新尺寸要对齐到 stride 的整数倍
    # 例如 stride=32，新尺寸必须是 32 的倍数
    # 这是因为特征图尺寸 = 输入尺寸 / stride，如果输入不是 stride 的倍数，
    # 特征图上会产生"小数格子"导致坐标映射出错
    new_w = max(round(w * scale / stride) * stride, stride)
    new_h = max(round(h * scale / stride) * stride, stride)

    # 计算灰边填充的偏移量
    # dw/dh 是左右/上下的填充宽度，居中放置缩放后的图像
    dw = (target_w - new_w) // 2
    dh = (target_h - new_h) // 2

    # 缩放图像
    resized = cv2.resize(img_bgr, (new_w, new_h))

    # 创建灰边画布（114 是 YOLOv5 官方使用的填充颜色值）
    canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    canvas[dh:dh + new_h, dw:dw + new_w] = resized

    # 归一化到 [0, 1] 并调整维度
    img = canvas.astype(np.float32) / 255.0          # [0, 1]
    img = np.transpose(img, (2, 0, 1))               # HWC → CHW
    img = np.expand_dims(img, 0)                     # CHW → NCHW
    return img


def yolo_rknn_nchw_float255(img_bgr: np.ndarray, input_size: tuple, stride: int) -> np.ndarray:
    """
    RKNN 推理路径的预处理。

    与 yolo_onnx_nchw_float01 几乎一样，唯一的区别是：
      不做 /255.0 归一化，保持数值范围在 [0, 255]。
    因为 RKNN 内部会根据 config 里的 mean_values=0, std_values=255 自行做归一化。

    参数：
        img_bgr:    OpenCV 读取的 BGR 图像
        input_size: (width, height)
        stride:     模型下采样步长
    返回：
        形状 (1, 3, H, W)，dtype float32，数值范围 [0, 255]
    """
    h, w = img_bgr.shape[:2]
    target_w, target_h = input_size

    scale = min(target_w / w, target_h / h)
    new_w = max(round(w * scale / stride) * stride, stride)
    new_h = max(round(h * scale / stride) * stride, stride)

    dw = (target_w - new_w) // 2
    dh = (target_h - new_h) // 2

    resized = cv2.resize(img_bgr, (new_w, new_h))
    canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    canvas[dh:dh + new_h, dw:dw + new_w] = resized

    # 注意：不做 /255.0，保持 [0, 255]
    img = canvas.astype(np.float32)
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)
    return img