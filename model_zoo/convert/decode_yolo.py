"""
YOLOv5 后处理：对 3 个 raw conv 输出做 decode 得到边界框，并绘制图片验证。

处理流程（对应 YOLOv5 原版 Detect 层里做的后处理）：
  1. Sigmoid：把网络的 raw 预测值压缩到 (0,1) 区间
  2. Box decode：将 cx, cy, w, h 的 raw 值解码为实际的像素坐标
  3. 置信度过滤：obj_conf * cls_scores 得到每个类的最终置信度，过滤低置信度框
  4. NMS（非极大值抑制）：对同一类物体的多个重叠框，只保留置信度最高的那个
  5. 画框：把最终检测结果绘制到图片上

注意：
  这里的 decode 是针对 3 个检测头的 raw conv 输出做的。
  如果你用的是 Ultralytics YOLOv5（单输出 [1,84,8400]），处理方式不同。
  关键是 ONNX 导出时是否在模型中包含了 decode 和后处理。
  本代码使用的是去掉 Detect 后处理的版本（3 个 raw conv 输出）。
"""

import glob
import os
import random
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from .preproc_utils import yolo_onnx_nchw_float01


# YOLOv5s 的 anchor 尺寸，从 models/yolov5s.yaml 中获取
# 3 个检测头（P3/P4/P5）各对应 3 个 anchor，总共 9 个 anchor
# 这些 anchor 是在 COCO 数据集上通过聚类得到的先验框尺寸
# P3/8  层（stride=8,  特征图 80x80）：检测小物体，用小 anchor
# P4/16 层（stride=16, 特征图 40x40）：检测中物体，用中 anchor
# P5/32 层（stride=32, 特征图 20x20）：检测大物体，用大 anchor
ANCHORS = [
    np.array([[10, 13], [16, 30], [33, 23]], dtype=np.float32),     # P3/8  小物体检测
    np.array([[30, 61], [62, 45], [59, 119]], dtype=np.float32),    # P4/16 中物体检测
    np.array([[116, 90], [156, 198], [373, 326]], dtype=np.float32), # P5/32 大物体检测
]

# COCO 数据集 80 个类别的名称，索引 0-79 对应模型的 cls_scores 输出
# YOLOv5 官方训练使用的是 COCO 数据集，所以这里直接用了 COCO 的 80 类
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',
]


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Sigmoid 激活函数，将网络输出压缩到 (0, 1) 区间。
    公式：sigmoid(x) = 1 / (1 + e^(-x))
    
    为什么需要 sigmoid？
      卷积层最后一层的输出是 raw 值（范围是负无穷到正无穷），
      但我们需要的是概率值（范围 0~1），所以用 sigmoid 做映射。
      YOLOv5 官方代码中，obj_conf 和 cls_scores 都是经过 sigmoid 的。
    """
    return 1.0 / (1.0 + np.exp(-x))


def decode_one_head(pred: np.ndarray, stride: int, anchors: np.ndarray) -> np.ndarray:
    """
    对单个检测头的输出进行 decode，得到边界框。

    每个检测头输出的形状：[1, 3, H, W, 85]
      3 = 每个格子有 3 个 anchor
      85 = 4 (cx, cy, w, h) + 1 (obj_conf) + 80 (cls_scores for COCO)
      H/W = 特征图尺寸，由输入尺寸 / stride 决定

    YOLOv5 的 decode 公式（与 YOLOv3 一致）：
      cx = (sigmoid(tx) * 2 - 0.5 + grid_x) * stride
      cy = (sigmoid(ty) * 2 - 0.5 + grid_y) * stride
      w  = (sigmoid(tw) * 2) ^ 2 * anchor_w
      h  = (sigmoid(th) * 2) ^ 2 * anchor_h
      
      * 2 - 0.5 是为了让 center 的偏移范围从 [0,1] 扩展到 [-0.5, 1.5]
      * 2 ^ 2 是为了让宽高比例范围从 [0,1] 扩展到 [0, 4]

    参数：
        pred:    某个检测头的 raw 输出，形状 [1, 3, H, W, 85]
        stride:  该检测头的下采样步长（如 8、16、32）
        anchors: 该检测头对应的 3 个 anchor，形状 [3, 2]（每个 anchor 的宽高）

    返回：
        形状 [N, 6] 的边界框数组，每行 [cx, cy, w, h, conf, cls_id]
        注意这里的 cx, cy, w, h 是相对于原始输入图像的像素坐标
    """
    bs, na, ny, nx, no = pred.shape  # batch_size, num_anchors, grid_h, grid_w, num_outputs

    # 创建网格坐标
    # 特征图上每个格子 (i, j) 对应原图上的一个区域
    # grid_x, grid_y 就是每个格子的坐标，用于把偏移量解码为实际位置
    grid_x, grid_y = np.meshgrid(np.arange(nx), np.arange(ny))
    grid = np.stack((grid_x, grid_y), axis=-1).astype(np.float32)

    # sigmoid 处理：把 raw 值映射到 (0, 1) 区间
    pred = sigmoid(pred)

    # box decode：
    # xy: 中心点坐标偏移
    #   pred[..., 0:2] 是 tx, ty，范围 (0, 1)
    #   * 2 - 0.5 把范围扩展到 (-0.5, 1.5)，可以让中心点跨格子预测
    #   + grid 加上格子坐标，得到特征图上的格子位置
    #   * stride 得到原图上的像素坐标
    xy = (pred[..., 0:2] * 2 - 0.5 + grid) * stride

    # wh: 宽高偏移
    #   pred[..., 2:4] 是 tw, th，范围 (0, 1)
    #   * 2 然后平方，把范围扩展到 (0, 4)，让预测更灵活
    #   * anchors 乘上先验框的宽高，得到实际宽高
    wh = (pred[..., 2:4] * 2) ** 2 * anchors.reshape(1, na, 1, 1, 2)

    # 计算每个类别的最终置信度
    # obj_conf: 该位置存在物体的概率（objectness score）
    obj_conf = pred[..., 4:5]
    # cls_scores: 80 个类别的分类概率
    cls_scores = pred[..., 5:] * obj_conf  # 分类概率乘上物体存在概率 = 最终置信度

    # 取分类概率最大的类别作为预测类别
    cls_id = np.argmax(cls_scores, axis=-1, keepdims=True)
    max_cls_conf = np.max(cls_scores, axis=-1, keepdims=True)

    # 拼接成 [cx, cy, w, h, conf, cls_id] 格式
    boxes = np.concatenate([xy, wh, max_cls_conf, cls_id.astype(np.float32)], axis=-1)
    # 展平：将 (batch, anchor, height, width, 6) → (batch, -1, 6)
    boxes = boxes.reshape(bs, -1, 6)

    return boxes


def decode_all_outputs(outputs, strides=(8, 16, 32)):
    """
    对 3 个检测头的输出分别做 decode，然后拼接所有结果。

    参数：
        outputs: 3 个 numpy 数组的列表，每个形状 [1, 3, H, W, 85]
        strides: 3 个检测头对应的下采样步长

    返回：
        形状 [N, 6] 的数组，N 是所有检测头输出的框总数（未过滤）
        每行 [cx, cy, w, h, conf, cls_id]
    """
    all_boxes = []
    for i, (out, stride) in enumerate(zip(outputs, strides)):
        boxes = decode_one_head(out, stride, ANCHORS[i])
        all_boxes.append(boxes)
    # 按第 1 维（检测框）拼接
    return np.concatenate(all_boxes, axis=1)[0]


def nms(boxes: np.ndarray, iou_thres: float = 0.45) -> np.ndarray:
    """
    非极大值抑制（Non-Maximum Suppression, NMS）。

    目的：
      同一个物体可能被多个 anchor 检测到，产生多个重叠的边界框。
      NMS 的作用就是：对同一类别的框，按置信度排序，只保留最高的那个，
      去掉和它重叠（IoU > 阈值）的其他框。

    算法步骤：
      1. 按置信度从高到低排序
      2. 选最高置信度的框，加入保留列表
      3. 计算该框与其他框的 IoU
      4. 去掉 IoU > 阈值的框（重叠太多的）
      5. 重复 2-4，直到没有框剩下

    参数：
        boxes:     [N, 6] 数组，每行 [cx, cy, w, h, conf, cls_id]
        iou_thres: IoU 阈值，高于此值的框会被抑制（去掉）

    返回：
        保留框的索引数组
    """
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)

    # 按置信度降序排序
    order = np.argsort(boxes[:, 4])[::-1]
    keep = []

    while len(order) > 0:
        # 取出当前置信度最高的框
        i = order[0]
        keep.append(i)

        # 把当前框的坐标从 (cx, cy, w, h) 转为 (x1, y1, x2, y2)
        cx, cy, w, h = boxes[i, :4]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        area = w * h

        # 剩余待处理的框
        others = order[1:]
        if len(others) == 0:
            break

        # 计算其他框的坐标
        cx_o = boxes[others, 0]
        cy_o = boxes[others, 1]
        w_o = boxes[others, 2]
        h_o = boxes[others, 3]

        x1_o = cx_o - w_o / 2
        y1_o = cy_o - h_o / 2
        x2_o = cx_o + w_o / 2
        y2_o = cy_o + h_o / 2

        # 计算交集区域
        inter_x1 = np.maximum(x1, x1_o)
        inter_y1 = np.maximum(y1, y1_o)
        inter_x2 = np.minimum(x2, x2_o)
        inter_y2 = np.minimum(y2, y2_o)

        inter_w = np.maximum(0, inter_x2 - inter_x1)
        inter_h = np.maximum(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        # 计算 IoU = 交集 / 并集
        area_o = w_o * h_o
        ious = inter_area / (area + area_o - inter_area + 1e-10)

        # 只保留 IoU 小于阈值的框（不重叠的框留着继续处理）
        mask = ious < iou_thres
        order = others[mask]

    return np.array(keep, dtype=np.int64)


def draw_boxes(img: np.ndarray, boxes: np.ndarray, class_names: list) -> np.ndarray:
    """
    在图像上绘制检测框。

    每个框绘制的内容：
      - 矩形框：用不同颜色区分不同类别
      - 标签：类别名称 + 置信度分数

    参数：
        img:        原始 BGR 图像
        boxes:      [N, 6] 数组，每行 [cx, cy, w, h, conf, cls_id]
        class_names:类别名称列表

    返回：
        绘制了检测框的图像
    """
    if len(boxes) == 0:
        return img

    canvas = img.copy()
    # 定义 8 种颜色用于不同类别（循环使用）
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
    ]

    for box in boxes:
        cx, cy, w, h, conf, cls_id = box
        cls_id = int(cls_id)

        # (cx, cy, w, h) → (x1, y1, x2, y2) 用于画矩形
        x1 = int(cx - w / 2)
        y1 = int(cy - h / 2)
        x2 = int(cx + w / 2)
        y2 = int(cy + h / 2)

        # 画矩形框
        color = colors[cls_id % len(colors)]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        # 画标签背景和文字
        label = f"{class_names[cls_id]}: {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(canvas, (x1, y1 - th - 4), (x1 + tw + 2, y1), color, -1)
        cv2.putText(canvas, label, (x1 + 1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return canvas


def verify_detection(cfg: dict) -> None:
    """
    用 ONNX 模型对测试图片做完整推理 → decode → NMS → 画框，验证检测效果。

    这是整个 pipeline 的最后一步（Step 4），用来直观地检查 RKNN 转换后的
    模型检测效果是否正确。目前使用的是 ONNX 模型推理，但 decode + NMS 的
    逻辑和 RKNN 推理后的后处理是完全一致的。

    处理流程：
      1. 读取测试图片（最多 5 张，从 test_coco/ 目录或 test_txt 中获取）
      2. LetterBox 预处理
      3. ONNXRuntime 推理
      4. 对 raw conv 输出做 reshape（区分单输出和多输出格式）
      5. decode → 置信度过滤（>0.25）→ NMS
      6. 在图片上画框并保存到 ./output/detections/ 目录

    参数：
        cfg: 模型配置字典
    """
    onnx_path = cfg["onnx_path"]
    input_size = cfg["input_size"]
    input_name = cfg["input_name"]
    stride = int(cfg.get("stride", 32))

    # 查找测试图片：优先用 test_coco/ 目录下的 jpg 图片，
    # 如果没有则用 test_txt 文件中的路径列表
    paths = sorted(glob.glob("./data/test/test_coco/*.jpg") or glob.glob("./data/calib/coco/*.jpg"))
    if not paths:
        txt = "./data/test_coco.txt"
        if os.path.isfile(txt):
            with open(txt, encoding="utf-8") as f:
                paths = [ln.strip() for ln in f if ln.strip()]
    if not paths:
        print("  [WARN] No test images found, skip detection verification")
        return

    # 创建输出目录
    out_dir = Path("./output/detections")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[INFO] Detection verification ({min(5, len(paths))} random images)...")

    # 每次运行时从图片列表中随机选取 5 张，保证每次检测的图片不一样
    selected_idx = random.sample(range(len(paths)), min(5, len(paths)))
    for path_idx, img_idx in enumerate(selected_idx, start=1):
        img_path = paths[img_idx]
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue

        # 预处理：LetterBox + 归一化
        x_np = yolo_onnx_nchw_float01(img_bgr, input_size, stride)

        # ONNX 推理
        ort_sess = ort.InferenceSession(str(onnx_path))
        onnx_outs = ort_sess.run(None, {input_name: x_np})

        # 对 ONNX 输出做 reshape：
        # YOLOv5s 的 3 个检测头输出形状：
        #   output0 [1, 255, 80, 80] → reshape 为 [1, 3, 80, 80, 85]
        #   output1 [1, 255, 40, 40] → reshape 为 [1, 3, 40, 40, 85]
        #   output2 [1, 255, 20, 20] → reshape 为 [1, 3, 20, 20, 85]
        # 255 = 3 (anchors) * 85 (4 + 1 + 80)
        reshaped = []
        for out in onnx_outs:
            if len(out.shape) == 4 and out.shape[1] == 255:
                bs, c, h, w = out.shape
                # 255 → 3 * 85，拆出 anchor 维度
                out = out.reshape(bs, 3, -1, h, w)
                # 调整维度顺序为 [batch, anchors, height, width, channels]
                out = np.transpose(out, (0, 1, 3, 4, 2))
            reshaped.append(out)

        # decode 所有检测头
        all_boxes = decode_all_outputs(reshaped)

        # 置信度过滤：只保留 conf > 0.25 的框
        mask = all_boxes[:, 4] >= 0.25
        filtered = all_boxes[mask]

        # NMS 去除重叠框
        keep = nms(filtered, iou_thres=0.45)
        final_boxes = filtered[keep] if len(keep) > 0 else np.empty((0, 6))

        # 绘制结果并保存
        result = draw_boxes(img_bgr, final_boxes, COCO_CLASSES)

        # 保存文件名带上原图名，避免覆盖，也方便知道对应哪张图
        base_name = Path(img_path).stem
        out_path = out_dir / f"result_{path_idx:02d}_{base_name}.jpg"
        cv2.imwrite(str(out_path), result)
        print(f"  [{path_idx}/5] saved: {out_path} ({len(final_boxes)} detections)")

    print("  [SUCCESS] Detection verification complete, results in ./output/detections/")