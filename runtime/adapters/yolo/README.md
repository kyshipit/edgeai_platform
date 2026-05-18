# YOLOv5 适配器（与 `runtime/cpp` 同步）

## 源文件对应关系

| 平台文件 | 正点原子例程 |
|----------|----------------|
| `yolo_postprocess.cpp` | `runtime/cpp/postprocess.cc` |
| `yolo_adapter.cpp` 中 `init_yolov5_model` / 推理 | `runtime/cpp/yolov5.cc`、`runtime/cpp/main.cc` |

**修改检测逻辑时：先改 `runtime/cpp/`，再覆盖 `yolo_postprocess.cpp`（或整文件复制）。**

## 模型与路径

- 板端仅使用 **`./model/yolov5.rknn`**、**`./model/coco_80_labels_list.txt`**
- **不要**用仓库根目录 `deploy/` 产物替换
- **`./model/` 目录勿随意更换文件**

## 后处理要点

- `post_process` 与 `cpp/postprocess.cc` 相同：固定解码 `output[0..2]`，要求 `n_output==3` 且 `dims[1]==255`（Init 会校验，不满足直接失败）
- **禁止**：9 路 box/conf/cls 分离解码、按 buffer 大小乱猜 output 索引

## 预处理

- OpenCV **BGR** 帧直接作为 `IMAGE_FORMAT_RGB888` 送 letterbox（与 cpp `main.cc` 一致，**不要** BGR2RGB）

## 平台扩展

- `GetAdapterSignals()`：`person_present` → `ModelCoordinator` 切换 SCRFD（与 cpp 无关）
