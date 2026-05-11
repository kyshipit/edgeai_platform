  ├── yolo_adapter.h/cpp       --> 包含 Init(), Preprocess(), Inference(), Postprocess()
  └── yolo_postprocess.cpp     --> 纯解码+NMS函数 (从 postprocess.cc 移入)