"""
准备校准与测试图片列表：
  - COCO val2017：检测校准 & 测试图片（YOLO 用）
  - Tiny ImageNet train：分类校准图（ResNet/ViT 用）
  - Tiny ImageNet test/images：分类测试图（直接引用，不复制）

数据目录结构：
  data/
  ├── calib/                     # 量化校准图片（从原始集复制 + 缩放）
  │   ├── coco/                   COCO 100 张 640x640（YOLO 用）
  │   └── imagenet/               ImageNet 100 张 224x224（ResNet/ViT 用）
  ├── test/                       # 精度校验测试图片
  │   └── test_coco/              COCO 100 张 640x640（YOLO 校验用）
  ├── raw/                        # 下载的原始资源（zip + 解压目录）
  │   ├── tiny-imagenet-200/      Tiny ImageNet 完整数据集（test_imagenet.txt 直接引用）
  │   └── annotations/            COCO 标注 JSON
  ├── calib_coco.txt             路径列表 → ./data/calib/coco/
  ├── calib_imagenet.txt         路径列表 → ./data/calib/imagenet/
  ├── test_coco.txt              路径列表 → ./data/test/test_coco/
  └── test_imagenet.txt          路径列表 → ./data/raw/tiny-imagenet-200/test/images/（直接引用）
"""

import glob
import json
import os
import random
import time
import zipfile

import cv2
import requests

TINY_IMAGENET_ZIP = "tiny-imagenet-200.zip"
TINY_IMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"


def ensure_annotations(annot_zip="data/raw/annotations_trainval2017.zip",
                       val_json="data/raw/annotations/instances_val2017.json"):
    """确保标注文件存在，如没有则自动下载并解压到 data/raw/annotations/。"""
    if os.path.exists(val_json):
        return

    # 如果 zip 不存在，下载到 data/raw/
    if not os.path.exists(annot_zip):
        print(f"未找到 {annot_zip}，开始下载 (241MB)...")
        url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        os.makedirs(os.path.dirname(annot_zip), exist_ok=True)
        for attempt in range(1, 4):
            try:
                print(f"  尝试 {attempt}/3 下载标注文件...")
                r = requests.get(url, stream=True, timeout=(15, 60))
                if r.status_code == 200:
                    with open(annot_zip, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                    print("  下载完成")
                    break
                else:
                    print(f"  HTTP {r.status_code}，重试...")
                    time.sleep(2)
            except Exception as e:
                print(f"  网络错误: {e}，重试...")
                time.sleep(2)
        else:
            raise RuntimeError("标注文件下载失败，请手动获取")

    print("解压标注文件...")
    extract_dir = os.path.dirname(val_json)  # data/raw/annotations/
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(annot_zip, "r") as zf:
        # COCO 标注 zip 包含 annotations/ 前缀，需要 strip
        for member in zf.namelist():
            if member.startswith("annotations/") or member.startswith("annotations"):
                member_name = member.split("/", 1)[1] if "/" in member else member
                if not member_name:
                    continue
                target = os.path.join(extract_dir, member_name)
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with zf.open(member) as src, open(target, "wb") as dst:
                    dst.write(src.read())
    print("解压完成")


def _list_images_in_dir(save_dir):
    return sorted(
        f for f in os.listdir(save_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )


def download_coco_images(save_dir, target_count, target_size,
                         val_json_path="data/raw/annotations/instances_val2017.json"):
    """
    从 COCO 2017 验证集标注中挑选图片，下载并缩放至 target_size。
    如果目录已有足够数量，仅进行缩放检查。
    返回：已就绪的图片文件名列表（长度不超过 target_count）。
    """
    ensure_annotations()

    os.makedirs(save_dir, exist_ok=True)
    existing = _list_images_in_dir(save_dir)

    if len(existing) >= target_count:
        print(f"[跳过] {save_dir} 已有 {len(existing)} 张图片，目标 {target_count}，直接缩放已有图片...")
        for f in existing[:target_count]:
            path = os.path.join(save_dir, f)
            img = cv2.imread(path)
            if img is not None:
                resized = cv2.resize(img, target_size)
                cv2.imwrite(path, resized)
                print(f"    已缩放: {path} -> {target_size}")
            else:
                print(f"    警告：无法读取 {path}，删除并准备重新下载")
                os.remove(path)
        existing_after = _list_images_in_dir(save_dir)
        if len(existing_after) >= target_count:
            return existing_after[:target_count]
        print(f"    缩放后有效图片 {len(existing_after)}，不足 {target_count}，继续下载...")

    with open(val_json_path, "r") as f:
        data = json.load(f)

    all_images = [img["file_name"] for img in data["images"]]
    random.shuffle(all_images)

    base_url = "http://images.cocodataset.org/val2017/"
    success = 0
    retries = 3

    for fn in all_images:
        if success >= target_count:
            break

        save_name = f"img_{success:04d}.jpg"
        path = os.path.join(save_dir, save_name)

        if os.path.exists(path):
            img_check = cv2.imread(path)
            if img_check is not None:
                print(f"[跳过已存在] {path}")
                success += 1
                continue
            os.remove(path)

        url = base_url + fn
        downloaded = False
        for attempt in range(1, retries + 1):
            try:
                print(f"[下载] ({attempt}/{retries}) {url} -> {path}")
                r = requests.get(url, timeout=(10, 30))
                if r.status_code == 200:
                    with open(path, "wb") as f:
                        f.write(r.content)
                    img = cv2.imread(path)
                    if img is not None:
                        resized = cv2.resize(img, target_size)
                        cv2.imwrite(path, resized)
                        success += 1
                        print(f"  ✓ 成功 [{success}/{target_count}] {fn} -> {target_size[0]}x{target_size[1]}")
                        downloaded = True
                        break
                    print("  ✗ 图片损坏，重试...")
                    os.remove(path)
                else:
                    print(f"  ✗ HTTP {r.status_code}，重试...")
            except Exception as e:
                print(f"  ✗ 网络异常: {e}，重试...")
                if os.path.exists(path):
                    os.remove(path)
                time.sleep(1)

        if not downloaded:
            print(f"  ✗✗ 放弃 {fn}，尝试下一张")

        time.sleep(0.2)

    print(f"[完成] {save_dir} 成功下载/验证 {success} 张图片（目标 {target_count}）")
    return _list_images_in_dir(save_dir)[:target_count]


def _tiny_imagenet_ready(extract_root: str) -> bool:
    if not os.path.isdir(extract_root):
        return False
    train_pat = os.path.join(extract_root, "train", "*", "images", "*.JPEG")
    test_dir = os.path.join(extract_root, "test", "images")
    if glob.glob(train_pat):
        return True
    if os.path.isdir(test_dir):
        for pat in ("*.JPEG", "*.jpeg", "*.jpg", "*.png"):
            if glob.glob(os.path.join(test_dir, pat)):
                return True
    return False


def ensure_tiny_imagenet(extract_root="data/raw/tiny-imagenet-200"):
    """下载并解压 Tiny ImageNet（ImageNet 子集，用于校准/测试）。"""
    if _tiny_imagenet_ready(extract_root):
        return extract_root

    zip_path = f"{extract_root}.zip"
    if not os.path.exists(zip_path):
        print(f"未找到 {zip_path}，开始下载 Tiny ImageNet (~240MB)...")
        os.makedirs(os.path.dirname(extract_root), exist_ok=True)
        for attempt in range(1, 4):
            try:
                print(f"  尝试 {attempt}/3 ...")
                r = requests.get(TINY_IMAGENET_URL, stream=True, timeout=(20, 120))
                if r.status_code == 200:
                    with open(zip_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                    print("  下载完成")
                    break
                print(f"  HTTP {r.status_code}，重试...")
                time.sleep(2)
            except Exception as e:
                print(f"  网络错误: {e}，重试...")
                time.sleep(2)
        else:
            raise RuntimeError("Tiny ImageNet 下载失败，请检查网络或手动放置 zip")

    print("解压 Tiny ImageNet...")
    os.makedirs(os.path.dirname(extract_root), exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(os.path.dirname(extract_root))
    print("解压完成")
    return extract_root


def download_imagenet_calib_images(save_dir, target_count, target_size,
                                   extract_root="data/raw/tiny-imagenet-200"):
    """
    使用 Tiny ImageNet 训练集图片（ImageNet 类别子集），缩放至 target_size，写入 save_dir。
    返回：文件名列表。
    """
    root = ensure_tiny_imagenet(extract_root)
    pattern = os.path.join(root, "train", "*", "images", "*.JPEG")
    all_jpeg = glob.glob(pattern)
    if not all_jpeg:
        raise RuntimeError(f"未在 {pattern} 找到 Tiny ImageNet 图片，请检查解压目录")

    random.shuffle(all_jpeg)
    os.makedirs(save_dir, exist_ok=True)

    success = 0
    for src in all_jpeg:
        if success >= target_count:
            break
        dst = os.path.join(save_dir, f"img_{success:04d}.jpg")
        img = cv2.imread(src)
        if img is None:
            continue
        resized = cv2.resize(img, target_size)
        cv2.imwrite(dst, resized)
        success += 1
        if success % 20 == 0:
            print(f"  已写入 {success}/{target_count} ...")

    print(f"[完成] {save_dir} 从 Tiny ImageNet 写入 {success} 张（目标 {target_count}）")
    return _list_images_in_dir(save_dir)[:target_count]


def generate_test_imagenet_txt(
    txt_file="./data/test_imagenet.txt",
    limit=100,
    extract_root="data/raw/tiny-imagenet-200",
):
    """
    从 Tiny ImageNet 的 test/images 目录收集图片路径，写入 txt（每行绝对路径）。
    不复制文件；分类脚本内会再 resize 到 224。
    """
    root = ensure_tiny_imagenet(extract_root)
    test_dir = os.path.join(root, "test", "images")
    if not os.path.isdir(test_dir):
        raise RuntimeError(f"未找到目录 {test_dir}，请确认 Tiny ImageNet 解压完整")

    paths = []
    for pat in ("*.JPEG", "*.jpeg", "*.jpg", "*.png"):
        paths.extend(glob.glob(os.path.join(test_dir, pat)))
    paths = sorted(set(paths))

    valid = []
    for path in paths:
        if cv2.imread(path) is None:
            continue
        valid.append(os.path.abspath(path))
        if len(valid) >= limit:
            break

    if not valid:
        raise RuntimeError(f"{test_dir} 下没有可读取的图片")

    os.makedirs(os.path.dirname(txt_file) or ".", exist_ok=True)
    with open(txt_file, "w", encoding="utf-8") as f:
        for p in valid:
            f.write(p + "\n")
    print(f"生成 {txt_file}，共 {len(valid)} 张（来源 {test_dir}）")


def generate_txt(save_dir, txt_file, limit):
    """生成绝对路径列表（只包含可正常读取的图片）"""
    valid = []
    for f in sorted(os.listdir(save_dir)):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(save_dir, f)
            img = cv2.imread(path)
            if img is not None:
                valid.append(os.path.abspath(path))
                if len(valid) >= limit:
                    break
    os.makedirs(os.path.dirname(txt_file) or ".", exist_ok=True)
    with open(txt_file, "w") as f:
        for p in valid:
            f.write(p + "\n")
    print(f"生成 {txt_file}，共 {len(valid)} 张有效图片")


def download_batch(save_dir, source, count, size_str, txt_name):
    """
    source: \"coco\" 使用 COCO val2017；\"imagenet\" 使用 Tiny ImageNet 训练图（ImageNet 风格校准集）。
    """
    target_w, target_h = map(int, size_str.split("/"))
    target_size = (target_w, target_h)
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n========== 开始处理: {save_dir} (来源 {source}, 尺寸 {target_size}) ==========")
    if source == "imagenet":
        download_imagenet_calib_images(save_dir, count, target_size)
    else:
        download_coco_images(save_dir, count, target_size)
    generate_txt(save_dir, txt_name, count)


if __name__ == "__main__":
    download_batch("./data/calib/coco", "coco", 100, "640/640", "./data/calib_coco.txt")
    download_batch("./data/calib/imagenet", "imagenet", 100, "224/224", "./data/calib_imagenet.txt")
    download_batch("./data/test/test_coco", "coco", 100, "640/640", "./data/test_coco.txt")
    generate_test_imagenet_txt("./data/test_imagenet.txt", limit=100)