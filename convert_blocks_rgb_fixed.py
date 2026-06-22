#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LabelMe -> YOLO segmentation dataset converter

固定配置：
- 数据集目录：
  /home/wxm/code/RobotStackDemo/datasets/blocks_rgb

转换规则：
1. 按图片遍历，不按 json 遍历。
2. 有同名 LabelMe json：正常转换。
3. 没有同名 json：生成空 label，作为负样本参加训练。
4. 自动跳过 *_sg.json。
5. 自动跳过 workspace。
6. 自动清理旧 images/train、images/val、labels/train、labels/val。
7. 自动删除 labels/*.cache。
"""

import json
import random
import shutil
from pathlib import Path
from PIL import Image


DATASET_DIR = Path("/home/wxm/code/RobotStackDemo/datasets/blocks_rgb")

INPUT_DIR = DATASET_DIR
OUTPUT_DIR = DATASET_DIR

TASK = "segment"
VAL_RATIO = 0.2
SEED = 42

CLASS_MAP = {
    "circle": 0,
    "rectangle": 1,
    "square blue": 2,
    "square red": 3,
    "square yellow": 4,
    "square green": 5,
    "semi square": 6,
    "triangle": 7,
    "semi circle": 8,
    "concave": 9,
}

SKIP_LABELS = {"workspace"}

IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp"]


def is_in_output_dir(path: Path) -> bool:
    parts = set(path.parts)
    return ("images" in parts and ("train" in parts or "val" in parts)) or (
        "labels" in parts and ("train" in parts or "val" in parts)
    )


def is_sg_json(json_path: Path) -> bool:
    return json_path.stem.endswith("_sg")


def find_labelme_json(img_path: Path):
    json_path = img_path.with_suffix(".json")
    if json_path.exists() and not is_sg_json(json_path):
        return json_path
    return None


def get_all_source_images(input_dir: Path):
    images = []
    for p in input_dir.rglob("*"):
        if not p.is_file():
            continue
        if is_in_output_dir(p):
            continue
        if p.suffix.lower() in IMG_EXTS:
            images.append(p)
    return sorted(images)


def get_image_size(img_path: Path, data):
    if data:
        w = data.get("imageWidth")
        h = data.get("imageHeight")
        if w and h:
            return int(w), int(h)

    with Image.open(img_path) as img:
        return img.size


def clamp(v, low, high):
    return max(low, min(high, float(v)))


def shape_to_bbox(points, img_w, img_h):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    x_min = clamp(min(xs), 0, img_w)
    y_min = clamp(min(ys), 0, img_h)
    x_max = clamp(max(xs), 0, img_w)
    y_max = clamp(max(ys), 0, img_h)

    if x_max <= x_min or y_max <= y_min:
        return None

    x_center = ((x_min + x_max) / 2) / img_w
    y_center = ((y_min + y_max) / 2) / img_h
    width = (x_max - x_min) / img_w
    height = (y_max - y_min) / img_h

    return x_center, y_center, width, height


def rectangle_to_polygon(points):
    if len(points) != 2:
        return points

    x1, y1 = points[0]
    x2, y2 = points[1]

    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def shape_to_segment(shape, img_w, img_h):
    points = shape.get("points", [])
    shape_type = shape.get("shape_type", "")

    if shape_type == "rectangle":
        points = rectangle_to_polygon(points)

    if len(points) < 3:
        return None

    seg = []
    for x, y in points:
        x = clamp(x, 0, img_w) / img_w
        y = clamp(y, 0, img_h) / img_h
        seg.extend([x, y])

    return seg


def convert_json_to_yolo_lines(json_path: Path, img_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    img_w, img_h = get_image_size(img_path, data)
    lines = []

    for shape in data.get("shapes", []):
        label = shape.get("label", "").strip()
        points = shape.get("points", [])

        if label in SKIP_LABELS:
            continue

        if label not in CLASS_MAP:
            print(f"[WARN] 跳过未知类别: {label} in {json_path}")
            continue

        if len(points) < 2:
            continue

        cls_id = CLASS_MAP[label]

        if TASK == "detect":
            bbox = shape_to_bbox(points, img_w, img_h)
            if bbox is None:
                continue

            x, y, w, h = bbox
            lines.append(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

        elif TASK == "segment":
            seg = shape_to_segment(shape, img_w, img_h)

            if seg is None:
                bbox = shape_to_bbox(points, img_w, img_h)
                if bbox is None:
                    continue

                x, y, w, h = bbox
                x1 = max(0, x - w / 2)
                y1 = max(0, y - h / 2)
                x2 = min(1, x + w / 2)
                y2 = min(1, y + h / 2)
                seg = [x1, y1, x2, y1, x2, y2, x1, y2]

            seg_text = " ".join(f"{v:.6f}" for v in seg)
            lines.append(f"{cls_id} {seg_text}")

        else:
            raise ValueError(f"未知任务类型: {TASK}")

    return lines


def write_data_yaml(output_dir: Path):
    id_to_name = {v: k for k, v in CLASS_MAP.items()}

    lines = [
        "# 数据集路径配置",
        f"path: {output_dir.resolve()}",
        "train: images/train",
        "val: images/val",
        "",
        "# 类别定义",
        f"nc: {len(CLASS_MAP)}",
        "names:",
    ]

    for i in range(len(CLASS_MAP)):
        lines.append(f"  {i}: {id_to_name[i]}")

    if TASK == "segment":
        lines.extend(["", "# 分割任务标记", "segment: True"])

    yaml_path = output_dir / "data.yaml"
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def clean_output_dirs(output_dir: Path):
    for sub in [
        output_dir / "images" / "train",
        output_dir / "images" / "val",
        output_dir / "labels" / "train",
        output_dir / "labels" / "val",
    ]:
        if sub.exists():
            shutil.rmtree(sub)
        sub.mkdir(parents=True, exist_ok=True)

    for cache in output_dir.rglob("*.cache"):
        if "labels" in cache.parts:
            cache.unlink()


def count_from_labelme_jsons(image_files):
    counts = {name: 0 for name in CLASS_MAP}
    skipped = {}
    no_json_images = 0
    json_images = 0
    empty_after_filter = 0

    for img_path in image_files:
        json_path = find_labelme_json(img_path)

        if json_path is None:
            no_json_images += 1
            continue

        json_images += 1

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        valid_count = 0

        for shape in data.get("shapes", []):
            label = shape.get("label", "").strip()

            if label in SKIP_LABELS:
                continue

            if label in counts:
                counts[label] += 1
                valid_count += 1
            else:
                skipped[label] = skipped.get(label, 0) + 1

        if valid_count == 0:
            empty_after_filter += 1

    return counts, skipped, json_images, no_json_images, empty_after_filter


def main():
    print("LabelMe -> YOLO 转换开始")
    print(f"输入目录: {INPUT_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"任务类型: {TASK}")
    print("遍历方式: 按图片遍历，未标注图片生成空 txt 作为负样本")

    image_files = get_all_source_images(INPUT_DIR)

    if not image_files:
        raise RuntimeError(f"未找到图片: {INPUT_DIR}")

    counts, skipped, json_images, no_json_images, empty_after_filter = count_from_labelme_jsons(image_files)

    print("\n图片统计:")
    print(f"源图片总数: {len(image_files)}")
    print(f"有同名 json 的图片: {json_images}")
    print(f"无 json 负样本图片: {no_json_images}")
    print(f"有 json 但过滤后为空的图片: {empty_after_filter}")

    print("\n类别实例数量:")
    for name, idx in CLASS_MAP.items():
        print(f"{idx}: {name}: {counts[name]}")

    if skipped:
        print("\n被跳过的未知类别:")
        for name, count in sorted(skipped.items()):
            print(f"{name}: {count}")

    clean_output_dirs(OUTPUT_DIR)

    random.seed(SEED)
    random.shuffle(image_files)

    val_count = int(len(image_files) * VAL_RATIO)
    val_set = set(image_files[:val_count])

    train_num = 0
    val_num = 0
    empty_label_num = 0

    for img_path in image_files:
        split = "val" if img_path in val_set else "train"

        img_out = OUTPUT_DIR / "images" / split / img_path.name
        label_out = OUTPUT_DIR / "labels" / split / f"{img_path.stem}.txt"

        img_out.parent.mkdir(parents=True, exist_ok=True)
        label_out.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(img_path, img_out)

        json_path = find_labelme_json(img_path)

        if json_path is None:
            lines = []
        else:
            lines = convert_json_to_yolo_lines(json_path, img_path)

        if len(lines) == 0:
            empty_label_num += 1

        label_out.write_text("\n".join(lines), encoding="utf-8")

        if split == "train":
            train_num += 1
        else:
            val_num += 1

    write_data_yaml(OUTPUT_DIR)

    print("\n转换完成")
    print(f"train 图片数: {train_num}")
    print(f"val 图片数: {val_num}")
    print(f"空 label 图片数: {empty_label_num}")
    print(f"data.yaml: {OUTPUT_DIR / 'data.yaml'}")


if __name__ == "__main__":
    main()