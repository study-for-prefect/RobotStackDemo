#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YOLO26 segmentation training for building blocks.

固定配置：
- 数据集：/home/wxm/code/RobotStackDemo/datasets/blocks_rgb/data.yaml
- 类别：10 类，来自 data.yaml
- 图片数量：约 172 张
- 任务：分割 segment
"""

from ultralytics import YOLO


DATA_YAML = "/home/wxm/code/RobotStackDemo/datasets/blocks_rgb/data.yaml"

# 从 YOLO26 nano 分割预训练模型开始训练
# 如果你本地 yolo26n-seg.pt 在当前目录，也可以改成 "./yolo26n-seg.pt"
MODEL_PATH = "yolo26n-seg.pt"

PROJECT = "/home/wxm/code/RobotStackDemo/runs/segment"
NAME = "building_block4"


def main():
    model = YOLO(MODEL_PATH)

    model.train(
        data=DATA_YAML,
        project=PROJECT,
        name="building_block5",

        task="segment",
        epochs=300,
        patience=0,
        batch=8,
        imgsz=960,
        workers=8,
        cache=False,
        plots=True,
        pretrained=True,

        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=5,
        cos_lr=True,

        mosaic=0.25,
        close_mosaic=80,
        mixup=0.0,
        copy_paste=0.03,

        hsv_h=0.0,
        hsv_s=0.10,
        hsv_v=0.15,

        degrees=5,
        translate=0.05,
        scale=0.25,
        shear=0.0,
        perspective=0.0,

        fliplr=0.5,
        flipud=0.0,

        overlap_mask=True,
        mask_ratio=4,
        single_cls=False,

        exist_ok=True,
    )


if __name__ == "__main__":
    main()
