import os
import cv2
import numpy as np


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DEFAULT_WEIGHT = os.path.join(
    PROJECT_ROOT,
    "runs",
    "segment",
    "building_block5",
    "weights",
    "best.pt",
)


def add_detector_args(parser):
    parser.add_argument("--detector-weight", default=DEFAULT_WEIGHT)
    parser.add_argument("--score-thresh", type=float, default=0.5)
    parser.add_argument("--candidate-score-thresh", type=float, default=0.15)
    parser.add_argument("--max-detections", type=int, default=30)
    parser.add_argument("--debug-topk", type=int, default=20)
    parser.add_argument("--detector-scale", type=float, default=1.0)
    parser.add_argument("--detector-device", default="cuda:0")
    parser.add_argument("--detector-iou", type=float, default=0.45)
    parser.add_argument("--detector-imgsz", type=int, default=960)
    return parser


def load_label_names(output_dir=None):
    return {
        0: "circle",
        1: "rectangle",
        2: "square blue",
        3: "square red",
        4: "square yellow",
        5: "square green",
        6: "semi square",
        7: "triangle",
    }


class DetectorModel:
    def __init__(self, args):
        from ultralytics import YOLO

        self.weight_path = getattr(args, "detector_weight", DEFAULT_WEIGHT)
        self.device = getattr(args, "detector_device", "cuda:0")
        self.iou = float(getattr(args, "detector_iou", 0.5))
        self.imgsz = int(getattr(args, "detector_imgsz", 1280))
        self.candidate_score_thresh = float(
            getattr(args, "candidate_score_thresh", 0.15)
        )

        if not os.path.exists(self.weight_path):
            raise FileNotFoundError(f"YOLO weight does not exist: {self.weight_path}")

        self.model = YOLO(self.weight_path)
        self.names = dict(self.model.names)

        print(f"[detector_runtime] Loaded YOLO weight: {self.weight_path}")
        print(f"[detector_runtime] imgsz={self.imgsz} iou={self.iou} device={self.device}")
        print(f"[detector_runtime] class names: {self.names}")

    def predict(self, frame_bgr, score_thresh, detector_scale, max_detections):
        if frame_bgr is None:
            raise ValueError("frame_bgr is None")

        original_h, original_w = frame_bgr.shape[:2]

        inference_bgr = frame_bgr
        scale = float(detector_scale)
        if scale <= 0:
            scale = 1.0

        if abs(scale - 1.0) > 1e-6:
            inference_bgr = cv2.resize(
                frame_bgr,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_LINEAR,
            )

        # Ultralytics 接受 BGR numpy 图像
        candidate_threshold = min(
            float(score_thresh),
            float(getattr(self, "candidate_score_thresh", 0.15)),
        )
        results = self.model.predict(
            source=inference_bgr,
            conf=candidate_threshold,
            iou=self.iou,
            device=self.device,
            imgsz=self.imgsz if self.imgsz > 0 else None,
            verbose=False,
        )

        if not results:
            return [], []

        result = results[0]
        boxes = result.boxes

        detections = []
        candidates = []

        if boxes is None or len(boxes) == 0:
            return detections, candidates

        xyxy = boxes.xyxy.detach().cpu().numpy()
        confs = boxes.conf.detach().cpu().numpy()
        clss = boxes.cls.detach().cpu().numpy().astype(int)

        masks_np = None
        mask_polygons = None
        if getattr(result, "masks", None) is not None and result.masks is not None:
            # shape: [N, H, W]，通常是推理尺寸下的 mask
            masks_np = result.masks.data.detach().cpu().numpy()
            mask_polygons = result.masks.xy

        order = np.argsort(-confs)

        for rank, i in enumerate(order):
            if len(detections) >= int(max_detections):
                break

            score = float(confs[i])
            if score < candidate_threshold:
                continue

            cls_id = int(clss[i])
            label = self.names.get(cls_id, str(cls_id))

            x1, y1, x2, y2 = xyxy[i].astype(float).tolist()

            if abs(scale - 1.0) > 1e-6:
                x1 /= scale
                y1 /= scale
                x2 /= scale
                y2 /= scale

            x1 = max(0.0, min(float(original_w - 1), x1))
            y1 = max(0.0, min(float(original_h - 1), y1))
            x2 = max(0.0, min(float(original_w - 1), x2))
            y2 = max(0.0, min(float(original_h - 1), y2))

            if x2 <= x1 or y2 <= y1:
                continue

            det = {
                "id": len(detections),
                "label": label,
                "label_id": cls_id,
                "class_id": cls_id,
                "score": score,
                "confidence": score,
                "primary_detector_passed": score >= float(score_thresh),
                "bbox": [x1, y1, x2, y2],
                "center_px": [(x1 + x2) * 0.5, (y1 + y2) * 0.5],
            }

            # 只提供轻量 mask 信息，避免 JSON 保存巨大 mask。
            if masks_np is not None and i < masks_np.shape[0]:
                mask_small = masks_np[i]
                mask = np.zeros((original_h, original_w), dtype=np.uint8)
                polygon = mask_polygons[i] if mask_polygons is not None and i < len(mask_polygons) else None
                if polygon is not None and len(polygon) >= 3:
                    polygon = np.asarray(polygon, dtype=np.float32)
                    if abs(scale - 1.0) > 1e-6:
                        polygon /= scale
                    polygon[:, 0] = np.clip(polygon[:, 0], 0, original_w - 1)
                    polygon[:, 1] = np.clip(polygon[:, 1], 0, original_h - 1)
                    cv2.fillPoly(mask, [np.round(polygon).astype(np.int32)], 1)
                    mask = mask.astype(bool)
                    mask_source = "polygon"
                else:
                    mask = cv2.resize(
                        mask_small.astype(np.float32),
                        (original_w, original_h),
                        interpolation=cv2.INTER_NEAREST,
                    ) > 0.5
                    mask_source = "resized_tensor"
                det["has_mask"] = True
                det["mask_shape"] = list(mask_small.shape)
                det["mask_area_px"] = int(np.count_nonzero(mask))
                det["mask_source"] = mask_source
                det["_mask_bool"] = mask
            else:
                det["has_mask"] = False

            detections.append(det)
            candidate = det.copy()
            candidate.pop("_mask_bool", None)
            candidates.append(candidate)

        return detections, candidates
