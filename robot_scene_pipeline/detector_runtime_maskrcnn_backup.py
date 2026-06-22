import os

import cv2

from .io_utils import PROJECT_ROOT, project_path


DEFAULT_CONFIG = os.path.join(PROJECT_ROOT, "models", "custom_output_detector", "config.yml")
DEFAULT_WEIGHT = os.path.join(PROJECT_ROOT, "models", "custom_output_detector", "model_final.pth")


def add_detector_args(parser):
    parser.add_argument("--config-file", default=DEFAULT_CONFIG)
    parser.add_argument("--weight", default=DEFAULT_WEIGHT)
    parser.add_argument("--score-thresh", type=float, default=0.5)
    parser.add_argument("--max-detections", type=int, default=30)
    parser.add_argument("--detector-scale", type=float, default=1.0)
    parser.add_argument("--debug-topk", type=int, default=30)
    parser.add_argument("--device", default="cuda")


def load_label_names(output_dir):
    import json

    label_file = os.path.join(output_dir, "labels.json")
    if not os.path.exists(label_file):
        return {}
    with open(label_file, "r", encoding="utf-8") as f:
        labels = json.load(f)
    if isinstance(labels, list):
        return {str(index): name for index, name in enumerate(labels)}
    return labels


def get_field(boxlist, *names):
    for name in names:
        if boxlist.has_field(name):
            return boxlist.get_field(name)
    raise KeyError("None of the requested fields exist: {}".format(names))


class DetectorModel:
    def __init__(self, args):
        from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

        import torch
        from maskrcnn_benchmark.config import cfg
        from maskrcnn_benchmark.data.transforms import build_transforms
        from maskrcnn_benchmark.modeling.detector import build_detection_model
        from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer

        self.torch = torch
        self.cfg = cfg
        self.config_file = project_path(args.config_file)
        self.weight = project_path(args.weight)
        cfg.merge_from_file(self.config_file)
        cfg.MODEL.WEIGHT = self.weight
        cfg.MODEL.ROI_HEADS.SCORE_THRESH = min(cfg.MODEL.ROI_HEADS.SCORE_THRESH, args.score_thresh)

        device = args.device
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA is not available; falling back to CPU.", flush=True)
            device = "cpu"
        cfg.MODEL.DEVICE = device
        if device == "cpu":
            cfg.DTYPE = "float32"
        cfg.freeze()

        self.device = cfg.MODEL.DEVICE
        self.model = build_detection_model(cfg)
        self.model.to(self.device)
        self.model.eval()
        checkpointer = DetectronCheckpointer(cfg, self.model, save_dir="")
        checkpointer.load(self.weight, with_optim=False)
        self.transforms = build_transforms(cfg, is_train=False)
        self.size_divisibility = cfg.DATALOADER.SIZE_DIVISIBILITY
        self.label_names = load_label_names(os.path.dirname(self.weight)) or load_label_names(cfg.OUTPUT_DIR)

    def predict(self, frame_bgr, score_thresh, detector_scale, max_detections):
        from PIL import Image
        from maskrcnn_benchmark.structures.bounding_box import BoxList
        from maskrcnn_benchmark.structures.image_list import to_image_list

        inference_bgr = frame_bgr
        scale = float(detector_scale)
        if scale > 1.0:
            inference_bgr = cv2.resize(
                frame_bgr,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC,
            )
        else:
            scale = 1.0

        image_rgb = cv2.cvtColor(inference_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        original_size = pil_image.size
        empty_target = BoxList(self.torch.empty((0, 4)), original_size, mode="xyxy")
        tensor, _ = self.transforms(pil_image, empty_target)
        with self.torch.no_grad():
            image_list = to_image_list([tensor], self.size_divisibility).to(self.device)
            prediction = self.model(image_list)[0].to("cpu")
        prediction = prediction.resize(original_size).convert("xyxy")

        labels = get_field(prediction, "pred_labels", "labels").tolist()
        scores = get_field(prediction, "pred_scores", "scores").tolist()
        boxes = (prediction.bbox / scale).tolist()

        candidates = []
        detections = []
        sorted_items = sorted(zip(boxes, labels, scores), key=lambda item: item[2], reverse=True)
        for rank, (box, label, score) in enumerate(sorted_items):
            label_id = int(label)
            label_name = self.label_names.get(str(label_id), str(label_id))
            candidate = {
                "rank": rank,
                "bbox": [float(v) for v in box],
                "label_id": label_id,
                "label": label_name,
                "confidence": float(score),
            }
            candidates.append(candidate)
            if score < score_thresh:
                continue
            detections.append(
                {
                    "id": len(detections),
                    "bbox": [float(v) for v in box],
                    "label_id": label_id,
                    "label": label_name,
                    "confidence": float(score),
                }
            )
            if len(detections) >= max_detections:
                break
        return detections, candidates
