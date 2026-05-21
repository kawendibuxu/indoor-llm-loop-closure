from dataclasses import dataclass
from typing import Any

import numpy as np

from indoor_loop.types import SegmentRecord


def build_segments_from_panoptic(
    panoptic_map: np.ndarray, segments_info: list[dict[str, Any]]
) -> list[SegmentRecord]:
    segments: list[SegmentRecord] = []
    for info in segments_info:
        segment_id = int(info["id"])
        mask = panoptic_map == segment_id
        if not np.any(mask):
            continue
        ys, xs = np.where(mask)
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        centroid = [float(xs.mean()), float(ys.mean())]
        segments.append(
            SegmentRecord(
                segment_id=str(segment_id),
                class_name=str(info["label_name"]),
                is_thing=str(info["label_name"]) not in {"wall", "floor", "ceiling"},
                score=float(info["score"]),
                area=int(mask.sum()),
                bbox_xyxy=bbox,
                centroid_xy=centroid,
            )
        )
    return segments


@dataclass(slots=True)
class OneFormerOutput:
    panoptic_map: np.ndarray
    segments: list[SegmentRecord]


class DemoOneFormerRunner:
    def predict(self, rgb: np.ndarray) -> OneFormerOutput:
        height, width = rgb.shape[:2]
        panoptic_map = np.zeros((height, width), dtype=np.int32)
        mid_x = max(width // 2, 1)
        panoptic_map[:, :mid_x] = 1
        panoptic_map[:, mid_x:] = 2
        segments = build_segments_from_panoptic(
            panoptic_map,
            [
                {"id": 1, "label_name": "bed", "score": 0.9},
                {"id": 2, "label_name": "desk", "score": 0.8},
            ],
        )
        return OneFormerOutput(panoptic_map=panoptic_map, segments=segments)


class RealOneFormerRunner:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def predict(self, rgb: np.ndarray) -> OneFormerOutput:
        try:
            from PIL import Image
            from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor
        except ImportError as exc:
            raise RuntimeError("transformers or OneFormer dependencies are unavailable") from exc

        processor = OneFormerProcessor.from_pretrained(self.model_name)
        model = OneFormerForUniversalSegmentation.from_pretrained(self.model_name)
        pil_image = Image.fromarray(rgb)
        inputs = processor(images=pil_image, task_inputs=["panoptic"], return_tensors="pt")
        outputs = model(**inputs)
        result = processor.post_process_panoptic_segmentation(
            outputs,
            target_sizes=[pil_image.size[::-1]],
        )[0]
        panoptic_map = result["segmentation"].cpu().numpy().astype(np.int32)
        segments = build_segments_from_panoptic(panoptic_map, result["segments_info"])
        return OneFormerOutput(panoptic_map=panoptic_map, segments=segments)


def run_oneformer_with_fallback(rgb: np.ndarray, model_name: str, use_demo: bool = False) -> OneFormerOutput:
    if use_demo:
        return DemoOneFormerRunner().predict(rgb)
    try:
        return RealOneFormerRunner(model_name).predict(rgb)
    except Exception:
        return DemoOneFormerRunner().predict(rgb)
