from dataclasses import dataclass
from typing import Any

import numpy as np

from indoor_loop.types import SegmentRecord


def build_segments_from_panoptic(
    panoptic_map: np.ndarray,
    segments_info: list[dict[str, Any]],
    label_lookup: dict[int, str] | None = None,
) -> list[SegmentRecord]:
    segments: list[SegmentRecord] = []
    for info in segments_info:
        segment_id = int(info["id"])
        label_id = int(info.get("label_id", segment_id))
        label_name = info.get("label_name")
        if label_name is None and label_lookup is not None:
            label_name = label_lookup.get(label_id)
        if label_name is None:
            label_name = f"class_{label_id}"
        mask = panoptic_map == segment_id
        if not np.any(mask):
            continue
        ys, xs = np.where(mask)
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        centroid = [float(xs.mean()), float(ys.mean())]
        segments.append(
            SegmentRecord(
                segment_id=str(segment_id),
                class_name=str(label_name),
                is_thing=str(label_name) not in {"wall", "floor", "ceiling"},
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
    run_mode: str = "demo"
    error_message: str | None = None


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
        return OneFormerOutput(
            panoptic_map=panoptic_map,
            segments=segments,
            run_mode="demo",
            error_message=None,
        )


class RealOneFormerRunner:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._processor = None
        self._model = None
        self._label_lookup: dict[int, str] | None = None

    def _ensure_loaded(self) -> None:
        if self._processor is not None and self._model is not None:
            return

        try:
            from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor
            from transformers.utils import logging as transformers_logging
        except ImportError as exc:
            raise RuntimeError("transformers or OneFormer dependencies are unavailable") from exc

        transformers_logging.set_verbosity_error()
        self._processor = OneFormerProcessor.from_pretrained(self.model_name)
        self._model = OneFormerForUniversalSegmentation.from_pretrained(self.model_name)
        self._model.eval()
        self._label_lookup = {
            int(label_id): str(label_name)
            for label_id, label_name in getattr(self._model.config, "id2label", {}).items()
        }

    def predict(self, rgb: np.ndarray) -> OneFormerOutput:
        from PIL import Image
        import torch

        self._ensure_loaded()
        pil_image = Image.fromarray(rgb)
        inputs = self._processor(images=pil_image, task_inputs=["panoptic"], return_tensors="pt")
        with torch.no_grad():
            outputs = self._model(**inputs)
            result = self._processor.post_process_panoptic_segmentation(
                outputs,
                target_sizes=[pil_image.size[::-1]],
            )[0]
        panoptic_map = result["segmentation"].cpu().numpy().astype(np.int32)
        segments = build_segments_from_panoptic(
            panoptic_map,
            result["segments_info"],
            label_lookup=self._label_lookup,
        )
        return OneFormerOutput(
            panoptic_map=panoptic_map,
            segments=segments,
            run_mode="real",
            error_message=None,
        )


_REAL_RUNNER_CACHE: dict[str, RealOneFormerRunner] = {}


def get_oneformer_runner(model_name: str, use_demo: bool = False) -> DemoOneFormerRunner | RealOneFormerRunner:
    if use_demo:
        return DemoOneFormerRunner()
    runner = _REAL_RUNNER_CACHE.get(model_name)
    if runner is None:
        runner = RealOneFormerRunner(model_name)
        _REAL_RUNNER_CACHE[model_name] = runner
    return runner


def run_oneformer_with_fallback(rgb: np.ndarray, model_name: str, use_demo: bool = False) -> OneFormerOutput:
    if use_demo:
        return DemoOneFormerRunner().predict(rgb)
    try:
        return get_oneformer_runner(model_name).predict(rgb)
    except Exception as exc:
        output = DemoOneFormerRunner().predict(rgb)
        return OneFormerOutput(
            panoptic_map=output.panoptic_map,
            segments=output.segments,
            run_mode="demo_fallback",
            error_message=f"{type(exc).__name__}: {exc}",
        )
