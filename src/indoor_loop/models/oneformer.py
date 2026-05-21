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
