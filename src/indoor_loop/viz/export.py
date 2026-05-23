import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from indoor_loop.types import SceneGraph, SegmentRecord


def write_scene_graph_json(graph: SceneGraph, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(graph.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def write_records_json(records: list[Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [record.model_dump() if hasattr(record, "model_dump") else record for record in records]
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def write_text_artifact(text: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def write_overlay_png(
    rgb: np.ndarray,
    segments: list[SegmentRecord],
    output_path: Path,
    graph: SceneGraph | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    centers_by_object_id: dict[str, tuple[float, float]] = {}
    for segment in segments:
        x1, y1, x2, y2 = segment.bbox_xyxy
        color = (0, 255, 0) if segment.is_thing else (0, 128, 255)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        draw.text((x1, max(y1 - 10, 0)), f"{segment.class_name}:{segment.segment_id}", fill=color, font=font)
        cx, cy = segment.centroid_xy
        draw.ellipse([cx - 2, cy - 2, cx + 2, cy + 2], fill=color)
        centers_by_object_id[f"{segment.class_name}-{segment.segment_id}"] = (cx, cy)

    if graph is not None:
        for node in graph.nodes:
            center = centers_by_object_id.get(node.object_id)
            if center is None:
                continue
            cx, cy = center
            draw.text((cx + 3, cy + 3), node.staticness_level, fill=(255, 255, 0), font=font)

        for relation in graph.relations:
            source = centers_by_object_id.get(relation.subject_id)
            target = centers_by_object_id.get(relation.object_id)
            if source is None or target is None:
                continue
            draw.line([source, target], fill=(255, 0, 0), width=1)
            mid_x = (source[0] + target[0]) / 2
            mid_y = (source[1] + target[1]) / 2
            draw.text((mid_x, mid_y), relation.predicate, fill=(255, 0, 0), font=font)

    image.save(output_path)
