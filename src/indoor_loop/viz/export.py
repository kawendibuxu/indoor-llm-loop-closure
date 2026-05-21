import json
from pathlib import Path

import numpy as np
from PIL import Image

from indoor_loop.types import SceneGraph


def write_scene_graph_json(graph: SceneGraph, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(graph.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def write_overlay_png(rgb: np.ndarray, _segments: list[object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(output_path)
