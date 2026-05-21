from pathlib import Path

import numpy as np

from indoor_loop.types import SceneGraph
from indoor_loop.viz.export import write_overlay_png, write_scene_graph_json


def test_write_scene_graph_json_creates_file(tmp_path: Path) -> None:
    graph = SceneGraph(nodes=[], relations=[])
    output_path = tmp_path / "scene_graph.json"

    write_scene_graph_json(graph, output_path)

    assert output_path.exists()
    assert output_path.read_text(encoding="utf-8").strip().startswith("{")


def test_write_overlay_png_creates_image(tmp_path: Path) -> None:
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    output_path = tmp_path / "overlay.png"

    write_overlay_png(rgb, [], output_path)

    assert output_path.exists()
