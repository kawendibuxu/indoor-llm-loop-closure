import json
from pathlib import Path

from indoor_loop.types import SceneGraph


def test_scene_graph_json_has_nodes_and_relations_lists(tmp_path: Path) -> None:
    frame_dir = tmp_path / "frame-000001"
    frame_dir.mkdir()
    scene_graph_path = frame_dir / "scene_graph.json"
    scene_graph_path.write_text(
        json.dumps(
            SceneGraph(nodes=[], relations=[]).model_dump(),
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    payload = json.loads(scene_graph_path.read_text(encoding="utf-8"))

    assert "nodes" in payload
    assert "relations" in payload
