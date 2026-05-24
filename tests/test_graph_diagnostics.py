import json
from pathlib import Path

from indoor_loop.diagnostics import summarize_graph_outputs
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


def test_summarize_graph_outputs_tracks_modes_and_errors(tmp_path: Path) -> None:
    real_frame = tmp_path / "frame-000001"
    real_frame.mkdir()
    (real_frame / "scene_graph.json").write_text(
        json.dumps({"nodes": [{"coarse_anchor_class": "bed"}], "relations": []}),
        encoding="utf-8",
    )
    (real_frame / "oneformer_mode.txt").write_text("real\n", encoding="utf-8")

    fallback_frame = tmp_path / "frame-000002"
    fallback_frame.mkdir()
    (fallback_frame / "scene_graph.json").write_text(
        json.dumps({"nodes": [{"coarse_anchor_class": "desk"}, {"coarse_anchor_class": "wall"}], "relations": []}),
        encoding="utf-8",
    )
    (fallback_frame / "oneformer_mode.txt").write_text("demo_fallback\n", encoding="utf-8")
    (fallback_frame / "oneformer_error.txt").write_text("RuntimeError: boom\n", encoding="utf-8")

    summary = summarize_graph_outputs(tmp_path, sample_count=5)

    assert summary["total_frames"] == 2
    assert summary["real_frame_count"] == 1
    assert summary["demo_fallback_count"] == 1
    assert summary["error_counts"]["RuntimeError: boom"] == 1
    assert summary["top_classes"][0][0] == "bed"


def test_summarize_graph_outputs_detects_relation_inconsistency(tmp_path: Path) -> None:
    frame_dir = tmp_path / "frame-000001"
    frame_dir.mkdir()
    (frame_dir / "scene_graph.json").write_text(
        json.dumps(
            {
                "nodes": [
                    {
                        "object_id": "a",
                        "coarse_anchor_class": "bed",
                        "center_3d": [0.0, 0.0, 0.0],
                    },
                    {
                        "object_id": "b",
                        "coarse_anchor_class": "desk",
                        "center_3d": [1.0, 0.0, 0.0],
                    },
                ],
                "relations": [
                    {"subject_id": "a", "predicate": "left_of", "object_id": "b", "confidence": 0.8},
                    {"subject_id": "a", "predicate": "right_of", "object_id": "b", "confidence": 0.8},
                ],
            }
        ),
        encoding="utf-8",
    )
    (frame_dir / "oneformer_mode.txt").write_text("real\n", encoding="utf-8")

    summary = summarize_graph_outputs(tmp_path, sample_count=5)

    assert summary["relation_issue_counts"]["pairwise_left_right_conflict"] == 1
    assert summary["sample_unstable_frames"] == []
