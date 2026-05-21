from pathlib import Path

import numpy as np

from indoor_loop.config import load_config
from indoor_loop.graph.builder import build_object_nodes_from_segments, build_scene_graph
from indoor_loop.io.scannet import load_scannet_frame
from indoor_loop.models.oneformer import run_oneformer_with_fallback
from indoor_loop.viz.export import write_overlay_png
from indoor_loop.types import ObjectNode, ObjectRelation, SceneGraph, SegmentRecord
from indoor_loop.viz.export import write_records_json


def test_sample_config_has_output_dir() -> None:
    config = load_config(Path("configs/scannet_sample.yaml"))
    assert config.output.output_dir == Path("outputs/sample_frame")


def test_demo_oneformer_output_matches_loaded_frame() -> None:
    config = load_config(Path("configs/scannet_sample.yaml"))
    frame = load_scannet_frame(
        config.data.rgb_path,
        config.data.depth_path,
        config.data.intrinsics_path,
    )

    output = run_oneformer_with_fallback(
        frame.rgb,
        model_name=config.models.oneformer_model_name,
        use_demo=True,
    )

    assert output.panoptic_map.shape == frame.depth.shape
    assert len(output.segments) == 2


def test_write_records_json_creates_separate_artifact(tmp_path: Path) -> None:
    output_path = tmp_path / "nodes.json"
    records = [
        ObjectNode(
            object_id="bed-1",
            coarse_anchor_class="bed",
            refined_label="bed",
            label_confidence=0.9,
            staticness_level="hard_static",
            center_3d=[0.0, 0.0, 2.0],
            size_3d=[1.0, 1.0, 1.0],
            height_from_floor=0.0,
            support_type="floor",
            wall_attachment=True,
            saliency_score=0.9,
            observation_quality=0.9,
        )
    ]

    write_records_json(records, output_path)

    assert output_path.exists()
    assert '"object_id": "bed-1"' in output_path.read_text(encoding="utf-8")


def test_single_frame_pipeline_produces_graph_objects() -> None:
    config = load_config(Path("configs/scannet_sample.yaml"))
    frame = load_scannet_frame(
        config.data.rgb_path,
        config.data.depth_path,
        config.data.intrinsics_path,
    )
    output = run_oneformer_with_fallback(
        frame.rgb,
        model_name=config.models.oneformer_model_name,
        use_demo=True,
    )
    nodes = build_object_nodes_from_segments(
        output.panoptic_map,
        output.segments,
        frame.depth,
        frame.intrinsics,
    )
    graph = build_scene_graph(
        nodes,
        near_distance_m=config.graph.near_distance_m,
        far_distance_m=config.graph.far_distance_m,
        max_relations_per_node=config.graph.max_relations_per_node,
    )

    assert len(graph.nodes) == 2
    assert len(graph.relations) >= 1
