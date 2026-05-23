from pathlib import Path

from indoor_loop.config import AppConfig
from indoor_loop.graph.builder import build_object_nodes_from_segments, build_scene_graph
from indoor_loop.io.scannet import load_scannet_frame
from indoor_loop.models.oneformer import run_oneformer_with_fallback
from indoor_loop.viz.export import (
    write_overlay_png,
    write_records_json,
    write_scene_graph_json,
    write_text_artifact,
)


def process_frame_graph(
    rgb_path: Path,
    depth_path: Path,
    intrinsics_path: Path,
    output_dir: Path,
    config: AppConfig,
) -> None:
    frame = load_scannet_frame(rgb_path, depth_path, intrinsics_path)
    oneformer_output = run_oneformer_with_fallback(
        frame.rgb,
        model_name=config.models.oneformer_model_name,
        use_demo=config.models.use_demo_oneformer,
    )
    nodes = build_object_nodes_from_segments(
        oneformer_output.panoptic_map,
        oneformer_output.segments,
        frame.depth,
        frame.intrinsics,
    )
    graph = build_scene_graph(
        nodes,
        near_distance_m=config.graph.near_distance_m,
        far_distance_m=config.graph.far_distance_m,
        max_relations_per_node=config.graph.max_relations_per_node,
    )

    write_records_json(oneformer_output.segments, output_dir / "segments.json")
    write_records_json(graph.nodes, output_dir / "nodes.json")
    write_records_json(graph.relations, output_dir / "relations.json")
    write_scene_graph_json(graph, output_dir / "scene_graph.json")
    write_overlay_png(frame.rgb, oneformer_output.segments, output_dir / "overlay.png", graph=graph)
    write_text_artifact(f"{oneformer_output.run_mode}\n", output_dir / "oneformer_mode.txt")
