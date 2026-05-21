import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from indoor_loop.config import load_config
from indoor_loop.graph.builder import build_object_nodes_from_segments, build_scene_graph
from indoor_loop.io.scannet import load_scannet_frame
from indoor_loop.models.oneformer import run_oneformer_with_fallback
from indoor_loop.viz.export import write_overlay_png, write_scene_graph_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    frame = load_scannet_frame(
        config.data.rgb_path,
        config.data.depth_path,
        config.data.intrinsics_path,
    )
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
    output_path = config.output.output_dir / "scene_graph.json"
    write_scene_graph_json(graph, output_path)
    write_overlay_png(
        frame.rgb,
        oneformer_output.segments,
        config.output.output_dir / "overlay.png",
        graph=graph,
    )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
