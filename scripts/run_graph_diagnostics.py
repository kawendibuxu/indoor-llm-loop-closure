import argparse
import json
import statistics
from pathlib import Path


def summarize_graph_outputs(output_root: Path) -> dict:
    scene_graph_paths = sorted(output_root.glob("*/scene_graph.json"))
    node_counts: list[int] = []
    relation_counts: list[int] = []
    frame_summaries: list[dict] = []

    for scene_graph_path in scene_graph_paths:
        payload = json.loads(scene_graph_path.read_text(encoding="utf-8"))
        nodes = payload.get("nodes", [])
        relations = payload.get("relations", [])
        node_count = len(nodes)
        relation_count = len(relations)
        node_counts.append(node_count)
        relation_counts.append(relation_count)
        frame_summaries.append(
            {
                "frame_dir": str(scene_graph_path.parent),
                "node_count": node_count,
                "relation_count": relation_count,
                "is_empty": node_count == 0,
            }
        )

    empty_frames = [item for item in frame_summaries if item["is_empty"]]
    sparse_frames = [item for item in frame_summaries if item["node_count"] <= 1]
    dense_frames = sorted(frame_summaries, key=lambda item: item["node_count"], reverse=True)[:10]

    return {
        "total_frames": len(frame_summaries),
        "mean_node_count": statistics.mean(node_counts) if node_counts else 0.0,
        "mean_relation_count": statistics.mean(relation_counts) if relation_counts else 0.0,
        "empty_frame_count": len(empty_frames),
        "sparse_frame_count": len(sparse_frames),
        "top_dense_frames": dense_frames,
        "sample_empty_frames": empty_frames[:10],
        "sample_sparse_frames": sparse_frames[:10],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    summary = summarize_graph_outputs(args.output_root)
    summary_path = args.output_root / "diagnostics_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
