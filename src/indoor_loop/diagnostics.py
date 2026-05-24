import json
import statistics
from collections import Counter
from pathlib import Path


def summarize_graph_outputs(output_root: Path, sample_count: int = 10) -> dict:
    scene_graph_paths = sorted(output_root.glob("*/scene_graph.json"))
    node_counts: list[int] = []
    relation_counts: list[int] = []
    frame_summaries: list[dict] = []
    class_counter: Counter[str] = Counter()
    mode_counter: Counter[str] = Counter()
    error_counter: Counter[str] = Counter()
    node_histogram: Counter[int] = Counter()
    relation_histogram: Counter[int] = Counter()

    for scene_graph_path in scene_graph_paths:
        frame_dir = scene_graph_path.parent
        payload = json.loads(scene_graph_path.read_text(encoding="utf-8"))
        nodes = payload.get("nodes", [])
        relations = payload.get("relations", [])
        node_count = len(nodes)
        relation_count = len(relations)
        node_counts.append(node_count)
        relation_counts.append(relation_count)
        node_histogram[node_count] += 1
        relation_histogram[relation_count] += 1

        for node in nodes:
            class_counter[str(node.get("coarse_anchor_class", "unknown"))] += 1

        mode_path = frame_dir / "oneformer_mode.txt"
        mode = mode_path.read_text(encoding="utf-8").strip() if mode_path.exists() else "missing"
        mode_counter[mode] += 1

        error_path = frame_dir / "oneformer_error.txt"
        error_message = error_path.read_text(encoding="utf-8").strip() if error_path.exists() else ""
        if error_message:
            error_counter[error_message] += 1

        frame_summaries.append(
            {
                "frame_dir": str(frame_dir),
                "node_count": node_count,
                "relation_count": relation_count,
                "mode": mode,
                "error": error_message or None,
                "is_empty": node_count == 0,
            }
        )

    empty_frames = [item for item in frame_summaries if item["is_empty"]]
    sparse_frames = [item for item in frame_summaries if item["node_count"] <= 1]
    dense_frames = sorted(frame_summaries, key=lambda item: (item["node_count"], item["relation_count"]), reverse=True)
    unstable_frames = [item for item in frame_summaries if item["mode"] != "real"]

    return {
        "total_frames": len(frame_summaries),
        "mean_node_count": statistics.mean(node_counts) if node_counts else 0.0,
        "mean_relation_count": statistics.mean(relation_counts) if relation_counts else 0.0,
        "median_node_count": statistics.median(node_counts) if node_counts else 0.0,
        "median_relation_count": statistics.median(relation_counts) if relation_counts else 0.0,
        "empty_frame_count": len(empty_frames),
        "sparse_frame_count": len(sparse_frames),
        "real_frame_count": mode_counter.get("real", 0),
        "demo_fallback_count": mode_counter.get("demo_fallback", 0),
        "demo_frame_count": mode_counter.get("demo", 0),
        "mode_counts": dict(mode_counter),
        "top_dense_frames": dense_frames[:sample_count],
        "sample_empty_frames": empty_frames[:sample_count],
        "sample_sparse_frames": sparse_frames[:sample_count],
        "sample_unstable_frames": unstable_frames[:sample_count],
        "top_classes": class_counter.most_common(20),
        "node_histogram": dict(sorted(node_histogram.items())),
        "relation_histogram": dict(sorted(relation_histogram.items())),
        "error_counts": dict(error_counter.most_common(20)),
    }
