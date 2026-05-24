import json
import statistics
from collections import Counter
from pathlib import Path
from math import sqrt


KNOWN_RELATION_PREDICATES = {"left_of", "right_of", "near", "far"}


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
    relation_issue_counter: Counter[str] = Counter()
    relation_support_counter: Counter[str] = Counter()

    for scene_graph_path in scene_graph_paths:
        frame_dir = scene_graph_path.parent
        payload = json.loads(scene_graph_path.read_text(encoding="utf-8"))
        nodes = payload.get("nodes", [])
        relations = payload.get("relations", [])
        relation_check = _analyze_relation_consistency(payload)
        relation_issue_counter.update(relation_check["issue_counts"])
        relation_support_counter.update(relation_check["support_counts"])
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
                "relation_issue_count": relation_check["issue_total"],
                "relation_support_rate": relation_check["support_rate"],
                "relation_issue_samples": relation_check["issue_samples"],
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
        "relation_issue_counts": dict(relation_issue_counter.most_common(20)),
        "relation_support_counts": dict(relation_support_counter.most_common(20)),
        "mean_relation_support_rate": _mean([item["relation_support_rate"] for item in frame_summaries]),
    }


def _analyze_relation_consistency(payload: dict, near_distance_m: float = 1.5, far_distance_m: float = 3.0) -> dict:
    nodes = payload.get("nodes", [])
    relations = payload.get("relations", [])
    nodes_by_id = {str(node.get("object_id")): node for node in nodes}

    issue_counts: Counter[str] = Counter()
    support_counts: Counter[str] = Counter()
    issue_samples: list[str] = []
    supported_count = 0
    total_count = 0

    seen_triplets: set[tuple[str, str, str]] = set()
    pair_predicates: dict[frozenset[str], set[str]] = {}

    for relation in relations:
        total_count += 1
        subject_id = str(relation.get("subject_id"))
        object_id = str(relation.get("object_id"))
        predicate = str(relation.get("predicate"))
        confidence = relation.get("confidence")
        triplet = (subject_id, predicate, object_id)

        if predicate not in KNOWN_RELATION_PREDICATES:
            issue_counts["unknown_predicate"] += 1
            issue_samples.append(f"unknown_predicate:{subject_id}-{predicate}-{object_id}")
            continue
        if subject_id not in nodes_by_id or object_id not in nodes_by_id:
            issue_counts["missing_node"] += 1
            issue_samples.append(f"missing_node:{subject_id}-{predicate}-{object_id}")
            continue
        if subject_id == object_id:
            issue_counts["self_relation"] += 1
            issue_samples.append(f"self_relation:{subject_id}-{predicate}-{object_id}")
            continue
        if triplet in seen_triplets:
            issue_counts["duplicate_relation"] += 1
            issue_samples.append(f"duplicate_relation:{subject_id}-{predicate}-{object_id}")
            continue
        seen_triplets.add(triplet)

        source = nodes_by_id[subject_id]
        target = nodes_by_id[object_id]
        dx = float(target["center_3d"][0]) - float(source["center_3d"][0])
        dz = float(target["center_3d"][2]) - float(source["center_3d"][2])
        distance = sqrt(dx * dx + dz * dz)

        pair_key = frozenset({subject_id, object_id})
        pair_predicates.setdefault(pair_key, set()).add(predicate)

        if predicate == "left_of":
            if dx <= 0:
                issue_counts["left_right_sign_mismatch"] += 1
                issue_samples.append(f"left_of_mismatch:{subject_id}->{object_id}:dx={dx:.3f}")
                continue
        elif predicate == "right_of":
            if dx >= 0:
                issue_counts["left_right_sign_mismatch"] += 1
                issue_samples.append(f"right_of_mismatch:{subject_id}->{object_id}:dx={dx:.3f}")
                continue
        elif predicate == "near":
            if distance > near_distance_m:
                issue_counts["near_distance_mismatch"] += 1
                issue_samples.append(f"near_mismatch:{subject_id}->{object_id}:dist={distance:.3f}")
                continue
        elif predicate == "far":
            if not (near_distance_m < distance <= far_distance_m):
                issue_counts["far_distance_mismatch"] += 1
                issue_samples.append(f"far_mismatch:{subject_id}->{object_id}:dist={distance:.3f}")
                continue

        supported_count += 1
        support_counts[predicate] += 1

    for pair_key, predicates in pair_predicates.items():
        if "left_of" in predicates and "right_of" in predicates:
            issue_counts["pairwise_left_right_conflict"] += 1
        if "near" in predicates and "far" in predicates:
            issue_counts["pairwise_near_far_conflict"] += 1

    support_rate = supported_count / total_count if total_count else 0.0
    return {
        "issue_counts": issue_counts,
        "issue_samples": issue_samples[:20],
        "issue_total": sum(issue_counts.values()),
        "support_counts": support_counts,
        "support_rate": support_rate,
    }


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0
