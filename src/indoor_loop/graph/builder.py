import numpy as np

from indoor_loop.graph.projection import project_mask_to_3d_bbox
from indoor_loop.graph.relations import pairwise_relations
from indoor_loop.graph.staticness import classify_staticness
from indoor_loop.models.relabel import map_to_anchor_class
from indoor_loop.types import ObjectNode, SceneGraph, SegmentRecord


def build_object_nodes_from_segments(
    panoptic_map: np.ndarray,
    segments: list[SegmentRecord],
    depth: np.ndarray,
    intrinsics: np.ndarray,
) -> list[ObjectNode]:
    nodes: list[ObjectNode] = []
    for segment in segments:
        staticness_level = classify_staticness(segment.class_name, observation_quality=segment.score)
        if staticness_level == "dynamic":
            continue
        mask = panoptic_map == int(segment.segment_id)
        center_3d, size_3d = project_mask_to_3d_bbox(mask, depth, intrinsics)
        nodes.append(
            ObjectNode(
                object_id=f"{segment.class_name}-{segment.segment_id}",
                coarse_anchor_class=map_to_anchor_class(segment.class_name),
                refined_label=segment.class_name,
                label_confidence=segment.score,
                staticness_level=staticness_level,
                center_3d=center_3d,
                size_3d=size_3d,
                height_from_floor=max(center_3d[1], 0.0),
                support_type="floor",
                wall_attachment=segment.class_name in {"bed", "cabinet", "window", "door", "wall"},
                saliency_score=min(1.0, segment.area / max(float(panoptic_map.size), 1.0)),
                observation_quality=segment.score,
            )
        )
    return nodes


def build_scene_graph(
    nodes: list[ObjectNode],
    near_distance_m: float,
    far_distance_m: float,
    max_relations_per_node: int,
) -> SceneGraph:
    relations = []
    for index, source in enumerate(nodes):
        local_relations = []
        for target in nodes[index + 1 :]:
            local_relations.extend(pairwise_relations(source, target, near_distance_m, far_distance_m))
        relations.extend(local_relations[:max_relations_per_node])
    return SceneGraph(nodes=nodes, relations=relations)
