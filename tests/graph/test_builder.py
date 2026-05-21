import numpy as np

from indoor_loop.graph.builder import build_object_nodes_from_segments, build_scene_graph
from indoor_loop.types import ObjectNode, SegmentRecord


def test_build_scene_graph_adds_spatial_relations() -> None:
    nodes = [
        ObjectNode(
            object_id="bed-1",
            coarse_anchor_class="bed",
            refined_label="bed",
            label_confidence=0.9,
            staticness_level="hard_static",
            center_3d=[0.0, 0.0, 2.0],
            size_3d=[2.0, 1.0, 1.0],
            height_from_floor=0.0,
            support_type="floor",
            wall_attachment=True,
            saliency_score=0.9,
            observation_quality=0.9,
        ),
        ObjectNode(
            object_id="desk-1",
            coarse_anchor_class="table_like",
            refined_label="desk",
            label_confidence=0.85,
            staticness_level="soft_static",
            center_3d=[1.0, 0.0, 2.2],
            size_3d=[1.2, 0.8, 0.7],
            height_from_floor=0.0,
            support_type="floor",
            wall_attachment=False,
            saliency_score=0.8,
            observation_quality=0.9,
        ),
    ]

    graph = build_scene_graph(nodes, near_distance_m=1.5, far_distance_m=3.0, max_relations_per_node=4)

    predicates = [relation.predicate for relation in graph.relations]
    assert "left_of" in predicates
    assert "near" in predicates


def test_build_object_nodes_from_segments_projects_static_nodes() -> None:
    panoptic_map = np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 2, 2],
            [0, 0, 2, 2],
        ],
        dtype=np.int32,
    )
    depth = np.ones((4, 4), dtype=np.float32) * 2.0
    intrinsics = np.array(
        [
            [2.0, 0.0, 1.5],
            [0.0, 2.0, 1.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    segments = [
        SegmentRecord(
            segment_id="1",
            class_name="bed",
            is_thing=True,
            score=0.9,
            area=4,
            bbox_xyxy=[0, 0, 1, 1],
            centroid_xy=[0.5, 0.5],
        ),
        SegmentRecord(
            segment_id="2",
            class_name="person",
            is_thing=True,
            score=0.9,
            area=4,
            bbox_xyxy=[2, 2, 3, 3],
            centroid_xy=[2.5, 2.5],
        ),
    ]

    nodes = build_object_nodes_from_segments(panoptic_map, segments, depth, intrinsics)

    assert len(nodes) == 1
    assert nodes[0].coarse_anchor_class == "bed"
    assert nodes[0].staticness_level == "hard_static"
    assert nodes[0].center_3d[2] == 2.0
