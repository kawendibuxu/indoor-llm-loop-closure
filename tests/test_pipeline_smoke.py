import numpy as np

from indoor_loop.graph.builder import build_object_nodes_from_segments, build_scene_graph
from indoor_loop.models.oneformer import run_oneformer_with_fallback


def test_pipeline_smoke_builds_non_empty_graph_from_fake_segments() -> None:
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    depth = np.ones((4, 4), dtype=np.float32) * 2.0
    intrinsics = np.array(
        [
            [2.0, 0.0, 1.5],
            [0.0, 2.0, 1.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    oneformer_output = run_oneformer_with_fallback(
        rgb,
        model_name="shi-labs/oneformer_ade20k_swin_large",
        use_demo=True,
    )
    nodes = build_object_nodes_from_segments(
        oneformer_output.panoptic_map,
        oneformer_output.segments,
        depth,
        intrinsics,
    )

    assert len(nodes) == 2
    graph = build_scene_graph(nodes, near_distance_m=1.5, far_distance_m=3.0, max_relations_per_node=4)
    assert len(graph.relations) >= 1
