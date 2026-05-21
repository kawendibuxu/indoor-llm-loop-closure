from indoor_loop.types import ObjectNode, ObjectRelation, SceneGraph


def test_scene_graph_serializes_expected_fields() -> None:
    node = ObjectNode(
        object_id="obj-1",
        coarse_anchor_class="bed",
        refined_label="bed",
        label_confidence=0.9,
        staticness_level="hard_static",
        center_3d=[1.0, 0.5, 2.0],
        size_3d=[2.0, 1.0, 1.5],
        height_from_floor=0.0,
        support_type="floor",
        wall_attachment=True,
        saliency_score=0.8,
        observation_quality=0.95,
    )
    relation = ObjectRelation(
        subject_id="obj-1",
        predicate="against_wall",
        object_id="wall-1",
        confidence=0.88,
    )

    graph = SceneGraph(nodes=[node], relations=[relation])
    payload = graph.model_dump()

    assert payload["nodes"][0]["coarse_anchor_class"] == "bed"
    assert payload["relations"][0]["predicate"] == "against_wall"
