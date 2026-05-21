from indoor_loop.graph.relations import pairwise_relations
from indoor_loop.types import ObjectNode


def test_pairwise_relations_emits_directional_and_distance_edges() -> None:
    source = ObjectNode(
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
    )
    target = ObjectNode(
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
    )

    relations = pairwise_relations(source, target, near_distance_m=1.5, far_distance_m=3.0)
    predicates = [relation.predicate for relation in relations]

    assert "left_of" in predicates
    assert "near" in predicates
