import math

from indoor_loop.types import ObjectNode, ObjectRelation


def pairwise_relations(
    a: ObjectNode, b: ObjectNode, near_distance_m: float, far_distance_m: float
) -> list[ObjectRelation]:
    relations: list[ObjectRelation] = []
    dx = b.center_3d[0] - a.center_3d[0]
    dz = b.center_3d[2] - a.center_3d[2]
    distance = math.sqrt(dx * dx + dz * dz)

    if dx > 0:
        relations.append(
            ObjectRelation(subject_id=a.object_id, predicate="left_of", object_id=b.object_id, confidence=0.8)
        )
    elif dx < 0:
        relations.append(
            ObjectRelation(subject_id=a.object_id, predicate="right_of", object_id=b.object_id, confidence=0.8)
        )

    if distance <= near_distance_m:
        relations.append(
            ObjectRelation(subject_id=a.object_id, predicate="near", object_id=b.object_id, confidence=0.9)
        )
    elif distance <= far_distance_m:
        relations.append(
            ObjectRelation(subject_id=a.object_id, predicate="far", object_id=b.object_id, confidence=0.6)
        )

    return relations
