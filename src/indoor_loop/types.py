from typing import Literal

from pydantic import BaseModel, Field


StaticnessLevel = Literal["hard_static", "soft_static", "dynamic"]


class SegmentRecord(BaseModel):
    segment_id: str
    class_name: str
    is_thing: bool
    score: float
    area: int
    bbox_xyxy: list[int]
    centroid_xy: list[float]


class ObjectNode(BaseModel):
    object_id: str
    coarse_anchor_class: str
    refined_label: str
    label_confidence: float
    staticness_level: StaticnessLevel
    center_3d: list[float]
    size_3d: list[float]
    height_from_floor: float
    support_type: str
    wall_attachment: bool
    saliency_score: float
    observation_quality: float


class ObjectRelation(BaseModel):
    subject_id: str
    predicate: str
    object_id: str
    confidence: float


class SceneGraph(BaseModel):
    nodes: list[ObjectNode] = Field(default_factory=list)
    relations: list[ObjectRelation] = Field(default_factory=list)
