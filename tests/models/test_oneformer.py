import numpy as np

from indoor_loop.models.oneformer import build_segments_from_panoptic


def test_build_segments_from_panoptic_extracts_bbox_area_and_centroid() -> None:
    panoptic_map = np.array(
        [
            [0, 1, 1],
            [0, 1, 1],
            [2, 2, 2],
        ],
        dtype=np.int32,
    )
    segments_info = [
        {"id": 1, "label_id": 7, "label_name": "bed", "score": 0.91, "was_fused": False},
        {"id": 2, "label_id": 12, "label_name": "floor", "score": 0.99, "was_fused": False},
    ]

    segments = build_segments_from_panoptic(panoptic_map, segments_info)

    assert [segment.segment_id for segment in segments] == ["1", "2"]
    assert segments[0].area == 4
    assert segments[0].bbox_xyxy == [1, 0, 2, 1]
    assert segments[1].class_name == "floor"
