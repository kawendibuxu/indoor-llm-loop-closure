import numpy as np

from indoor_loop.models.oneformer import (
    DemoOneFormerRunner,
    build_segments_from_panoptic,
    run_oneformer_with_fallback,
)


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


def test_run_oneformer_with_fallback_uses_demo_runner_when_requested() -> None:
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)

    output = run_oneformer_with_fallback(
        rgb,
        model_name="shi-labs/oneformer_ade20k_swin_large",
        use_demo=True,
    )

    assert output.panoptic_map.shape == (4, 4)
    assert [segment.class_name for segment in output.segments] == ["bed", "desk"]


def test_demo_runner_is_available_without_model_weights() -> None:
    rgb = np.zeros((6, 8, 3), dtype=np.uint8)
    output = DemoOneFormerRunner().predict(rgb)

    assert output.panoptic_map.shape == (6, 8)
    assert len(output.segments) == 2
