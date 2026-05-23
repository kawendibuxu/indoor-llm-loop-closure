from pathlib import Path

import numpy as np
from PIL import Image

from indoor_loop.types import ObjectNode, ObjectRelation, SceneGraph, SegmentRecord
from indoor_loop.viz.export import write_overlay_png, write_scene_graph_json, write_text_artifact


def test_write_scene_graph_json_creates_file(tmp_path: Path) -> None:
    graph = SceneGraph(nodes=[], relations=[])
    output_path = tmp_path / "scene_graph.json"

    write_scene_graph_json(graph, output_path)

    assert output_path.exists()
    assert output_path.read_text(encoding="utf-8").strip().startswith("{")


def test_write_overlay_png_creates_image(tmp_path: Path) -> None:
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    output_path = tmp_path / "overlay.png"

    write_overlay_png(rgb, [], output_path)

    assert output_path.exists()


def test_write_overlay_png_draws_annotations() -> None:
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    segments = [
        SegmentRecord(
            segment_id="1",
            class_name="bed",
            is_thing=True,
            score=0.9,
            area=64,
            bbox_xyxy=[2, 2, 12, 12],
            centroid_xy=[7.0, 7.0],
        ),
        SegmentRecord(
            segment_id="2",
            class_name="desk",
            is_thing=True,
            score=0.8,
            area=64,
            bbox_xyxy=[18, 2, 28, 12],
            centroid_xy=[23.0, 7.0],
        ),
    ]
    graph = SceneGraph(
        nodes=[
            ObjectNode(
                object_id="bed-1",
                coarse_anchor_class="bed",
                refined_label="bed",
                label_confidence=0.9,
                staticness_level="hard_static",
                center_3d=[0.0, 0.0, 2.0],
                size_3d=[1.0, 1.0, 1.0],
                height_from_floor=0.0,
                support_type="floor",
                wall_attachment=True,
                saliency_score=0.9,
                observation_quality=0.9,
            ),
            ObjectNode(
                object_id="desk-2",
                coarse_anchor_class="table_like",
                refined_label="desk",
                label_confidence=0.8,
                staticness_level="soft_static",
                center_3d=[1.0, 0.0, 2.0],
                size_3d=[1.0, 1.0, 1.0],
                height_from_floor=0.0,
                support_type="floor",
                wall_attachment=False,
                saliency_score=0.8,
                observation_quality=0.8,
            ),
        ],
        relations=[
            ObjectRelation(
                subject_id="bed-1",
                predicate="left_of",
                object_id="desk-2",
                confidence=0.8,
            )
        ],
    )
    output_path = Path("outputs/test_overlay_annotations.png")

    write_overlay_png(rgb, segments, output_path, graph=graph)

    image = np.asarray(Image.open(output_path))
    assert image.sum() > 0


def test_write_text_artifact_creates_file(tmp_path: Path) -> None:
    output_path = tmp_path / "oneformer_mode.txt"

    write_text_artifact("real\n", output_path)

    assert output_path.exists()
    assert output_path.read_text(encoding="utf-8") == "real\n"
