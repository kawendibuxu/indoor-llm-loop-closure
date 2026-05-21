import numpy as np

from indoor_loop.graph.projection import project_mask_to_3d_bbox


def test_project_mask_to_3d_bbox_returns_center_and_size() -> None:
    depth = np.ones((4, 4), dtype=np.float32) * 2.0
    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:3] = True
    intrinsics = np.array(
        [
            [2.0, 0.0, 1.5],
            [0.0, 2.0, 1.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    center, size = project_mask_to_3d_bbox(mask, depth, intrinsics)

    assert len(center) == 3
    assert len(size) == 3
    assert center[2] == 2.0
    assert size[2] == 0.0
