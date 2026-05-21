from pathlib import Path

import numpy as np
from PIL import Image

from indoor_loop.io.scannet import load_scannet_frame


def test_load_scannet_frame_reads_rgb_depth_and_intrinsics(tmp_path: Path) -> None:
    rgb_path = tmp_path / "rgb.jpg"
    depth_path = tmp_path / "depth.png"
    intrinsics_path = tmp_path / "intrinsics.txt"

    Image.fromarray(np.full((4, 5, 3), 64, dtype=np.uint8)).save(rgb_path)
    Image.fromarray(np.full((4, 5), 1200, dtype=np.uint16)).save(depth_path)
    intrinsics_path.write_text("577.0 0.0 320.0\n0.0 577.0 240.0\n0.0 0.0 1.0\n", encoding="utf-8")

    frame = load_scannet_frame(rgb_path, depth_path, intrinsics_path)

    assert frame.rgb.shape == (4, 5, 3)
    assert frame.depth.shape == (4, 5)
    assert frame.depth.dtype == np.float32
    assert np.isclose(frame.depth[0, 0], 1.2)
    assert frame.intrinsics.shape == (3, 3)
