from pathlib import Path

from indoor_loop.io.scannet import normalize_scannet_exports


def test_normalize_scannet_exports_creates_triplet_layout(tmp_path: Path) -> None:
    source_root = tmp_path / "scene0000_00"
    color_dir = source_root / "color"
    depth_dir = source_root / "depth"
    intrinsic_dir = source_root / "intrinsic"
    color_dir.mkdir(parents=True)
    depth_dir.mkdir(parents=True)
    intrinsic_dir.mkdir(parents=True)

    (color_dir / "0.jpg").write_bytes(b"jpg")
    (depth_dir / "0.png").write_bytes(b"png")
    (intrinsic_dir / "intrinsic_depth.txt").write_text(
        "1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n",
        encoding="utf-8",
    )

    output_root = tmp_path / "prepared"
    count = normalize_scannet_exports(source_root, output_root)

    assert count == 1
    assert (output_root / "color" / "frame-000000.color.jpg").exists()
    assert (output_root / "depth" / "frame-000000.depth.png").exists()
    assert (output_root / "intrinsics" / "frame-000000.intrinsics.txt").exists()
