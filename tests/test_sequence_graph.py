from pathlib import Path

from indoor_loop.io.scannet import collect_sequence_triplets


def test_collect_sequence_triplets_matches_color_depth_intrinsics(tmp_path: Path) -> None:
    color_dir = tmp_path / "color"
    depth_dir = tmp_path / "depth"
    intrinsics_dir = tmp_path / "intrinsics"
    color_dir.mkdir()
    depth_dir.mkdir()
    intrinsics_dir.mkdir()

    (color_dir / "frame-000001.color.jpg").write_bytes(b"a")
    (depth_dir / "frame-000001.depth.png").write_bytes(b"b")
    (intrinsics_dir / "frame-000001.intrinsics.txt").write_text("1 0 0\n0 1 0\n0 0 1\n", encoding="utf-8")

    triplets = collect_sequence_triplets(color_dir, depth_dir, intrinsics_dir)

    assert len(triplets) == 1
    assert triplets[0][0].name == "frame-000001.color.jpg"
