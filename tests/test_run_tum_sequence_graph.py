from pathlib import Path

from indoor_loop.io.tum import collect_tum_triplets


def test_collect_tum_triplets_supports_association_layout(tmp_path: Path) -> None:
    rgb_dir = tmp_path / "rgb"
    depth_dir = tmp_path / "depth"
    rgb_dir.mkdir()
    depth_dir.mkdir()
    (rgb_dir / "1305031102.175304.png").write_bytes(b"rgb")
    (depth_dir / "1305031102.160407.png").write_bytes(b"depth")
    assoc = tmp_path / "associate.txt"
    assoc.write_text(
        "1305031102.175304 rgb/1305031102.175304.png 1305031102.160407 depth/1305031102.160407.png\n",
        encoding="utf-8",
    )

    triplets = collect_tum_triplets(assoc, tmp_path, intrinsics_path=tmp_path / "intrinsics.txt")

    assert len(triplets) == 1
    assert triplets[0][0].name == "1305031102.175304.png"
