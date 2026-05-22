from pathlib import Path


def collect_tum_triplets(
    associate_path: Path,
    dataset_root: Path,
    intrinsics_path: Path,
) -> list[tuple[Path, Path, Path]]:
    triplets: list[tuple[Path, Path, Path]] = []
    for line in associate_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rgb_timestamp, rgb_rel, depth_timestamp, depth_rel = line.split()
        del rgb_timestamp, depth_timestamp
        rgb_path = dataset_root / rgb_rel
        depth_path = dataset_root / depth_rel
        if rgb_path.exists() and depth_path.exists():
            triplets.append((rgb_path, depth_path, intrinsics_path))
    return triplets


def write_shared_intrinsics(output_path: Path, fx: float, fy: float, cx: float, cy: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        f"{fx} 0.0 {cx}\n0.0 {fy} {cy}\n0.0 0.0 1.0\n",
        encoding="utf-8",
    )
