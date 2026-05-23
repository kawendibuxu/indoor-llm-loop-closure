from pathlib import Path


def collect_tum_triplets(
    rgb_txt_path: Path,
    depth_txt_path: Path,
    dataset_root: Path,
    intrinsics_path: Path,
) -> list[tuple[Path, Path, Path]]:
    rgb_entries = _read_tum_index(rgb_txt_path)
    depth_entries = _read_tum_index(depth_txt_path)

    triplets: list[tuple[Path, Path, Path]] = []
    for rgb_timestamp, rgb_rel in rgb_entries:
        depth_rel = _nearest_depth_match(rgb_timestamp, depth_entries)
        if depth_rel is None:
            continue
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


def _read_tum_index(index_path: Path) -> list[tuple[float, str]]:
    entries: list[tuple[float, str]] = []
    for line in index_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        timestamp_str, relative_path = line.split(maxsplit=1)
        entries.append((float(timestamp_str), relative_path))
    return entries


def _nearest_depth_match(rgb_timestamp: float, depth_entries: list[tuple[float, str]]) -> str | None:
    if not depth_entries:
        return None
    best_timestamp, best_rel = min(depth_entries, key=lambda item: abs(item[0] - rgb_timestamp))
    if abs(best_timestamp - rgb_timestamp) > 0.05:
        return None
    return best_rel
