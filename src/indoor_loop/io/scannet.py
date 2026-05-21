from dataclasses import dataclass
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(slots=True)
class FrameData:
    rgb: np.ndarray
    depth: np.ndarray
    intrinsics: np.ndarray


def load_scannet_frame(rgb_path: Path, depth_path: Path, intrinsics_path: Path) -> FrameData:
    rgb = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
    depth_raw = np.asarray(Image.open(depth_path), dtype=np.uint16)
    depth = depth_raw.astype(np.float32) / 1000.0
    intrinsics = np.loadtxt(intrinsics_path, dtype=np.float32)
    return FrameData(rgb=rgb, depth=depth, intrinsics=intrinsics)


def collect_sequence_triplets(
    color_dir: Path,
    depth_dir: Path,
    intrinsics_dir: Path,
) -> list[tuple[Path, Path, Path]]:
    triplets: list[tuple[Path, Path, Path]] = []
    for rgb_path in sorted(color_dir.glob("*.jpg")):
        stem = rgb_path.name.replace(".color.jpg", "")
        depth_path = depth_dir / f"{stem}.depth.png"
        intrinsics_path = intrinsics_dir / f"{stem}.intrinsics.txt"
        if depth_path.exists() and intrinsics_path.exists():
            triplets.append((rgb_path, depth_path, intrinsics_path))
    return triplets


def normalize_scannet_exports(source_root: Path, output_root: Path) -> int:
    color_src = source_root / "color"
    depth_src = source_root / "depth"
    intrinsic_src = source_root / "intrinsic" / "intrinsic_depth.txt"

    color_out = output_root / "color"
    depth_out = output_root / "depth"
    intrinsics_out = output_root / "intrinsics"
    color_out.mkdir(parents=True, exist_ok=True)
    depth_out.mkdir(parents=True, exist_ok=True)
    intrinsics_out.mkdir(parents=True, exist_ok=True)

    count = 0
    for color_path in sorted(color_src.glob("*.jpg")):
        stem = color_path.stem
        depth_path = depth_src / f"{stem}.png"
        if not depth_path.exists():
            continue
        frame_name = f"frame-{int(stem):06d}"
        shutil.copy2(color_path, color_out / f"{frame_name}.color.jpg")
        shutil.copy2(depth_path, depth_out / f"{frame_name}.depth.png")
        shutil.copy2(intrinsic_src, intrinsics_out / f"{frame_name}.intrinsics.txt")
        count += 1
    return count
