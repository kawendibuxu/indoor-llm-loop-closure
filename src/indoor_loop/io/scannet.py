from dataclasses import dataclass
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
