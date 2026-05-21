import numpy as np


def project_mask_to_3d_bbox(
    mask: np.ndarray, depth: np.ndarray, intrinsics: np.ndarray
) -> tuple[list[float], list[float]]:
    ys, xs = np.where(mask & np.isfinite(depth) & (depth > 0))
    zs = depth[ys, xs]
    if zs.size == 0:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]

    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    xs_3d = (xs.astype(np.float32) - cx) * zs / fx
    ys_3d = (ys.astype(np.float32) - cy) * zs / fy

    points = np.stack([xs_3d, ys_3d, zs], axis=1)
    center = np.median(points, axis=0).tolist()
    size = (points.max(axis=0) - points.min(axis=0)).tolist()
    return [float(v) for v in center], [float(v) for v in size]
