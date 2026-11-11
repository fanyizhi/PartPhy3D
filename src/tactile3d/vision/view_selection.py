"""Select actual calibrated views for broad azimuth coverage."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def select_azimuth_views(poses: np.ndarray, count: int) -> list[dict]:
    """Match distinct cameras to equally spaced azimuths near 30° elevation."""
    poses = np.asarray(poses, dtype=float)
    if (
        poses.ndim != 3
        or poses.shape[1:] != (4, 4)
        or len(poses) < count
        or count not in (3, 4)
        or not np.isfinite(poses).all()
    ):
        raise ValueError("At least three/four finite camera-to-world matrices required")
    positions = poses[:, :3, 3]
    directions = -poses[:, :3, 2]
    lengths = np.linalg.norm(directions, axis=1)
    if np.any(lengths < 1e-12):
        raise ValueError("Camera optical directions cannot be zero")
    directions /= lengths[:, None]
    projectors = np.eye(3)[None] - np.einsum("ni,nj->nij", directions, directions)
    center = np.linalg.lstsq(
        projectors.sum(0), np.einsum("nij,nj->i", projectors, positions), rcond=None
    )[0]
    relative = positions - center
    azimuth = np.rad2deg(np.arctan2(relative[:, 1], relative[:, 0])) % 360
    elevation = np.rad2deg(
        np.arctan2(relative[:, 2], np.linalg.norm(relative[:, :2], axis=1))
    )
    targets = np.arange(count) * 360 / count
    angular = np.abs((azimuth[None] - targets[:, None] + 180) % 360 - 180)
    cost = angular + 0.25 * np.abs(elevation[None] - 30)
    rows, columns = linear_sum_assignment(cost)
    return [
        {
            "frame_index": int(j),
            "target_azimuth_degrees": float(targets[i]),
            "actual_azimuth_degrees": float(azimuth[j]),
            "elevation_degrees": float(elevation[j]),
            "estimated_lookat_center": center.tolist(),
        }
        for i, j in zip(rows, columns)
    ]
