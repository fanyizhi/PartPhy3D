"""Fit a camera-space mesh similarity from calibrated foreground extents."""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares


def projected_bounds(
    vertices: np.ndarray, pose: np.ndarray, camera: dict
) -> np.ndarray:
    """Return pixel-space [xmin, ymin, xmax, ymax] for the visible half-space."""
    local = (vertices - pose[:3, 3]) @ pose[:3, :3]
    if np.any(local[:, 2] >= -1e-8):
        raise ValueError("Mesh projects through or behind the camera")
    u = camera["cx"] - camera["fl_x"] * local[:, 0] / local[:, 2]
    v = camera["cy"] + camera["fl_y"] * local[:, 1] / local[:, 2]
    return np.array([u.min(), v.min(), u.max(), v.max()])


def fit_silhouette_extents(
    vertices: np.ndarray,
    poses: np.ndarray,
    camera: dict,
    bounds: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Fit isotropic scale and translation from the designated input views only."""
    vertices = np.asarray(vertices, dtype=float)
    bounds = np.asarray(bounds, dtype=float)
    poses = np.asarray(poses, dtype=float)
    if not np.isfinite(vertices).all() or vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("Expected finite Nx3 mesh vertices")
    if (
        len(poses) < 3
        or poses.shape != (len(poses), 4, 4)
        or bounds.shape != (len(poses), 4)
    ):
        raise ValueError("Expected at least three aligned poses and foreground bounds")
    span = np.ptp(vertices, axis=0)
    length = float(span.max())
    center = vertices.min(axis=0) + span / 2
    extents = bounds[:, 2:] - bounds[:, :2]
    if length <= 0 or not np.isfinite(bounds).all() or np.any(extents <= 0):
        raise ValueError("Foreground and mesh extents must be finite and nondegenerate")
    normalizer = np.tile(extents, (1, 2))

    def transform(parameters: np.ndarray) -> np.ndarray:
        return (
            (vertices - center) * np.exp(parameters[0])
            + center
            + length * parameters[1:]
        )

    def residual(parameters: np.ndarray) -> np.ndarray:
        moved = transform(parameters)
        projected = np.array([projected_bounds(moved, pose, camera) for pose in poses])
        return ((projected - bounds) / normalizer).ravel()

    initial = np.zeros(4)
    lower = np.array([-np.log(2), -0.5, -0.5, -0.5])
    result = least_squares(
        residual, initial, bounds=(lower, -lower), loss="soft_l1", f_scale=0.05
    )
    if not result.success:
        raise RuntimeError(result.message)
    return transform(result.x), {
        "isotropic_scale": float(np.exp(result.x[0])),
        "translation_camera_units": (length * result.x[1:]).tolist(),
        "scale_center_camera_units": center.tolist(),
        "extent_rmse_before": float(np.sqrt(np.mean(residual(initial) ** 2))),
        "extent_rmse_after": float(np.sqrt(np.mean(result.fun**2))),
        "solver_active_bounds": result.active_mask.tolist(),
    }
