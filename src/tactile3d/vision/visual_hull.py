"""Conservative foreground carving in the calibrated ABO camera frame."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation
from scipy.optimize import linprog


def build_visual_hull(poses, camera, masks, resolution: int = 144):
    """Intersect silhouette cones on an isotropic grid, then extract their boundary."""
    if resolution < 32 or len(poses) != len(masks) or len(masks) < 3:
        raise ValueError("Visual hull requires at least three calibrated masks")
    rows, bounds = [], []
    fx, fy, cx, cy = (camera[key] for key in ("fl_x", "fl_y", "cx", "cy"))
    for pose, mask in zip(poses, masks, strict=True):
        if mask.shape != (camera["h"], camera["w"]):
            raise ValueError("Foreground dimensions differ from camera calibration")
        y, x = np.where(mask)
        if not len(x):
            raise ValueError("Empty foreground")
        a, b, forward = pose[:3, :3].T
        forward = -forward
        origin = pose[:3, 3]
        # Convert the four projected bounding-box limits and positive depth to halfspaces.
        u, v = fx * a, -fy * b
        for edge in (
            u - (x.max() + 2 - cx) * forward,
            -u + (x.min() - 2 - cx) * forward,
            v - (y.max() + 2 - cy) * forward,
            -v + (y.min() - 2 - cy) * forward,
            -forward,
        ):
            rows.append(edge)
            bounds.append(edge @ origin)
    corners = []
    for direction in (*np.eye(3), *-np.eye(3)):
        result = linprog(direction, A_ub=rows, b_ub=bounds, bounds=(None, None))
        if not result.success:
            raise ValueError("Silhouette frusta have no bounded intersection")
        corners.append(result.fun)
    lower, upper = np.array(corners[:3]), -np.array(corners[3:])
    pitch = float((upper - lower).max()) / resolution
    if not np.isfinite(pitch) or pitch <= 0:
        raise ValueError("Degenerate visual-hull extent")
    shape = tuple(np.ceil((upper - lower) / pitch).astype(int) + 1)
    indices = np.indices(shape, dtype=np.int32).reshape(3, -1).T
    active = np.arange(len(indices))
    for pose, mask in zip(poses, masks, strict=True):
        xyz = lower + pitch * indices[active]
        local = (xyz - pose[:3, 3]) @ pose[:3, :3]
        front = local[:, 2] < -1e-8
        depth = np.where(front, -local[:, 2], 1)
        u = np.rint(cx + fx * local[:, 0] / depth).astype(int)
        v = np.rint(cy - fy * local[:, 1] / depth).astype(int)
        good = front & (u >= 0) & (u < mask.shape[1]) & (v >= 0) & (v < mask.shape[0])
        allowed = binary_dilation(mask, iterations=1)
        good[good] = allowed[v[good], u[good]]
        active = active[good]
        if not len(active):
            raise ValueError("Silhouette carving removed all voxels")
    occupied = np.zeros(shape, dtype=bool)
    occupied.flat[active] = True

    from skimage.measure import marching_cubes
    import trimesh

    padded = np.pad(occupied, 1)
    vertices, faces, _, _ = marching_cubes(padded, level=0.5, allow_degenerate=False)
    mesh = trimesh.Trimesh(
        vertices=lower + pitch * (vertices - 1), faces=faces[:, ::-1], process=False
    )
    return mesh, pitch
