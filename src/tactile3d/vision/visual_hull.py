"""Conservative foreground carving in the calibrated ABO camera frame."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, map_coordinates
from scipy.optimize import linprog


def build_visual_hull(
    poses,
    camera,
    masks,
    resolution: int = 144,
    *,
    boundary_offset_px: float | None = None,
):
    """Extract a hull from binary cones or bilinearly sampled signed contours.

    The optional offset is measured in source-image pixels. Zero preserves the
    interpolated mask boundary. Omitting it retains legacy binary carving.
    """
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
    active = np.arange(np.prod(shape), dtype=np.int64)
    field = (
        None
        if boundary_offset_px is None
        else np.full(len(active), np.inf, dtype=np.float32)
    )
    if field is not None and not (
        np.isfinite(boundary_offset_px) and abs(boundary_offset_px) <= 1.5
    ):
        raise ValueError("Boundary offset must be within 1.5 input pixels")
    for pose, mask in zip(poses, masks, strict=True):
        if field is None:
            allowed = binary_dilation(mask, iterations=1)
        else:
            # Pixel centers sit half a pixel away from an axis-aligned mask edge.
            signed = np.where(
                mask,
                distance_transform_edt(mask) - 0.5,
                0.5 - distance_transform_edt(~mask),
            )
        survivors = []
        for start in range(0, len(active), 500_000):
            subset = active[start : start + 500_000]
            indices = np.column_stack(np.unravel_index(subset, shape))
            xyz = lower + pitch * indices
            local = (xyz - pose[:3, 3]) @ pose[:3, :3]
            front = local[:, 2] < -1e-8
            depth = np.where(front, -local[:, 2], 1)
            u = cx + fx * local[:, 0] / depth
            v = cy - fy * local[:, 1] / depth
            if field is None:
                ix, iy = np.rint(u).astype(int), np.rint(v).astype(int)
                good = (
                    front
                    & (ix >= 0)
                    & (ix < mask.shape[1])
                    & (iy >= 0)
                    & (iy < mask.shape[0])
                )
                good[good] = allowed[iy[good], ix[good]]
            else:
                distance = map_coordinates(
                    signed, [v, u], order=1, mode="constant", cval=-1e6
                )
                distance += boundary_offset_px
                field[subset] = np.minimum(field[subset], distance)
                # Keep one projected voxel of negative space for isosurface interpolation.
                margin = 1 + np.sqrt(3) * max(fx, fy) * pitch / depth
                good = front & (distance > -margin)
            survivors.append(subset[good])
        active = np.concatenate(survivors)
        if not len(active):
            raise ValueError("Silhouette carving removed all voxels")

    from skimage.measure import marching_cubes
    import trimesh

    if field is None:
        occupied = np.zeros(shape, dtype=bool)
        occupied.flat[active] = True
        padded = np.pad(occupied, 1)
        level = 0.5
    else:
        occupied = np.full(shape, -2.0, dtype=np.float32)
        occupied.flat[active] = field[active]
        if not (occupied > 0).any():
            raise ValueError(
                "Subpixel silhouette intersection has no positive interior"
            )
        padded = np.pad(occupied, 1, constant_values=-2)
        level = 0.0
    vertices, faces, _, _ = marching_cubes(padded, level=level, allow_degenerate=False)
    mesh = trimesh.Trimesh(
        vertices=lower + pitch * (vertices - 1), faces=faces[:, ::-1], process=False
    )
    return mesh, pitch
