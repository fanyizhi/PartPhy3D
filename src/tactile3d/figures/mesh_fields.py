"""Shared shaded mesh panels with pixel-level depth and smooth normals."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from tactile3d.figures.bound_material import (
    ELEVATION_DEGREES,
    MARGIN_RATIO,
    raster_face_ids,
)


def render_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray,
    azimuth: float = -55,
    resolution: int = 400,
) -> np.ndarray:
    """Render face RGB values using one fixed orthographic light and camera."""
    if colors.shape != (len(faces), 3) or not np.isfinite(colors).all():
        raise ValueError("Finite RGB colors must match mesh faces")
    geometry = SimpleNamespace(vertices_m=vertices, faces=faces)
    ids = raster_face_ids(geometry, azimuth, resolution)
    triangles = vertices[faces]
    vectors = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    normals = np.zeros_like(vertices, dtype=float)
    np.add.at(normals, faces.ravel(), np.repeat(vectors, 3, axis=0))
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    az, el = np.deg2rad([azimuth, ELEVATION_DEGREES])
    camera = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    right = np.array([-np.sin(az), np.cos(az), 0.0])
    up = np.cross(camera, right)
    centered = vertices - (vertices.min(0) + vertices.max(0)) / 2
    plane = centered @ np.column_stack((right, up))
    span = np.ptp(plane, axis=0).max()
    plane *= resolution * (1 - 2 * MARGIN_RATIO) / span
    plane[:, 1] *= -1
    plane += resolution / 2
    y, x = np.nonzero(ids >= 0)
    tri = faces[ids[y, x]]
    a, b, c = plane[tri].transpose(1, 0, 2)
    u, v = b - a, c - a
    p = np.column_stack((x, y)) + 0.5 - a
    det = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    wb = (p[:, 0] * v[:, 1] - p[:, 1] * v[:, 0]) / det
    wc = (u[:, 0] * p[:, 1] - u[:, 1] * p[:, 0]) / det
    n = np.einsum("ij,ijk->ik", np.column_stack((1 - wb - wc, wb, wc)), normals[tri])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    key = 0.7 * camera - 0.65 * right + 0.95 * up
    key /= np.linalg.norm(key)
    half_vector = key + camera
    half_vector /= np.linalg.norm(half_vector)
    shade = (
        0.48
        + 0.47 * np.maximum(n @ key, 0)
        + 0.05 * np.maximum(n @ half_vector, 0) ** 24
    )
    result = np.ones((resolution, resolution, 3), dtype=float)
    result[y, x] = np.clip(colors[ids[y, x]] * shade[:, None], 0, 1)
    return np.rint(result * 255).astype(np.uint8)
