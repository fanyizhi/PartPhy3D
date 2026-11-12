"""Observable assembly geometry; strength is deliberately an unmeasured state."""

from __future__ import annotations

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def texture_face_rgb(mesh: trimesh.Trimesh) -> np.ndarray:
    """TextureVisuals.to_color returns unbound vertex colors in some releases."""
    if mesh.visual.kind == "texture":
        vertex_rgba = np.asarray(mesh.visual.to_color().vertex_colors)
        return vertex_rgba[np.asarray(mesh.faces), :3].mean(axis=1) / 255.0
    return np.asarray(mesh.visual.face_colors)[:, :3] / 255.0


def face_components(mesh: trimesh.Trimesh) -> tuple[int, np.ndarray]:
    adjacent = mesh.face_adjacency
    graph = coo_matrix(
        (np.ones(len(adjacent)), (adjacent[:, 0], adjacent[:, 1])),
        shape=(len(mesh.faces), len(mesh.faces)),
    )
    return connected_components(graph, directed=False)


def evidence_groups(areas, colors, *, area_fraction=0.01):
    """Compress VLM evidence only; never union or hull the physical components.

    Large candidates stay separate. Small candidates share coarse RGB buckets;
    a group may still be mixed and must be checked by the visual model.
    """
    areas, colors = np.asarray(areas, float), np.asarray(colors, float)
    if (
        areas.ndim != 1
        or not len(areas)
        or colors.shape != (len(areas), 3)
        or not np.isfinite(areas).all()
        or not np.isfinite(colors).all()
        or np.any(areas <= 0)
        or np.any(colors < 0)
        or np.any(colors > 1)
        or not 0 < area_fraction < 1
    ):
        raise ValueError("Positive areas and finite RGB colors required")
    large = areas >= areas.sum() * area_fraction
    groups = [[int(i)] for i in np.flatnonzero(large)]
    buckets = {}
    for i in np.flatnonzero(~large):
        key = tuple(np.floor(colors[i] / 0.1 + 1e-9).astype(int))
        buckets.setdefault(key, []).append(int(i))
    groups.extend(buckets[key] for key in sorted(buckets))
    return groups


def interface_geometry(
    source: trimesh.Trimesh,
    target: trimesh.Trimesh,
    *,
    top_band_m: float = 0.05,
    near_threshold_m: float = 0.003,
    samples: int = 2048,
    seed: int = 17,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Unsigned point-to-triangle distances on an upper attachment band.

    This measures sampled visible surface proximity, not penetration, bond
    strength, fastener capacity or a joint safety factor.
    """
    if not len(source.faces) or not len(target.faces):
        raise ValueError("Both interface meshes must have faces")
    if top_band_m <= 0 or near_threshold_m <= 0 or samples < 32:
        raise ValueError("Positive lengths and at least 32 samples are required")
    clipped = source.slice_plane(
        plane_origin=[0, float(source.bounds[1, 1] - top_band_m), 0],
        plane_normal=[0, 1, 0],
        cap=False,
    )
    if not len(clipped.faces):
        raise ValueError("The attachment band contains no surface")
    rng = np.random.default_rng(seed)
    ids = rng.choice(
        len(clipped.faces), size=samples, p=clipped.area_faces / clipped.area
    )
    uv = rng.random((samples, 2))
    uv[uv.sum(axis=1) > 1] = 1 - uv[uv.sum(axis=1) > 1]
    triangles = clipped.triangles[ids]
    points = (
        triangles[:, 0]
        + uv[:, :1] * (triangles[:, 1] - triangles[:, 0])
        + uv[:, 1:] * (triangles[:, 2] - triangles[:, 0])
    )
    distances = []
    for start in range(0, samples, 32):
        _, d, _ = trimesh.proximity.closest_point_naive(
            target, points[start : start + 32]
        )
        distances.extend(d)
    distances = np.asarray(distances)
    result = {
        "method": "area-uniform samples on upper band to target triangles (unsigned)",
        "sample_count": samples,
        "top_band_m": top_band_m,
        "near_threshold_m": near_threshold_m,
        "sample_min_distance_mm": float(distances.min() * 1000),
        "sample_q05_distance_mm": float(np.quantile(distances, 0.05) * 1000),
        "sample_median_distance_mm": float(np.median(distances) * 1000),
        "near_sample_fraction": float(np.mean(distances <= near_threshold_m)),
        "geometric_observation": (
            "surface proximity detected at the configured threshold"
            if np.any(distances <= near_threshold_m)
            else "no near surface detected among the sampled points"
        ),
        "penetration": "not evaluated",
        "mechanical_quality": "unverified",
        "connection_type": "unknown: not encoded in the visible mesh",
        "strength_safety_factor": None,
        "required_evidence": [
            "joint construction",
            "fasteners or bond geometry",
            "material measurements",
            "load case and supports",
        ],
    }
    return result, points, distances


def crop_mesh(mesh: trimesh.Trimesh, center: np.ndarray, half_size: np.ndarray):
    result = mesh.copy()
    for axis in range(3):
        for sign in (1, -1):
            normal = np.zeros(3)
            normal[axis] = sign
            origin = np.array(center, dtype=float)
            origin[axis] -= sign * half_size[axis]
            result = result.slice_plane(origin, normal, cap=False)
    return result
