"""Component-preserving QEM with explicit, capacity-limited face budgets.

The sensitivity is a ranking proxy, not a certified physical error bound.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import trimesh


def allocate_faces(
    capacities: np.ndarray, weights: np.ndarray, total: int
) -> np.ndarray:
    """Allocate an exact integer budget without exceeding input face counts."""
    caps = np.asarray(capacities)
    weights = np.asarray(weights, dtype=float)
    if (
        caps.ndim != 1
        or not len(caps)
        or caps.dtype.kind not in "iu"
        or np.any(caps <= 0)
        or weights.shape != caps.shape
        or not np.isfinite(weights).all()
        or np.any(weights <= 0)
        or type(total) is not int
    ):
        raise ValueError(
            "Positive integer capacities and finite positive weights required"
        )
    floor = np.minimum(caps, 4)
    if not int(floor.sum()) <= total <= int(caps.sum()):
        raise ValueError("Budget must cover component floors and not exceed capacity")
    result = floor.copy()
    while int(result.sum()) < total:
        available = caps - result
        active = available > 0
        quota = (total - int(result.sum())) * weights * active / weights[active].sum()
        add = np.minimum(available, np.floor(quota).astype(int))
        if not add.any():
            ids = np.flatnonzero(active)
            order = ids[np.argsort(-quota[ids], kind="stable")]
            add[order[: total - int(result.sum())]] = 1
        result += add
    return result


def component_sensitivity(
    meshes: list[trimesh.Trimesh], densities: np.ndarray
) -> np.ndarray:
    """Rank thin geometry and density-weighted boundary-moment sensitivity.

    Unknown bulk densities are NaN and contribute no physical term. They retain
    the geometry term. Surface textile density must not be used as bulk density.
    """
    density = np.asarray(densities, dtype=float)
    if (
        density.shape != (len(meshes),)
        or np.any(np.isinf(density))
        or np.any(density < 0)
    ):
        raise ValueError("Density vector must contain nonnegative values or NaN")
    bounds = np.array([m.bounds for m in meshes])
    low, high = bounds[:, 0].min(0), bounds[:, 1].max(0)
    diagonal = float(np.linalg.norm(high - low))
    if diagonal <= 0:
        raise ValueError("Nondegenerate geometry required")
    center = (low + high) / 2
    geometric, physical = [], []
    for mesh, rho in zip(meshes, density):
        extents = np.sort(mesh.extents)
        thin = min(diagonal / max(float(extents[0]), diagonal * 0.01), 30.0)
        geometric.append(np.sqrt(len(mesh.faces)) * (1 + thin / 10))
        radius2 = np.sum((mesh.triangles_center - center) ** 2, axis=1)
        physical.append(
            float(rho * np.sum(mesh.area_faces * (1 + radius2 / diagonal**2)))
            if np.isfinite(rho)
            else 0.0
        )
    physical = np.asarray(physical)
    scale = np.median(physical[physical > 0]) if np.any(physical > 0) else 1.0
    return np.asarray(geometric) * (1 + np.minimum(physical / scale, 5.0))


@dataclass
class ComponentLOD:
    mesh: trimesh.Trimesh
    face_components: np.ndarray
    budgets: list[int]
    seconds: float


def simplify_components(
    meshes: list[trimesh.Trimesh], total: int, densities: np.ndarray | None = None
) -> ComponentLOD:
    """Preserve component identity while using QEM inside each component."""
    start = perf_counter()
    capacities = np.array([len(m.faces) for m in meshes], dtype=int)
    weights = (
        capacities.astype(float)
        if densities is None
        else component_sensitivity(meshes, densities)
    )
    budgets = allocate_faces(capacities, weights, total)
    pieces, labels = [], []
    for index, (mesh, budget) in enumerate(zip(meshes, budgets)):
        reduced = (
            mesh.copy()
            if budget >= len(mesh.faces)
            else mesh.simplify_quadric_decimation(face_count=int(budget))
        )
        if not len(reduced.faces) or not np.isfinite(reduced.vertices).all():
            raise ValueError("QEM returned invalid component geometry")
        pieces.append(reduced)
        labels.extend([index] * len(reduced.faces))
    return ComponentLOD(
        trimesh.util.concatenate(pieces),
        np.asarray(labels, dtype=int),
        budgets.tolist(),
        perf_counter() - start,
    )
