"""Category-conditioned geometric role candidates for upholstered furniture.

These explicit development rules do not infer hidden construction. They use no
component IDs, role annotations or mass labels.
"""

from __future__ import annotations

import numpy as np


def furniture_role_candidates(
    bounds: np.ndarray, object_bounds: np.ndarray, material: str, watertight: bool
) -> dict:
    """Return roles consistent with observed extent, location and ground support.

    Coordinates are z-up, x across furniture width and y along seat length.
    Category and axis convention are explicit inputs to this furniture template.
    """
    bounds = np.asarray(bounds, dtype=float)
    whole = np.asarray(object_bounds, dtype=float)
    if (
        bounds.shape != (2, 3)
        or whole.shape != (2, 3)
        or not np.isfinite(bounds).all()
        or not np.isfinite(whole).all()
    ):
        raise ValueError("Finite component and object bounds are required")
    size = bounds[1] - bounds[0]
    extent = whole[1] - whole[0]
    if np.any(size < 0) or np.any(extent <= 0):
        raise ValueError("Positive object extents and ordered bounds required")
    center = bounds.mean(0)
    diagonal = np.linalg.norm(extent)
    tolerance = 1e-3 * diagonal
    on_ground = abs(bounds[0, 2] - whole[0, 2]) <= tolerance
    role, evidence = None, []
    if size[2] < 1e-3 * diagonal and size[:2].max() > 0.3 * extent[:2].max():
        role, evidence = "underside", ["near-planar horizontal sheet"]
    elif material in {"wood", "metal", "plastic", "geometry_only"}:
        if on_ground and size[2] > 1.5 * max(size[0], size[1]):
            role, evidence = "leg", [
                "reaches common ground plane",
                "elongated vertically",
            ]
        elif (
            not on_ground
            and size[:2].max() > 3 * max(size[:2].min(), tolerance)
            and size[2] < 0.5 * size[:2].max()
        ):
            role, evidence = "frame_rail", [
                "horizontal elongated support",
                "does not reach ground",
            ]
    if role is None and material in {"fabric", "geometry_only"}:
        lateral = abs(center[0] - whole.mean(0)[0]) / (extent[0] / 2)
        if lateral > 0.55 and size[0] < 0.3 * extent[0]:
            role, evidence = "armrest", [
                "lateral placement",
                "narrow transverse extent",
            ]
        elif (
            size[0] > 0.6 * extent[0]
            and size[1] > 0.5 * extent[1]
            and size[2] < 0.3 * size[1]
        ):
            role, evidence = "seat", ["broad horizontal support"]
        elif watertight and size[2] > 0.5 * size[1]:
            role, evidence = "cushion", [
                "separate closed upholstered region",
                "upright cushion proportions",
            ]
        elif size[2] > 0.7 * size[1] and size[0] > 0.5 * extent[0]:
            role, evidence = "backrest", ["wide upright panel"]
    return {
        "allowed_roles": [role] if role else ["unknown"],
        "geometry_resolved": role is not None,
        "evidence": evidence,
        "scope": "upholstered furniture template; no hidden structure conclusion",
    }
