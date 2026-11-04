from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass

import numpy as np
import trimesh


@dataclass(frozen=True)
class LODConstraints:
    """Maximum errors allowed when selecting a reduced mesh."""

    volume_relative: float = 0.04
    center_of_mass_relative: float = 0.0075
    inertia_relative: float = 0.05
    surface_area_relative: float = 0.01


def select_physics_constrained_lod(
    mesh: trimesh.Trimesh,
    target_faces: Iterable[int] = (25000, 12500, 5000, 2500),
    *,
    constraints: LODConstraints | None = None,
    aggression: int = 7,
) -> tuple[trimesh.Trimesh, dict]:
    """Select the coarsest QEM candidate that preserves physical proxies.

    Signed-tetrahedra mass properties are exact only for a valid water-tight
    mesh. For an open input they are retained as cross-resolution proxies and
    the report explicitly marks that limitation.
    """

    constraints = constraints or LODConstraints()
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        raise ValueError("A non-empty triangular mesh is required")

    reference = _properties(mesh)
    candidates: list[tuple[trimesh.Trimesh, dict]] = []
    for target in sorted({int(value) for value in target_faces}, reverse=True):
        if target < 4 or target >= len(mesh.faces):
            continue
        reduced = mesh.simplify_quadric_decimation(
            face_count=target,
            aggression=int(np.clip(aggression, 0, 10)),
        )
        row = _candidate_report(reduced, reference, constraints)
        row["requested_faces"] = target
        candidates.append((reduced, row))

    passing = [item for item in candidates if item[1]["passes_constraints"]]
    if passing:
        selected_mesh, selected = min(passing, key=lambda item: len(item[0].faces))
    else:
        selected_mesh = mesh.copy()
        selected = _candidate_report(selected_mesh, reference, constraints)
        selected["requested_faces"] = len(mesh.faces)
        selected["fallback_reason"] = "No simplified candidate met all physical constraints."

    report = {
        "method": "physics_constrained_qem_candidate_selection",
        "constraints": asdict(constraints),
        "reference": reference,
        "input_is_watertight": bool(mesh.is_watertight),
        "mass_property_semantics": (
            "exact enclosed mass properties"
            if mesh.is_watertight and mesh.is_volume
            else "signed-tetrahedra cross-resolution proxy; not exact enclosed volume"
        ),
        "candidates": [row for _, row in candidates],
        "selected": selected,
        "face_reduction_fraction": 1.0 - len(selected_mesh.faces) / len(mesh.faces),
    }
    return selected_mesh, report


def _properties(mesh: trimesh.Trimesh) -> dict[str, float | int | list[float]]:
    return {
        "faces": len(mesh.faces),
        "vertices": len(mesh.vertices),
        "absolute_signed_volume": abs(float(mesh.volume)),
        "surface_area": float(mesh.area),
        "center_of_mass": np.asarray(mesh.center_mass, dtype=float).tolist(),
        "inertia_frobenius": float(np.linalg.norm(mesh.moment_inertia)),
        "bounding_box_diagonal": float(np.linalg.norm(mesh.extents)),
    }


def _candidate_report(
    mesh: trimesh.Trimesh,
    reference: dict[str, float | int | list[float]],
    constraints: LODConstraints,
) -> dict:
    current = _properties(mesh)
    volume_error = _relative_error(
        float(current["absolute_signed_volume"]),
        float(reference["absolute_signed_volume"]),
    )
    area_error = _relative_error(
        float(current["surface_area"]),
        float(reference["surface_area"]),
    )
    inertia_error = _relative_error(
        float(current["inertia_frobenius"]),
        float(reference["inertia_frobenius"]),
    )
    center_error = float(
        np.linalg.norm(
            np.asarray(current["center_of_mass"], dtype=float)
            - np.asarray(reference["center_of_mass"], dtype=float)
        )
        / max(float(reference["bounding_box_diagonal"]), 1e-12)
    )
    errors = {
        "volume_relative": volume_error,
        "center_of_mass_relative": center_error,
        "inertia_relative": inertia_error,
        "surface_area_relative": area_error,
    }
    limits = asdict(constraints)
    return {
        **current,
        "errors": errors,
        "passes_constraints": all(errors[name] <= limits[name] for name in errors),
    }


def _relative_error(value: float, reference: float) -> float:
    return abs(value - reference) / max(abs(reference), 1e-12)
