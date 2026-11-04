from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import trimesh

from tactile3d.mesh.io import load_triangle_mesh
from tactile3d.mesh.repair import repair_mesh
from tactile3d.utils.io import write_json


def analyze_mesh_volume(
    mesh_path: str | Path,
    mesh_unit_to_meter: float,
    *,
    repair_strategy: str = "fill_only",
    convex_hull_proxy: bool = False,
    density_kg_m3: float | None = None,
    fill_fraction: float = 1.0,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate and integrate the enclosed volume of a triangular mesh."""

    if not math.isfinite(mesh_unit_to_meter) or mesh_unit_to_meter <= 0:
        raise ValueError("mesh_unit_to_meter must be positive")
    if density_kg_m3 is not None and (
        not math.isfinite(density_kg_m3) or density_kg_m3 <= 0
    ):
        raise ValueError("density_kg_m3 must be positive")
    if not 0 <= fill_fraction <= 1:
        raise ValueError("fill_fraction must lie in [0, 1]")

    path = Path(mesh_path).expanduser().resolve()
    mesh = _load_mesh(path)
    initial = _diagnostics(mesh)
    repaired, repair_report = repair_mesh(mesh, strategy=repair_strategy)
    final = _diagnostics(repaired)

    closed = bool(
        repaired.is_watertight
        and repaired.is_volume
        and abs(float(repaired.volume)) > 0
    )
    envelope_proxy = bool(repair_report.get("approximate_envelope"))
    hull_proxy = bool(repair_report.get("used_convex_hull"))
    exact = closed and not (envelope_proxy or hull_proxy)
    method = "enclosed_signed_tetrahedra" if closed else "unavailable"
    volume_units3: float | None = abs(float(repaired.volume)) if closed else None
    warning = None
    if closed and (envelope_proxy or hull_proxy):
        method = "voxel_envelope_proxy" if envelope_proxy else "convex_hull_proxy"
        warning = "Approximate repaired geometry; this is not an exact input volume."
    if not closed and convex_hull_proxy:
        volume_units3 = abs(float(repaired.convex_hull.volume))
        method = "convex_hull_proxy"
        warning = "Convex-hull volume fills all concavities and is not an exact object volume."
    elif not closed:
        warning = "Mesh is not a valid closed volume; no numeric volume is reported."

    volume_m3 = (
        volume_units3 * mesh_unit_to_meter**3 if volume_units3 is not None else None
    )
    occupied_m3 = volume_m3 * fill_fraction if volume_m3 is not None else None
    result: dict[str, Any] = {
        "schema_version": 1,
        "mesh": str(path),
        "mesh_unit_to_meter": float(mesh_unit_to_meter),
        "initial": initial,
        "repair": repair_report,
        "final": final,
        "volume": {
            "method": method,
            "is_exact_enclosed_volume": exact,
            "is_valid_closed_mesh": closed,
            "mesh_units3": volume_units3,
            "m3": volume_m3,
            "liters": volume_m3 * 1000.0 if volume_m3 is not None else None,
            "cm3": volume_m3 * 1_000_000.0 if volume_m3 is not None else None,
        },
        "occupancy": {
            "fill_fraction": float(fill_fraction),
            "occupied_volume_m3": occupied_m3,
            "semantics": "uniform effective occupied fraction, not inferred hidden geometry",
        },
        "warning": warning,
    }
    if density_kg_m3 is not None:
        result["uniform_material_mass"] = {
            "density_kg_m3": float(density_kg_m3),
            "mass_kg": density_kg_m3 * occupied_m3 if occupied_m3 is not None else None,
        }
    if output_path is not None:
        write_json(output_path, result)
    return result


def _diagnostics(mesh: trimesh.Trimesh) -> dict[str, Any]:
    return {
        "vertices": len(mesh.vertices),
        "faces": len(mesh.faces),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "is_volume": bool(mesh.is_volume),
        "surface_area_mesh_units2": float(mesh.area),
    }


def _load_mesh(path: Path) -> trimesh.Trimesh:
    return load_triangle_mesh(path)
