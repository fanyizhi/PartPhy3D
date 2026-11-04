from __future__ import annotations

import trimesh


def repair_mesh(mesh: trimesh.Trimesh, strategy: str = "fill_then_hull") -> tuple[trimesh.Trimesh, dict]:
    repaired = mesh.copy()
    report = {
        "initial_watertight": bool(repaired.is_watertight),
        "strategy": strategy,
        "used_convex_hull": False,
    }

    if strategy == "none":
        report["final_watertight"] = bool(repaired.is_watertight)
        report["volume_mesh_units3"] = (
            float(abs(repaired.volume)) if repaired.is_volume else 0.0
        )
        return repaired, report

    trimesh.repair.fix_normals(repaired)
    trimesh.repair.fix_inversion(repaired)
    trimesh.repair.fill_holes(repaired)
    repaired.remove_unreferenced_vertices()

    if strategy == "fill_only":
        report["final_watertight"] = bool(repaired.is_watertight)
        report["volume_mesh_units3"] = float(abs(repaired.volume)) if repaired.is_volume else 0.0
        return repaired, report

    if not repaired.is_watertight and strategy == "fill_then_hull":
        repaired = repaired.convex_hull
        report["used_convex_hull"] = True

    report["final_watertight"] = bool(repaired.is_watertight)
    report["volume_mesh_units3"] = float(abs(repaired.volume)) if repaired.is_volume else 0.0
    return repaired, report
