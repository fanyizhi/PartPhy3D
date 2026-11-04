from __future__ import annotations

import numpy as np
import trimesh
from scipy import ndimage

from tactile3d.mesh.state_surface import part_surface
from tactile3d.physics.part_state import Material, PartPhysicalState, PhysicalObject

VOXEL_RESOLUTION = 48
VOXEL_CLOSING_ITERATIONS = 1


def voxel_envelope(
    mesh: trimesh.Trimesh, resolution: int = VOXEL_RESOLUTION
) -> tuple[trimesh.Trimesh, dict]:
    """显式体素闭运算包络；这是近似外包络，不是恢复真实内部。

    Args:
        mesh: 含面的有限坐标输入。
        resolution: 最长轴的网格分辨率，范围8到96。

    Returns:
        体素边界Mesh及修复假设；保留原始Mesh供独立几何评价。
    """
    if not 8 <= resolution <= 96 or not len(mesh.faces):
        raise ValueError("体素包络需要三角面和8到96的分辨率")
    pitch = float(np.max(mesh.extents)) / resolution
    if not np.isfinite(pitch) or pitch <= 0:
        raise ValueError("无法从退化几何建立体素包络")
    voxels = mesh.voxelized(pitch)
    # 边界留空，避免闭运算把数组边缘的物体截断。
    padding = VOXEL_CLOSING_ITERATIONS + 1
    shell = np.pad(voxels.matrix, padding)
    sealed = ndimage.binary_closing(shell, iterations=VOXEL_CLOSING_ITERATIONS)
    filled = ndimage.binary_fill_holes(sealed)
    indices = np.argwhere(filled) - padding
    if not len(indices):
        raise ValueError("体素包络为空")
    origin = np.asarray(voxels.transform[:3, 3])
    centers = indices * pitch + origin
    part = PartPhysicalState(
        "envelope",
        "envelope",
        centers,
        pitch,
        (Material("envelope_proxy", 1.0, "morphological_assumption"),),
        np.ones((len(centers), 1)),
        evidence=("outer_envelope_not_internal_material_truth",),
    )
    result = part_surface(PhysicalObject("envelope", (part,), origin), "envelope")
    if not result.is_watertight or not result.is_volume:
        raise ValueError("体素包络仍非有效封闭体积，停止而不使用凸包")
    return result, {
        "approximate_envelope": True,
        "true_internal_geometry_recovered": False,
        "resolution": resolution,
        "pitch_mesh_units": pitch,
        "closing_iterations": VOXEL_CLOSING_ITERATIONS,
        "solid_cells": len(indices),
        "assumption": "surface voxelization + closing + exterior flood fill",
    }


def repair_mesh(
    mesh: trimesh.Trimesh, strategy: str = "fill_then_hull"
) -> tuple[trimesh.Trimesh, dict]:
    if strategy not in {"none", "fill_only", "fill_then_hull", "voxel_envelope"}:
        raise ValueError(f"Unknown repair strategy: {strategy}")
    repaired = mesh.copy()
    report = {
        "initial_watertight": bool(repaired.is_watertight),
        "strategy": strategy,
        "used_convex_hull": False,
    }

    if strategy == "voxel_envelope":
        repaired, extra = voxel_envelope(repaired)
        report.update(extra)
        report["final_watertight"] = bool(repaired.is_watertight)
        report["volume_mesh_units3"] = float(repaired.volume)
        return repaired, report

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
        report["volume_mesh_units3"] = (
            float(abs(repaired.volume)) if repaired.is_volume else 0.0
        )
        return repaired, report

    if not repaired.is_watertight and strategy == "fill_then_hull":
        repaired = repaired.convex_hull
        report["used_convex_hull"] = True

    report["final_watertight"] = bool(repaired.is_watertight)
    report["volume_mesh_units3"] = (
        float(abs(repaired.volume)) if repaired.is_volume else 0.0
    )
    return repaired, report
