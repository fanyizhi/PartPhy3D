from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from scipy.spatial import ConvexHull, QhullError, cKDTree

from tactile3d.mesh.io import load_triangle_mesh
from tactile3d.utils.io import ensure_dir, write_json


@dataclass(frozen=True)
class VolumeEstimate:
    value_mesh_units3: float
    method: str


@dataclass(frozen=True)
class RigidMassProperties:
    mass_kg: float
    center_of_mass_m: np.ndarray
    inertia_com_kg_m2: np.ndarray
    volume: VolumeEstimate


def replace_mesh_part(
    source_path: str | Path,
    face_labels_path: str | Path,
    donor_path: str | Path,
    target_part_id: int,
    output_dir: str | Path,
    *,
    mesh_unit_to_meter: float,
    source_density_kg_m3: float,
    donor_density_kg_m3: float,
    use_icp: bool = True,
    vertical_axis: int = 2,
) -> dict[str, Any]:
    """Replace a labelled component and recompute first-order physical properties.

    This retrieval baseline preserves the donor shape using uniform scaling. It
    exports an assembly rather than silently applying a destructive boolean. A
    later constrained meshing stage can fuse interfaces when the two parts are
    intentionally intersecting.
    """

    if mesh_unit_to_meter <= 0:
        raise ValueError("mesh_unit_to_meter must be positive")
    if source_density_kg_m3 <= 0 or donor_density_kg_m3 <= 0:
        raise ValueError("material densities must be positive")
    if vertical_axis not in {0, 1, 2}:
        raise ValueError("vertical_axis must be 0, 1, or 2")

    source = _load_mesh(source_path, process=False)
    donor = _load_mesh(donor_path)
    labels = np.load(face_labels_path)
    if labels.ndim != 1 or len(labels) != len(source.faces):
        raise ValueError("face_labels must contain exactly one integer per source face")
    target_mask = labels.astype(np.int64) == int(target_part_id)
    if not np.any(target_mask) or np.all(target_mask):
        raise ValueError("target part must select some, but not all, source faces")

    target = source.submesh([target_mask], append=True, repair=False)
    remaining = source.submesh([~target_mask], append=True, repair=False)
    aligned, alignment = _align_donor(donor, target, use_icp)
    assembly = trimesh.util.concatenate([remaining, aligned])

    source_properties = _rigid_mass_properties(source, source_density_kg_m3, mesh_unit_to_meter)
    target_properties = _rigid_mass_properties(target, source_density_kg_m3, mesh_unit_to_meter)
    donor_properties = _rigid_mass_properties(aligned, donor_density_kg_m3, mesh_unit_to_meter)
    source_volume = source_properties.volume
    target_volume = target_properties.volume
    donor_volume = donor_properties.volume
    scale3 = mesh_unit_to_meter**3
    old_mass = source_properties.mass_kg
    removed_mass = target_properties.mass_kg
    added_mass = donor_properties.mass_kg
    new_mass = max(old_mass - removed_mass + added_mass, 0.0)

    source_com = source_properties.center_of_mass_m
    target_com = target_properties.center_of_mass_m
    donor_com = donor_properties.center_of_mass_m
    moment = old_mass * source_com - removed_mass * target_com + added_mass * donor_com
    new_com = moment / max(new_mass, 1e-12)
    new_inertia = (
        _shift_inertia(
            source_properties.inertia_com_kg_m2,
            old_mass,
            source_com,
            new_com,
        )
        - _shift_inertia(
            target_properties.inertia_com_kg_m2,
            removed_mass,
            target_com,
            new_com,
        )
        + _shift_inertia(
            donor_properties.inertia_com_kg_m2,
            added_mass,
            donor_com,
            new_com,
        )
    )
    new_inertia = 0.5 * (new_inertia + new_inertia.T)
    source_principal = np.linalg.eigvalsh(source_properties.inertia_com_kg_m2)
    new_principal = np.linalg.eigvalsh(new_inertia)
    rigid_checks = _rigid_property_checks(
        old_mass,
        removed_mass,
        added_mass,
        new_mass,
        new_com,
        new_inertia,
    )
    interface = _interface_metrics(source, labels.astype(np.int64), target_part_id, aligned)
    stability = _stability(assembly, new_com / mesh_unit_to_meter, vertical_axis)
    components = len(assembly.split(only_watertight=False))

    warnings = []
    for name, estimate in (
        ("source", source_volume),
        ("target", target_volume),
        ("donor", donor_volume),
    ):
        if estimate.method != "enclosed_volume":
            warnings.append(f"{name} volume uses {estimate.method}; mass is an approximation")
    if interface["num_interface_vertices"] == 0:
        warnings.append(
            "No shared labelled interface was found; parts may be disconnected components"
        )
    if components > 1:
        warnings.append(
            "Edited result is a multi-component assembly, not a fused manufacturing mesh"
        )
    if not rigid_checks["all_passed"]:
        warnings.append(
            "Reconstructed rigid-body properties failed at least one analytical consistency check"
        )

    output_dir = ensure_dir(Path(output_dir).expanduser().resolve())
    metric_remaining = remaining.copy()
    metric_target = target.copy()
    metric_donor = aligned.copy()
    metric_assembly = assembly.copy()
    for mesh in (metric_remaining, metric_target, metric_donor, metric_assembly):
        mesh.apply_scale(mesh_unit_to_meter)
    metric_remaining.export(output_dir / "source_remaining.ply")
    metric_target.export(output_dir / "source_removed_part.ply")
    metric_donor.export(output_dir / "donor_aligned.ply")
    metric_assembly.export(output_dir / "edited_assembly.ply")

    report = {
        "schema_version": 1,
        "method": "PartPhy3D-RetrievalBaseline",
        "source_mesh": str(Path(source_path).expanduser().resolve()),
        "donor_mesh": str(Path(donor_path).expanduser().resolve()),
        "target_part_id": int(target_part_id),
        "mesh_unit_to_meter": float(mesh_unit_to_meter),
        "alignment": alignment,
        "geometry": {
            "edited_watertight": bool(assembly.is_watertight),
            "connected_components": components,
            "interface": interface,
        },
        "physics": {
            "assumptions": [
                "source material is represented by one effective density",
                "the removed part volume is exact only when that labelled submesh is closed",
                "static support uses the convex hull of lowest vertices and ignores deformation",
            ],
            "source_density_kg_m3": float(source_density_kg_m3),
            "donor_density_kg_m3": float(donor_density_kg_m3),
            "source_volume": _volume_json(source_volume, scale3),
            "removed_volume": _volume_json(target_volume, scale3),
            "donor_volume": _volume_json(donor_volume, scale3),
            "mass_before_kg": float(old_mass),
            "removed_mass_kg": float(removed_mass),
            "added_mass_kg": float(added_mass),
            "mass_after_kg": float(new_mass),
            "center_of_mass_before_m": source_com.tolist(),
            "center_of_mass_after_m": new_com.tolist(),
            "inertia_tensor_before_kg_m2": source_properties.inertia_com_kg_m2.tolist(),
            "inertia_tensor_after_kg_m2": new_inertia.tolist(),
            "principal_inertia_before_kg_m2": source_principal.tolist(),
            "principal_inertia_after_kg_m2": new_principal.tolist(),
            "rigid_body_consistency": rigid_checks,
            "stability": stability,
        },
        "warnings": warnings,
    }
    write_json(output_dir / "edit_report.json", report)
    return report


def _align_donor(
    donor: trimesh.Trimesh, target: trimesh.Trimesh, use_icp: bool
) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    aligned = donor.copy()
    donor_size = float(np.linalg.norm(aligned.extents))
    target_size = float(np.linalg.norm(target.extents))
    if donor_size <= 0 or target_size <= 0:
        raise ValueError("Source target and donor must have non-zero extent")
    uniform_scale = target_size / donor_size
    aligned.apply_scale(uniform_scale)
    aligned.apply_translation(target.centroid - aligned.centroid)
    icp_cost = None
    if use_icp:
        donor_points = _limited_vertices(aligned.vertices)
        target_points = _limited_vertices(target.vertices)
        matrix, _, icp_cost = trimesh.registration.icp(
            donor_points, target_points, max_iterations=40, reflection=False, scale=False
        )
        aligned.apply_transform(matrix)
    return aligned, {
        "uniform_scale": float(uniform_scale),
        "used_icp": bool(use_icp),
        "icp_cost": float(icp_cost) if icp_cost is not None else None,
        "transform_semantics": "donor is uniformly scaled and rigidly registered to removed part",
    }


def _limited_vertices(vertices: np.ndarray, limit: int = 5000) -> np.ndarray:
    if len(vertices) <= limit:
        return np.asarray(vertices)
    indices = np.linspace(0, len(vertices) - 1, limit, dtype=np.int64)
    return np.asarray(vertices)[indices]


def _volume(mesh: trimesh.Trimesh) -> VolumeEstimate:
    if mesh.is_watertight and mesh.is_volume and abs(float(mesh.volume)) > 0:
        return VolumeEstimate(abs(float(mesh.volume)), "enclosed_volume")
    hull = mesh.convex_hull
    return VolumeEstimate(abs(float(hull.volume)), "convex_hull_proxy")


def _centroid(mesh: trimesh.Trimesh) -> np.ndarray:
    if mesh.is_watertight and mesh.is_volume:
        return np.asarray(mesh.center_mass, dtype=np.float64)
    return np.asarray(mesh.centroid, dtype=np.float64)


def _rigid_mass_properties(
    mesh: trimesh.Trimesh,
    density_kg_m3: float,
    mesh_unit_to_meter: float,
) -> RigidMassProperties:
    estimate = _volume(mesh)
    proxy = mesh if estimate.method == "enclosed_volume" else mesh.convex_hull
    scale3 = mesh_unit_to_meter**3
    mass = density_kg_m3 * estimate.value_mesh_units3 * scale3
    center = _centroid(proxy) * mesh_unit_to_meter
    inertia = (
        np.asarray(proxy.moment_inertia, dtype=np.float64) * density_kg_m3 * mesh_unit_to_meter**5
    )
    return RigidMassProperties(
        mass_kg=float(mass),
        center_of_mass_m=center,
        inertia_com_kg_m2=inertia,
        volume=estimate,
    )


def _shift_inertia(
    inertia_com: np.ndarray,
    mass: float,
    source_com: np.ndarray,
    destination: np.ndarray,
) -> np.ndarray:
    displacement = np.asarray(source_com) - np.asarray(destination)
    parallel_axis = float(np.dot(displacement, displacement)) * np.eye(3) - np.outer(
        displacement, displacement
    )
    return np.asarray(inertia_com) + mass * parallel_axis


def _rigid_property_checks(
    mass_before: float,
    removed_mass: float,
    added_mass: float,
    mass_after: float,
    center_of_mass: np.ndarray,
    inertia: np.ndarray,
) -> dict[str, Any]:
    expected_mass = mass_before - removed_mass + added_mass
    eigenvalues = np.linalg.eigvalsh(inertia)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    symmetric = bool(np.allclose(inertia, inertia.T, rtol=1e-8, atol=1e-12))
    positive_semidefinite = bool(np.all(eigenvalues >= -1e-9 * scale))
    moments = np.maximum(eigenvalues, 0.0)
    triangle_inequality = bool(moments[0] + moments[1] + 1e-9 * scale >= moments[2])
    checks = {
        "positive_mass": bool(mass_after > 0),
        "finite_center_of_mass": bool(np.all(np.isfinite(center_of_mass))),
        "inertia_symmetric": symmetric,
        "inertia_positive_semidefinite": positive_semidefinite,
        "principal_moment_triangle_inequality": triangle_inequality,
        "mass_balance_residual_kg": float(mass_after - expected_mass),
        "principal_moments_kg_m2": moments.tolist(),
    }
    checks["all_passed"] = bool(
        checks["positive_mass"]
        and checks["finite_center_of_mass"]
        and symmetric
        and positive_semidefinite
        and triangle_inequality
        and abs(checks["mass_balance_residual_kg"]) <= 1e-9 * max(mass_after, 1.0)
    )
    return checks


def _volume_json(estimate: VolumeEstimate, scale3: float) -> dict[str, Any]:
    return {"value_m3": float(estimate.value_mesh_units3 * scale3), "method": estimate.method}


def _interface_metrics(
    source: trimesh.Trimesh,
    labels: np.ndarray,
    target_part_id: int,
    donor: trimesh.Trimesh,
) -> dict[str, Any]:
    adjacency = source.face_adjacency
    if len(adjacency) == 0:
        interface_vertices = np.empty((0,), dtype=np.int64)
    else:
        crossing = (labels[adjacency[:, 0]] == target_part_id) != (
            labels[adjacency[:, 1]] == target_part_id
        )
        edges = source.face_adjacency_edges[crossing]
        interface_vertices = (
            np.unique(edges.reshape(-1)) if len(edges) else np.empty((0,), dtype=np.int64)
        )
    if len(interface_vertices) == 0:
        return {
            "num_interface_vertices": 0,
            "mean_gap_mesh_units": None,
            "max_gap_mesh_units": None,
        }
    distances = cKDTree(donor.vertices).query(source.vertices[interface_vertices])[0]
    return {
        "num_interface_vertices": len(interface_vertices),
        "mean_gap_mesh_units": float(np.mean(distances)),
        "max_gap_mesh_units": float(np.max(distances)),
    }


def _stability(mesh: trimesh.Trimesh, center_of_mass: np.ndarray, vertical_axis: int) -> dict:
    planar_axes = [axis for axis in range(3) if axis != vertical_axis]
    height = mesh.vertices[:, vertical_axis]
    tolerance = max(float(np.ptp(height)) * 0.01, 1e-8)
    support = mesh.vertices[height <= float(height.min()) + tolerance][:, planar_axes]
    support = np.unique(np.round(support, decimals=10), axis=0)
    if len(support) < 3:
        return {"is_statically_supported": None, "support_margin_mesh_units": None}
    try:
        hull = ConvexHull(support)
    except QhullError:
        return {"is_statically_supported": None, "support_margin_mesh_units": None}
    point = center_of_mass[planar_axes]
    signed = hull.equations[:, :-1] @ point + hull.equations[:, -1]
    norm = np.linalg.norm(hull.equations[:, :-1], axis=1)
    margin = float(np.min(-signed / np.maximum(norm, 1e-12)))
    return {
        "is_statically_supported": bool(np.all(signed <= 1e-9)),
        "support_margin_mesh_units": margin,
    }


def _load_mesh(path: str | Path, process: bool = True) -> trimesh.Trimesh:
    return load_triangle_mesh(path, process=process)
