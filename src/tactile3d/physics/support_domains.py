"""Disjoint component envelopes and shells with conserved geometric moments."""

from __future__ import annotations

from dataclasses import dataclass

import manifold3d as manifold
import numpy as np
import trimesh
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, HalfspaceIntersection


def solid_from_mesh(mesh: trimesh.Trimesh) -> manifold.Manifold:
    result = manifold.Manifold(
        manifold.Mesh64(
            vert_properties=np.array(
                mesh.vertices, dtype=np.float64, order="C", copy=True
            ),
            tri_verts=np.array(mesh.faces, dtype=np.uint64, order="C", copy=True),
        )
    )
    if result.status() != manifold.Error.NoError:
        raise ValueError(f"Invalid closed domain: {result.status()}")
    return result


def domain_mesh(solid: manifold.Manifold) -> trimesh.Trimesh:
    raw = solid.to_mesh64()
    return trimesh.Trimesh(raw.vert_properties[:, :3], raw.tri_verts, process=False)


def convex_envelope(mesh: trimesh.Trimesh, thickness: float, assembly_center):
    """Explicit completion hypothesis; planar patches become inward prisms."""
    vertices = np.asarray(mesh.vertices)
    center = vertices.mean(axis=0)
    _, singular, axes = np.linalg.svd(vertices - center, full_matrices=False)
    rank = int(np.sum(singular > max(singular[0] * 1e-7, 1e-10)))
    if rank < 2:
        raise ValueError("A line or point cannot define a material domain")
    if rank == 2:
        if not np.isfinite(thickness) or thickness <= 0:
            raise ValueError("A planar manufactured part requires positive thickness")
        normal = axes[-1]
        if normal @ (np.asarray(assembly_center) - center) < 0:
            normal = -normal
        vertices = np.vstack([vertices, vertices + normal * thickness])
        method = "planar_patch_inward_prism_hypothesis"
    else:
        method = "component_convex_completion_hypothesis"
    return domain_mesh(manifold.Manifold.hull_points(vertices)), method


def eroded_convex(envelope: trimesh.Trimesh, thickness: float) -> manifold.Manifold:
    """Intersect inward-offset support planes, with an explicit empty-core case."""
    if not np.isfinite(thickness) or thickness < 0:
        raise ValueError("Shell thickness must be finite and nonnegative")
    if thickness == 0:
        return solid_from_mesh(envelope)
    planes = ConvexHull(envelope.vertices).equations
    normals = np.linalg.norm(planes[:, :3], axis=1)
    fit = linprog(
        [0, 0, 0, -1],
        A_ub=np.c_[planes[:, :3], normals],
        b_ub=-planes[:, 3],
        bounds=[(None, None)] * 3 + [(0, None)],
        method="highs",
    )
    if not fit.success:
        raise ValueError("Cannot certify an interior point for the convex envelope")
    if fit.x[3] <= thickness * (1 + 1e-8):
        return manifold.Manifold()
    shifted = planes.copy()
    shifted[:, 3] += thickness * normals
    interior = HalfspaceIntersection(shifted, fit.x[:3]).intersections
    # Half-space vertices can coincide on triangulated coplanar facets. Build
    # the solid hull directly, rather than passing a near-degenerate Qhull
    # triangulation through a second mesh-to-solid conversion.
    return manifold.Manifold.hull_points(interior)


@dataclass
class SupportDomain:
    component_id: int
    material: str
    envelope: trimesh.Trimesh
    inner: manifold.Manifold
    priority: int
    source: str


def unit_moments(solid: manifold.Manifold) -> dict:
    volume = float(solid.volume())
    if volume < -1e-10:
        raise ValueError("Negative material volume")
    if volume <= 1e-14:
        return {
            "volume_m3": 0.0,
            "first_m4": [0.0] * 3,
            "inertia_origin_m5": np.zeros((3, 3)).tolist(),
        }
    mesh = domain_mesh(solid)
    center = np.asarray(mesh.center_mass)
    inertia = np.asarray(mesh.moment_inertia) + volume * (
        np.eye(3) * float(center @ center) - np.outer(center, center)
    )
    return {
        "volume_m3": volume,
        "first_m4": (volume * center).tolist(),
        "inertia_origin_m5": inertia.tolist(),
    }


def partition_domains(domains: list[SupportDomain]) -> dict:
    """Crop overlaps before integration; cropping never creates an artificial coat."""
    if len({d.component_id for d in domains}) != len(domains):
        raise ValueError("Component IDs must be unique")
    occupied = manifold.Manifold()
    parts = []
    raw_volume = 0.0
    for domain in sorted(domains, key=lambda d: (d.priority, d.component_id)):
        outer = solid_from_mesh(domain.envelope)
        raw_volume += float(outer.volume())
        assigned = outer - occupied
        core = assigned ^ domain.inner
        shell = assigned - core
        assigned_moments, core_moments, shell_moments = (
            unit_moments(solid) for solid in (assigned, core, shell)
        )
        if not np.isclose(
            assigned_moments["volume_m3"],
            core_moments["volume_m3"] + shell_moments["volume_m3"],
            rtol=2e-6,
            atol=1e-10,
        ):
            raise ValueError("Shell and core do not partition the assigned volume")
        parts.append(
            {
                "component_id": domain.component_id,
                "material": domain.material,
                "geometry_source": domain.source,
                "assigned": assigned_moments,
                "core": core_moments,
                "shell": shell_moments,
            }
        )
        occupied = occupied + outer
    union_volume = float(occupied.volume())
    if not np.isclose(
        sum(p["assigned"]["volume_m3"] for p in parts),
        union_volume,
        rtol=2e-6,
        atol=1e-10,
    ):
        raise ValueError("Disjoint domains do not conserve union volume")
    return {
        "parts": parts,
        "union_volume_m3": union_volume,
        "raw_envelope_sum_m3": raw_volume,
        "overlap_removed_m3": raw_volume - union_volume,
    }


def assemble_properties(weighted_moments: list[tuple[dict, float]]) -> dict:
    """Combine fixed shell/core moments under material or homogenized-core densities."""
    if any(not np.isfinite(rho) or rho < 0 for _, rho in weighted_moments):
        raise ValueError("Densities must be finite and nonnegative")
    mass = sum(m["volume_m3"] * rho for m, rho in weighted_moments)
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError("Positive finite total mass required")
    first = sum(
        (np.asarray(m["first_m4"]) * rho for m, rho in weighted_moments), np.zeros(3)
    )
    origin = sum(
        (np.asarray(m["inertia_origin_m5"]) * rho for m, rho in weighted_moments),
        np.zeros((3, 3)),
    )
    center = first / mass
    inertia = origin - mass * (np.eye(3) * (center @ center) - np.outer(center, center))
    if np.linalg.eigvalsh(inertia).min() < -1e-7:
        raise ValueError("Inertia is not positive semidefinite")
    return {
        "mass_kg": float(mass),
        "center_of_mass_m": center.tolist(),
        "inertia_kg_m2": inertia.tolist(),
    }
