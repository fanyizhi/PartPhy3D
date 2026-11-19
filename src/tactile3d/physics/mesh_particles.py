"""Adaptive material-point quadrature of a closed metric mesh."""

from dataclasses import dataclass
from itertools import combinations, product

import numpy as np
import trimesh

from tactile3d.physics.materials import MaterialPrior


@dataclass(frozen=True)
class InteriorDomain:
    """Closed metric subdomain. The label ``void`` reserves air without exporting particles."""

    mesh: trimesh.Trimesh
    material: str


@dataclass(frozen=True)
class Construction:
    surface: str
    core: str
    thickness_m: float = 0.0
    frame: str | None = None
    frame_boxes: tuple = ()  # metric lower/upper XYZ bounds, in mesh coordinates
    support_top_m: float | None = None

    @classmethod
    def from_generated_partition(cls, surface: str, core: str = "void"):
        """Use an explicit inner mesh for the void or a second material."""
        if not surface or surface == core or surface == "void" or not core:
            raise ValueError(
                "Generated partition requires distinct shell and core regions"
            )
        return cls(surface=surface, core=core)

    @classmethod
    def from_visual_hypothesis(
        cls, surface: str, structure: str, *, thickness_m=0.0, core=None
    ):
        """Accept a selected visual hypothesis; unresolved constructions cannot emit particles."""
        if structure == "solid":
            return cls.from_evidence(
                surface=surface, solid=True, core=core, thickness_m=thickness_m
            )
        if structure == "hollow" and core in (None, "void"):
            return cls.from_evidence(
                surface=surface, solid=False, core="void", thickness_m=thickness_m
            )
        if structure in {"layered", "foam_filled"} and core not in (None, "void"):
            return cls.from_evidence(
                surface=surface, solid=False, core=core, thickness_m=thickness_m
            )
        raise ValueError("Unresolved or contradictory visual construction")

    @classmethod
    def from_evidence(
        cls,
        *,
        surface,
        solid,
        core=None,
        thickness_m=0.0,
        frame=None,
        frame_boxes=(),
        support_top_m=None,
    ):
        if not surface or surface == "void":
            raise ValueError("Outer material must be known")
        if support_top_m is not None and (not frame or not np.isfinite(support_top_m)):
            raise ValueError(
                "Support boundary requires a frame material and finite height"
            )
        if (
            solid
            and core in (None, surface)
            and thickness_m == 0
            and not frame
            and not frame_boxes
        ):
            return cls(surface, surface)
        if (
            not solid
            and core
            and core != surface
            and np.isfinite(thickness_m)
            and thickness_m > 0
            and bool(frame) == bool(frame_boxes)
        ):
            bounds = np.asarray(frame_boxes, dtype=float)
            if frame and (
                bounds.ndim != 3
                or bounds.shape[1:] != (2, 3)
                or not np.isfinite(bounds).all()
                or np.any(bounds[:, 0] >= bounds[:, 1])
            ):
                raise ValueError("Frame boxes need finite positive metric bounds")
            return cls(surface, core, thickness_m, frame, frame_boxes, support_top_m)
        raise ValueError("Construction evidence is missing or contradictory")


def sample_mesh_particles(
    mesh: trimesh.Trimesh,
    construction: Construction,
    priors: list[MaterialPrior],
    *,
    finest_m: float,
    bulk_m: float,
    poisson: dict[str, float],
    inner_mesh: trimesh.Trimesh | None = None,
) -> dict:
    """Return disjoint quadrature points with SI particle fields and stable material IDs."""
    import igl

    if not mesh.is_volume or not np.isfinite(mesh.vertices).all():
        raise ValueError("Particle sampling requires an outward, closed metric mesh")
    if not np.isfinite([finest_m, bulk_m]).all() or not 0 < finest_m <= bulk_m:
        raise ValueError("Require finite 0 < finest_m <= bulk_m")
    explicit = (
        construction.core != construction.surface and construction.thickness_m == 0
    )
    if explicit != (inner_mesh is not None):
        raise ValueError("Generated partition requires exactly one explicit inner mesh")
    if inner_mesh is not None:
        if not inner_mesh.is_volume or not np.isfinite(inner_mesh.vertices).all():
            raise ValueError("Inner interface must be an outward, closed metric mesh")
        if np.any(inner_mesh.bounds[0] <= mesh.bounds[0]) or np.any(
            inner_mesh.bounds[1] >= mesh.bounds[1]
        ):
            raise ValueError("Inner interface must be strictly enclosed")
        outside = trimesh.boolean.difference([inner_mesh, mesh], engine="manifold")
        if not outside.is_empty and abs(outside.volume) > 1e-8 * inner_mesh.volume:
            raise ValueError("Inner interface intersects the exterior")
    support_top = construction.support_top_m
    if (
        support_top is not None
        and not mesh.bounds[0, 2] < support_top < mesh.bounds[1, 2]
    ):
        raise ValueError("Support boundary must lie inside mesh z bounds")
    labels = list(
        dict.fromkeys(
            (
                name
                for name in (
                    construction.surface,
                    construction.core,
                    construction.frame,
                )
                if name and name != "void"
            )
        )
    )
    materials = {prior.name: prior for prior in priors}
    density = np.array([materials[name].density_kg_m3.mean for name in labels])
    youngs = np.array([materials[name].youngs_modulus_pa.mean for name in labels])
    nu = np.array([poisson[name] for name in labels])
    if not (
        np.isfinite([density, youngs, nu]).all()
        and np.all(density > 0)
        and np.all(youngs > 0)
        and np.all((-1 < nu) & (nu < 0.5))
    ):
        raise ValueError("Invalid material constitutive parameters")

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    core_id = labels.index(construction.core) if construction.core != "void" else -1
    if inner_mesh is not None:
        inner_vertices = np.asarray(inner_mesh.vertices, dtype=np.float64)
        inner_faces = np.asarray(inner_mesh.faces, dtype=np.int64)
        inner_in_outer = igl.signed_distance(
            inner_vertices, vertices, faces, igl.SIGNED_DISTANCE_TYPE_PSEUDONORMAL
        )[0]
        if not np.all(inner_in_outer < 0):
            raise ValueError("Inner interface touches or exits the exterior")
    positions = mesh.bounds.mean(0)[None]
    size = mesh.extents.max() + finest_m
    offsets = np.array(list(product((-0.25, 0.25), repeat=3)))
    xyz, volumes, ids = [], [], []
    resolved_void = False
    while len(positions):
        depth = -igl.signed_distance(
            positions, vertices, faces, igl.SIGNED_DISTANCE_TYPE_PSEUDONORMAL
        )[0]
        if not np.isfinite(depth).all():
            raise ValueError("Mesh signed-distance query produced non-finite values")
        radius = np.sqrt(3) * size / 2
        refine = np.abs(depth) <= radius
        domain = np.full(len(depth), core_id, dtype=np.int16)
        if inner_mesh is not None:
            inner_depth = -igl.signed_distance(
                positions,
                inner_vertices,
                inner_faces,
                igl.SIGNED_DISTANCE_TYPE_PSEUDONORMAL,
            )[0]
            if not np.isfinite(inner_depth).all():
                raise ValueError(
                    "Inner signed-distance query produced non-finite values"
                )
            domain[inner_depth <= 0] = labels.index(construction.surface)
            refine |= np.abs(inner_depth) <= radius
        elif construction.thickness_m:
            shell = depth <= construction.thickness_m
            domain[shell] = labels.index(construction.surface)
            refine |= np.abs(depth - construction.thickness_m) <= radius
            if construction.frame:
                for lower, upper in construction.frame_boxes:
                    signed_box_distance = np.max(
                        np.maximum(lower - positions, positions - upper), axis=1
                    )
                    domain[(signed_box_distance <= 0) & ~shell] = labels.index(
                        construction.frame
                    )
                    refine |= np.abs(signed_box_distance) <= size / 2
                if support_top is not None:
                    height = positions[:, 2] - support_top
                    domain[height < 0] = labels.index(construction.frame)
                    refine |= np.abs(height) <= size / 2
        inside = depth > 0
        split = (depth >= -radius) & (size > finest_m) & (refine | (size > bulk_m))
        accepted = inside & ~split & (domain >= 0)
        if construction.core == "void" and np.any(inside & ~split & (domain < 0)):
            resolved_void = True
        if np.any(accepted):
            xyz.append(positions[accepted])
            volumes.append(np.full(accepted.sum(), size**3))
            ids.append(domain[accepted])
        positions = (positions[split, None, :] + offsets * size).reshape(-1, 3)
        size /= 2
    if not xyz:
        raise ValueError("No positive-volume cells at requested resolution")
    if construction.core == "void" and not resolved_void:
        raise ValueError("Sampling did not resolve the declared cavity")
    material_id = np.concatenate(ids)
    if len(np.unique(material_id)) != len(labels):
        raise ValueError("Sampling did not resolve all declared material domains")
    volume_m3 = np.concatenate(volumes)
    if inner_mesh is not None:
        # The declared boundaries supply exact per-region volume independently of sampling.
        exact = [mesh.volume - inner_mesh.volume]
        if construction.core != "void":
            exact.append(inner_mesh.volume)
        sampled = np.bincount(material_id, weights=volume_m3)
        volume_m3 *= (np.asarray(exact) / sampled)[material_id]
    return dict(
        position_m=np.concatenate(xyz),
        volume_m3=volume_m3,
        mass_kg=volume_m3 * density[material_id],
        density_kg_m3=density[material_id],
        youngs_pa=youngs[material_id],
        poisson=nu[material_id],
        material_id=material_id,
        material_names=np.asarray(labels),
    )


def sample_disjoint_material_domains(
    outer: trimesh.Trimesh,
    outer_material: str,
    interior: tuple[InteriorDomain, ...],
    priors: list[MaterialPrior],
    *,
    finest_m: float,
    bulk_m: float,
    poisson: dict[str, float],
) -> dict[str, np.ndarray]:
    """Export SI particle fields from disjoint interior meshes and the outer remainder.

    Interior domains may share a boundary. Positive-volume overlaps and material
    domains outside the exterior are rejected before any points are sampled.
    """
    if (
        not outer.is_volume
        or not np.isfinite(outer.vertices).all()
        or not interior
        or not outer_material
        or outer_material == "void"
    ):
        raise ValueError(
            "Explicit partition needs a closed exterior and internal domains"
        )
    outer_min, outer_max = outer.bounds
    for region in interior:
        child = region.mesh
        if (
            not region.material
            or not child.is_volume
            or not np.isfinite(child.vertices).all()
        ):
            raise ValueError(
                "Each internal domain needs a material and closed finite geometry"
            )
        if np.any(child.bounds[0] <= outer_min) or np.any(child.bounds[1] >= outer_max):
            raise ValueError("Internal domain bounds must lie inside the exterior")
        outside = trimesh.boolean.difference([child, outer], engine="manifold")
        if not outside.is_empty and abs(outside.volume) > 1e-8 * child.volume:
            raise ValueError("An internal domain extends outside the exterior")
    for a, b in combinations(interior, 2):
        # Box rejection skips Boolean intersection for separated or face-touching pieces.
        if np.any(
            np.minimum(a.mesh.bounds[1], b.mesh.bounds[1])
            <= np.maximum(a.mesh.bounds[0], b.mesh.bounds[0])
        ):
            continue
        overlap = trimesh.boolean.intersection([a.mesh, b.mesh], engine="manifold")
        if not overlap.is_empty and abs(overlap.volume) > 1e-8 * min(
            a.mesh.volume, b.mesh.volume
        ):
            raise ValueError("Explicit internal domains overlap")

    remainder = trimesh.boolean.difference(
        [outer, *(r.mesh for r in interior)], engine="manifold"
    )
    if not remainder.is_volume:
        raise ValueError("Explicit domains leave no closed outer material")
    partition_volume = remainder.volume + sum(r.mesh.volume for r in interior)
    if not np.isclose(partition_volume, outer.volume, rtol=1e-6, atol=0):
        raise ValueError("Explicit domain volumes do not conserve the exterior volume")

    physical = [
        r
        for r in (InteriorDomain(remainder, outer_material), *interior)
        if r.material != "void"
    ]
    names = list(dict.fromkeys(r.material for r in physical))
    material_ids = {name: i for i, name in enumerate(names)}
    chunks = []
    for region in physical:
        data = sample_mesh_particles(
            region.mesh,
            Construction.from_evidence(surface=region.material, solid=True),
            priors,
            finest_m=finest_m,
            bulk_m=bulk_m,
            poisson=poisson,
        )
        # Correct each domain independently, retaining its local material density.
        factor = region.mesh.volume / data["volume_m3"].sum()
        data["mass_kg"] *= factor
        data["volume_m3"] *= factor
        data["material_id"][:] = material_ids[region.material]
        chunks.append(data)
    return {
        name: np.concatenate([chunk[name] for chunk in chunks])
        for name in chunks[0]
        if name != "material_names"
    } | {"material_names": np.asarray(names)}
