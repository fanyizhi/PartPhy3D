"""Adaptive material-point quadrature of a closed metric mesh."""

from dataclasses import dataclass
from itertools import product

import numpy as np
import trimesh

from tactile3d.physics.materials import MaterialPrior


@dataclass(frozen=True)
class Construction:
    surface: str
    core: str
    thickness_m: float = 0.0
    frame: str | None = None
    frame_boxes: tuple = ()  # metric lower/upper XYZ bounds, in mesh coordinates
    support_top_m: float | None = None

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
) -> dict:
    """Return disjoint quadrature points with SI particle fields and stable material IDs."""
    import igl

    if not mesh.is_volume or not np.isfinite(mesh.vertices).all():
        raise ValueError("Particle sampling requires an outward, closed metric mesh")
    if not np.isfinite([finest_m, bulk_m]).all() or not 0 < finest_m <= bulk_m:
        raise ValueError("Require finite 0 < finest_m <= bulk_m")
    support_top = construction.support_top_m
    if (
        support_top is not None
        and not mesh.bounds[0, 2] < support_top < mesh.bounds[1, 2]
    ):
        raise ValueError("Support boundary must lie inside mesh z bounds")
    labels = list(
        dict.fromkeys(
            filter(None, (construction.surface, construction.core, construction.frame))
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
    positions = mesh.bounds.mean(0)[None]
    size = mesh.extents.max() + finest_m
    offsets = np.array(list(product((-0.25, 0.25), repeat=3)))
    xyz, volumes, ids = [], [], []
    while len(positions):
        depth = -igl.signed_distance(
            positions, vertices, faces, igl.SIGNED_DISTANCE_TYPE_PSEUDONORMAL
        )[0]
        if not np.isfinite(depth).all():
            raise ValueError("Mesh signed-distance query produced non-finite values")
        radius = np.sqrt(3) * size / 2
        refine = np.abs(depth) <= radius
        domain = np.full(len(depth), labels.index(construction.core), dtype=np.int16)
        if construction.thickness_m:
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
        accepted = inside & ~split
        if np.any(accepted):
            xyz.append(positions[accepted])
            volumes.append(np.full(accepted.sum(), size**3))
            ids.append(domain[accepted])
        positions = (positions[split, None, :] + offsets * size).reshape(-1, 3)
        size /= 2
    if not xyz:
        raise ValueError("No positive-volume cells at requested resolution")
    material_id = np.concatenate(ids)
    if len(np.unique(material_id)) != len(labels):
        raise ValueError("Sampling did not resolve all declared material domains")
    volume_m3 = np.concatenate(volumes)
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
