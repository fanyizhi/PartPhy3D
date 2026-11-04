"""Explicit part states and a canonical volume quadrature shared by mesh LODs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class CanonicalPartState:
    """Metric cube cells; part densities include the effect of material voids."""

    centers_m: np.ndarray
    part_ids: np.ndarray
    pitch_m: float
    density_kg_m3: np.ndarray

    def __post_init__(self) -> None:
        centers = np.array(self.centers_m, dtype=float, copy=True)
        ids = np.array(self.part_ids, copy=True)
        density = np.array(self.density_kg_m3, dtype=float, copy=True)
        if centers.ndim != 2 or centers.shape[1] != 3 or not len(centers):
            raise ValueError("centers_m must be a nonempty (N, 3) array")
        if ids.shape != (len(centers),) or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError("part_ids must contain one integer per cell")
        if density.ndim != 1 or not len(density) or np.any(density < 0):
            raise ValueError("Part densities must be a nonnegative vector")
        if np.any(ids < 0) or np.any(ids >= len(density)):
            raise ValueError("Part IDs must index the density vector")
        if not np.isfinite(self.pitch_m) or self.pitch_m <= 0:
            raise ValueError("pitch_m must be finite and positive")
        if not np.isfinite(centers).all() or not np.isfinite(density).all():
            raise ValueError("Coordinates and density must be finite")
        for array in (centers, ids, density):
            array.flags.writeable = False
        object.__setattr__(self, "centers_m", centers)
        object.__setattr__(self, "part_ids", ids)
        object.__setattr__(self, "density_kg_m3", density)

    @property
    def part_volumes_m3(self) -> np.ndarray:
        return np.bincount(self.part_ids, minlength=len(self.density_kg_m3)) * self.pitch_m**3

    def face_lookup(self, triangle_centers_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Appearance lookup only; surface faces are never assigned volume."""
        distance, index = cKDTree(self.centers_m).query(triangle_centers_m)
        return self.part_ids[index], distance

    def integrate(self) -> dict:
        cell_mass = self.density_kg_m3[self.part_ids] * self.pitch_m**3
        total = float(cell_mass.sum())
        if total <= 0:
            raise ValueError("Center of mass requires positive total mass")
        center = np.einsum("i,ij->j", cell_mass, self.centers_m) / total
        offsets = self.centers_m - center
        second = np.einsum("i,ij,ik->jk", cell_mass, offsets, offsets)
        inertia = np.trace(second) * np.eye(3) - second
        # Each quadrature cell is a finite cube, not a point mass.
        inertia += np.eye(3) * total * self.pitch_m**2 / 6.0
        return {
            "mass_kg": total,
            "center_of_mass_m": center.tolist(),
            "inertia_kg_m2": inertia.tolist(),
            "part_mass_kg": (self.part_volumes_m3 * self.density_kg_m3).tolist(),
        }


def moment_disagreement(reference: dict, candidate: dict, diagonal_m: float) -> dict:
    if diagonal_m <= 0:
        raise ValueError("diagonal_m must be positive")
    ref_inertia = np.asarray(reference["inertia_kg_m2"])
    return {
        "mass_error_percent": 100 * abs(candidate["mass_kg"] - reference["mass_kg"])
        / reference["mass_kg"],
        "com_error_mm": 1000 * float(np.linalg.norm(
            np.asarray(candidate["center_of_mass_m"]) - reference["center_of_mass_m"]
        )),
        "com_error_percent_bbox": 100 * float(np.linalg.norm(
            np.asarray(candidate["center_of_mass_m"]) - reference["center_of_mass_m"]
        )) / diagonal_m,
        "inertia_tensor_error_percent": 100 * float(np.linalg.norm(
            np.asarray(candidate["inertia_kg_m2"]) - ref_inertia
        )) / max(float(np.linalg.norm(ref_inertia)), 1e-12),
    }
