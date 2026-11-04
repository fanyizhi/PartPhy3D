from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from tactile3d.mesh.segmentation import PartSegmentation
from tactile3d.physics.materials import MaterialPrior, material_probabilities, sample_material


@dataclass
class PartEstimate:
    part_id: int
    volume_m3: float
    material_probs: dict[str, float]
    sampled_material: str
    density_kg_m3: float
    youngs_modulus_pa: float
    mass_kg: float


def estimate_parts(
    mesh: trimesh.Trimesh,
    segmentation: PartSegmentation,
    priors: list[MaterialPrior],
    material_temperature: float,
    rng: np.random.Generator,
    image_features: dict | None = None,
) -> list[PartEstimate]:
    total_volume = float(abs(mesh.volume)) if mesh.is_volume else 0.0
    image_features = image_features or {}
    estimates: list[PartEstimate] = []
    for part_id, fraction in segmentation.part_volume_fraction.items():
        face_mask = segmentation.face_labels == part_id
        area = float(mesh.area_faces[face_mask].sum())
        compactness = area / (mesh.area + 1e-12)
        part_features = {
            "brightness": image_features.get("brightness", 0.5),
            "roughness": image_features.get("roughness", 0.15),
            "compactness": compactness,
        }
        probs = material_probabilities(part_features, priors, material_temperature)
        material, density, youngs = sample_material(probs, priors, rng)
        volume = total_volume * float(fraction)
        estimates.append(
            PartEstimate(
                part_id=int(part_id),
                volume_m3=volume,
                material_probs=probs,
                sampled_material=material,
                density_kg_m3=density,
                youngs_modulus_pa=youngs,
                mass_kg=density * volume,
            )
        )
    return estimates


def parts_to_json(parts: list[PartEstimate]) -> list[dict]:
    return [
        {
            "part_id": p.part_id,
            "volume_m3": p.volume_m3,
            "material_probs": p.material_probs,
            "sampled_material": p.sampled_material,
            "density_kg_m3": p.density_kg_m3,
            "youngs_modulus_pa": p.youngs_modulus_pa,
            "mass_kg": p.mass_kg,
        }
        for p in parts
    ]
