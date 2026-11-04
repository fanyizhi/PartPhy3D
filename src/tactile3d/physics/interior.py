from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from tactile3d.physics.materials import MaterialPrior

STRUCTURES = ("solid", "hollow", "layered", "foam_filled", "frame_supported", "partial_infill")
ROLES = ("shell", "core", "frame", "infill", "void")


@dataclass(frozen=True)
class Interval:
    low: float
    mode: float
    high: float

    def sample(self, rng: np.random.Generator) -> float:
        if self.high <= self.low:
            return float(self.low)
        mode = float(np.clip(self.mode, self.low, self.high))
        return float(rng.triangular(self.low, mode, self.high))

    def to_json(self) -> dict[str, float]:
        return {"low": self.low, "mode": self.mode, "high": self.high}


@dataclass(frozen=True)
class InteriorComponent:
    material: str
    role: str
    volume_fraction: Interval
    fill_ratio: Interval

    def to_json(self) -> dict[str, Any]:
        return {
            "material": self.material,
            "role": self.role,
            "volume_fraction": self.volume_fraction.to_json(),
            "fill_ratio": self.fill_ratio.to_json(),
        }


@dataclass(frozen=True)
class InteriorHypothesis:
    name: str
    structure: str
    probability: float
    wall_thickness_m: Interval
    components: tuple[InteriorComponent, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "structure": self.structure,
            "probability": self.probability,
            "wall_thickness_m": self.wall_thickness_m.to_json(),
            "components": [component.to_json() for component in self.components],
        }


@dataclass(frozen=True)
class PartInteriorProposal:
    part_id: int
    object_category: str
    source: str
    surface_material_probabilities: dict[str, float]
    hypotheses: tuple[InteriorHypothesis, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "part_id": self.part_id,
            "object_category": self.object_category,
            "source": self.source,
            "surface_material_probabilities": self.surface_material_probabilities,
            "interior_hypotheses": [hypothesis.to_json() for hypothesis in self.hypotheses],
        }


@dataclass(frozen=True)
class PartPhysicsSample:
    mass_kg: float
    occupied_volume_m3: float
    effective_density_kg_m3: float
    volume_weighted_youngs_pa: float
    structure: str
    material_mass_kg: dict[str, float]


def solid_proposal(
    part_id: int,
    material_names: list[str] | tuple[str, ...],
    surface_probabilities: np.ndarray,
    source: str = "surface_only",
) -> PartInteriorProposal:
    probabilities = _probability_dict(material_names, surface_probabilities)
    return PartInteriorProposal(
        part_id=part_id,
        object_category="unknown",
        source=source,
        surface_material_probabilities=probabilities,
        hypotheses=(
            InteriorHypothesis(
                name="uniform solid",
                structure="solid",
                probability=1.0,
                wall_thickness_m=Interval(0.0, 0.0, 0.0),
                components=(),
            ),
        ),
    )


def sample_part_physics(
    volume_m3: float,
    area_m2: float,
    surface_probabilities: np.ndarray,
    proposal: PartInteriorProposal,
    priors: list[MaterialPrior],
    rng: np.random.Generator,
    density_scale: float = 1.0,
) -> PartPhysicsSample:
    hypothesis_probabilities = _normalize_array(
        np.asarray([item.probability for item in proposal.hypotheses], dtype=np.float64)
    )
    hypothesis = proposal.hypotheses[int(rng.choice(len(proposal.hypotheses), p=hypothesis_probabilities))]
    prior_by_name = {prior.name: prior for prior in priors}
    surface_material = priors[int(rng.choice(len(priors), p=_normalize_array(surface_probabilities)))].name

    allocations = _sample_volume_allocations(
        volume_m3,
        area_m2,
        hypothesis,
        surface_material,
        rng,
    )
    mass = 0.0
    occupied_volume = 0.0
    youngs_volume_sum = 0.0
    material_mass: dict[str, float] = {}
    for material, component_volume in allocations.items():
        if component_volume <= 0 or material == "void":
            continue
        prior = prior_by_name.get(material, prior_by_name[surface_material])
        density = prior.density_kg_m3.sample(rng) * density_scale
        youngs = prior.youngs_modulus_pa.sample(rng)
        component_mass = density * component_volume
        mass += component_mass
        occupied_volume += component_volume
        youngs_volume_sum += youngs * component_volume
        material_mass[prior.name] = material_mass.get(prior.name, 0.0) + component_mass
    effective_density = mass / max(volume_m3, 1e-12)
    effective_youngs = youngs_volume_sum / max(occupied_volume, 1e-12)
    return PartPhysicsSample(
        mass_kg=float(mass),
        occupied_volume_m3=float(occupied_volume),
        effective_density_kg_m3=float(effective_density),
        volume_weighted_youngs_pa=float(effective_youngs),
        structure=hypothesis.structure,
        material_mass_kg=material_mass,
    )


def expected_material_volumes(
    volume_m3: float,
    area_m2: float,
    surface_probabilities: np.ndarray,
    proposal: PartInteriorProposal,
    material_names: list[str] | tuple[str, ...],
) -> dict[str, float]:
    """Deterministic first-moment approximation used by mass calibration."""

    result = {name: 0.0 for name in material_names}
    hypothesis_probabilities = _normalize_array(
        np.asarray([item.probability for item in proposal.hypotheses], dtype=np.float64)
    )
    surface_probabilities = _normalize_array(surface_probabilities)
    for hypothesis, hypothesis_probability in zip(
        proposal.hypotheses, hypothesis_probabilities, strict=True
    ):
        for surface_index, surface_probability in enumerate(surface_probabilities):
            allocations = _mean_volume_allocations(
                volume_m3,
                area_m2,
                hypothesis,
                material_names[surface_index],
            )
            weight = float(hypothesis_probability * surface_probability)
            for material, component_volume in allocations.items():
                if material in result:
                    result[material] += weight * component_volume
    return result


def _sample_volume_allocations(
    volume_m3: float,
    area_m2: float,
    hypothesis: InteriorHypothesis,
    surface_material: str,
    rng: np.random.Generator,
) -> dict[str, float]:
    return _volume_allocations(
        volume_m3,
        area_m2,
        hypothesis,
        surface_material,
        wall_thickness_m=hypothesis.wall_thickness_m.sample(rng),
        component_values=[
            (component.volume_fraction.sample(rng), component.fill_ratio.sample(rng))
            for component in hypothesis.components
        ],
    )


def _mean_volume_allocations(
    volume_m3: float,
    area_m2: float,
    hypothesis: InteriorHypothesis,
    surface_material: str,
) -> dict[str, float]:
    return _volume_allocations(
        volume_m3,
        area_m2,
        hypothesis,
        surface_material,
        wall_thickness_m=hypothesis.wall_thickness_m.mode,
        component_values=[
            (component.volume_fraction.mode, component.fill_ratio.mode)
            for component in hypothesis.components
        ],
    )


def _volume_allocations(
    volume_m3: float,
    area_m2: float,
    hypothesis: InteriorHypothesis,
    surface_material: str,
    wall_thickness_m: float,
    component_values: list[tuple[float, float]],
) -> dict[str, float]:
    volume_m3 = max(float(volume_m3), 0.0)
    if hypothesis.structure == "solid" and not hypothesis.components:
        return {surface_material: volume_m3}

    components = list(hypothesis.components)
    shell_components = [
        (component, values)
        for component, values in zip(components, component_values, strict=True)
        if component.role == "shell"
    ]
    inner_components = [
        (component, values)
        for component, values in zip(components, component_values, strict=True)
        if component.role != "shell"
    ]
    shell_volume = float(np.clip(area_m2 * max(wall_thickness_m, 0.0), 0.0, volume_m3))
    if hypothesis.structure == "solid":
        shell_volume = 0.0
    elif shell_volume <= 0 and hypothesis.structure in {"hollow", "layered", "foam_filled"}:
        shell_volume = min(volume_m3 * 0.05, volume_m3)

    allocations: dict[str, float] = {}
    if shell_volume > 0:
        if shell_components:
            weights = _bounded_component_weights(shell_components)
            for (component, values), weight in zip(shell_components, weights, strict=True):
                fill_ratio = float(np.clip(values[1], 0.0, 1.0))
                _add_volume(allocations, component.material, shell_volume * weight * fill_ratio)
        else:
            _add_volume(allocations, surface_material, shell_volume)

    inner_volume = max(volume_m3 - shell_volume, 0.0)
    if hypothesis.structure == "hollow" and not inner_components:
        return allocations
    if hypothesis.structure == "solid" and components:
        inner_volume = volume_m3
    if not inner_components:
        if hypothesis.structure == "solid":
            _add_volume(allocations, surface_material, inner_volume)
        return allocations

    fractions = np.asarray(
        [float(np.clip(values[0], 0.0, 1.0)) for _, values in inner_components],
        dtype=np.float64,
    )
    non_void_total = float(
        sum(fraction for fraction, (component, _) in zip(fractions, inner_components, strict=True) if component.role != "void")
    )
    scale = 1.0 / non_void_total if non_void_total > 1.0 else 1.0
    for fraction, (component, values) in zip(fractions, inner_components, strict=True):
        if component.role == "void" or component.material == "void":
            continue
        fill_ratio = float(np.clip(values[1], 0.0, 1.0))
        _add_volume(allocations, component.material, inner_volume * fraction * scale * fill_ratio)
    return allocations


def _bounded_component_weights(
    components: list[tuple[InteriorComponent, tuple[float, float]]],
) -> np.ndarray:
    values = np.asarray([max(item[1][0], 0.0) for item in components], dtype=np.float64)
    return _normalize_array(values)


def _add_volume(values: dict[str, float], material: str, volume: float) -> None:
    values[material] = values.get(material, 0.0) + max(float(volume), 0.0)


def _normalize_array(values: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    total = float(values.sum())
    if total <= 0:
        return np.full(len(values), 1.0 / len(values), dtype=np.float64)
    return values / total


def _probability_dict(
    material_names: list[str] | tuple[str, ...], probabilities: np.ndarray
) -> dict[str, float]:
    normalized = _normalize_array(probabilities)
    return {name: float(normalized[index]) for index, name in enumerate(material_names)}
