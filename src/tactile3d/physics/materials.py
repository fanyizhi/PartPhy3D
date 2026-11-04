from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


@dataclass(frozen=True)
class ScalarPrior:
    mean: float
    std: float
    min: float
    max: float

    def sample(self, rng: np.random.Generator) -> float:
        mean = max(float(self.mean), 1e-12)
        std = max(float(self.std), 0.0)
        sigma2 = np.log1p((std / mean) ** 2)
        mu = np.log(mean) - 0.5 * sigma2
        value = rng.lognormal(mu, np.sqrt(sigma2)) if std > 0 else mean
        return float(np.clip(value, float(self.min), float(self.max)))

    @property
    def second_moment(self) -> float:
        return float(self.mean**2 + self.std**2)


@dataclass(frozen=True)
class MaterialPrior:
    name: str
    density_kg_m3: ScalarPrior
    youngs_modulus_pa: ScalarPrior


def load_material_priors(path: str | Path) -> list[MaterialPrior]:
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    priors: list[MaterialPrior] = []
    for name, values in raw.get("materials", {}).items():
        priors.append(
            MaterialPrior(
                name=name,
                density_kg_m3=_scalar_prior(values["density_kg_m3"]),
                youngs_modulus_pa=_scalar_prior(values["youngs_modulus_pa"]),
            )
        )
    if not priors:
        raise ValueError(f"No material priors found in {path}")
    return priors


def _scalar_prior(values: dict) -> ScalarPrior:
    return ScalarPrior(
        mean=float(values["mean"]),
        std=float(values["std"]),
        min=float(values["min"]),
        max=float(values["max"]),
    )


def material_probabilities(part_features: dict, priors: list[MaterialPrior], temperature: float) -> dict[str, float]:
    brightness = float(part_features.get("brightness", 0.5))
    roughness = float(part_features.get("roughness", 0.15))
    compactness = float(part_features.get("compactness", 0.5))

    scores = []
    for prior in priors:
        name = prior.name
        score = 0.0
        if name in {"steel", "aluminum"}:
            score += 1.2 * brightness + 0.4 * compactness - 0.7 * roughness
        elif name == "rubber":
            score += 0.7 * (1.0 - brightness) + 0.7 * compactness + 0.2 * roughness
        elif name == "foam":
            score += 1.1 * roughness + 0.4 * (1.0 - compactness)
        elif name == "wood":
            score += 0.5 * roughness + 0.2 * brightness
        else:
            score += 0.3 * compactness
        scores.append(score)

    logits = np.asarray(scores, dtype=np.float64) / max(float(temperature), 1e-6)
    logits -= logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()
    return {prior.name: float(prob) for prior, prob in zip(priors, probs)}


def sample_material(
    probs: dict[str, float],
    priors: list[MaterialPrior],
    rng: np.random.Generator,
) -> tuple[str, float, float]:
    names = [p.name for p in priors]
    p = np.asarray([probs[name] for name in names], dtype=np.float64)
    p /= p.sum()
    idx = int(rng.choice(len(priors), p=p))
    prior = priors[idx]
    return (
        prior.name,
        prior.density_kg_m3.sample(rng),
        prior.youngs_modulus_pa.sample(rng),
    )
