from pathlib import Path

from tactile3d.physics.materials import load_material_priors, material_probabilities


def test_material_priors_load():
    priors = load_material_priors(Path("configs/material_priors.yaml"))
    assert len(priors) >= 4
    assert all(p.density_kg_m3.mean > 0 for p in priors)


def test_material_probabilities_sum_to_one():
    priors = load_material_priors(Path("configs/material_priors.yaml"))
    probs = material_probabilities({"brightness": 0.5, "roughness": 0.1}, priors, temperature=1.0)
    assert abs(sum(probs.values()) - 1.0) < 1e-6
