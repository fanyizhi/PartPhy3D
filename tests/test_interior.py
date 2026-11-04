from pathlib import Path

import numpy as np

from tactile3d.physics.interior import expected_material_volumes, sample_part_physics
from tactile3d.physics.materials import load_material_priors
from tactile3d.vision.vlm_interior import parse_part_proposal


def _one_hot(names: list[str], selected: str) -> np.ndarray:
    result = np.zeros(len(names), dtype=np.float64)
    result[names.index(selected)] = 1.0
    return result


def test_hollow_shell_is_lighter_than_solid() -> None:
    priors = load_material_priors(Path("configs/material_priors.yaml"))
    names = [prior.name for prior in priors]
    plastic = _one_hot(names, "plastic")
    solid = parse_part_proposal(
        {
            "surface_materials": [{"material": "plastic", "probability": 1.0}],
            "interior_hypotheses": [
                {"name": "solid", "structure": "solid", "probability": 1.0}
            ],
        },
        0,
        names,
        plastic,
        "test",
    )
    hollow = parse_part_proposal(
        {
            "surface_materials": [{"material": "plastic", "probability": 1.0}],
            "interior_hypotheses": [
                {
                    "name": "thin shell",
                    "structure": "hollow",
                    "probability": 1.0,
                    "wall_thickness_mm": 2.0,
                    "components": [
                        {
                            "material": "plastic",
                            "role": "shell",
                            "volume_fraction": 1.0,
                            "fill_ratio": 1.0,
                        }
                    ],
                }
            ],
        },
        0,
        names,
        plastic,
        "test",
    )
    rng = np.random.default_rng(4)
    solid_samples = [
        sample_part_physics(0.01, 0.3, plastic, solid, priors, rng).mass_kg
        for _ in range(256)
    ]
    hollow_samples = [
        sample_part_physics(0.01, 0.3, plastic, hollow, priors, rng).mass_kg
        for _ in range(256)
    ]
    assert np.mean(hollow_samples) < 0.1 * np.mean(solid_samples)


def test_layered_upholstery_assigns_hidden_foam_volume() -> None:
    priors = load_material_priors(Path("configs/material_priors.yaml"))
    names = [prior.name for prior in priors]
    leather = _one_hot(names, "leather")
    proposal = parse_part_proposal(
        {
            "object_category": "sofa",
            "surface_materials": [{"material": "leather", "probability": 1.0}],
            "interior_hypotheses": [
                {
                    "name": "leather over foam",
                    "structure": "layered",
                    "probability": 3.0,
                    "wall_thickness_mm": {"low": 1.0, "mode": 1.5, "high": 2.0},
                    "components": [
                        {
                            "material": "leather",
                            "role": "shell",
                            "volume_fraction": 1.0,
                            "fill_ratio": 1.0,
                        },
                        {
                            "material": "polyurethane foam",
                            "role": "core",
                            "volume_fraction": 1.0,
                            "fill_ratio": 1.0,
                        },
                    ],
                }
            ],
        },
        2,
        names,
        leather,
        "test",
    )
    expected = expected_material_volumes(0.25, 2.0, leather, proposal, names)
    assert proposal.object_category == "sofa"
    assert proposal.hypotheses[0].probability == 1.0
    assert expected["leather"] == 0.003
    assert expected["foam"] == 0.247
    assert abs(sum(expected.values()) - 0.25) < 1e-12
