from __future__ import annotations

import numpy as np
import pytest

from tactile3d.experiments.observability import (
    audit_observability,
    normalized_response,
    recover_parameters,
)


def test_distinct_and_duplicate_interventions_have_different_information() -> None:
    report = audit_observability()
    ranks = {item["case"]: item["rank"] for item in report["cases"]}
    assert ranks == {
        "original_only": 1,
        "one_edit": 2,
        "two_distinct_edits": 3,
        "two_duplicate_edits": 2,
        "two_near_duplicate_edits": 3,
    }
    np.testing.assert_allclose(report["masses_kg"], [0.65, 1.0, 1.75])
    np.testing.assert_allclose(report["alternative_masses_kg"], [0.65, 1.0, 1.6])
    assert report["recovered"]["density_kg_m3"] == pytest.approx(500)
    assert report["recovered"]["fill_fraction"] == pytest.approx(0.5)
    assert report["recovered"]["rest_mass_kg"] == pytest.approx(0.4)
    assert report["jacobian_absolute_error"] < 1e-8
    assert not report["rgb_only_test_identifiability_established"]


def test_unknown_envelope_volume_leaves_fill_ambiguous() -> None:
    masses = np.array([0.65, 1.0, 1.75])
    densities = np.array([1200.0, 2700.0])
    first = recover_parameters(masses, densities, 0.001)
    second = recover_parameters(masses, densities, 0.002)
    assert first["occupied_volume_m3"] == second["occupied_volume_m3"]
    assert first["density_kg_m3"] == second["density_kg_m3"]
    assert first["fill_fraction"] == pytest.approx(2 * second["fill_fraction"])


def test_full_rank_can_still_be_poorly_conditioned() -> None:
    report = audit_observability()
    cases = {item["case"]: item for item in report["cases"]}
    assert cases["two_near_duplicate_edits"]["condition_number"] > (
        100 * cases["two_distinct_edits"]["condition_number"]
    )
    noise = report["illustrative_noise"]
    assert (
        noise["one_kg_m3_density_gap_fill_sigma"]
        > 1000 * (noise["distinct_edits_fill_sigma"])
    )


@pytest.mark.parametrize("volume,fraction", [(0.001, 1.0), (0.00073, 0.123)])
def test_zero_remaining_mass_accepts_rounding_at_legal_boundary(
    volume: float, fraction: float
) -> None:
    densities = np.array([500.0, 1200.0, 2700.0])
    result = recover_parameters(volume * fraction * densities, densities[1:], volume)
    assert result["rest_mass_kg"] == pytest.approx(0, abs=1e-14)
    assert result["fill_fraction"] == pytest.approx(fraction)
    assert result["density_kg_m3"] == pytest.approx(densities[0])


@pytest.mark.parametrize(
    "parameters",
    ([0, 0.5, 0.4], [0.5, 0, 0.4], [0.5, 1.1, 0.4], [0.5, 0.5, -1]),
)
def test_invalid_physical_states_are_rejected(parameters: list[float]) -> None:
    with pytest.raises(ValueError):
        normalized_response(np.array(parameters), np.array([1.2]))


def test_degenerate_and_incompatible_observations_are_rejected() -> None:
    with pytest.raises(ValueError, match="密度相同"):
        recover_parameters(np.array([0.65, 1.0, 1.0]), np.array([1200, 1200]), 0.001)
    with pytest.raises(ValueError, match="正材料体积"):
        recover_parameters(np.array([0.65, 1.0, 0.9]), np.array([1200, 2700]), 0.001)
