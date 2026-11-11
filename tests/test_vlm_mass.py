import numpy as np
import pytest

from tactile3d.experiments.vlm_mass import integrate_construction


def row(core, thickness=0.0, region=0):
    return {
        "region_id": region,
        "surface_material": "cover",
        "wall_thickness_m": thickness,
        "core": core,
    }


POINTS = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
OWNERS = np.array([0, 0])
DENSITIES = {"cover": 1000.0, "foam": 40.0, "steel": 7850.0}


def test_solid_and_hollow_use_distinct_material_volumes():
    solid = integrate_construction(
        POINTS, OWNERS, 0.1, [row([{"material": "cover", "fraction": 1.0}])], DENSITIES
    )
    hollow = integrate_construction(POINTS, OWNERS, 0.1, [row([], 0.01)], DENSITIES)
    assert solid["moments"]["mass_kg"] == pytest.approx(2.0)
    assert hollow["moments"]["mass_kg"] == pytest.approx(0.2)
    assert hollow["occupied_volume_m3"] == pytest.approx(0.0002)


def test_tiny_screws_fraction_is_not_an_entire_metal_region():
    phases = [
        {"material": "steel", "fraction": 0.001},
        {"material": "foam", "fraction": 0.999},
    ]
    result = integrate_construction(POINTS, OWNERS, 0.1, [row(phases)], DENSITIES)
    assert result["moments"]["mass_kg"] == pytest.approx(
        0.002 * (0.001 * 7850 + 0.999 * 40)
    )
    assert sum(result["material_volumes_m3"].values()) == pytest.approx(0.002)


@pytest.mark.parametrize(
    "phases",
    [
        [{"material": "steel", "fraction": 1.1}],
        [{"material": "steel", "fraction": -0.1}],
        [{"material": "steel", "fraction": float("nan")}],
        [{"material": "unsupported", "fraction": 0.5}],
    ],
)
def test_invalid_core_is_rejected_instead_of_clipped(phases):
    with pytest.raises(ValueError, match="Core fractions"):
        integrate_construction(POINTS, OWNERS, 0.1, [row(phases)], DENSITIES)


def test_missing_or_duplicate_regions_are_rejected():
    with pytest.raises(ValueError, match="every occupied region"):
        integrate_construction(POINTS, np.array([0, 1]), 0.1, [row([])], DENSITIES)
    with pytest.raises(ValueError, match="every occupied region"):
        integrate_construction(
            POINTS, np.array([0, 1]), 0.1, [row([]), row([])], DENSITIES
        )


def test_material_library_thickness_is_a_paired_parameter_source_ablation():
    direct = integrate_construction(POINTS, OWNERS, 0.1, [row([], 0.02)], DENSITIES)
    library = integrate_construction(
        POINTS, OWNERS, 0.1, [row([], 0.02)], DENSITIES, {"cover": 0.005}
    )
    assert direct["moments"]["mass_kg"] == pytest.approx(0.4)
    assert library["moments"]["mass_kg"] == pytest.approx(0.1)


def test_two_centimeter_cloth_is_rejected_against_material_bounds():
    with pytest.raises(ValueError, match="outside material bounds"):
        integrate_construction(
            POINTS,
            OWNERS,
            0.1,
            [row([], 0.02)],
            DENSITIES,
            thickness_bounds={"cover": [0.002, 0.005]},
        )
    result = integrate_construction(
        POINTS,
        OWNERS,
        0.1,
        [row([], 0.003)],
        DENSITIES,
        thickness_bounds={"cover": [0.002, 0.005]},
    )
    assert result["moments"]["mass_kg"] == pytest.approx(0.06)
