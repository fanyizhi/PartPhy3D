import numpy as np
import pytest
import trimesh

from tactile3d.experiments.vlm_mass import (
    apply_beam_inventory,
    construct,
    properties,
    validate_hypothesis,
)
from tactile3d.physics.fields import integrate_construction


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


def beam_case(count=2, length=1.0, section=10):
    base = {
        "upholstery": [
            {"role": "seat", "foam_fraction": 0.8, "hidden_wood_fraction": 0.15}
        ],
        "underside": {"kind": "manufactured_panel", "thickness_mm": 15},
    }
    proposal = {
        "upholstery": [
            {
                "role": "seat",
                "beams": [
                    {
                        "count": count,
                        "length_m": length,
                        "section_component_id": section,
                    }
                ],
            }
        ]
    }
    return (
        base,
        proposal,
        {"seat": {"diagonal_m": 2.0, "core_volume_m3": 0.1}},
        {10: {"area_m2": 0.001}},
    )


def test_beam_volume_uses_observed_section_and_preserves_other_parameters():
    base, proposal, roles, sections = beam_case()
    answer = apply_beam_inventory(base, proposal, roles, sections)
    assert answer["upholstery"][0]["hidden_wood_fraction"] == pytest.approx(0.02)
    assert answer["upholstery"][0]["foam_fraction"] == 0.8
    assert answer["underside"] == base["underside"]
    assert base["upholstery"][0]["hidden_wood_fraction"] == 0.15


@pytest.mark.parametrize(
    "count,length,section",
    [
        (1.5, 1, 10),
        (-1, 1, 10),
        (2, 3, 10),
        (2, float("nan"), 10),
        (2, 1, 99),
        (100, 1, 10),
    ],
)
def test_invalid_beam_inventory_is_rejected_without_clipping(count, length, section):
    with pytest.raises(ValueError):
        apply_beam_inventory(*beam_case(count, length, section))


def test_empty_beam_inventory_is_explicit_absence_hypothesis():
    base, proposal, roles, sections = beam_case()
    proposal["upholstery"][0]["beams"] = []
    assert (
        apply_beam_inventory(base, proposal, roles, sections)["upholstery"][0][
            "hidden_wood_fraction"
        ]
        == 0
    )


@pytest.mark.parametrize("foam,wood", [(0.9, 0.2), (-0.1, 0.1), (float("nan"), 0)])
def test_component_core_rejects_invalid_phase_volumes(foam, wood):
    answer = {
        "upholstery": [
            {"role": "seat", "foam_fraction": foam, "hidden_wood_fraction": wood}
        ],
        "underside": {"kind": "manufactured_panel", "thickness_mm": 15},
    }
    with pytest.raises(ValueError, match="Core fractions"):
        validate_hypothesis(answer, ["seat"])


@pytest.mark.parametrize(
    "kind,thickness", [("visual_cap", 15), ("manufactured_panel", 0)]
)
def test_planar_component_cannot_silently_gain_or_lose_material(kind, thickness):
    answer = {
        "upholstery": [
            {"role": "seat", "foam_fraction": 0.8, "hidden_wood_fraction": 0.05}
        ],
        "underside": {"kind": kind, "thickness_mm": thickness},
    }
    with pytest.raises(ValueError, match="panel requires thickness"):
        validate_hypothesis(answer, ["seat"])


def test_qem_removed_component_loses_its_mass_without_dropping_the_object():
    first, second = trimesh.creation.box(extents=[0.2] * 3), trimesh.creation.box(
        extents=[0.2] * 3
    )
    second.apply_translation([1, 0, 0])
    parts = [
        {"component_id": i, "surface_material": "wood", "functional_role": "leg"}
        for i in range(2)
    ]
    answer = {"upholstery": [], "underside": {"kind": "visual_cap", "thickness_mm": 0}}
    missing = [first, trimesh.Trimesh()]
    with pytest.raises(ValueError, match="Missing reference support"):
        construct(missing, parts, answer, 0.0035)
    before = construct([first, second], parts, answer, 0.0035)
    after = construct(missing, parts, answer, 0.0035, allow_removed=True)
    density = {"wood": 500.0}
    assert properties(before, parts, answer, density)["mass_kg"] == pytest.approx(8.0)
    assert properties(after, parts, answer, density)["mass_kg"] == pytest.approx(4.0)
    assert after["excluded_geometry_hypotheses"][0]["component_id"] == 1


def test_planar_upholstery_has_fabric_volume_without_a_fictitious_foam_core():
    patch = trimesh.Trimesh(
        [[0, 0, 0], [2, 0, 0], [2, 1, 0], [0, 1, 0]],
        [[0, 1, 2], [0, 2, 3]],
        process=False,
    )
    parts = [
        {"component_id": 0, "surface_material": "fabric", "functional_role": "seat"}
    ]
    answer = {
        "upholstery": [
            {"role": "seat", "foam_fraction": 0.9, "hidden_wood_fraction": 0.1}
        ],
        "underside": {"kind": "visual_cap", "thickness_mm": 0},
    }
    partition = construct([patch], parts, answer, 0.0035)
    result = properties(
        partition, parts, answer, {"wood": 500, "foam": 40, "fabric": 200}
    )
    assert result["mass_kg"] == pytest.approx(1.4)
    assert partition["parts"][0]["core"]["volume_m3"] == 0


def test_flattened_wood_leg_does_not_inherit_the_underside_panel_thickness():
    box = trimesh.creation.box(extents=[0.2] * 3)
    flat_leg = trimesh.Trimesh(
        [[1, 0, 0], [1.1, 0, 0], [1.1, 0.1, 0]], [[0, 1, 2]], process=False
    )
    parts = [
        {"component_id": i, "surface_material": "wood", "functional_role": "leg"}
        for i in range(2)
    ]
    answer = {
        "upholstery": [],
        "underside": {"kind": "manufactured_panel", "thickness_mm": 15},
    }
    partition = construct([box, flat_leg], parts, answer, 0.0035, allow_removed=True)
    result = properties(partition, parts, answer, {"wood": 500})
    assert result["mass_kg"] == pytest.approx(4.0)
    assert "three-dimensional" in partition["excluded_geometry_hypotheses"][0]["reason"]


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
