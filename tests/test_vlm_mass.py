import json

import numpy as np
import pytest
import trimesh

from tactile3d.experiments.vlm_mass import (
    apply_beam_inventory,
    construct,
    density_edit_check,
    evaluate_saved_prediction,
    properties,
    seal_prediction_files,
    validate_hypothesis,
)
from tactile3d.physics.fields import integrate_construction
from tactile3d.physics.part_io import file_sha256


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


def test_role_map_keeps_the_broad_upholstery_role_and_freezes_other_parameters():
    from tactile3d.experiments.vlm_mass import apply_fraction_map

    base = {
        "upholstery": [
            {"role": name, "foam_fraction": 0.9, "hidden_wood_fraction": 0.05}
            for name in ["seat", "upholstery"]
        ],
        "underside": {"kind": "visual_cap", "thickness_mm": 0},
    }
    result = apply_fraction_map(base, {"wood_fraction_by_id": {"R0": 0.02, "R1": 0.07}})
    assert [r["hidden_wood_fraction"] for r in result["upholstery"]] == [0.02, 0.07]
    assert result["underside"] == base["underside"]
    assert base["upholstery"][1]["hidden_wood_fraction"] == 0.05
    for invalid in [{"R0": 0.02}, {"R0": 0.02, "R1": 0.2}, {"R0": False, "R1": 0.01}]:
        with pytest.raises(ValueError):
            apply_fraction_map(base, {"wood_fraction_by_id": invalid})


def test_unresolved_surface_material_cannot_silently_receive_a_density():
    from tactile3d.experiments.vlm_mass import validate_grounded_roles

    mesh = trimesh.creation.box()
    with pytest.raises(ValueError, match="Unresolved surface material"):
        validate_grounded_roles(
            [mesh],
            [
                {
                    "component_id": 0,
                    "surface_material": "mixed",
                    "functional_role": "frame",
                }
            ],
        )


def test_closed_geometry_policy_does_not_fill_the_concave_empty_corner():
    import manifold3d as manifold

    from tactile3d.physics.support_domains import domain_mesh

    solid = manifold.Manifold.cube((2, 1, 1)) + manifold.Manifold.cube((1, 2, 1))
    mesh = domain_mesh(solid)
    parts = [
        {
            "component_id": 0,
            "surface_material": "fabric",
            "functional_role": "upholstery",
        }
    ]
    answer = {
        "upholstery": [
            {"role": "upholstery", "foam_fraction": 0.9, "hidden_wood_fraction": 0.07}
        ],
        "underside": {"kind": "visual_cap", "thickness_mm": 0},
    }
    convex = construct([mesh], parts, answer, 0.1)
    closed = construct([mesh], parts, answer, 0.1, geometry_policy="closed")
    assert convex["union_volume_m3"] == pytest.approx(3.5)
    assert closed["union_volume_m3"] == pytest.approx(3.0)


def test_strict_material_lookup_does_not_assign_screw_density_to_plastic():
    from pathlib import Path

    import yaml

    from tactile3d.experiments.vlm_mass import resolve_native_priors

    info = {
        "caption": "a chair",
        "candidate_materials_density": "(wood: 400-800 kg/m^3);(foam: 20-100 kg/m^3);(fabric: 100-300 kg/m^3);(screws: 7000-9000 kg/m^3)",
        "thickness": "(fabric: 0.2-0.5 cm)",
    }
    cfg = yaml.safe_load(Path("configs/ten_object_pilot.yaml").read_text())
    densities, thickness, sources = resolve_native_priors(
        info, cfg, {"wood", "foam", "fabric", "plastic", "metal"}
    )
    assert densities["plastic"] == 1050.0
    assert densities["metal"] == 8000.0
    assert densities["wood"] == 600.0
    assert thickness == pytest.approx(0.0035)
    assert sources["plastic"]["source"] == "existing_declared_material_library_mean"
    assert sources["metal"]["name"] == "screws"


def test_inward_closed_concave_solid_is_reoriented_without_convex_completion():
    import manifold3d as manifold

    from tactile3d.physics.support_domains import domain_mesh

    mesh = domain_mesh(
        manifold.Manifold.cube((2, 1, 1)) + manifold.Manifold.cube((1, 2, 1))
    )
    mesh.invert()
    vertices, faces = mesh.vertices.copy(), mesh.faces.copy()
    parts = [
        {"component_id": 0, "surface_material": "plastic", "functional_role": "frame"}
    ]
    answer = {"upholstery": [], "underside": {"kind": "visual_cap", "thickness_mm": 0}}
    result = construct([mesh], parts, answer, 0.0035, geometry_policy="closed")
    assert result["union_volume_m3"] == pytest.approx(3.0)
    assert "reversed globally inward winding" in result["parts"][0]["geometry_source"]
    np.testing.assert_array_equal(mesh.vertices, vertices)
    np.testing.assert_array_equal(mesh.faces, faces)


def test_whole_upholstered_frame_cannot_be_accepted_as_a_leg():
    from tactile3d.experiments.vlm_mass import validate_grounded_roles

    leg = trimesh.creation.box(extents=[0.04, 0.04, 0.3])
    body = trimesh.creation.box(extents=[0.8, 1.0, 0.7])
    body.apply_translation([0, 0, 0.6])
    validate_grounded_roles(
        [leg, body],
        [
            {"component_id": 0, "functional_role": "leg"},
            {"component_id": 1, "functional_role": "frame"},
        ],
    )
    with pytest.raises(ValueError, match="rebind components"):
        validate_grounded_roles(
            [leg, body],
            [
                {"component_id": 0, "functional_role": "leg"},
                {"component_id": 1, "functional_role": "leg"},
            ],
        )
    with pytest.raises(ValueError, match="every native component"):
        validate_grounded_roles(
            [leg, body], [{"component_id": 0, "functional_role": "leg"}]
        )


def test_saved_prediction_evaluation_resolves_scene_suffix_and_rejects_tampering(
    tmp_path,
):
    output, root = tmp_path / "run", tmp_path / "data"
    output.mkdir()
    root.mkdir()
    path = output / "predictions.json"
    path.write_text(json.dumps([{"scene_id": "ASIN_MARKET", "mass_kg": 2.0}]))
    (output / "prediction_seal.json").write_text(
        json.dumps({"predictions.json": file_sha256(path)})
    )
    (root / "filtered_product_weights.json").write_text(json.dumps({"ASIN": 2.5}))
    report = evaluate_saved_prediction(output, root)
    assert report["target_mass_kg"] == 2.5
    assert report["metrics"]["ade_kg"] == pytest.approx(0.5)
    seal_prediction_files(output)
    evaluate_saved_prediction(output, root)
    seal = json.loads((output / "prediction_seal.json").read_text())
    assert "report.json" not in seal and "prediction_seal.json" not in seal
    path.write_text(json.dumps([{"scene_id": "ASIN_MARKET", "mass_kg": 2.5}]))
    with pytest.raises(ValueError, match="seal mismatch"):
        evaluate_saved_prediction(output, root)


def test_metal_fastener_retains_its_own_density_instead_of_wood_density():
    mesh = trimesh.creation.box(extents=[0.01] * 3)
    parts = [
        {"component_id": 0, "surface_material": "metal", "functional_role": "fastener"}
    ]
    answer = {"upholstery": [], "underside": {"kind": "visual_cap", "thickness_mm": 0}}
    partition = construct([mesh], parts, answer, 0.0035)
    result = properties(partition, parts, answer, {"metal": 7500, "wood": 500})
    assert result["mass_kg"] == pytest.approx(0.0075)


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
    assert density_edit_check(
        partition, parts, answer, {"wood": 500, "foam": 40, "fabric": 200}, result
    )["edit_status"].startswith("not applicable")


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
