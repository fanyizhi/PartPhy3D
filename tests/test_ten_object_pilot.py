from pathlib import Path

import numpy as np
import pytest
import trimesh
import yaml

from tactile3d.experiments.ten_objects import (
    construction_profile,
    geometric_support,
    materials_from_info,
    metrics,
    parse_ranges,
    predict_physics,
)


def config():
    return yaml.safe_load(Path("configs/ten_object_pilot.yaml").read_text())


def test_cached_ranges_accept_fixed_values_and_ranges_and_reject_bad_order():
    parsed = parse_ranges("(steel: 7850 kg/m^3);(wood: 300-700 kg/m^3)")
    assert parsed == {"steel": (7850, 7850), "wood": (300, 700)}
    with pytest.raises(ValueError):
        parse_ranges("(wood: 700-300 kg/m^3)")


def test_missing_internal_material_is_declared_fallback_not_a_cached_prediction():
    info = {"caption": "a blue ottoman with wooden legs",
            "candidate_materials_density": "(fabric: 100-300 kg/m^3);(wood: 300-700 kg/m^3)",
            "thickness": "(fabric: .2-.5 cm);(wood: 2-4 cm)"}
    materials, profile = materials_from_info(info, config())
    assert profile == "seating"
    foam = next(m for m in materials if m["kind"] == "foam")
    assert foam["source"] == "declared_global_fallback_prior"
    assert construction_profile("a usb hub") == "electronics"


def test_component_support_does_not_fill_gap_between_two_disconnected_solids():
    a = trimesh.creation.box(extents=[.2, .2, .2])
    b = a.copy()
    b.apply_translation([1, 0, 0])
    mesh = trimesh.util.concatenate([a, b])
    selected, count = geometric_support(mesh, np.array([[0., 0, 0], [.5, 0, 0], [1, 0, 0]]),
                                         .01, config())
    assert count == 2
    assert selected.tolist() == [True, False, True]


def test_mass_samples_match_disjoint_field_integration_without_labels():
    cfg = config()
    cfg["samples"] = 128
    info = {"caption": "a usb hub", "candidate_materials_density": "(plastic: 1000-2000 kg/m^3)",
            "thickness": "(plastic: 0.2-0.5 cm)"}
    materials, profile = materials_from_info(info, cfg)
    points = np.array([[0., 0, 0], [.01, 0, 0], [0, .01, 0], [.01, .01, 0]])
    fields = predict_physics(points, np.array([0, 0, 1, 1]),
        np.array([[.1, .1, .1], [.2, .2, .2]]), materials, profile, .01, cfg)
    assert fields["moments"]["mass_kg"] == pytest.approx(fields["part_samples"].sum(axis=0).mean())
    assert np.all((fields["occupied"] >= 0) & (fields["occupied"] <= 1))
    assert np.isfinite(fields["density"]).all()


def test_metrics_are_object_level_and_percentage_is_explicit():
    summary = metrics([{"mass_kg": 2, "target_mass_kg": 1, "interval_hit": True},
                       {"mass_kg": 1, "target_mass_kg": 2, "interval_hit": False}])
    assert summary["ADE_kg"] == 1
    assert summary["APE_percent"] == 75
    assert summary["MnRE"] == .5
    assert metrics([]) is None


def test_artificial_region_boundary_does_not_create_an_extra_physical_shell():
    cfg = config()
    cfg["samples"] = 128
    cfg["construction"]["generic_core_occupancy"] = [.25-1e-12, .25, .25+1e-12]
    grid = np.stack(np.meshgrid(np.arange(6), np.arange(6), np.arange(6),
                                indexing="ij"), axis=-1).reshape(-1, 3)*.01
    materials = [{"name": "wood", "kind": "wood", "density": [1000, 1000],
                  "thickness_m": [.002, .002]}]
    whole = predict_physics(grid, np.zeros(len(grid), int), np.array([[.5, .5, .5]]),
                            materials, "generic", .01, cfg)
    split = predict_physics(grid, (grid[:, 0] >= .025).astype(int),
        np.array([[.5, .5, .5], [.5, .5, .5]]), materials, "generic", .01, cfg)
    np.testing.assert_allclose(whole["density"], split["density"], rtol=1e-9)
    assert whole["moments"]["mass_kg"] == pytest.approx(split["moments"]["mass_kg"], rel=1e-9)
