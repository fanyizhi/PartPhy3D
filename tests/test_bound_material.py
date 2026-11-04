from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tactile3d.experiments.bound_material_demo import (
    ALUMINUM,
    DEFAULT_PART,
    analytic_reference,
    make_chair_fixture,
    run_material_edit,
)
from tactile3d.figures.bound_material import raster_face_ids, shade_surface
from tactile3d.physics.material_binding import (
    BoundSurface,
    MaterialRegion,
    edit_bound_material,
    geometry_signature,
    load_bound_asset,
    save_bound_asset,
)
from tactile3d.physics.part_editing import EditSpec
from tactile3d.physics.part_io import load_object, save_object
from tactile3d.physics.part_state import Material, PhysicalObject


def test_bound_change_matches_independent_reference_and_preserves_other_parts() -> None:
    before, surface = make_chair_fixture()
    after, report = edit_bound_material(
        before, surface, DEFAULT_PART, "bulk", ALUMINUM, 0
    )
    for state, edited in ((before, False), (after, True)):
        for name, value in analytic_reference(edited).items():
            np.testing.assert_allclose(
                state.total.properties()[name], value, atol=1e-12
            )
    assert before.total.mass_kg == pytest.approx(5.12)
    assert after.total.mass_kg == pytest.approx(6.432)
    assert report["delta_mass_kg"] == pytest.approx(1.312)
    assert after.recomputed_part_ids == (DEFAULT_PART,)
    for part in before.parts:
        if part.part_id != DEFAULT_PART:
            assert after.part(part.part_id) is part
    np.testing.assert_array_equal(
        after.part(DEFAULT_PART).fractions, before.part(DEFAULT_PART).fractions
    )
    assert after.part(DEFAULT_PART).materials[0].youngs_modulus_pa == 69e9
    assert geometry_signature(before) == geometry_signature(after)


@pytest.mark.parametrize("azimuth", [-55, -125, 45])
def test_visible_changes_follow_same_region_in_each_view(azimuth: float) -> None:
    before, surface = make_chair_fixture()
    after, _ = edit_bound_material(before, surface, DEFAULT_PART, "bulk", ALUMINUM, 0)
    ids = raster_face_ids(surface, azimuth, 128)
    original, edited = (shade_surface(surface, state, ids) for state in (before, after))
    selected = surface.selected_faces(DEFAULT_PART, "bulk")
    visible = (ids >= 0) & selected[np.maximum(ids, 0)]
    changed = np.any(original != edited, axis=2)
    assert visible.any()
    np.testing.assert_array_equal(changed, visible)


def interior_fixture() -> tuple[PhysicalObject, BoundSurface]:
    before, surface = make_chair_fixture()
    leg = before.part(DEFAULT_PART)
    x = leg.centers_local_m
    core = np.all((x > x.min(0) + 0.005) & (x < x.max(0) - 0.005), axis=1)
    leg = replace(
        leg,
        materials=(leg.materials[0], Material("foam", 40)),
        fractions=np.column_stack((~core, core)).astype(float),
    )
    state = PhysicalObject(
        before.object_id,
        tuple(leg if p.part_id == DEFAULT_PART else p for p in before.parts),
        before.grid_origin_m,
    )
    bound = replace(
        surface,
        regions=(*surface.regions, MaterialRegion(DEFAULT_PART, "filling", 1)),
        geometry_sha256=geometry_signature(state),
    )
    return state, bound


def test_hidden_filling_edit_changes_physics_without_recoloring_outer_surface() -> None:
    before, surface = interior_fixture()
    after, report = edit_bound_material(
        before,
        surface,
        DEFAULT_PART,
        "filling",
        Material("rubber", 900, display_rgba=(230, 90, 40, 255)),
        0,
    )
    assert report["delta_mass_kg"] > 0
    assert report["selected_faces"] == report["changed_faces"] == 0
    np.testing.assert_array_equal(surface.face_rgba(before), surface.face_rgba(after))
    assert (
        after.part(DEFAULT_PART).materials[0] is before.part(DEFAULT_PART).materials[0]
    )


@pytest.mark.parametrize("change", ["instance", "geometry", "occupancy"])
def test_stale_geometry_binding_is_rejected(change: str) -> None:
    state, surface = interior_fixture()
    part = state.part(DEFAULT_PART)
    if change == "instance":
        part = replace(part, instance_id="new_instance")
    elif change == "geometry":
        transform = np.eye(4)
        transform[0, 3] = 0.01
        part = replace(part, local_to_world=transform)
    else:
        part = replace(part, fractions=part.fractions[:, ::-1])
    other = PhysicalObject(
        state.object_id,
        tuple(part if p.part_id == DEFAULT_PART else p for p in state.parts),
        state.grid_origin_m,
    )
    with pytest.raises(ValueError, match="过期"):
        edit_bound_material(other, surface, DEFAULT_PART, "bulk", ALUMINUM, 0)


def test_invalid_region_and_version_roll_back_source() -> None:
    state, surface = make_chair_fixture()
    original = surface.face_rgba(state).copy()
    for region, version in (("missing", 0), ("bulk", 7), ("bulk", None)):
        with pytest.raises(ValueError):
            edit_bound_material(state, surface, DEFAULT_PART, region, ALUMINUM, version)
    assert state.version == 0
    np.testing.assert_array_equal(surface.face_rgba(state), original)


@pytest.mark.parametrize(
    "properties",
    [
        {"youngs_modulus_pa": -3},
        {"poisson_ratio": 0.5},
        {"poisson_ratio": float("nan")},
        {"display_rgba": (0.3, 0, 0, 255)},
        {"display_rgba": (True, 0, 0, 255)},
    ],
)
def test_invalid_physical_or_display_parameters_fail(
    properties: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        Material("invalid", 900, **properties)


def test_bound_asset_and_edit_spec_roundtrip_keep_material_properties(
    tmp_path: Path,
) -> None:
    before, surface = make_chair_fixture()
    after, _ = edit_bound_material(before, surface, DEFAULT_PART, "bulk", ALUMINUM, 0)
    save_bound_asset(after, surface, tmp_path / "asset")
    restored, bound = load_bound_asset(tmp_path / "asset")
    assert restored.part(DEFAULT_PART).materials[0].to_dict() == ALUMINUM.to_dict()
    np.testing.assert_array_equal(bound.face_rgba(restored), surface.face_rgba(after))
    spec = EditSpec(
        DEFAULT_PART, "material_change", material_name="wood", new_material=ALUMINUM
    )
    assert (
        EditSpec.from_dict(spec.to_dict()).new_material.to_dict() == ALUMINUM.to_dict()
    )


def test_legacy_schema_one_still_loads_without_inventing_elastic_properties(
    tmp_path: Path,
) -> None:
    state, _ = make_chair_fixture()
    save_object(state, tmp_path / "old")
    path = tmp_path / "old/state.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = 1
    for part in payload["parts"]:
        for material in part["materials"]:
            for key in ("youngs_modulus_pa", "poisson_ratio", "display_rgba"):
                material.pop(key, None)
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_object(tmp_path / "old")
    assert loaded.total.mass_kg == pytest.approx(state.total.mass_kg)
    assert loaded.parts[0].materials[0].youngs_modulus_pa is None


def test_full_export_replays_from_original_asset(tmp_path: Path) -> None:
    result = run_material_edit(tmp_path / "first", figures=False)
    replay = run_material_edit(
        tmp_path / "replayed",
        tmp_path / "first/before",
        tmp_path / "first/edit_spec.json",
        figures=False,
    )
    assert result["after"] == replay["after"]
    assert (tmp_path / "replayed/physics.csv").is_file()
    with pytest.raises(FileExistsError):
        run_material_edit(tmp_path / "first", figures=False)


def test_corrupt_surface_and_incomplete_labels_are_rejected(tmp_path: Path) -> None:
    state, surface = make_chair_fixture()
    with pytest.raises(ValueError, match="负索引"):
        BoundSurface(
            surface.vertices_m,
            surface.faces,
            np.full(len(surface.faces), -1),
            surface.regions,
            surface.geometry_sha256,
            "test",
        )
    save_bound_asset(state, surface, tmp_path / "asset")
    file = tmp_path / "asset/surface.npz"
    with file.open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="SHA256"):
        load_bound_asset(tmp_path / "asset")
