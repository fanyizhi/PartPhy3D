from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tactile3d.experiments.part_edit_demo import make_example
from tactile3d.mesh.state_surface import part_surface
from tactile3d.physics.part_editing import (
    EditSpec,
    apply_edit,
    verify_full_recompute,
)
from tactile3d.physics.part_io import (
    import_development_field,
    load_object,
    save_object,
)
from tactile3d.physics.part_state import (
    Interface,
    Material,
    PartPhysicalState,
    PhysicalObject,
    bind_surface_points,
)


@pytest.fixture
def example() -> tuple[PhysicalObject, PartPhysicalState]:
    return make_example()


@pytest.mark.parametrize("translation", [(1.0, 2.0, 3.0), (2.0, -3.0, 4.0)])
def test_box_mass_full_inertia_rotation_and_translation(
    example: tuple[PhysicalObject, PartPhysicalState],
    translation: tuple[float, float, float],
) -> None:
    source, _ = example
    body = source.part("body")
    mass = 600 * 0.20 * 0.12 * 0.04
    inertia = (
        mass / 12 * np.diag([0.12**2 + 0.04**2, 0.20**2 + 0.04**2, 0.20**2 + 0.12**2])
    )
    assert body.moments().mass_kg == pytest.approx(mass)
    np.testing.assert_allclose(
        body.moments().properties()["inertia_kg_m2"], inertia, atol=1e-14
    )
    rotation = np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    moved = replace(body, local_to_world=transform)
    result = PhysicalObject("moved", (moved,), source.grid_origin_m).total.properties()
    np.testing.assert_allclose(result["center_of_mass_m"], translation, atol=1e-14)
    np.testing.assert_allclose(
        result["inertia_kg_m2"], rotation @ inertia @ rotation.T, atol=2e-14
    )


def test_material_edit_preserves_occupancy_and_reuses_only_unchanged_cache(
    example: tuple[PhysicalObject, PartPhysicalState],
) -> None:
    source, _ = example
    original = source.total.properties()
    result = apply_edit(
        source,
        EditSpec(
            "leg",
            "material_change",
            expected_part_version=0,
            material_name="wood",
            new_material=Material("steel", 7850),
        ),
    )
    assert result.state.total.mass_kg == pytest.approx(0.576 + 0.00032 * 7850)
    assert result.state._cache[0] is source._cache[0]
    assert result.state.parts[0] is source.parts[0]
    assert result.state.recomputed_part_ids == ("leg",)
    np.testing.assert_array_equal(
        result.state.part("leg").fractions, source.part("leg").fractions
    )
    assert source.total.properties() == original
    assert source.part("leg").version == 0
    assert result.state.part("leg").version == 1
    verify_full_recompute(result.state)


def test_hollow_box_matches_analytic_volume_and_inertia(
    example: tuple[PhysicalObject, PartPhysicalState],
) -> None:
    source, _ = example
    result = apply_edit(
        source,
        EditSpec(
            "leg",
            "construction_change",
            material_name="wood",
            construction="hollow_box",
            wall_thickness_m=0.01,
        ),
    )
    part = result.state.part("leg")
    outer, inner = np.array([0.04, 0.04, 0.20]), np.array([0.02, 0.02, 0.18])
    outer_mass, inner_mass = 650 * np.prod(outer), 650 * np.prod(inner)
    expected_mass = outer_mass - inner_mass
    expected_inertia = outer_mass / 12 * (
        np.sum(outer**2) * np.eye(3) - np.diag(outer**2)
    ) - inner_mass / 12 * (np.sum(inner**2) * np.eye(3) - np.diag(inner**2))
    assert part.moments().mass_kg == pytest.approx(expected_mass)
    np.testing.assert_allclose(
        part.moments().properties()["inertia_kg_m2"], expected_inertia, atol=1e-14
    )
    mesh = part_surface(result.state, "leg")
    assert mesh.is_watertight
    assert mesh.volume == pytest.approx(expected_mass / 650)
    verify_full_recompute(result.state)


def test_replace_keeps_slot_changes_instance_and_has_analytic_mass(
    example: tuple[PhysicalObject, PartPhysicalState],
) -> None:
    source, donor = example
    result = apply_edit(source, EditSpec("leg", "replace_part"), donor)
    assert result.state.part("leg").instance_id == donor.instance_id
    assert result.state.part("leg").part_id == "leg"
    assert result.state.total.mass_kg == pytest.approx(0.576 + 0.000104 * 2700)
    assert result.report["interface_checks"][0]["position_m"] < 1e-12
    mesh = part_surface(result.state, "leg")
    assert mesh.is_watertight
    assert mesh.volume == pytest.approx(0.000104)
    verify_full_recompute(result.state)


def test_failure_is_atomic_for_stale_version_and_bad_interface(
    example: tuple[PhysicalObject, PartPhysicalState],
) -> None:
    source, donor = example
    before = source.total.properties()
    with pytest.raises(ValueError, match="版本过期"):
        apply_edit(
            source, EditSpec("leg", "replace_part", expected_part_version=5), donor
        )
    interface = Interface(
        "mount", donor.interfaces[0].frame_local, np.array([0.03, 0.04])
    )
    with pytest.raises(ValueError, match="不兼容"):
        apply_edit(
            source,
            EditSpec("leg", "replace_part"),
            replace(donor, interfaces=(interface,)),
        )
    assert source.total.properties() == before
    assert source.version == 0


def test_replacement_rejects_overlap_and_arbitrary_angle(
    example: tuple[PhysicalObject, PartPhysicalState],
) -> None:
    source, donor = example
    points = np.vstack([source.part("leg").centers_local_m, [0.005, 0.005, 0.105]])
    colliding = replace(
        donor,
        centers_local_m=points,
        materials=(Material("steel", 7850),),
        fractions=np.ones((len(points), 1)),
    )
    with pytest.raises(ValueError, match="重叠"):
        apply_edit(source, EditSpec("leg", "replace_part"), colliding)
    frame = np.array(donor.interfaces[0].frame_local)
    angle = np.pi / 4
    frame[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    rotated = replace(
        donor, interfaces=(Interface("mount", frame, np.array([0.04, 0.04])),)
    )
    with pytest.raises(ValueError, match="重体素化"):
        apply_edit(source, EditSpec("leg", "replace_part"), rotated)


def test_hollow_rejects_subvoxel_wall_and_nonbox_geometry(
    example: tuple[PhysicalObject, PartPhysicalState],
) -> None:
    source, _ = example
    with pytest.raises(ValueError, match="整数体素"):
        apply_edit(
            source,
            EditSpec(
                "leg",
                "construction_change",
                material_name="wood",
                construction="hollow_box",
                wall_thickness_m=0.005,
            ),
        )
    old = source.part("leg")
    incomplete = replace(
        old, centers_local_m=old.centers_local_m[1:], fractions=old.fractions[1:]
    )
    state = PhysicalObject("nonbox", (incomplete,), source.grid_origin_m)
    with pytest.raises(ValueError, match="完整矩形"):
        apply_edit(
            state,
            EditSpec(
                "leg",
                "construction_change",
                material_name="wood",
                construction="hollow_box",
                wall_thickness_m=0.01,
            ),
        )


def test_read_only_copy_and_input_validation(
    example: tuple[PhysicalObject, PartPhysicalState],
) -> None:
    source, _ = example
    part = source.part("leg")
    with pytest.raises(ValueError):
        part.fractions[0, 0] = 0
    with pytest.raises(ValueError, match="占用"):
        replace(part, fractions=part.fractions * 0.5)
    with pytest.raises(ValueError, match="密度"):
        Material("invalid", -1)
    duplicate = replace(part, part_id="other", instance_id="other")
    with pytest.raises(ValueError, match="重叠"):
        PhysicalObject("bad", (part, duplicate), source.grid_origin_m)


def test_part_scoped_binding_rejects_cross_part_nearest_neighbor(
    example: tuple[PhysicalObject, PartPhysicalState],
) -> None:
    source, _ = example
    points = source.part("leg").centers_world_m[:3]
    indices, distance = bind_surface_points(source.part("leg"), points, 0.01)
    assert indices.tolist() == [0, 1, 2]
    np.testing.assert_array_equal(distance, 0)
    with pytest.raises(ValueError, match="绑定距离"):
        bind_surface_points(source.part("leg"), np.array([[0.095, 0.055, 0.235]]), 0.01)


def test_storage_round_trip_and_tamper_detection(
    example: tuple[PhysicalObject, PartPhysicalState],
    tmp_path: Path,
) -> None:
    source, _ = example
    folder = tmp_path / "state"
    save_object(source, folder)
    result = load_object(folder)
    assert result.total.properties() == source.total.properties()
    assert result.part("leg").interfaces[0].name == "mount"
    with pytest.raises(FileExistsError):
        save_object(source, folder)
    with (folder / "cells.npz").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="校验失败"):
        load_object(folder)


def test_strict_edit_spec_and_donor_relative_path(tmp_path: Path) -> None:
    path = tmp_path / "edit.json"
    path.write_text(
        json.dumps(
            {
                "target_part_id": "leg",
                "operation": "replace_part",
                "donor_state_ref": "donor",
            }
        ),
        encoding="utf-8",
    )
    assert EditSpec.from_json(path).donor_state_ref == str(
        (tmp_path / "donor").resolve()
    )
    path.write_text(
        '{"target_part_id":"leg","operation":"replace_part","typo":1}', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="未知字段"):
        EditSpec.from_json(path)


def test_import_does_not_multiply_occupancy_twice(tmp_path: Path) -> None:
    file = tmp_path / "field.npz"
    np.savez(
        file,
        centers_m=np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]),
        part_ids=np.array([0, 1]),
        pitch_m=0.1,
        density_mean_kg_m3=np.array([100.0, 300.0]),
        occupied_fraction=np.array([0.2, 0.5]),
    )
    state = import_development_field(file)
    assert state.total.mass_kg == pytest.approx(0.4)
    np.testing.assert_allclose(
        state.total.properties()["center_of_mass_m"], [0.075, 0, 0]
    )
    assert state.part("region_00").construction == "unresolved_effective_density_only"


@pytest.mark.parametrize(
    "invalid",
    [
        {"target_part_id": None, "operation": "replace_part"},
        {
            "target_part_id": "leg",
            "operation": "replace_part",
            "expected_part_version": True,
        },
        {
            "target_part_id": "leg",
            "operation": "material_change",
            "material_name": "wood",
            "new_material": {"name": "steel"},
        },
    ],
)
def test_invalid_edit_types_have_clear_errors(invalid: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        EditSpec(**invalid)


def test_matching_declared_interface_without_material_is_rejected(
    example: tuple[PhysicalObject, PartPhysicalState],
) -> None:
    source, donor = example
    fractions = np.array(donor.fractions)
    fractions[donor.centers_local_m[:, 2] > 0.08] = [0, 1]
    unsupported = replace(donor, fractions=fractions)
    with pytest.raises(ValueError, match="材料支持"):
        apply_edit(source, EditSpec("leg", "replace_part"), unsupported)
