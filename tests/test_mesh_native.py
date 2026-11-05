from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tactile3d.experiments.mesh_native_demo import (
    EDIT_MATERIAL,
    analytic_box,
    check_example,
    make_mesh_example,
    run_mesh_demo,
)
from tactile3d.physics.mesh_asset_io import (
    import_mesh_regions,
    load_mesh_asset,
    save_mesh_asset,
)
from tactile3d.physics.mesh_native import (
    ClosedMesh,
    MeshPhysicalObject,
    MeshRegion,
    closed_shell,
    verify_mesh_recompute,
)
from tactile3d.physics.part_state import Material
from tactile3d.research.construction import mass_forward
from tactile3d.research.data import analytic_volumes, make_dataset
from tactile3d.research.learning import state_loss
from tactile3d.research.mesh_templates import (
    box_domain,
    mesh_candidate_volumes,
    mesh_state_from_logits,
)
from tactile3d.research.native_learning import run_native_learning


def test_solid_cavity_layered_and_material_edit_match_independent_boxes() -> None:
    states = list(make_mesh_example())
    edited, report = states[-1].edit_material("body", "shell", EDIT_MATERIAL, 0)
    assert max(check_example([*states, edited]).values()) < 1e-12
    assert report["delta_mass_kg"] == pytest.approx(15.3408)
    assert report["reintegrated_meshes"] == 0
    assert max(verify_mesh_recompute(edited).values()) == 0
    for old, new in zip(states[-1].regions, edited.regions, strict=True):
        assert old.domain is new.domain
        assert old.domain.moments is new.domain.moments
        assert old.fill_fraction == new.fill_fraction


def test_asymmetric_rotated_assembly_has_correct_com_and_full_inertia() -> None:
    rotation = Rotation.from_euler("xyz", [0.4, -0.3, 0.7]).as_matrix()
    sizes = [(0.2, 0.3, 0.4), (0.1, 0.2, 0.3)]
    centers = np.array([[0.2, -0.1, 0.3], [0.8, 0.3, -0.2]])
    densities = [700.0, 2700.0]
    mass_inertia = [analytic_box(s, d) for s, d in zip(sizes, densities)]
    masses = np.array([item[0] for item in mass_inertia])
    expected_center = masses @ centers / masses.sum()
    expected_inertia = np.zeros((3, 3))
    for (mass, own), center in zip(mass_inertia, centers):
        offset = center - expected_center
        expected_inertia += rotation @ own @ rotation.T + mass * (
            (offset @ offset) * np.eye(3) - np.outer(offset, offset)
        )
    assert np.abs(expected_inertia - np.diag(np.diag(expected_inertia))).max() > 0.01
    for translation in (np.zeros(3), np.array([1e5, -2e5, 3e5])):
        regions = []
        for index, (size, density, center) in enumerate(zip(sizes, densities, centers)):
            base = box_domain(size)
            domain = ClosedMesh(
                base.vertices_m @ rotation.T + center + translation,
                base.faces,
                "independent_rigid_transform",
            )
            regions.append(
                MeshRegion(str(index), "bulk", domain, Material("specified", density))
            )
        state = MeshPhysicalObject("assembly", tuple(regions))
        props = state.properties
        assert props["mass_kg"] == pytest.approx(masses.sum(), rel=2e-9)
        np.testing.assert_allclose(
            props["center_of_mass_m"], expected_center + translation, atol=1e-9
        )
        np.testing.assert_allclose(
            props["inertia_kg_m2"], expected_inertia, rtol=2e-8, atol=1e-10
        )


def test_open_reversed_and_overlapping_regions_are_rejected() -> None:
    box = box_domain((1.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="闭合"):
        ClosedMesh(box.vertices_m, box.faces[:-1], "open")
    with pytest.raises(ValueError, match="有向体积"):
        ClosedMesh(box.vertices_m, box.faces[:, ::-1], "reversed")
    first = MeshRegion("a", "bulk", box, Material("a", 1000))
    second = MeshRegion(
        "b", "bulk", box_domain((1, 1, 1), (0.5, 0, 0)), Material("b", 100)
    )
    with pytest.raises(ValueError, match="重叠"):
        MeshPhysicalObject("overlap", (first, second))
    # 接触界面可共享，只有正体积交集才是材料重复计量。
    touching = MeshRegion(
        "c", "bulk", box_domain((1, 1, 1), (1, 0, 0)), Material("c", 100)
    )
    assert (
        MeshPhysicalObject("touching", (first, touching)).properties["mass_kg"] == 1100
    )


def test_cavity_outside_nonconvex_outer_is_rejected_by_geometry() -> None:
    from tactile3d.physics.mesh_native import boolean_volume

    first = box_domain((0.4, 0.4, 0.4), (-0.3, 0, 0))
    second = box_domain((0.4, 0.4, 0.4), (0.3, 0, 0))
    outer = boolean_volume(first, second, "union")
    cavity = box_domain((0.1, 0.1, 0.1))
    with pytest.raises(ValueError, match="穿出"):
        closed_shell(outer, cavity)
    with pytest.raises(ValueError, match="严格"):
        closed_shell(first, second)


def test_fraction_is_effective_density_and_edit_does_not_change_it() -> None:
    region = MeshRegion("a", "bulk", box_domain((1, 1, 1)), Material("a", 100), 0.4)
    state = MeshPhysicalObject("fraction", (region,))
    updated, report = state.edit_material("a", "bulk", Material("b", 1000), 0)
    assert state.properties["mass_kg"] == 40
    assert updated.properties["mass_kg"] == 400
    assert report["delta_mass_kg"] == 360
    assert updated.regions[0].fill_fraction == 0.4
    with pytest.raises(ValueError, match="版本"):
        updated.edit_material("a", "bulk", Material("c", 2000), 0)
    with pytest.raises(ValueError, match="占用"):
        MeshRegion("a", "bad", region.domain, region.material, 1.01)
    with pytest.raises(ValueError, match="标量"):
        MeshRegion("a", "bad", region.domain, Material("field", [10, 20]))


def test_public_properties_and_mesh_copies_cannot_mutate_cache() -> None:
    state = make_mesh_example()[-1]
    original = state.properties
    external = state.properties
    external["center_of_mass_m"][0] = 2000
    external["mass_kg"] = 0
    state.regions[0].domain.mesh().vertices[:] = 100
    assert state.properties == original
    with pytest.raises(ValueError, match="read-only"):
        state.regions[0].domain.vertices_m[0, 0] = 20


def test_mesh_asset_replays_ignores_cached_physics_and_detects_tamper(
    tmp_path: Path,
) -> None:
    state = make_mesh_example()[-1]
    directory = tmp_path / "asset"
    save_mesh_asset(state, directory)
    payload_path = directory / "asset.json"
    payload = json.loads(payload_path.read_text())
    payload["physics"]["mass_kg"] = -999
    payload_path.write_text(json.dumps(payload))
    restored = load_mesh_asset(directory)
    assert restored.properties == state.properties
    assert restored.regions[0].material.to_dict() == state.regions[0].material.to_dict()
    with pytest.raises(FileExistsError):
        save_mesh_asset(state, directory)
    geometry = directory / "region_000.npz"
    geometry.write_bytes(geometry.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="SHA256"):
        load_mesh_asset(directory)


def test_user_mesh_import_requires_explicit_metric_scale(tmp_path: Path) -> None:
    box_domain((200, 300, 400)).mesh().export(tmp_path / "outer.ply")
    box_domain((160, 260, 360)).mesh().export(tmp_path / "inner.ply")
    row = {
        "part_id": "body",
        "region_id": "shell",
        "outer_mesh": "outer.ply",
        "cavity_mesh": "inner.ply",
        "material": Material("shell", 1000).to_dict(),
        "provenance": "specified_design_mm",
    }
    payload = {"object_id": "mm", "mesh_unit_to_meter": 0.001, "regions": [row]}
    path = tmp_path / "input.json"
    path.write_text(json.dumps(payload))
    state = import_mesh_regions(path)
    assert state.properties["mass_kg"] == pytest.approx(9.024)
    payload["mesh_unit_to_meter"] = 0
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="比例"):
        import_mesh_regions(path)


def test_mesh_csg_volumes_and_map_assets_match_independent_reference() -> None:
    units = np.array([0.003, 0.013, 0.027])
    volumes = mesh_candidate_volumes(units)
    np.testing.assert_allclose(volumes, analytic_volumes(units), rtol=1e-12)
    priors = np.array([[550.0, 90.0]])
    for candidate in range(4):
        logits = np.zeros((1, 8))
        logits[0, candidate] = 3
        state = mesh_state_from_logits("MAP", units[1], logits, priors)
        prediction = mass_forward(logits, priors, volumes[1:2], hard=True)[0]
        assert state.properties["mass_kg"] == pytest.approx(prediction[0, 0])


def test_direct_section_preserves_cavity_orientation_and_hole_area() -> None:
    hollow = make_mesh_example()[1]
    contours = hollow.regions[0].domain.xy_section(0.2)
    signed_areas = []
    for xy in contours:
        rolled = np.roll(xy, -1, axis=0)
        signed_areas.append(
            float(np.sum(xy[:, 0] * rolled[:, 1] - xy[:, 1] * rolled[:, 0]) / 2)
        )
    assert len(contours) == 2
    assert min(signed_areas) < 0 < max(signed_areas)
    assert sum(signed_areas) == pytest.approx(0.2 * 0.3 - 0.16 * 0.26)
    assert hollow.regions[0].domain.xy_section(10) == ()


def test_mesh_basis_response_gradient_matches_finite_difference() -> None:
    inputs, labels = make_dataset(60, 32)
    logits = np.random.default_rng(47).normal(0, 0.4, (2, 8))
    arguments = (
        inputs["priors"][:2],
        mesh_candidate_volumes(inputs["pitch_m"][:2]),
        labels["mass_kg"][:2],
        labels["construction"][:2],
        1.0,
    )
    _, gradient = state_loss(logits, *arguments)
    numerical = np.empty_like(logits)
    step = 1e-6
    for index in np.ndindex(logits.shape):
        plus, minus = logits.copy(), logits.copy()
        plus[index] += step
        minus[index] -= step
        numerical[index] = (
            state_loss(plus, *arguments)[0] - state_loss(minus, *arguments)[0]
        ) / (2 * step)
    np.testing.assert_allclose(gradient, numerical, atol=2e-8, rtol=2e-5)


def test_native_paths_run_with_voxel_entrypoints_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Mesh原生路径不应进入体素模块")

    monkeypatch.setattr("trimesh.Trimesh.voxelized", forbidden)
    monkeypatch.setattr("tactile3d.research.construction.candidate_masks", forbidden)
    monkeypatch.setattr(
        "tactile3d.physics.part_state.PartPhysicalState.__post_init__", forbidden
    )
    result = run_mesh_demo(tmp_path / "native", figures=False)
    assert result["voxelization_used"] is False
    learned = run_native_learning(
        tmp_path / "learning", epochs=2, seeds=(7,), figures=False
    )
    assert learned["status"] == "complete"
    assert learned["voxelization_used"] is False
    assert (tmp_path / "learning/seed_7/original/asset.json").is_file()


def test_mesh_demo_roundtrip_replay_and_pixel_binding(tmp_path: Path) -> None:
    first = tmp_path / "first"
    report = run_mesh_demo(first)
    assert all(row["outside_target_changed"] == 0 for row in report["views"])
    assert all(row["changed_pixels"] > 0 for row in report["views"])
    replay = run_mesh_demo(
        tmp_path / "replay",
        asset=first / "original",
        plan=first / "edit_spec.json",
        figures=False,
    )
    assert replay["after"] == report["after"]
