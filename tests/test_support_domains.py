import numpy as np
import pytest
import trimesh

from tactile3d.physics.support_domains import (
    SupportDomain,
    assemble_properties,
    convex_envelope,
    domain_mesh,
    eroded_convex,
    partition_domains,
    solid_from_mesh,
)


def test_convex_layer_recovers_cube_shell_volume():
    mesh = trimesh.creation.box()
    inner = eroded_convex(mesh, 0.1)
    result = partition_domains([SupportDomain(0, "cover", mesh, inner, 1, "test")])
    part = result["parts"][0]
    assert part["shell"]["volume_m3"] == pytest.approx(1 - 0.8**3, rel=1e-6)
    assert part["core"]["volume_m3"] == pytest.approx(0.8**3, rel=1e-6)
    properties = assemble_properties([(part["shell"], 200.0), (part["core"], 40.0)])
    assert properties["mass_kg"] == pytest.approx(118.08, rel=1e-6)


def test_triangulated_support_planes_and_readonly_solid_roundtrip_remain_manifold():
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=0.1)
    inner = eroded_convex(mesh, 0.0035)
    recovered = solid_from_mesh(domain_mesh(inner))
    assert 0 < recovered.volume() < mesh.volume
    assert recovered.volume() == pytest.approx(inner.volume(), rel=1e-10)
    assert domain_mesh(recovered).is_volume


def test_embedded_support_removes_overlap_without_creating_internal_fabric_skin():
    cushion, frame = trimesh.creation.box(), trimesh.creation.box(extents=[0.2] * 3)
    result = partition_domains(
        [
            SupportDomain(1, "fabric", cushion, eroded_convex(cushion, 0.1), 1, "test"),
            SupportDomain(0, "wood", frame, solid_from_mesh(frame), 0, "test"),
        ]
    )
    parts = {p["component_id"]: p for p in result["parts"]}
    assert result["union_volume_m3"] == pytest.approx(1)
    assert result["overlap_removed_m3"] == pytest.approx(0.008, rel=1e-6)
    assert parts[1]["shell"]["volume_m3"] == pytest.approx(1 - 0.8**3, rel=1e-6)
    assert parts[1]["core"]["volume_m3"] == pytest.approx(0.8**3 - 0.2**3, rel=1e-6)
    properties = assemble_properties(
        [(parts[0]["assigned"], 600), (parts[1]["shell"], 200), (parts[1]["core"], 40)]
    )
    assert properties["mass_kg"] == pytest.approx(122.56, rel=1e-6)
    np.testing.assert_allclose(properties["center_of_mass_m"], 0, atol=1e-8)


def test_planar_patch_requires_explicit_thickness_and_extrudes_inward():
    mesh = trimesh.Trimesh(
        [[0, 0, 0], [2, 0, 0], [2, 1, 0], [0, 1, 0]],
        [[0, 1, 2], [0, 2, 3]],
        process=False,
    )
    with pytest.raises(ValueError, match="requires positive thickness"):
        convex_envelope(mesh, 0, [1, 0.5, 1])
    hull, source = convex_envelope(mesh, 0.01, [1, 0.5, 1])
    assert hull.volume == pytest.approx(0.02)
    assert hull.bounds[0, 2] == pytest.approx(0)
    assert hull.bounds[1, 2] == pytest.approx(0.01)
    assert "prism" in source


def test_fixed_geometry_density_edit_has_linear_mass_change_and_translation_invariant_inertia():
    box = trimesh.creation.box(extents=[0.2, 0.3, 0.4])

    def domain(mesh):
        return partition_domains(
            [SupportDomain(0, "wood", mesh, solid_from_mesh(mesh), 0, "test")]
        )["parts"][0]["assigned"]

    first = domain(box)
    before, after = assemble_properties([(first, 500)]), assemble_properties(
        [(first, 2700)]
    )
    assert after["mass_kg"] - before["mass_kg"] == pytest.approx(0.024 * 2200, rel=1e-6)
    box.apply_translation([1, 2, 3])
    translated = assemble_properties([(domain(box), 500)])
    np.testing.assert_allclose(
        translated["inertia_kg_m2"], before["inertia_kg_m2"], rtol=1e-5, atol=1e-7
    )


def test_two_part_material_edit_updates_center_and_parallel_axis_inertia():
    left = trimesh.creation.box(extents=[0.2] * 3)
    right = left.copy()
    right.apply_translation([1, 0, 0])
    parts = partition_domains(
        [
            SupportDomain(i, "wood", mesh, solid_from_mesh(mesh), 0, "test")
            for i, mesh in enumerate([left, right])
        ]
    )["parts"]
    result = assemble_properties(
        [(parts[0]["assigned"], 500), (parts[1]["assigned"], 2700)]
    )
    assert result["mass_kg"] == pytest.approx(25.6)
    np.testing.assert_allclose(result["center_of_mass_m"], [0.84375, 0, 0], atol=1e-10)
    np.testing.assert_allclose(
        result["inertia_kg_m2"],
        np.diag(
            [25.6 * 0.2**2 / 6, 25.6 * 0.2**2 / 6 + 3.375, 25.6 * 0.2**2 / 6 + 3.375]
        ),
        atol=1e-9,
    )
