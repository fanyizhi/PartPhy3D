import numpy as np
import pytest
import trimesh

from tactile3d.mesh.task_lod import (
    allocate_faces,
    component_sensitivity,
    simplify_components,
    simplify_with_lineage,
)
from tactile3d.vision.progressive import followup_focus, validate_answer


def test_allocation_saturates_small_parts_and_spends_exact_budget():
    caps = np.array([4, 12, 100])
    result = allocate_faces(caps, np.array([100.0, 10.0, 1.0]), 40)
    assert result.sum() == 40
    assert np.all(result <= caps)
    assert result[0] == 4 and result[1] == 12


def test_impossible_budget_rejected():
    with pytest.raises(ValueError):
        allocate_faces(np.array([12, 12]), np.ones(2), 7)


def test_component_identity_survives_simplification():
    left = trimesh.creation.icosphere(subdivisions=2)
    right = left.copy()
    right.apply_translation([4, 0, 0])
    result = simplify_components([left, right], 240)
    assert len(result.mesh.faces) <= 240
    assert set(result.face_components) == {0, 1}
    assert result.mesh.is_watertight
    assert len(result.mesh.split()) == 2


def test_followup_targets_textile_volume_confusion():
    previous = {"components": [{"surface_material": "fabric", "construction": "solid"}]}
    assert "fibers" in followup_focus(previous)


def test_duplicate_assignments_rejected():
    with pytest.raises(ValueError):
        validate_answer({"components": [{"component_id": 0}, {"component_id": 0}]}, 2)


def test_sensitivity_is_invariant_to_coordinate_units_and_origin():
    first = trimesh.creation.box(extents=[1, 0.08, 0.08])
    second = trimesh.creation.icosphere(subdivisions=1)
    second.apply_translation([2, 0, 1])
    densities = np.array([650.0, np.nan])
    reference = component_sensitivity([first, second], densities)
    transformed = []
    for mesh in (first, second):
        copy = mesh.copy()
        copy.apply_scale(1000)
        copy.apply_translation([100, -300, 800])
        transformed.append(copy)
    assert np.allclose(component_sensitivity(transformed, densities), reference)


def test_global_qem_lineage_preserves_spatial_component_identity():
    left = trimesh.creation.icosphere(subdivisions=2)
    right = left.copy()
    right.apply_translation([3, 0, 0])
    mesh = trimesh.util.concatenate([left, right])
    labels = np.repeat([0, 1], [len(left.faces), len(right.faces)])
    reduced = simplify_with_lineage(mesh, labels, 240)
    centers = reduced.mesh.triangles_center
    assert np.all(centers[reduced.face_components == 0, 0] < 1.1)
    assert np.all(centers[reduced.face_components == 1, 0] > 1.9)
