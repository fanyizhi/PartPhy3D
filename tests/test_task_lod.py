import numpy as np
import pytest
import trimesh

from tactile3d.mesh.task_lod import allocate_faces, simplify_components
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
