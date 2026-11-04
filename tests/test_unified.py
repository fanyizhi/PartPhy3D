import numpy as np
import pytest

from tactile3d.physics.unified import CanonicalPartState, moment_disagreement


def test_two_cubes_mass_centroid_and_full_inertia():
    state = CanonicalPartState(np.array([[0., 0., 0.], [2., 0., 0.]]),
                               np.array([0, 1]), 1., np.array([1., 3.]))
    result = state.integrate()
    assert result["mass_kg"] == pytest.approx(4.)
    assert result["center_of_mass_m"] == pytest.approx([1.5, 0., 0.])
    np.testing.assert_allclose(result["inertia_kg_m2"],
                               np.diag([2/3, 3+2/3, 3+2/3]))
    labels, _ = state.face_lookup(np.array([[0.1, 0., 0.], [2.2, 0., 0.]]))
    assert labels.tolist() == [0, 1]
    assert state.integrate() == result


def test_inertia_direction_change_is_not_hidden_by_equal_norm():
    ref = {"mass_kg": 1., "center_of_mass_m": [0., 0., 0.],
           "inertia_kg_m2": np.diag([1., 2., 3.]).tolist()}
    changed = {**ref, "inertia_kg_m2": np.diag([3., 2., 1.]).tolist()}
    errors = moment_disagreement(ref, changed, 1.)
    assert errors["mass_error_percent"] == 0
    assert errors["inertia_tensor_error_percent"] > 0


def test_canonical_state_rejects_invalid_material_index():
    with pytest.raises(ValueError, match="Part IDs"):
        CanonicalPartState(np.zeros((1, 3)), np.array([2]), 1., np.array([1.]))
