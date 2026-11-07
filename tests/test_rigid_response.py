from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tactile3d.physics.rigid_response import free_flight, impulse_response


def test_impulse_has_independent_linear_and_angular_momentum_solution() -> None:
    velocity, omega = impulse_response(
        2,
        np.diag([2, 3, 4]),
        np.array([1, 0, 0]),
        np.array([0, 2, 0]),
    )
    np.testing.assert_allclose(velocity, [0, 1, 0])
    np.testing.assert_allclose(omega, [0, 0, 0.5])
    with pytest.raises(ValueError, match="正定"):
        impulse_response(2, np.zeros((3, 3)), np.zeros(3), np.ones(3))
    with pytest.raises(ValueError, match="三角不等式"):
        impulse_response(2, np.diag([1, 1, 5]), np.zeros(3), np.ones(3))


def test_sphere_rotation_and_ballistic_trajectory_match_analytic_solution() -> None:
    times = np.linspace(0, 0.5, 41)
    gravity = np.array([0, 0, -9.81])
    output = free_flight(
        2,
        np.eye(3) * 0.2,
        np.array([0, 0, 2]),
        np.array([1, 0, 0]),
        np.array([0, 0, 3]),
        times,
        gravity,
    )
    np.testing.assert_allclose(output["position_m"][:, 0], times)
    np.testing.assert_allclose(output["position_m"][:, 2], 2 - 4.905 * times**2)
    np.testing.assert_allclose(
        Rotation.from_quat(output["quaternion_xyzw"]).as_rotvec()[:, 2],
        3 * times,
        atol=1e-8,
    )
    assert output["relative_energy_drift"] < 1e-8


def test_asymmetric_body_conserves_world_angular_momentum_and_energy() -> None:
    output = free_flight(
        1,
        np.diag([0.2, 0.3, 0.4]),
        np.zeros(3),
        np.zeros(3),
        np.array([1, 2, 3]),
        np.linspace(0, 2, 80),
        np.zeros(3),
    )
    assert output["relative_angular_momentum_drift"] < 1e-8
    assert output["relative_energy_drift"] < 1e-8
    assert output["quaternion_norm_error"] < 1e-8
