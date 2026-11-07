"""刚体自由飞行与冲量响应诊断，不含接触、变形或材料参数识别。"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation

RTOL = 1e-10
ATOL = 1e-12


def validate_inertia(mass_kg: float, inertia_kg_m2: np.ndarray) -> np.ndarray:
    inertia = np.asarray(inertia_kg_m2, dtype=float)
    if not np.isfinite(mass_kg) or mass_kg <= 0:
        raise ValueError("刚体质量需要有限正数")
    if (
        inertia.shape != (3, 3)
        or not np.isfinite(inertia).all()
        or not np.allclose(inertia, inertia.T, atol=ATOL, rtol=RTOL)
        or np.linalg.eigvalsh(inertia).min() <= 0
    ):
        raise ValueError("质心惯量必须为有限、对称正定的3×3矩阵")
    eigenvalues = np.linalg.eigvalsh(inertia)
    if eigenvalues[-1] > eigenvalues[:-1].sum() * (1 + RTOL) + ATOL:
        raise ValueError("主惯量不满足三角不等式，不能对应非负密度刚体")
    return inertia


def impulse_response(
    mass_kg: float,
    inertia_kg_m2: np.ndarray,
    point_from_com_m: np.ndarray,
    impulse_ns: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """初始静止且坐标一致时，计算冲量带来的线速度和角速度。

    Args:
        mass_kg: 刚体总质量。
        inertia_kg_m2: 作用时刻世界坐标下绕质心惯量。
        point_from_com_m: 冲量作用点相对质心的位置。
        impulse_ns: 世界坐标冲量，单位N·s。

    Returns:
        线速度增量m/s和世界角速度增量rad/s。
    """
    inertia = validate_inertia(mass_kg, inertia_kg_m2)
    point, impulse = np.asarray(point_from_com_m), np.asarray(impulse_ns)
    if (
        point.shape != (3,)
        or impulse.shape != (3,)
        or not np.isfinite(np.concatenate([point, impulse])).all()
    ):
        raise ValueError("冲量与相对作用点需要有限三维向量")
    return impulse / mass_kg, np.linalg.solve(inertia, np.cross(point, impulse))


def free_flight(
    mass_kg: float,
    inertia_body_kg_m2: np.ndarray,
    initial_position_m: np.ndarray,
    initial_velocity_m_s: np.ndarray,
    initial_omega_rad_s: np.ndarray,
    times_s: np.ndarray,
    gravity_m_s2: np.ndarray,
) -> dict[str, Any]:
    """初始姿态为单位旋转；无接触、恒重力且无外力矩的自由刚体。

    Args:
        mass_kg: 对应完整模拟刚体的质量。
        inertia_body_kg_m2: 刚体坐标中绕质心惯量。
        initial_position_m: 初始世界质心。
        initial_velocity_m_s: 初始世界线速度。
        initial_omega_rad_s: 初始刚体坐标角速度。
        times_s: 从0开始、严格递增的采样时间。
        gravity_m_s2: 恒定世界重力加速度。

    Returns:
        质心轨迹、姿态、角动量、能量及相对漂移；这是求解器一致性诊断。
    """
    inertia = validate_inertia(mass_kg, inertia_body_kg_m2)
    vectors = [
        np.asarray(v, dtype=float)
        for v in (
            initial_position_m,
            initial_velocity_m_s,
            initial_omega_rad_s,
            gravity_m_s2,
        )
    ]
    if any(v.shape != (3,) or not np.isfinite(v).all() for v in vectors):
        raise ValueError("初始条件与重力必须为有限三维向量")
    times = np.asarray(times_s, dtype=float)
    if (
        times.ndim != 1
        or len(times) < 2
        or times[0] != 0
        or not np.isfinite(times).all()
        or np.any(np.diff(times) <= 0)
    ):
        raise ValueError("采样时间必须从0开始并严格递增")
    position, velocity, omega, gravity = vectors

    def derivative(time_s: float, state: np.ndarray) -> np.ndarray:
        angular = state[:3]
        qv, qw = state[3:6], state[6]
        # q表示从刚体到世界的旋转，右乘刚体角速度四元数。
        dqv = 0.5 * (qw * angular + np.cross(qv, angular))
        dqw = -0.5 * np.dot(qv, angular)
        acceleration = np.linalg.solve(inertia, -np.cross(angular, inertia @ angular))
        return np.r_[acceleration, dqv, dqw]

    solution = solve_ivp(
        derivative,
        (0, times[-1]),
        np.r_[omega, 0, 0, 0, 1],
        method="DOP853",
        t_eval=times,
        rtol=RTOL,
        atol=ATOL,
    )
    if not solution.success or solution.y.shape != (7, len(times)):
        raise RuntimeError(f"刚体积分失败：{solution.message}")
    angular = solution.y[:3].T
    quaternion = solution.y[3:].T
    quaternion_norm_error = float(
        np.max(np.abs(np.linalg.norm(quaternion, axis=1) - 1))
    )
    rotation = Rotation.from_quat(quaternion).as_matrix()
    trajectory = (
        position + times[:, None] * velocity + 0.5 * times[:, None] ** 2 * gravity
    )
    linear = velocity + times[:, None] * gravity
    angular_momentum = np.einsum("nij,nj->ni", rotation, angular @ inertia.T)
    energy = (
        0.5 * mass_kg * np.sum(linear**2, axis=1)
        + 0.5 * np.einsum("ni,ij,nj->n", angular, inertia, angular)
        - mass_kg * (trajectory @ gravity)
    )
    momentum_scale = max(float(np.linalg.norm(angular_momentum[0])), ATOL)
    energy_scale = max(abs(float(energy[0])), float(np.max(np.abs(energy))), ATOL)
    return {
        "times_s": times,
        "position_m": trajectory,
        "quaternion_xyzw": quaternion,
        "omega_body_rad_s": angular,
        "world_angular_momentum": angular_momentum,
        "energy_j": energy,
        "relative_angular_momentum_drift": float(
            np.linalg.norm(angular_momentum - angular_momentum[0], axis=1).max()
            / momentum_scale
        ),
        "relative_energy_drift": float(
            np.max(np.abs(energy - energy[0])) / energy_scale
        ),
        "quaternion_norm_error": quaternion_norm_error,
    }
