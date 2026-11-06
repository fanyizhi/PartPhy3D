"""仅由已标定观测构造InstantMesh相机，不读取参考Mesh或质量。"""

from __future__ import annotations

from typing import Any

import numpy as np

CANONICAL_AZIMUTHS = np.arange(30.0, 360.0, 60.0)
CANONICAL_ELEVATIONS = np.array([20.0, -10.0] * 3)
MODEL_CAMERA_RADIUS = 4.0
MAX_CONDITION = 1e8


def calibrated_observed_cameras(
    camera: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """将六个观测相机映射到固定模型空间，保留原始投影。

    相机需要OpenGL c2w、Z向上和无畸变针孔内参。相似变换不能产生
    米制标定；输出Mesh只能还原到输入相机单位。

    Args:
        camera: ABO transforms.json的内容。

    Returns:
        六个视角索引、形状为(6, 16)的模型相机、坐标变换审计记录。

    Raises:
        ValueError: 标定非刚体、退化、含畸变或缺少足够观测。
    """
    frames = camera.get("frames", [])
    if len(frames) < 6:
        raise ValueError("实际多视角诊断至少需要六个已标定观测")
    poses = np.asarray([f["transform_matrix"] for f in frames], dtype=np.float64)
    if poses.shape != (len(frames), 4, 4) or not np.isfinite(poses).all():
        raise ValueError("相机矩阵需要有限的N×4×4数组")
    rotations = poses[:, :3, :3]
    if (
        not np.allclose(poses[:, 3, :], [0, 0, 0, 1], atol=1e-6)
        or not np.allclose(
            rotations.transpose(0, 2, 1) @ rotations, np.eye(3), atol=1e-5
        )
        or not np.allclose(np.linalg.det(rotations), 1, atol=1e-5)
    ):
        raise ValueError("c2w需要右手刚体变换，不可包含缩放或反射")
    if any(
        abs(float(camera.get(k, 0))) > 1e-10
        for k in ("k1", "k2", "k3", "k4", "p1", "p2")
    ):
        raise ValueError("请先显式去畸变，再输入观测相机")
    width, height = float(camera["w"]), float(camera["h"])
    intrinsic = np.array(
        [camera["fl_x"], camera["fl_y"], camera["cx"], camera["cy"]],
        dtype=np.float64,
    )
    if (
        not np.isfinite([width, height]).all()
        or min(width, height) <= 0
        or not np.isfinite(intrinsic).all()
        or min(intrinsic[:2]) <= 0
    ):
        raise ValueError("图像尺寸和焦距须为有限正数")
    origins, directions = poses[:, :3, 3], -rotations[:, :, 2]
    # 通过视轴最小二乘求公共中心，不借助参考几何对齐重建结果。
    projectors = np.eye(3) - directions[:, :, None] * directions[:, None, :]
    system = projectors.sum(axis=0)
    condition = float(np.linalg.cond(system))
    if not np.isfinite(condition) or condition > MAX_CONDITION:
        raise ValueError("视轴退化，不能稳定估计相机归一化中心")
    center = np.linalg.solve(system, np.einsum("nij,nj->i", projectors, origins))
    offsets = origins - center
    radii = np.linalg.norm(offsets, axis=1)
    if np.min(radii) <= 1e-8:
        raise ValueError("相机位于归一化中心，无法确定视角")
    scale = MODEL_CAMERA_RADIUS / float(radii.mean())
    azimuth, elevation = np.deg2rad(CANONICAL_AZIMUTHS), np.deg2rad(
        CANONICAL_ELEVATIONS
    )
    targets = np.column_stack(
        (
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        )
    )
    directions_from_center = offsets / radii[:, None]
    selected: list[int] = []
    for target in targets:
        scores = directions_from_center @ target
        scores[selected] = -np.inf
        selected.append(int(np.argmax(scores)))
    indices = np.asarray(selected, dtype=np.int64)
    normalized = poses[indices].copy()
    normalized[:, :3, 3] = (normalized[:, :3, 3] - center) * scale
    intrinsic /= [width, height, width, height]
    cameras = np.column_stack(
        (normalized[:, :3, :].reshape(6, 12), np.broadcast_to(intrinsic, (6, 4)))
    ).astype(np.float32)
    residuals = np.einsum("nij,nj->ni", projectors, center - origins)
    return (
        indices,
        cameras,
        {
            "source_frame": "OpenGL_c2w_Z_up_camera_units",
            "center_camera_units": center.tolist(),
            "model_units_per_camera_unit": scale,
            "camera_unit_to_meter": None,
            "center_estimation_condition": condition,
            "view_axis_rms_residual": float(np.sqrt(np.mean(residuals**2))),
            "selected_frame_indices": selected,
            "selection": "greedy_nearest_six_canonical_directions_without_replacement",
            "reference_mesh_used": False,
            "mass_labels_used": False,
            "input_distribution_changed_from_official_single_image_pipeline": True,
        },
    )
