"""相机归一化应保持投影，且不能用退化标定制造尺度。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from tactile3d.research.observed_cameras import calibrated_observed_cameras


def camera_fixture() -> dict[str, Any]:
    center = np.array([0.3, -0.2, 0.5])
    frames = []
    for angle in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        z = np.array([np.cos(angle), np.sin(angle), 0.3])
        z /= np.linalg.norm(z)
        x = np.cross([0, 0, 1], z)
        x /= np.linalg.norm(x)
        pose = np.eye(4)
        pose[:3, :3] = np.column_stack((x, np.cross(z, x), z))
        pose[:3, 3] = center + 2.7 * z
        frames.append({"transform_matrix": pose.tolist()})
    return {
        "w": 1024,
        "h": 768,
        "fl_x": 1400,
        "fl_y": 1390,
        "cx": 511.5,
        "cy": 383.5,
        "frames": frames,
    }


def test_observed_camera_similarity_preserves_projection() -> None:
    source = camera_fixture()
    indices, cameras, report = calibrated_observed_cameras(source)
    assert len(set(indices)) == 6
    assert cameras.shape == (6, 16)
    assert report["camera_unit_to_meter"] is None
    assert report["reference_mesh_used"] is False
    center = np.asarray(report["center_camera_units"])
    np.testing.assert_allclose(center, [0.3, -0.2, 0.5], atol=1e-12)
    point = center + np.array([0.1, 0.05, -0.1])
    model_point = (point - center) * report["model_units_per_camera_unit"]
    for row, index in zip(cameras, indices, strict=True):
        old = np.asarray(source["frames"][index]["transform_matrix"])
        new = row[:12].reshape(3, 4)
        before = old[:3, :3].T @ (point - old[:3, 3])
        after = new[:3, :3].T @ (model_point - new[:3, 3])
        np.testing.assert_allclose(
            before[:2] / before[2], after[:2] / after[2], atol=1e-7
        )


def test_observed_camera_rejects_distortion_and_degenerate_rays() -> None:
    source = camera_fixture()
    source["k1"] = 0.1
    with pytest.raises(ValueError, match="去畸变"):
        calibrated_observed_cameras(source)
    source["k1"] = 0
    source["frames"] = [source["frames"][0]] * 6
    with pytest.raises(ValueError, match="退化"):
        calibrated_observed_cameras(source)
