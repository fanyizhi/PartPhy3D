"""坐标错配会污染多视角特征，使用已知投影与体积不变量做回归。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image

from tactile3d.abo500.dataset import CameraView
from tactile3d.mesh.frames import GLTF_TO_Z_UP, mesh_in_camera_frame
from tactile3d.vision.multiview import collect_part_observations


def test_rotation_preserves_volume_and_does_not_mutate_source() -> None:
    mesh = trimesh.creation.box(extents=(2, 3, 4))
    mesh.apply_translation((1, 2, 3))
    vertices = mesh.vertices.copy()
    result, report = mesh_in_camera_frame(mesh, "gltf_y_up")
    np.testing.assert_array_equal(mesh.vertices, vertices)
    np.testing.assert_allclose(result.center_mass, (1, -3, 2))
    np.testing.assert_allclose(result.volume, mesh.volume)
    np.testing.assert_allclose(result.area, mesh.area)
    assert np.linalg.det(GLTF_TO_Z_UP[:3, :3]) == pytest.approx(1)
    assert report["length_scale_changed"] is False
    with pytest.raises(ValueError, match="不能自动猜测"):
        mesh_in_camera_frame(mesh, "unknown")


def test_declared_gltf_rotation_restores_visible_part(
    tmp_path: Path,
) -> None:
    axis = np.linspace(-0.5, 0.5, 8)
    x, y = np.meshgrid(axis, axis)
    world = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, -2.0)))
    faces = []
    for row in range(7):
        for column in range(7):
            a = row * 8 + column
            faces.extend([(a, a + 1, a + 9), (a, a + 9, a + 8)])
    source_vertices = world @ GLTF_TO_Z_UP[:3, :3]
    mesh = trimesh.Trimesh(vertices=source_vertices, faces=faces, process=False)
    image_path = tmp_path / "view.png"
    Image.new("RGB", (100, 100), (100, 120, 140)).save(image_path)
    camera = CameraView(image_path, np.eye(4), 80, 80, 50, 50, 100, 100)
    labels = np.zeros(len(faces), dtype=int)
    with pytest.raises(ValueError, match="没有可见面"):
        collect_part_observations(mesh, labels, (camera,))
    before = collect_part_observations(
        mesh, labels, (camera,), allow_global_fallback=True
    )
    aligned, _ = mesh_in_camera_frame(mesh, "gltf_y_up")
    after = collect_part_observations(aligned, labels, (camera,))
    assert before[0][0].visible_faces == 0
    assert before[0][0].is_global_context
    assert after[0]
    assert after[0][0].visible_faces > 0
    assert not after[0][0].is_global_context
