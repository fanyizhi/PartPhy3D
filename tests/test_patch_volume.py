"""薄层构造与解析惯量、孔洞、输入不变和非法折返的独立核对。"""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from tactile3d.physics.patch_volume import extrude_patch, piecewise_patch_layer


def test_rectangle_patch_has_analytic_box_moments() -> None:
    mesh = trimesh.Trimesh(
        vertices=[[0, 0, 0], [2, 0, 0], [2, 3, 0], [0, 3, 0]],
        faces=[[0, 1, 2], [0, 2, 3]],
        process=False,
    )
    original = mesh.vertices.copy()
    domain = extrude_patch(mesh, np.arange(2), 0.1, "specified_test_dimensions")
    np.testing.assert_allclose(domain.moments.volume_m3, 0.6)
    np.testing.assert_allclose(domain.moments.center_m, [1, 1.5, -0.05])
    np.testing.assert_allclose(
        domain.moments.central_second_m5,
        np.diag(0.6 * np.array([2, 3, 0.1]) ** 2 / 12),
        atol=1e-14,
    )
    np.testing.assert_array_equal(mesh.vertices, original)
    with pytest.raises(ValueError, match="唯一"):
        extrude_patch(mesh, np.array([0, 0]), 0.1, "invalid_repeat")


def test_hole_is_not_filled_by_patch_closure() -> None:
    vertices = np.array(
        [
            [-2, -2, 0],
            [2, -2, 0],
            [2, 2, 0],
            [-2, 2, 0],
            [-1, -1, 0],
            [1, -1, 0],
            [1, 1, 0],
            [-1, 1, 0],
        ]
    )
    faces = []
    for i in range(4):
        j = (i + 1) % 4
        faces.extend([[i, j, j + 4], [i, j + 4, i + 4]])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    domain = extrude_patch(mesh, np.arange(8), 0.2, "specified_annulus")
    assert domain.mesh().is_watertight
    np.testing.assert_allclose(domain.moments.volume_m3, (16 - 4) * 0.2)


def test_folded_surface_is_rejected() -> None:
    mesh = trimesh.creation.box()
    with pytest.raises(ValueError, match="折返|方向"):
        extrude_patch(mesh, np.arange(len(mesh.faces)), 0.01, "invalid_whole_box")


def test_piecewise_box_skin_matches_analytic_shell_without_double_count() -> None:
    mesh = trimesh.creation.box(extents=[2, 3, 4])
    thickness = 0.1
    domain, pieces = piecewise_patch_layer(
        mesh,
        np.arange(len(mesh.faces)),
        thickness,
        "specified_box_skin",
    )
    assert pieces == 6
    expected = 24 - np.prod(np.array([2, 3, 4]) - 2 * thickness)
    np.testing.assert_allclose(domain.moments.volume_m3, expected, rtol=1e-10)
    np.testing.assert_allclose(domain.moments.center_m, 0, atol=1e-12)
