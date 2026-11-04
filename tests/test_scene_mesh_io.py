from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import trimesh

from tactile3d.mesh.io import load_triangle_mesh
from tactile3d.mesh.volume import analyze_mesh_volume


def test_scene_instances_keep_transform_scale_and_multiplicity(tmp_path: Path) -> None:
    scene = trimesh.Scene()
    box = trimesh.creation.box(extents=[1, 2, 3])
    scene.add_geometry(box, geom_name="shared", node_name="first")
    transform = np.diag([2.0, 2.0, 2.0, 1.0])
    transform[:3, 3] = [10, -3, 4]
    scene.graph.update(frame_to="second", matrix=transform, geometry="shared")
    file = tmp_path / "instanced.glb"
    scene.export(file)
    mesh = load_triangle_mesh(file)
    assert len(mesh.faces) == 24
    assert mesh.volume == pytest.approx(54)
    np.testing.assert_allclose(mesh.center_mass, np.array([10, -3, 4]) * 8 / 9)
    report = analyze_mesh_volume(file, 0.1, repair_strategy="none")
    assert report["volume"]["m3"] == pytest.approx(0.054)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0, 0.0])
def test_nonfinite_or_nonpositive_scale_fails_before_file_access(value: float) -> None:
    with pytest.raises(ValueError):
        analyze_mesh_volume("missing.glb", value)
