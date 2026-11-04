from __future__ import annotations

import numpy as np
import pytest
import trimesh

from tactile3d.mesh.adaptive_construction import refine_surface


@pytest.mark.parametrize("fraction", [0.001, 0.25, 0.63, 1.0])
def test_conforming_refinement_retains_watertight_orientation(fraction: float) -> None:
    mesh = trimesh.creation.icosphere(subdivisions=1)
    project = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)
    result, report = refine_surface(mesh, project, lambda x: np.zeros(len(x)), fraction)
    assert result.is_watertight and result.is_winding_consistent
    assert result.euler_number == 2
    assert result.volume > mesh.volume
    assert report["field_query_points"] == len(result.vertices) - len(mesh.vertices)
    if fraction == 1:
        assert len(result.faces) == 4 * len(mesh.faces)


def test_bad_projection_and_open_input_are_rejected() -> None:
    mesh = trimesh.creation.icosphere(subdivisions=1)
    with pytest.raises(ValueError, match="投影"):
        refine_surface(
            mesh, lambda x: np.full_like(x, np.nan), lambda x: np.zeros(len(x))
        )
    mesh.update_faces(np.arange(len(mesh.faces)) > 0)
    with pytest.raises(ValueError, match="水密"):
        refine_surface(mesh, lambda x: x, lambda x: np.zeros(len(x)))
