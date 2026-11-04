import numpy as np
import pytest
import trimesh

from tactile3d.mesh.assembly import interface_geometry, texture_face_rgb


def test_upper_band_proximity_distinguishes_separated_surfaces():
    source = trimesh.creation.box(extents=[1, 1, 1])
    target = trimesh.creation.box(extents=[1, 1, 1])
    target.apply_translation([0, 1.01, 0])
    report, _, distance = interface_geometry(source, target, top_band_m=0.1, samples=128)
    assert distance.min() == pytest.approx(0.01)
    assert report["near_sample_fraction"] == 0
    assert report["mechanical_quality"] == "unverified"
    assert report["strength_safety_factor"] is None


def test_texture_vertex_colors_do_not_require_bound_color_visual():
    from PIL import Image

    mesh = trimesh.creation.box()
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=np.full((len(mesh.vertices), 2), 0.5),
        image=Image.new("RGB", (2, 2), (128, 64, 32)),
    )
    rgb = texture_face_rgb(mesh)
    assert rgb.shape == (len(mesh.faces), 3)
    np.testing.assert_allclose(rgb, np.tile(np.array([128, 64, 32])/255, (len(mesh.faces), 1)))
