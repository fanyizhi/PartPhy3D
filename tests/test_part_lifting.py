"""防止背面染色和不可见表面的无证据填充。"""

import numpy as np
import trimesh

from tactile3d.vision.part_lifting import (
    add_view_votes,
    finalize_votes,
    mask_labels,
    rasterize_faces,
)


def test_occlusion_and_unknown_faces_are_preserved() -> None:
    vertices = np.array(
        [
            [-0.5, -0.5, -1],
            [0.5, -0.5, -1],
            [0, 0.5, -1],
            [-0.5, -0.5, -2],
            [0.5, -0.5, -2],
            [0, 0.5, -2],
        ]
    )
    mesh = trimesh.Trimesh(
        vertices=vertices, faces=[[0, 1, 2], [3, 4, 5]], process=False
    )
    camera = {"fl_x": 32, "fl_y": 32, "cx": 16, "cy": 16, "w": 32, "h": 32}
    ids, depth = rasterize_faces(mesh, np.eye(4), camera, 32)
    assert (ids == 0).any()
    assert not (ids == 1).any()
    np.testing.assert_allclose(depth[ids == 0], 1)
    votes = np.zeros((2, 3))
    add_view_votes(votes, mesh, np.eye(4), camera, np.ones((32, 32), np.int32))
    labels, _, support = finalize_votes(mesh, votes)
    assert labels.tolist() == [1, 0]
    assert support.tolist() == [1, 0]
    cached_votes = np.zeros((2, 3))
    add_view_votes(
        cached_votes,
        mesh,
        np.eye(4),
        camera,
        np.ones((32, 32), np.int32),
        cached_face_ids=ids,
    )
    np.testing.assert_allclose(cached_votes, votes)


def test_mask_quality_and_foreground_are_not_material_probabilities() -> None:
    masks = np.ones((2, 32, 32), dtype=bool)
    foreground = np.ones((32, 32), dtype=bool)
    foreground[0] = False
    labels = mask_labels(masks, np.array([0.8, 0.5]), foreground)
    assert labels[0].sum() == 0
    assert np.all(labels[1:] == 1)


def test_nested_small_region_is_not_erased_by_parent_quality() -> None:
    masks = np.zeros((2, 32, 32), dtype=bool)
    masks[0] = True
    masks[1, 10:14, 10:14] = True
    labels = mask_labels(
        masks,
        np.array([0.99, 0.9]),
        np.ones((32, 32), dtype=bool),
        specificity_power=0.25,
    )
    assert np.all(labels[10:14, 10:14] == 2)
    assert labels[0, 0] == 1
    nested = mask_labels(
        masks, np.array([0.99, 0.7]), np.ones((32, 32), dtype=bool), prefer_nested=True
    )
    assert np.all(nested[10:14, 10:14] == 2)
