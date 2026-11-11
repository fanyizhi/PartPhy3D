import numpy as np
import trimesh

from tactile3d.mesh.contacts import closest_surface_pair


def test_separated_boxes_have_known_surface_gap():
    first = trimesh.creation.box()
    second = first.copy()
    second.apply_translation([3, 0, 0])
    result = closest_surface_pair(first, second)
    assert np.isclose(result["distance_m"], 2)
    assert np.isclose(
        np.linalg.norm(np.array(result["point_first_m"]) - result["point_second_m"]), 2
    )


def test_piercing_triangles_detect_zero_distance_without_vertex_contact():
    first = trimesh.Trimesh(
        [[-2, -2, 0], [2, -2, 0], [0, 2, 0]], [[0, 1, 2]], process=False
    )
    second = trimesh.Trimesh(
        [[0, 0, -1], [0, 0, 1], [0, 3, 1]], [[0, 1, 2]], process=False
    )
    assert closest_surface_pair(first, second)["distance_m"] == 0


def test_contact_pair_is_symmetric():
    first = trimesh.creation.icosphere(subdivisions=1)
    second = first.copy()
    second.apply_translation([3, 0.2, 0.1])
    assert np.isclose(
        closest_surface_pair(first, second)["distance_m"],
        closest_surface_pair(second, first)["distance_m"],
    )
