import numpy as np

from tactile3d.vision.role_constraints import furniture_role_candidates


def test_vertical_ground_support_and_horizontal_rail_are_distinguished():
    whole = np.array([[-0.5, -1, 0], [0.5, 1, 1]])
    leg = np.array([[0.35, 0.8, 0], [0.42, 0.87, 0.3]])
    rail = np.array([[-0.4, 0.8, 0.2], [0.4, 0.85, 0.28]])
    assert furniture_role_candidates(leg, whole, "wood", True)["allowed_roles"] == [
        "leg"
    ]
    assert furniture_role_candidates(rail, whole, "wood", False)["allowed_roles"] == [
        "frame_rail"
    ]


def test_role_does_not_depend_on_world_scale_or_origin():
    whole = np.array([[-0.5, -1, 0], [0.5, 1, 1]])
    leg = np.array([[0.35, 0.8, 0], [0.42, 0.87, 0.3]])
    transform = lambda x: x * 1000 + np.array([100, 300, -700])
    assert (
        furniture_role_candidates(leg, whole, "wood", True)["allowed_roles"]
        == furniture_role_candidates(transform(leg), transform(whole), "wood", True)[
            "allowed_roles"
        ]
    )
