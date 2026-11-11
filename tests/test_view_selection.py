import numpy as np

from tactile3d.vision.view_selection import select_azimuth_views


def test_four_view_selection_covers_distinct_cardinal_directions():
    angles = np.deg2rad([0, 90, 180, 270, 45])
    poses = np.tile(np.eye(4), (5, 1, 1))
    for pose, angle in zip(poses, angles):
        position = np.array([np.cos(angle), np.sin(angle), 0.6])
        pose[:3, 3] = position
        pose[:3, 2] = position / np.linalg.norm(position)
    chosen = select_azimuth_views(poses, 4)
    assert {row["frame_index"] for row in chosen} == {0, 1, 2, 3}
    assert all(
        abs(
            (row["actual_azimuth_degrees"] - row["target_azimuth_degrees"] + 180) % 360
            - 180
        )
        < 1e-9
        for row in chosen
    )
