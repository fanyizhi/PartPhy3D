"""将共同网格的已占用体素导出为显式边界，保留内部空腔。"""

from __future__ import annotations

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from tactile3d.physics.part_state import PhysicalObject

NEIGHBORS = np.array(
    [
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1],
    ],
    dtype=np.int64,
)
FACE_CORNERS = np.array(
    [
        [[1, -1, -1], [1, 1, -1], [1, 1, 1], [1, -1, 1]],
        [[-1, -1, -1], [-1, -1, 1], [-1, 1, 1], [-1, 1, -1]],
        [[-1, 1, -1], [-1, 1, 1], [1, 1, 1], [1, 1, -1]],
        [[-1, -1, -1], [1, -1, -1], [1, -1, 1], [-1, -1, 1]],
        [[-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]],
        [[-1, -1, -1], [-1, 1, -1], [1, 1, -1], [1, -1, -1]],
    ],
    dtype=np.int64,
)
NEIGHBOR_TOLERANCE = 1e-6


def part_surface(state: PhysicalObject, part_id: str) -> trimesh.Trimesh:
    """提取正密度单元边界，不使用凸包填空腔或改变原始密度场。

    Args:
        state: 已检查共同网格和体积互斥的物体。
        part_id: 明确部件槽位。

    Returns:
        体素边界Mesh；不是带原始纹理的重建表面。
    """
    part = state.part(part_id)
    indices = state.grid_indices(part)[part.density_kg_m3 > 0]
    if not len(indices):
        raise ValueError("全空腔部件没有可导出的材料边界")
    tree = cKDTree(indices)
    quads = []
    for offset, corners in zip(NEIGHBORS, FACE_CORNERS, strict=True):
        distance, _ = tree.query(indices + offset)
        boundary = indices[distance > NEIGHBOR_TOLERANCE]
        if len(boundary):
            quads.append(2 * boundary[:, None, :] + corners[None, :, :])
    vertices, inverse = np.unique(
        np.concatenate(quads).reshape(-1, 3), axis=0, return_inverse=True
    )
    faces = inverse.reshape(-1, 4)
    triangles = np.concatenate([faces[:, [0, 1, 2]], faces[:, [0, 2, 3]]])
    world = state.grid_origin_m + vertices * (part.pitch_m / 2)
    return trimesh.Trimesh(vertices=world, faces=triangles, process=False)
