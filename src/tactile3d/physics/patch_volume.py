"""将可见部件表面转为显式假设的闭合薄层，拒绝折返和非流形边界。"""

from __future__ import annotations

import manifold3d as manifold
import numpy as np
import trimesh
from numpy.typing import NDArray

from tactile3d.physics.mesh_native import ClosedMesh, positive_scalar

MIN_NORMAL_COSINE = 0.2


def extrude_patch(
    mesh_m: trimesh.Trimesh,
    face_indices: NDArray[np.int64],
    thickness_m: float,
    provenance: str,
) -> ClosedMesh:
    """以共同方向将有向表面片封成材料域，厚度为轴向而非法向厚度。

    Args:
        mesh_m: 输入坐标按调用方给定尺度转为米；尺度来源须另存。
        face_indices: 唯一的非空面索引，不会自动选择最大连通片。
        thickness_m: 显式指定轴向厚度，不由材料名称或目标质量反推。
        provenance: 厚度和构造假设的来源，不允许为空。

    Returns:
        带正向闭合边界的薄层域；仍需与物体外包域检查包含关系。
    """
    thickness = positive_scalar(thickness_m, "薄层轴向厚度m")
    selected = np.asarray(face_indices)
    if (
        selected.ndim != 1
        or not selected.size
        or selected.dtype.kind not in "iu"
        or selected.min() < 0
        or selected.max() >= len(mesh_m.faces)
        or len(np.unique(selected)) != len(selected)
    ):
        raise ValueError("需要非空、唯一且合法的表面面索引")
    if not isinstance(provenance, str) or not provenance.strip():
        raise ValueError("必须记录薄层构造和厚度假设的来源")
    normal = (mesh_m.face_normals[selected] * mesh_m.area_faces[selected, None]).sum(0)
    length = np.linalg.norm(normal)
    if not np.isfinite(length) or length <= 0:
        raise ValueError("表面片没有稳定的公共挤出方向")
    normal /= length
    alignment = mesh_m.face_normals[selected] @ normal
    if np.min(alignment) < MIN_NORMAL_COSINE:
        raise ValueError("表面片包含折返或接近竖直的面，不能用共同方向建立薄层")
    used, inverse = np.unique(mesh_m.faces[selected], return_inverse=True)
    top = inverse.reshape(-1, 3)
    points = mesh_m.vertices[used]
    edges = top[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
    _, groups, counts = np.unique(
        np.sort(edges, axis=1), axis=0, return_inverse=True, return_counts=True
    )
    if np.any(counts > 2):
        raise ValueError("表面片含非流形边，不可封成合法材料域")
    boundary = edges[counts[groups] == 1]
    if not len(boundary):
        raise ValueError("该表面片没有开放边界，不能套用局部薄层构造")
    offset = len(points)
    a, b = boundary.T
    sides = np.vstack(
        [
            np.column_stack([a, a + offset, b + offset]),
            np.column_stack([a, b + offset, b]),
        ]
    )
    faces = np.vstack([top, top[:, ::-1] + offset, sides])
    vertices = np.vstack([points, points - normal * thickness])
    domain = ClosedMesh(vertices, faces, provenance)
    expected = thickness * np.sum(mesh_m.area_faces[selected] * alignment)
    if not np.isclose(domain.moments.volume_m3, expected, rtol=1e-8, atol=1e-14):
        raise ValueError("薄层体积与投影面积×轴向厚度不一致，检查自交与方向")
    return domain


def piecewise_patch_layer(
    mesh_m: trimesh.Trimesh,
    face_indices: NDArray[np.int64],
    thickness_m: float,
    provenance: str,
) -> tuple[ClosedMesh, int]:
    """按固定方向和连通性分片后取并集，避免弯折区域挤出重叠重复计质。

    每个局部片的轴向厚度相同，方向不同；这是假设的工程薄层，
    不是等距偏移曲面。所有输入面必须参与，任何子片失败则整体失败。

    Returns:
        并集域与构造子片数；最终仍应与外包域取交集。
    """
    selected = np.asarray(face_indices)
    if (
        selected.ndim != 1
        or not selected.size
        or selected.dtype.kind not in "iu"
        or selected.min() < 0
        or selected.max() >= len(mesh_m.faces)
        or len(np.unique(selected)) != len(selected)
    ):
        raise ValueError("分片薄层需要唯一合法的面索引")
    positive_scalar(thickness_m, "薄层厚度m")
    directions = np.array(
        [
            (x, y, z)
            for x in (-1, 0, 1)
            for y in (-1, 0, 1)
            for z in (-1, 0, 1)
            if (x, y, z) != (0, 0, 0)
        ],
        dtype=float,
    )
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    assignment = np.argmax(mesh_m.face_normals[selected] @ directions.T, axis=1)
    membership = np.full(len(mesh_m.faces), -1, dtype=int)
    membership[selected] = assignment
    adjacency = mesh_m.face_adjacency
    domains = []
    for group in np.unique(assignment):
        nodes = selected[assignment == group]
        edges = adjacency[(membership[adjacency] == group).all(1)]
        for component in trimesh.graph.connected_components(edges, nodes=nodes):
            try:
                domains.append(
                    extrude_patch(
                        mesh_m,
                        np.asarray(component),
                        thickness_m,
                        provenance,
                    )
                )
            except ValueError:
                # 分片边界可能在顶点处夹接；改成逐三角柱并集，不删除失败面。
                domains.extend(
                    extrude_patch(mesh_m, np.array([face]), thickness_m, provenance)
                    for face in component
                )
    united = manifold.Manifold.batch_boolean(
        [domain._solid for domain in domains], manifold.OpType.Add
    )
    if united.is_empty() or united.status() != manifold.Error.NoError:
        raise ValueError("分片薄层并集失败，拒绝输出重叠体积")
    payload = united.to_mesh64()
    return ClosedMesh(
        payload.vert_properties[:, :3],
        payload.tri_verts,
        "piecewise_union:" + provenance,
    ), len(domains)
