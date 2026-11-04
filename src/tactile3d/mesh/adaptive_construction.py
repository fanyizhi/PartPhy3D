"""共享边细分的最终曲面构造；需要上游提供可查询的投影场。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import trimesh
from numpy.typing import NDArray

Array = NDArray[np.float64]
MAX_FACES = 100000
CURVATURE_WEIGHT = 4.0
TASK_WEIGHT = 6.0


def refine_surface(
    mesh: trimesh.Trimesh,
    project: Callable[[Array], Array],
    importance: Callable[[Array], Array],
    fraction: float = 0.25,
) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    """按曲率与任务区域选择共享边，用红绿细分避免T形裂缝。

    Args:
        mesh: 已定向、水密的粗曲面。
        project: 把新顶点投影至隐式曲面的回调；不能用未披露真值实现。
        importance: 各共享边中点的非负任务权重。
        fraction: 每轮细分的共享边比例；1表示均匀细分。

    Returns:
        新水密Mesh与新增顶点查询数。只统计此阶段，不含上游推理。
    """
    if (
        not 0 < fraction <= 1
        or not mesh.is_watertight
        or not mesh.is_winding_consistent
    ):
        raise ValueError("需要水密、定向一致的Mesh以及(0,1]内的边比例")
    if len(mesh.faces) * 4 > MAX_FACES:
        raise ValueError("可能超过面数预算，请降低轮数或增加明确预算")
    faces, vertices = np.asarray(mesh.faces), np.asarray(mesh.vertices)
    edges = np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    unique, inverse = np.unique(edges, axis=0, return_inverse=True)
    midpoint = vertices[unique].mean(1)
    weight = np.asarray(importance(midpoint), dtype=float)
    if (
        weight.shape != (len(unique),)
        or np.any(weight < 0)
        or not np.isfinite(weight).all()
    ):
        raise ValueError("任务权重必须为逐边有限非负数")
    normal_sum = np.zeros((len(unique), 3))
    np.add.at(normal_sum, inverse, np.repeat(mesh.face_normals, 3, axis=0))
    curvature = np.clip(1 - np.linalg.norm(normal_sum, axis=1) / 2, 0, 1)
    length = np.linalg.norm(vertices[unique[:, 0]] - vertices[unique[:, 1]], axis=1)
    score = length * (1 + CURVATURE_WEIGHT * curvature + TASK_WEIGHT * weight)
    count = max(1, int(np.ceil(fraction * len(unique))))
    selected = np.argsort(-score, kind="stable")[:count]
    projected = np.asarray(project(midpoint[selected]), dtype=float)
    if projected.shape != (count, 3) or not np.isfinite(projected).all():
        raise ValueError("投影场必须返回有限(N,3)坐标")
    if np.any(
        np.linalg.norm(projected - midpoint[selected], axis=1) > length[selected]
    ):
        raise ValueError("投影位移超过边长，粗网格无法可靠表达该曲面")
    vertex_ids = np.full(len(unique), -1, dtype=int)
    vertex_ids[selected] = np.arange(count) + len(vertices)
    mids = vertex_ids[inverse].reshape(-1, 3)
    output = []
    for (a, b, c), (u, v, w) in zip(faces, mids, strict=True):
        flags = (u >= 0, v >= 0, w >= 0)
        number = sum(flags)
        if number == 0:
            output.append((a, b, c))
        elif number == 3:
            output.extend(((a, u, w), (u, b, v), (w, v, c), (u, v, w)))
        elif number == 1:
            if u >= 0:
                output.extend(((a, u, c), (u, b, c)))
            elif v >= 0:
                output.extend(((b, v, a), (v, c, a)))
            else:
                output.extend(((c, w, b), (w, a, b)))
        elif u < 0:
            output.extend(((c, w, v), (b, v, a), (v, w, a)))
        elif v < 0:
            output.extend(((a, u, w), (c, w, b), (w, u, b)))
        else:
            output.extend(((b, v, u), (a, u, c), (u, v, c)))
    result = trimesh.Trimesh(
        np.vstack((vertices, projected)), np.asarray(output), process=False
    )
    if (
        not result.is_watertight
        or not result.is_winding_consistent
        or result.volume <= 0
    ):
        raise ValueError("细分导致无效曲面，拒绝导出")
    if np.any(result.area_faces <= np.finfo(float).eps):
        raise ValueError("投影产生退化三角形")
    return result, {
        "input_faces": len(faces),
        "output_faces": len(result.faces),
        "field_query_points": count,
        "fraction": fraction,
        "upstream_inference_included": False,
    }
