"""经遮挡检查的多视角部件投票；不把表面标签当作内部材料域。"""

from __future__ import annotations

from typing import Any

import numpy as np
import trimesh

DEPTH_EPSILON = 1e-8
NORMAL_COSINE = np.cos(np.deg2rad(45.0))


def rasterize_faces(
    mesh: trimesh.Trimesh, pose: np.ndarray, camera: dict[str, Any], size: int
) -> tuple[np.ndarray, np.ndarray]:
    """用透视正确的深度测试保留前表面，避免背面被错误赋材质。"""
    if size < 32 or pose.shape != (4, 4):
        raise ValueError("需要至少32像素的分辨率和4×4相机矩阵")
    local = (mesh.vertices - pose[:3, 3]) @ pose[:3, :3]
    depth = -local[:, 2]
    safe = np.maximum(depth, DEPTH_EPSILON)
    uv = (
        np.column_stack(
            (
                (camera["fl_x"] * local[:, 0] / safe + camera["cx"]) / camera["w"],
                (-camera["fl_y"] * local[:, 1] / safe + camera["cy"]) / camera["h"],
            )
        )
        * size
    )
    ids = np.full((size, size), -1, np.int32)
    buffer = np.full((size, size), np.inf)
    for index, face in enumerate(mesh.faces):
        if np.any(depth[face] <= DEPTH_EPSILON):
            continue
        triangle = uv[face]
        low = np.maximum(np.floor(triangle.min(0)).astype(int), 0)
        high = np.minimum(np.ceil(triangle.max(0)).astype(int) + 1, size)
        if np.any(high <= low):
            continue
        a, b, c = triangle
        den = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(den) < 1e-9:
            continue
        x, y = np.meshgrid(
            np.arange(low[0], high[0]) + 0.5, np.arange(low[1], high[1]) + 0.5
        )
        w0 = ((b[1] - c[1]) * (x - c[0]) + (c[0] - b[0]) * (y - c[1])) / den
        w1 = ((c[1] - a[1]) * (x - c[0]) + (a[0] - c[0]) * (y - c[1])) / den
        w2 = 1 - w0 - w1
        reciprocal = w0 / depth[face[0]] + w1 / depth[face[1]] + w2 / depth[face[2]]
        z = np.divide(
            1, reciprocal, out=np.full_like(reciprocal, np.inf), where=reciprocal > 0
        )
        current = buffer[low[1] : high[1], low[0] : high[0]]
        update = (w0 >= 0) & (w1 >= 0) & (w2 >= 0) & (z < current)
        current[update] = z[update]
        ids[low[1] : high[1], low[0] : high[0]][update] = index
    return ids, buffer


def mask_labels(
    masks: np.ndarray,
    quality: np.ndarray,
    foreground: np.ndarray,
    minimum_quality: float = 0.65,
    specificity_power: float = 0.0,
    prefer_nested: bool = False,
) -> np.ndarray:
    """重叠像素按SAM质量分数解决，0始终表示未知；分数不是材料概率。"""
    if masks.ndim != 3 or not len(masks) or masks.shape[1:] != foreground.shape:
        raise ValueError("掩码与前景尺寸不一致")
    if quality.shape != (len(masks),) or not np.isfinite(quality).all():
        raise ValueError("每个掩码需要有限质量分数")
    if not 0 <= minimum_quality <= 1 or np.any((quality < 0) | (quality > 1)):
        raise ValueError("质量分数与门槛需要位于0到1")
    weights = (
        masks.astype(float)
        * np.where(quality >= minimum_quality, quality, 0)[:, None, None]
    )
    if not 0 <= specificity_power <= 1:
        raise ValueError("掩码特异性权重需要位于0到1")
    area = masks.sum(axis=(1, 2))
    # 父区域的高SAM分数不应覆盖已定位的小部件；此规则不是概率校准。
    specificity = (max(float(area.max()), 1) / np.maximum(area, 1)) ** specificity_power
    weights *= specificity[:, None, None]
    if prefer_nested:
        for child in range(len(masks)):
            if quality[child] < minimum_quality or area[child] == 0:
                continue
            for parent in range(len(masks)):
                if area[child] >= 0.7 * area[parent]:
                    continue
                containment = (masks[child] & masks[parent]).sum() / area[child]
                if containment >= 0.95:
                    weights[parent, masks[child]] = 0
    best = weights.argmax(axis=0)
    return np.where((weights.max(0) > 0) & foreground, best + 1, 0).astype(np.int32)


def add_view_votes(
    votes: np.ndarray,
    mesh: trimesh.Trimesh,
    pose: np.ndarray,
    camera: dict[str, Any],
    labels: np.ndarray,
    cached_face_ids: np.ndarray | None = None,
) -> np.ndarray:
    """像素面积归一化避免近视角独占投票；补充通过深度检验的面中心观测。"""
    size = labels.shape[0]
    if labels.shape != (size, size) or votes.shape[0] != len(mesh.faces):
        raise ValueError("投票维度与Mesh/图像不一致")
    if labels.min() < 0 or labels.max() >= votes.shape[1]:
        raise ValueError("部件标签超出投票范围")
    if cached_face_ids is None:
        ids, depth = rasterize_faces(mesh, pose, camera, size)
    else:
        ids = np.asarray(cached_face_ids)
        if ids.shape != labels.shape or ids.dtype.kind not in "iu":
            raise ValueError("缓存可见面编号的尺寸或类型错误")
        if ids.min() < -1 or ids.max() >= len(mesh.faces):
            raise ValueError("缓存面编号越界")
        depth = np.full(labels.shape, np.inf)
        y, x = np.where(ids >= 0)
        face = ids[y, x]
        triangles = (mesh.triangles[face] - pose[:3, 3]) @ pose[:3, :3]
        normals = np.cross(
            triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
        )
        rays = np.column_stack(
            (
                ((x + 0.5) / size * camera["w"] - camera["cx"]) / camera["fl_x"],
                -((y + 0.5) / size * camera["h"] - camera["cy"]) / camera["fl_y"],
                -np.ones(len(x)),
            )
        )
        # 对固定可见面做射线-平面求交，避免每次语义迭代重复光栅化全部三角面。
        numerator = np.einsum("ij,ij->i", normals, triangles[:, 0])
        denominator = np.einsum("ij,ij->i", normals, rays)
        depth[y, x] = np.divide(
            numerator,
            denominator,
            out=np.full(len(x), np.inf),
            where=np.abs(denominator) > 1e-12,
        )
    valid = ids >= 0
    counts = np.bincount(
        ids[valid] * votes.shape[1] + labels[valid], minlength=votes.size
    ).reshape(votes.shape)
    seen = counts.sum(1)
    contribution = counts / np.maximum(seen[:, None], 1)
    centers = mesh.triangles_center
    local = (centers - pose[:3, 3]) @ pose[:3, :3]
    z = -local[:, 2]
    safe = np.maximum(z, DEPTH_EPSILON)
    uv = (
        np.column_stack(
            (
                (camera["fl_x"] * local[:, 0] / safe + camera["cx"]) / camera["w"],
                (-camera["fl_y"] * local[:, 1] / safe + camera["cy"]) / camera["h"],
            )
        )
        * size
    )
    bounded = ((uv >= 0) & (uv < size)).all(1) & (z > DEPTH_EPSILON)
    bounded &= np.einsum("ij,ij->i", mesh.face_normals, pose[:3, 3] - centers) > 0
    indices = np.flatnonzero(bounded & (seen == 0))
    xy = uv[indices].astype(int)
    visible = z[indices] <= depth[xy[:, 1], xy[:, 0]] * 1.005 + 1e-6
    indices, xy = indices[visible], xy[visible]
    contribution[indices, labels[xy[:, 1], xy[:, 0]]] = 1
    votes += contribution
    return ids


def finalize_votes(
    mesh: trimesh.Trimesh,
    votes: np.ndarray,
    iterations: int = 3,
    smoothing_weight: float = 0.15,
    minimum_margin: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """仅在已观测面内平滑，保留尖锐接口和不可见区域的未知状态。

    Returns:
        面标签、投票一致性、观测视角数。一致性不是校准概率。
    """
    if votes.ndim != 2 or votes.shape[0] != len(mesh.faces):
        raise ValueError("无效逐面投票数组")
    if not np.isfinite(votes).all() or np.any(votes < 0):
        raise ValueError("投票必须为有限非负值")
    if votes.shape[1] < 2 or iterations < 0:
        raise ValueError("至少需要未知类与一个部件，迭代次数不可为负")
    if not 0 <= smoothing_weight <= 1 or not 0 <= minimum_margin <= 1:
        raise ValueError("平滑权重与拒绝门槛需要位于0到1")
    support = votes.sum(1)
    scores = votes / np.maximum(support[:, None], 1)
    edges = mesh.face_adjacency
    compatible = (
        np.einsum(
            "ij,ij->i", mesh.face_normals[edges[:, 0]], mesh.face_normals[edges[:, 1]]
        )
        >= NORMAL_COSINE
    )
    edges = edges[compatible & (support[edges] > 0).all(1)]
    fixed = scores.max(1) >= 0.8
    for _ in range(iterations):
        neighbor = np.zeros_like(scores)
        degree = np.zeros(len(scores))
        for a, b in ((edges[:, 0], edges[:, 1]), (edges[:, 1], edges[:, 0])):
            np.add.at(neighbor, a, scores[b])
            np.add.at(degree, a, 1)
        blended = (
            1 - smoothing_weight
        ) * scores + smoothing_weight * neighbor / np.maximum(degree[:, None], 1)
        scores[~fixed & (degree > 0)] = blended[~fixed & (degree > 0)]
    order = np.sort(scores, axis=1)
    margin = order[:, -1] - order[:, -2]
    labels = scores.argmax(1).astype(np.int32)
    labels[(support == 0) | (margin < minimum_margin)] = 0
    return labels, scores.max(1), support
