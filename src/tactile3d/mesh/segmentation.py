from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh
from scipy.spatial import cKDTree


@dataclass
class PartSegmentation:
    face_labels: np.ndarray
    num_parts: int
    part_area_fraction: dict[int, float]
    part_volume_fraction: dict[int, float]


def _kmeans(features: np.ndarray, k: int, rng: np.random.Generator, max_iters: int) -> np.ndarray:
    n = features.shape[0]
    k = max(1, min(k, n))
    centers = features[rng.choice(n, size=k, replace=False)]
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(max_iters):
        distances = ((features[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
        new_labels = distances.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for idx in range(k):
            members = features[labels == idx]
            if len(members):
                centers[idx] = members.mean(axis=0)
    return labels


def segment_mesh(
    mesh: trimesh.Trimesh,
    num_parts: int,
    cfg: dict,
    rng: np.random.Generator,
) -> PartSegmentation:
    face_centers = mesh.triangles_center
    face_normals = mesh.face_normals
    face_areas = mesh.area_faces[:, None]

    weights = cfg.get("segmentation", {}).get("feature_weights", {})
    pos_w = float(weights.get("position", 1.0))
    normal_w = float(weights.get("normal", 0.55))
    area_w = float(weights.get("area", 0.15))

    pos = _standardize(face_centers) * pos_w
    normals = _standardize(face_normals) * normal_w
    areas = _standardize(np.log(face_areas + 1e-12)) * area_w
    features = np.concatenate([pos, normals, areas], axis=1)

    labels = _kmeans(
        features,
        k=num_parts,
        rng=rng,
        max_iters=int(cfg.get("segmentation", {}).get("max_iters", 40)),
    )
    labels = _smooth_face_labels(
        labels,
        mesh.face_adjacency,
        iterations=int(cfg.get("segmentation", {}).get("smoothing_iters", 3)),
    )
    labels = _compact_labels(labels)

    total_area = float(mesh.area) + 1e-12
    part_area_fraction: dict[int, float] = {}
    for part_id in range(int(labels.max()) + 1):
        area = float(mesh.area_faces[labels == part_id].sum())
        part_area_fraction[part_id] = area / total_area

    part_volume_fraction = _voxel_volume_fractions(
        mesh,
        labels,
        resolution=int(cfg.get("segmentation", {}).get("voxel_resolution", 48)),
        fallback=part_area_fraction,
    )
    return PartSegmentation(
        face_labels=labels,
        num_parts=len(part_area_fraction),
        part_area_fraction=part_area_fraction,
        part_volume_fraction=part_volume_fraction,
    )


def _standardize(x: np.ndarray) -> np.ndarray:
    mu = x.mean(axis=0, keepdims=True)
    sigma = x.std(axis=0, keepdims=True) + 1e-8
    return (x - mu) / sigma


def _smooth_face_labels(labels: np.ndarray, adjacency: np.ndarray, iterations: int) -> np.ndarray:
    if iterations <= 0 or len(adjacency) == 0:
        return labels
    result = labels.copy()
    neighbors: list[list[int]] = [[] for _ in range(len(result))]
    for left, right in adjacency:
        neighbors[int(left)].append(int(right))
        neighbors[int(right)].append(int(left))
    for _ in range(iterations):
        updated = result.copy()
        for face_id, adjacent in enumerate(neighbors):
            if not adjacent:
                continue
            votes = np.bincount(result[np.asarray(adjacent, dtype=np.int64)])
            winner = int(votes.argmax())
            if votes[winner] > len(adjacent) / 2:
                updated[face_id] = winner
        if np.array_equal(updated, result):
            break
        result = updated
    return result


def _voxel_volume_fractions(
    mesh: trimesh.Trimesh,
    labels: np.ndarray,
    resolution: int,
    fallback: dict[int, float],
) -> dict[int, float]:
    """Assign filled interior voxels to the nearest labelled surface face.

    Surface-area ratios are a poor volume proxy for thin and nested parts. This
    calculation preserves the exact mesh volume while using a filled voxel grid
    only to estimate how that volume should be divided among surface parts.
    """

    if not mesh.is_watertight or resolution < 8:
        return _normalize(fallback)
    extent = float(np.max(mesh.extents))
    if extent <= 0:
        return _normalize(fallback)
    try:
        pitch = extent / float(resolution)
        points = mesh.voxelized(pitch).fill().points
        if len(points) == 0:
            return _normalize(fallback)
        tree = cKDTree(mesh.triangles_center)
        nearest = tree.query(points, workers=-1)[1]
        voxel_labels = labels[np.asarray(nearest, dtype=np.int64)]
        counts = np.bincount(voxel_labels, minlength=int(labels.max()) + 1).astype(np.float64)
        if counts.sum() <= 0:
            return _normalize(fallback)
        return {part_id: float(count / counts.sum()) for part_id, count in enumerate(counts) if count > 0}
    except (MemoryError, RuntimeError, ValueError):
        return _normalize(fallback)


def _normalize(values: dict[int, float]) -> dict[int, float]:
    total = float(sum(values.values()))
    if total <= 0:
        return {key: 0.0 for key in values}
    return {key: float(value / total) for key, value in values.items()}


def _compact_labels(labels: np.ndarray) -> np.ndarray:
    unique = np.unique(labels)
    mapping = {int(old): new for new, old in enumerate(unique)}
    return np.asarray([mapping[int(label)] for label in labels], dtype=np.int64)
