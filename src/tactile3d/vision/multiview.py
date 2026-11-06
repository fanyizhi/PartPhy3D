from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from tactile3d.abo500.dataset import CameraView


@dataclass
class ViewObservation:
    image_path: Path
    crop: Image.Image
    quality: float
    visible_faces: int
    mean_rgb: np.ndarray
    std_rgb: np.ndarray
    saturation: float
    texture: float
    is_global_context: bool = False


def collect_part_observations(
    mesh: trimesh.Trimesh,
    face_labels: np.ndarray,
    views: tuple[CameraView, ...],
    max_views: int = 12,
    crop_padding: float = 0.12,
    zbuffer_size: int = 256,
    allow_global_fallback: bool = False,
) -> dict[int, list[ViewObservation]]:
    part_ids = sorted(int(x) for x in np.unique(face_labels))
    observations: dict[int, list[ViewObservation]] = {part_id: [] for part_id in part_ids}
    centers = mesh.triangles_center
    normals = mesh.face_normals

    for view in views:
        with Image.open(view.image_path) as source:
            image = source.convert("RGB")
        width, height = image.size
        sx = width / float(view.width)
        sy = height / float(view.height)
        fx, fy = view.fx * sx, view.fy * sy
        cx, cy = view.cx * sx, view.cy * sy
        projected = _project(centers, view.c2w, fx, fy, cx, cy)
        camera_position = view.c2w[:3, 3]
        to_camera = camera_position[None, :] - centers
        distances = np.linalg.norm(to_camera, axis=1, keepdims=True) + 1e-12
        facing = np.sum(normals * (to_camera / distances), axis=1)
        valid = (
            (projected[:, 2] > 1e-6)
            & (projected[:, 0] >= 0)
            & (projected[:, 0] < width)
            & (projected[:, 1] >= 0)
            & (projected[:, 1] < height)
            & (facing > 0)
        )
        visible = valid & _zbuffer_visibility(projected, valid, width, height, zbuffer_size)
        pixels = np.asarray(image, dtype=np.float32) / 255.0

        for part_id in part_ids:
            indices = np.flatnonzero(visible & (face_labels == part_id))
            if len(indices) == 0:
                continue
            uv = projected[indices, :2]
            x = np.clip(np.rint(uv[:, 0]).astype(np.int64), 0, width - 1)
            y = np.clip(np.rint(uv[:, 1]).astype(np.int64), 0, height - 1)
            sampled = pixels[y, x]
            bounds = _crop_bounds(uv, width, height, crop_padding)
            crop = image.crop(bounds)
            crop_array = np.asarray(crop, dtype=np.float32) / 255.0
            mean_rgb = sampled.mean(axis=0)
            std_rgb = sampled.std(axis=0)
            saturation = float((sampled.max(axis=1) - sampled.min(axis=1)).mean())
            quality = float(len(indices) * np.clip(facing[indices], 0, 1).mean())
            observations[part_id].append(
                ViewObservation(
                    image_path=view.image_path,
                    crop=crop,
                    quality=quality,
                    visible_faces=len(indices),
                    mean_rgb=mean_rgb,
                    std_rgb=std_rgb,
                    saturation=saturation,
                    texture=_texture_score(crop_array),
                )
            )

    for part_id in part_ids:
        observations[part_id].sort(key=lambda item: item.quality, reverse=True)
        observations[part_id] = observations[part_id][:max_views]
        if not observations[part_id]:
            if not allow_global_fallback:
                raise ValueError(
                    f"部件{part_id}没有可见面；检查坐标、相机或视角覆盖。"
                    "全局图像不能静默替代部件观测。"
                )
            observations[part_id] = _fallback_observations(views, max_views)
    return observations


def _project(
    points: np.ndarray,
    c2w: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    world_to_camera = np.linalg.inv(c2w)
    homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    camera = (world_to_camera @ homogeneous.T).T[:, :3]
    depth = -camera[:, 2]
    safe_depth = np.maximum(depth, 1e-12)
    u = fx * camera[:, 0] / safe_depth + cx
    v = fy * (-camera[:, 1]) / safe_depth + cy
    return np.stack([u, v, depth], axis=1)


def _zbuffer_visibility(
    projected: np.ndarray,
    valid: np.ndarray,
    width: int,
    height: int,
    max_size: int,
) -> np.ndarray:
    visible = np.zeros(len(projected), dtype=bool)
    indices = np.flatnonzero(valid)
    if len(indices) == 0:
        return visible
    grid_w = min(width, max_size)
    grid_h = min(height, max_size)
    gx = np.clip((projected[indices, 0] * grid_w / width).astype(np.int64), 0, grid_w - 1)
    gy = np.clip((projected[indices, 1] * grid_h / height).astype(np.int64), 0, grid_h - 1)
    flat = gy * grid_w + gx
    depth_buffer = np.full(grid_w * grid_h, np.inf, dtype=np.float64)
    np.minimum.at(depth_buffer, flat, projected[indices, 2])
    nearest = depth_buffer[flat]
    tolerance = np.maximum(1e-4, nearest * 0.025)
    visible[indices] = projected[indices, 2] <= nearest + tolerance
    return visible


def _crop_bounds(
    uv: np.ndarray,
    width: int,
    height: int,
    padding_fraction: float,
) -> tuple[int, int, int, int]:
    low = uv.min(axis=0)
    high = uv.max(axis=0)
    size = np.maximum(high - low, 12.0)
    low -= size * padding_fraction
    high += size * padding_fraction
    left = int(np.clip(np.floor(low[0]), 0, width - 1))
    top = int(np.clip(np.floor(low[1]), 0, height - 1))
    right = int(np.clip(np.ceil(high[0]) + 1, left + 1, width))
    bottom = int(np.clip(np.ceil(high[1]) + 1, top + 1, height))
    return left, top, right, bottom


def _texture_score(image: np.ndarray) -> float:
    if image.size == 0 or min(image.shape[:2]) < 2:
        return 0.0
    gray = image.mean(axis=2)
    dx = np.abs(np.diff(gray, axis=1)).mean()
    dy = np.abs(np.diff(gray, axis=0)).mean()
    return float(0.5 * (dx + dy))


def _fallback_observations(
    views: tuple[CameraView, ...],
    max_views: int,
) -> list[ViewObservation]:
    result: list[ViewObservation] = []
    for view in views[:max_views]:
        with Image.open(view.image_path) as source:
            crop = source.convert("RGB")
        array = np.asarray(crop, dtype=np.float32) / 255.0
        pixels = array.reshape(-1, 3)
        result.append(
            ViewObservation(
                image_path=view.image_path,
                crop=crop,
                quality=1e-3,
                visible_faces=0,
                is_global_context=True,
                mean_rgb=pixels.mean(axis=0),
                std_rgb=pixels.std(axis=0),
                saturation=float((pixels.max(axis=1) - pixels.min(axis=1)).mean()),
                texture=_texture_score(array),
            )
        )
    return result
