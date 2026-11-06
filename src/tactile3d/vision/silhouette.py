"""针孔相机下的三角轮廓并集，用于几何投影检查。"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image, ImageDraw


def silhouette(
    vertices: np.ndarray,
    faces: np.ndarray,
    pose: np.ndarray,
    camera: dict[str, Any],
    resolution: int,
) -> np.ndarray:
    """轮廓只需投影并集；不能用于带遮挡的逐面材质可见性。"""
    if resolution <= 0:
        raise ValueError("轮廓分辨率必须为正")
    local = (vertices - pose[:3, 3]) @ pose[:3, :3]
    depth = -local[:, 2]
    positive = depth > 1e-8
    safe_depth = np.where(positive, depth, 1)
    uv = (
        np.column_stack(
            (
                (camera["fl_x"] * local[:, 0] / safe_depth + camera["cx"])
                / camera["w"],
                (-camera["fl_y"] * local[:, 1] / safe_depth + camera["cy"])
                / camera["h"],
            )
        )
        * resolution
    )
    canvas = Image.new("L", (resolution, resolution))
    painter = ImageDraw.Draw(canvas)
    for face in faces[np.all(positive[faces], axis=1)]:
        painter.polygon([tuple(point) for point in uv[face]], fill=255)
    return np.asarray(canvas) > 0


def iou(first: np.ndarray, second: np.ndarray) -> float:
    """两个空轮廓不构成有效的几何验收。"""
    if first.shape != second.shape:
        raise ValueError("轮廓尺寸不一致")
    union = np.count_nonzero(first | second)
    if union == 0:
        raise ValueError("空轮廓不能用于坐标验收")
    return float(np.count_nonzero(first & second) / union)
