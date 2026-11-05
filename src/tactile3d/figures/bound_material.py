"""同一绑定Mesh的正交深度渲染和材料编辑图，不修改输入照片。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image

from tactile3d.physics.material_binding import BoundSurface
from tactile3d.physics.part_state import PhysicalObject

RESOLUTION = 300
ELEVATION_DEGREES = 22.0
AZIMUTHS = (-55.0, -125.0, 45.0)
MARGIN_RATIO = 0.10
FIGURE_DPI = 200
MIN_RESOLUTION = 32
MAX_RESOLUTION = 2048
TRIANGLE_EPSILON = 1e-12
LIGHT_DIRECTION = (-0.4, -0.6, 1.0)
AMBIENT_WEIGHT = 0.6
DIFFUSE_WEIGHT = 0.4
VIEW_FIGURE_SIZE = (7.0, 8.5)
PHYSICS_FIGURE_SIZE = (10, 3.3)
STATE_COLORS = ("#997047", "#7189A4")
COM_BAR_WIDTH = 0.32


class SurfaceGeometry(Protocol):
    vertices_m: np.ndarray
    faces: np.ndarray


def raster_face_ids(
    surface: SurfaceGeometry, azimuth: float, resolution: int = RESOLUTION
) -> np.ndarray:
    """正交相机逐像素深度测试，前后图复用完全相同的面可见性。

    Returns:
        每像素可见三角面ID；背景为-1。
    """
    if (
        type(resolution) is not int
        or not MIN_RESOLUTION <= resolution <= MAX_RESOLUTION
        or not np.isfinite(azimuth)
    ):
        raise ValueError("渲染需要有限方位角和32到2048的整数分辨率")
    az, el = np.deg2rad([azimuth, ELEVATION_DEGREES])
    direction = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    right = np.array([-np.sin(az), np.cos(az), 0.0])
    up = np.cross(direction, right)
    vertices = (
        surface.vertices_m - (surface.vertices_m.min(0) + surface.vertices_m.max(0)) / 2
    )
    coordinates = vertices @ np.column_stack((right, up, direction))
    span = max(np.ptp(coordinates[:, :2], axis=0))
    if span <= 0:
        raise ValueError("退化表面无法投影")
    uv = coordinates[:, :2] / span * resolution * (1 - 2 * MARGIN_RATIO)
    uv[:, 1] *= -1
    uv += resolution / 2
    # 正交相机的深度可线性插值；朝观察者的坐标越大，表面越靠前。
    depth = np.full((resolution, resolution), -np.inf)
    ids = np.full((resolution, resolution), -1, dtype=np.int32)
    for index, face in enumerate(surface.faces):
        triangle = uv[face]
        lower = np.maximum(0, np.floor(triangle.min(0)).astype(int))
        upper = np.minimum(resolution, np.ceil(triangle.max(0)).astype(int) + 1)
        if np.any(upper <= lower):
            continue
        a, b, c = triangle
        denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(denominator) <= TRIANGLE_EPSILON:
            continue
        x, y = np.meshgrid(
            np.arange(lower[0], upper[0]) + 0.5, np.arange(lower[1], upper[1]) + 0.5
        )
        w0 = ((b[1] - c[1]) * (x - c[0]) + (c[0] - b[0]) * (y - c[1])) / denominator
        w1 = ((c[1] - a[1]) * (x - c[0]) + (a[0] - c[0]) * (y - c[1])) / denominator
        w2 = 1 - w0 - w1
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        candidate = (
            w0 * coordinates[face[0], 2]
            + w1 * coordinates[face[1], 2]
            + w2 * coordinates[face[2], 2]
        )
        current = depth[lower[1] : upper[1], lower[0] : upper[0]]
        update = inside & (candidate > current)
        current[update] = candidate[update]
        ids[lower[1] : upper[1], lower[0] : upper[0]][update] = index
    return ids


def shade_surface(
    surface: BoundSurface, state: PhysicalObject, face_ids: np.ndarray
) -> np.ndarray:
    return shade_face_colors(surface, surface.face_rgba(state), face_ids)


def shade_face_colors(
    surface: SurfaceGeometry, rgba: np.ndarray, face_ids: np.ndarray
) -> np.ndarray:
    """独立于物理表示的示意色着色；体素和Mesh资产共用同一深度逻辑。"""
    if face_ids.ndim != 2 or not face_ids.size or face_ids.dtype.kind not in "iu":
        raise ValueError("渲染需要二维整数面编号")
    if face_ids.min() < -1 or face_ids.max() >= len(surface.faces):
        raise ValueError("面编号超出绑定表面")
    triangles = surface.vertices_m[surface.faces]
    normal = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    light = np.array(LIGHT_DIRECTION)
    light /= np.linalg.norm(light)
    brightness = AMBIENT_WEIGHT + DIFFUSE_WEIGHT * np.abs(normal @ light)
    if rgba.shape != (len(surface.faces), 4):
        raise ValueError("RGBA必须与每个三角面一一对应")
    color = np.rint(rgba[:, :3] * brightness[:, None]).astype(np.uint8)
    result = np.full((*face_ids.shape, 3), 255, dtype=np.uint8)
    visible = face_ids >= 0
    result[visible] = color[face_ids[visible]]
    return result


def render_comparison(
    before: PhysicalObject,
    after: PhysicalObject,
    surface: BoundSurface,
    part_id: str,
    region_id: str,
    output: Path,
) -> list[dict[str, Any]]:
    """输出多视角图片、可见面、目标掩码及非目标像素不变核对。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = surface.selected_faces(part_id, region_id)
    region = next(
        r for r in surface.regions if (r.part_id, r.region_id) == (part_id, region_id)
    )
    names = [
        state.part(part_id).materials[region.material_slot].name
        for state in (before, after)
    ]
    figure, axes = plt.subplots(
        len(AZIMUTHS), 2, figsize=VIEW_FIGURE_SIZE, layout="constrained"
    )
    rows = []
    for index, azimuth in enumerate(AZIMUTHS):
        ids = raster_face_ids(surface, azimuth)
        original, edited = (
            shade_surface(surface, state, ids) for state in (before, after)
        )
        # 背景ID为-1，先约束索引再屏蔽，避免把最后一个面误当背景材质。
        mask = (ids >= 0) & selected[np.maximum(ids, 0)]
        changed = np.any(original != edited, axis=2)
        outside = int((changed & ~mask).sum())
        if outside:
            raise ValueError("存在非目标区域像素变化，渲染与状态绑定不一致")
        prefix = f"view_{index:02d}"
        np.save(output / f"{prefix}_face_ids.npy", ids, allow_pickle=False)
        Image.fromarray(mask.astype(np.uint8) * 255).save(
            output / f"{prefix}_target.png"
        )
        for column, (pixels, label) in enumerate(
            ((original, "before"), (edited, "after"))
        ):
            Image.fromarray(pixels).save(output / f"{prefix}_{label}.png")
            axes[index, column].imshow(pixels)
            axes[index, column].set(
                xlabel="u (pixel)",
                ylabel="v (pixel)",
                xticks=[0, RESOLUTION - 1],
                yticks=[0, RESOLUTION - 1],
            )
            axes[index, column].set_title(
                f"{label.capitalize()}: {names[column]} | azimuth {azimuth:g}°",
                fontsize=10,
            )
        rows.append(
            {
                "view": index,
                "azimuth_degrees": azimuth,
                "visible_target_pixels": int(mask.sum()),
                "changed_pixels": int(changed.sum()),
                "outside_target_changed": outside,
            }
        )
    figure.suptitle(
        f"{part_id}/{region_id}\nSchematic colors; geometry and occupancy fixed",
        fontsize=11,
    )
    figure.savefig(output / "material_edit_views.svg")
    figure.savefig(output / "material_edit_views.png", dpi=FIGURE_DPI)
    plt.close(figure)
    return rows


def plot_physics(
    before: PhysicalObject, after: PhysicalObject, report: dict[str, Any], output: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=PHYSICS_FIGURE_SIZE, layout="constrained")
    colors = STATE_COLORS
    axes[0].bar(
        ["Before", "After"], [before.total.mass_kg, after.total.mass_kg], color=colors
    )
    axes[0].set(ylabel="Object mass (kg)")
    positions = np.arange(3)
    for offset, state, color, label in zip(
        (-COM_BAR_WIDTH / 2, COM_BAR_WIDTH / 2),
        (before, after),
        colors,
        ("Before", "After"),
        strict=True,
    ):
        axes[1].bar(
            positions + offset,
            np.array(state.total.properties()["center_of_mass_m"]) * 1000,
            width=COM_BAR_WIDTH,
            color=color,
            label=label,
        )
    axes[1].set(
        xticks=positions, xticklabels=["x", "y", "z"], ylabel="COM coordinate (mm)"
    )
    axes[1].legend()
    values = [
        report[key].get("youngs_modulus_pa") for key in ("old_material", "new_material")
    ]
    if all(value is not None for value in values):
        axes[2].bar(["Before", "After"], np.array(values) / 1e9, color=colors)
    else:
        axes[2].text(
            0.5, 0.5, "Not specified", ha="center", transform=axes[2].transAxes
        )
    axes[2].set(
        ylabel="Target material Young's modulus (GPa)",
        xlabel="Declared values; not stiffness",
    )
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(output / "material_edit_physics.svg")
    figure.savefig(output / "material_edit_physics.png", dpi=FIGURE_DPI)
    plt.close(figure)
