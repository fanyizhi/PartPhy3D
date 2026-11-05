"""Mesh原生材料状态的外观与物理图，所有图具有数值来源。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from tactile3d.figures.bound_material import (
    AZIMUTHS,
    FIGURE_DPI,
    raster_face_ids,
    shade_face_colors,
)
from tactile3d.physics.mesh_asset_io import display_mesh, face_colors
from tactile3d.physics.mesh_native import MeshPhysicalObject

VIEW_SIZE = (7.0, 8.5)
PHYSICS_SIZE = (10, 3.5)
STATE_COLORS = ("#997047", "#7189A4")


def plot_mesh_edit(
    before: MeshPhysicalObject,
    after: MeshPhysicalObject,
    part_id: str,
    region_id: str,
    output: Path,
) -> list[dict[str, Any]]:
    """重用同一Mesh深度图，核对换材只影响目标可见面。

    Returns:
        逐视角可见目标、变化及非目标变化像素数量。
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface = display_mesh(before)
    if any(
        a.domain is not b.domain
        for a, b in zip(before.regions, after.regions, strict=True)
    ):
        raise ValueError("该对比只支持保持区域几何不变的材料编辑")
    target = next(
        i
        for i, r in enumerate(before.regions)
        if (r.part_id, r.region_id) == (part_id, region_id)
    )
    colors = [face_colors(state, surface) for state in (before, after)]
    other_faces = surface.face_regions != target
    if not np.array_equal(colors[0][other_faces], colors[1][other_faces]):
        raise ValueError("非目标区域材质被改写")
    figure, axes = plt.subplots(
        len(AZIMUTHS), 2, figsize=VIEW_SIZE, layout="constrained"
    )
    rows = []
    for view, azimuth in enumerate(AZIMUTHS):
        ids = raster_face_ids(surface, azimuth)
        visible = (ids >= 0) & (surface.face_regions[np.maximum(ids, 0)] == target)
        pixels = [shade_face_colors(surface, color, ids) for color in colors]
        changed = np.any(pixels[0] != pixels[1], axis=2)
        outside = int(np.count_nonzero(changed & ~visible))
        if outside:
            raise ValueError("换材造成非目标像素改变，检查区域绑定")
        for column, (label, image) in enumerate(
            zip(("Before", "After"), pixels, strict=True)
        ):
            axes[view, column].imshow(image)
            axes[view, column].set(
                xlabel="u (pixel)",
                ylabel="v (pixel)",
                title=f"{label} | azimuth {azimuth:g}°",
            )
            Image.fromarray(image).save(output / f"view_{view:02d}_{label.lower()}.png")
        np.save(output / f"view_{view:02d}_face_ids.npy", ids, allow_pickle=False)
        rows.append(
            {
                "view": view,
                "visible_target_pixels": int(visible.sum()),
                "changed_pixels": int(changed.sum()),
                "outside_target_changed": outside,
            }
        )
    figure.suptitle(
        "Mesh-native region edit | schematic colors, no voxelization", fontsize=11
    )
    figure.savefig(output / "mesh_edit_views.svg")
    figure.savefig(output / "mesh_edit_views.png", dpi=FIGURE_DPI)
    plt.close(figure)
    return rows


def plot_mesh_physics(
    states: list[MeshPhysicalObject], labels: list[str], output: Path
) -> None:
    """展示已知/预测状态的质量与中心惯量，弹性常数不充当结构刚度。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(states) != len(labels) or not states:
        raise ValueError("物理图的状态与标签必须非空且数量一致")
    figure, axes = plt.subplots(1, 3, figsize=PHYSICS_SIZE, layout="constrained")
    x = np.arange(len(states))
    axes[0].bar(x, [s.properties["mass_kg"] for s in states], color=STATE_COLORS[1])
    axes[0].set(xticks=x, xticklabels=labels, ylabel="Mass (kg)")
    for component, label in enumerate(("x", "y", "z")):
        axes[1].plot(
            x,
            [s.properties["center_of_mass_m"][component] for s in states],
            "o-",
            label=label,
        )
        axes[2].plot(
            x,
            [s.properties["inertia_kg_m2"][component][component] for s in states],
            "o-",
            label=f"I{label}{label}",
        )
    for axis, label in zip(
        axes[1:], ("COM coordinate (m)", "Central inertia (kg m²)"), strict=True
    ):
        axis.set(xticks=x, xticklabels=labels, ylabel=label)
        axis.legend()
    for axis in axes:
        axis.set_xlabel("Physical state")
        axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(output / "mesh_physics.svg")
    figure.savefig(output / "mesh_physics.png", dpi=FIGURE_DPI)
    plt.close(figure)


def plot_mesh_sections(
    states: list[MeshPhysicalObject], labels: list[str], output: Path
) -> None:
    """直接绘制材料域的平面截面，空腔保留为空白，未知弹性标灰。

    Args:
        states: 共享坐标系的物理状态；只显示中央XY截面，不改写体域。
        labels: 各行名称。
        output: 已存在的结果目录。
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize
    from matplotlib.patches import Patch, PathPatch
    from matplotlib.path import Path as PlotPath

    if not states or len(states) != len(labels):
        raise ValueError("截面状态与标签必须非空且数量一致")
    vertices = np.concatenate([r.domain.vertices_m for s in states for r in s.regions])
    lower, upper = vertices.min(0), vertices.max(0)
    height = float((lower[2] + upper[2]) / 2)
    fields = (
        ("density_kg_m3", "Density", "kg m$^{-3}$", "viridis"),
        ("fill_fraction", "Macro fill", "Fraction", "Blues"),
        ("youngs_modulus_pa", "Young's modulus", "Pa", "magma"),
    )
    figure, axes = plt.subplots(
        len(states),
        3,
        squeeze=False,
        figsize=(10, 3.3 * len(states)),
        layout="constrained",
    )
    for column, (name, title, units, cmap) in enumerate(fields):
        values = [
            getattr(r, name) if name == "fill_fraction" else getattr(r.material, name)
            for state in states
            for r in state.regions
        ]
        known = [float(value) for value in values if value is not None]
        norm = (
            Normalize(0, 1)
            if name == "fill_fraction"
            else LogNorm(
                min(known) if known else 1, max(known) * (1 + 1e-6) if known else 10
            )
        )
        mapper = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        for row, state in enumerate(states):
            axis = axes[row, column]
            for region in state.regions:
                contours = region.domain.xy_section(height)
                if not contours:
                    continue
                coordinates, codes = [], []
                for contour in contours:
                    coordinates.extend(np.vstack((contour, contour[0])))
                    codes.extend(
                        [PlotPath.MOVETO]
                        + [PlotPath.LINETO] * (len(contour) - 1)
                        + [PlotPath.CLOSEPOLY]
                    )
                value = (
                    getattr(region, name)
                    if name == "fill_fraction"
                    else getattr(region.material, name)
                )
                axis.add_patch(
                    PathPatch(
                        PlotPath(coordinates, codes),
                        facecolor="#cccccc" if value is None else mapper.to_rgba(value),
                        edgecolor="#444444",
                        linewidth=0.5,
                        hatch="///" if value is None else None,
                    )
                )
            margin = (upper - lower)[:2] * 0.06
            axis.set(
                xlim=(lower[0] - margin[0], upper[0] + margin[0]),
                ylim=(lower[1] - margin[1], upper[1] + margin[1]),
                xlabel="x (m)",
                ylabel="y (m)",
                aspect="equal",
            )
            axis.set_title(f"{labels[row]} | {title}", fontsize=10)
            axis.set_xlabel("x (m)", fontsize=9)
            axis.set_ylabel("y (m)", fontsize=9)
            axis.tick_params(labelsize=8)
        colorbar = figure.colorbar(mapper, ax=list(axes[:, column]), shrink=0.75)
        colorbar.set_label(units, fontsize=9)
        colorbar.ax.tick_params(labelsize=8)
    figure.legend(
        handles=[Patch(facecolor="#cccccc", hatch="///", label="Unknown, not zero")],
        fontsize=8,
        loc="outside lower center",
    )
    figure.suptitle(
        f"Specified material regions | XY section z = {height:.4g} m", fontsize=11
    )
    figure.savefig(output / "mesh_property_sections.png", dpi=FIGURE_DPI)
    figure.savefig(output / "mesh_property_sections.svg")
    plt.close(figure)
