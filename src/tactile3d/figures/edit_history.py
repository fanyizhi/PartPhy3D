"""编辑事务的数值与边界可视化，不将合成结果标成真实物体精度。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from tactile3d.mesh.state_surface import part_surface
from tactile3d.physics.edit_program import ProgramResult

PALETTE = ("#5A76A8", "#C96A6A", "#4FA38F", "#8573B5", "#D1A14B")
PREVIEW_DPI = 180
MAX_GEOMETRY_PANELS = 4
MAX_SLICE_PIXELS = 2_000_000
PLOT_STYLE = {
    "font.family": "DejaVu Serif",
    "font.size": 10,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
}


def save_figure(figure: Figure, stem: Path) -> None:
    for extension in ("svg", "pdf", "png"):
        figure.savefig(
            stem.with_suffix(f".{extension}"), dpi=PREVIEW_DPI, bbox_inches="tight"
        )


def preview_state_indices(count: int) -> tuple[int, ...]:
    if type(count) is not int or count <= 0:
        raise ValueError("预览需要正整数状态数量")
    return tuple(
        int(index)
        for index in np.unique(
            np.linspace(0, count - 1, min(count, MAX_GEOMETRY_PANELS))
            .round()
            .astype(int)
        )
    )


def plot_edit_history(
    result: ProgramResult,
    output: Path,
    reference: list[dict[str, Any]] | None = None,
) -> None:
    """导出质量、质心、主惯量和缓存计数，保留离散编辑的真实数值。

    Args:
        result: 已通过全量重算的事务。
        output: 已创建的输出目录。
        reference: 可选解析合成真值，不能传入模型自身预测冒充真值。
    """
    properties = [state.total.properties() for state in result.states]
    positions = np.arange(len(properties))
    labels = [f"S{index}" for index in positions]
    mass = np.array([row["mass_kg"] for row in properties])
    centers = np.array([row["center_of_mass_m"] for row in properties])
    inertia = np.array([np.linalg.eigvalsh(row["inertia_kg_m2"]) for row in properties])
    with plt.rc_context(PLOT_STYLE):
        figure, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
        axes[0, 0].plot(
            positions, mass, "o-", color=PALETTE[0], label="Voxel integration"
        )
        if reference is not None:
            axes[0, 0].plot(
                positions,
                [row["mass_kg"] for row in reference],
                "x",
                color="#222222",
                markersize=9,
                label="Analytic synthetic reference",
            )
        axes[0, 0].set_ylabel("Mass (kg)")
        for index, name in enumerate(("x", "y", "z")):
            axes[0, 1].plot(
                positions,
                centers[:, index],
                "o-",
                color=PALETTE[index],
                label=f"COM {name}",
            )
            axes[1, 0].plot(
                positions,
                inertia[:, index],
                "o-",
                color=PALETTE[index],
                label=f"Principal {index + 1}",
            )
        axes[0, 1].set_ylabel("COM coordinate (m)")
        axes[1, 0].set_ylabel("Principal inertia (kg m²)")
        steps = np.arange(1, len(result.states))
        for offset, key, color, label in (
            (-0.18, "recomputed_parts", PALETTE[1], "Recomputed"),
            (0.18, "reused_parts", PALETTE[0], "Reused"),
        ):
            axes[1, 1].bar(
                steps + offset,
                [len(row[key]) for row in result.reports],
                width=0.36,
                color=color,
                label=label,
            )
        axes[1, 1].set_ylabel("Parts (count)")
        axes[1, 1].set_xticks(steps, labels[1:])
        maximum_parts = max(
            len(row[key])
            for row in result.reports
            for key in ("recomputed_parts", "reused_parts")
        )
        axes[1, 1].set_ylim(0, max(1, maximum_parts) * 1.35 + 0.2)
        axes[1, 1].yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        for index, axis in enumerate(axes.flat):
            axis.set_xlabel("Edit state")
            if index < 3:
                axis.set_xticks(positions, labels)
            axis.legend(loc="best", fontsize=8)
            axis.text(
                -0.13,
                1.04,
                "abcd"[index],
                transform=axis.transAxes,
                fontweight="bold",
                fontsize=12,
            )
        save_figure(figure, output / "physical_updates")
        plt.close(figure)
    plot_geometry(result, output)
    plot_density_slices(result, output)


def plot_geometry(result: ProgramResult, output: Path) -> None:
    """显示共同坐标系中的体素边界；超过四个状态时等距选取并记录编号。"""
    selected = preview_state_indices(len(result.states))
    points = np.concatenate(
        [
            part.centers_world_m
            for index in selected
            for part in result.states[index].parts
        ]
    )
    pitch = result.states[0].parts[0].pitch_m
    lower, upper = points.min(axis=0) - pitch, points.max(axis=0) + pitch
    with plt.rc_context(PLOT_STYLE):
        figure = plt.figure(figsize=(11, 8), layout="constrained")
        for panel, state_index in enumerate(selected, start=1):
            state = result.states[state_index]
            axis = figure.add_subplot(
                2, 2, panel, projection="3d", computed_zorder=False
            )
            for index, part in enumerate(state.parts):
                if not np.any(part.density_kg_m3 > 0):
                    continue
                mesh = part_surface(state, part.part_id)
                axis.add_collection3d(
                    Poly3DCollection(
                        mesh.triangles,
                        facecolors=PALETTE[index % len(PALETTE)],
                        edgecolors="none",
                        alpha=0.9,
                        label=part.part_id,
                        zorder=1,
                    )
                )
            axis.scatter(
                *state.total.properties()["center_of_mass_m"],
                marker="x",
                color="#222222",
                s=50,
                label="COM (overlay)",
                zorder=10,
            )
            axis.set(
                xlabel="x (m)",
                ylabel="y (m)",
                zlabel="z (m)",
                xlim=(lower[0], upper[0]),
                ylim=(lower[1], upper[1]),
                zlim=(lower[2], upper[2]),
                title=f"S{state_index}: {state.total.mass_kg:.4f} kg",
            )
            axis.set_box_aspect(upper - lower)
            axis.locator_params(nbins=3)
            axis.legend(loc="upper left", fontsize=8)
        save_figure(figure, output / "geometry_updates")
        plt.close(figure)


def plot_density_slices(result: ProgramResult, output: Path) -> None:
    """显示第一个编辑目标的真实体素截面，以区分外形相同的实心和空心。"""
    selected = preview_state_indices(len(result.states))
    target = result.program.steps[0].target_part_id
    slices = []
    for index in selected:
        state = result.states[index]
        part = state.part(target)
        grid = state.grid_indices(part)
        layers = np.unique(grid[:, 1])
        middle = (layers.min() + layers.max()) / 2
        layer = layers[np.argmin(np.abs(layers - middle))]
        mask = grid[:, 1] == layer
        points = grid[mask][:, [0, 2]]
        lower, upper = points.min(axis=0), points.max(axis=0)
        counts = upper - lower + 1
        if int(counts[0]) * int(counts[1]) > MAX_SLICE_PIXELS:
            raise ValueError("截面超过显示预算，请显式减小空间域或调整预算")
        image = np.full((counts[1], counts[0]), np.nan)
        offsets = points - lower
        image[offsets[:, 1], offsets[:, 0]] = part.density_kg_m3[mask]
        origin = state.grid_origin_m[[0, 2]]
        edges = origin + np.array([lower - 0.5, upper + 0.5]) * part.pitch_m
        slices.append((index, image, edges))
    maximum = max(float(np.nanmax(image)) for _, image, _ in slices)
    with plt.rc_context(PLOT_STYLE):
        figure, axes = plt.subplots(2, 2, figsize=(9, 8), constrained_layout=True)
        artist = None
        for axis, (index, image, edges) in zip(axes.flat, slices):
            artist = axis.imshow(
                image,
                origin="lower",
                interpolation="nearest",
                cmap="viridis",
                vmin=0,
                vmax=max(maximum, 1.0),
                extent=(edges[0, 0], edges[1, 0], edges[0, 1], edges[1, 1]),
                aspect="equal",
            )
            axis.set(xlabel="x (m)", ylabel="z (m)", title=f"S{index}: {target}")
            axis.set_xticks([edges[0, 0], edges[1, 0]])
            axis.locator_params(axis="y", nbins=5)
        for axis in list(axes.flat)[len(slices) :]:
            axis.set_visible(False)
        figure.colorbar(
            artist,
            ax=list(axes.flat)[: len(slices)],
            label="Effective density (kg/m³)",
            shrink=0.8,
        )
        save_figure(figure, output / "density_slices")
        plt.close(figure)
