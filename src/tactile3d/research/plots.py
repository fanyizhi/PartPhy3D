"""与CSV一一对应的结果图，不平滑、不补写论文结果。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {
    "mass_only": "#777777",
    "shared_response": "#0072B2",
    "direct_response": "#D55E00",
}


def plot_physical_fields(
    inputs: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    predictions: dict[str, np.ndarray],
    output: Path,
) -> None:
    """同一米制切片上的真值、MAP状态与编辑状态，密度和占用分开着色。"""
    from tactile3d.research.construction import (
        GRID_SHAPE,
        TEST_EDIT,
        candidate_masks,
        decode_state,
    )

    index = int(np.flatnonzero(inputs["split"] == 2)[0])
    decoded = decode_state(predictions["shared_response_logits"], inputs["priors"])
    mask = candidate_masks()
    actual = mask[int(labels["construction"][index])] * labels["fill"][index]
    predicted = (
        mask[int(decoded["probability"][index].argmax())] * decoded["fill"][index]
    )
    rho = decoded["density"][index].copy()
    edited_rho = rho.copy()
    edited_rho[TEST_EDIT[0]] = TEST_EDIT[1]
    density = (
        actual @ labels["density"][index],
        predicted @ rho,
        predicted @ edited_rho,
    )
    fill = (actual.sum(1), predicted.sum(1), predicted.sum(1))
    pitch = inputs["pitch_m"][index]
    extent = (0, GRID_SHAPE[0] * pitch, 0, GRID_SHAPE[2] * pitch)
    x, z = np.meshgrid(
        (np.arange(GRID_SHAPE[0]) + 0.5) * pitch,
        (np.arange(GRID_SHAPE[2]) + 0.5) * pitch,
    )
    columns = [x.ravel(), z.ravel()]
    for values in (*density, *fill):
        columns.append(values.reshape(GRID_SHAPE)[:, GRID_SHAPE[1] // 2, :].T.ravel())
    np.savetxt(
        output.with_suffix(".csv"),
        np.column_stack(columns),
        delimiter=",",
        header="x_m,z_m,truth_density,predicted_density,steel_density,truth_fill,predicted_fill,steel_fill",
        comments="",
    )
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.7), layout="constrained")
    images = []
    for row, fields in enumerate((density, fill)):
        for column, values in enumerate(fields):
            section = values.reshape(GRID_SHAPE)[:, GRID_SHAPE[1] // 2, :].T
            image = axes[row, column].imshow(
                section,
                origin="lower",
                extent=extent,
                interpolation="nearest",
                cmap="viridis",
                vmin=0,
                vmax=max(v.max() for v in density) if row == 0 else 1,
            )
            axes[row, column].set(xlabel="x (m)", ylabel="z (m)")
            if column == 2:
                images.append(image)
    for axis, label in zip(
        axes[0],
        ("Synthetic reference", "Learned MAP state", "Explicit steel edit"),
        strict=True,
    ):
        axis.set_title(label, fontsize=10)
    fig.colorbar(
        images[0], ax=axes[0].tolist(), label="Effective density (kg/m³)", shrink=0.8
    )
    fig.colorbar(
        images[1], ax=axes[1].tolist(), label="Macro occupied fraction", shrink=0.8
    )
    fig.savefig(output.with_suffix(".svg"))
    fig.savefig(output.with_suffix(".png"), dpi=240)
    plt.close(fig)


def plot_results(
    output: Path,
    metrics: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    history: list[dict[str, Any]],
) -> None:
    """导出学习曲线、质量散点、分种子误差及校准区间。"""
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    seed = min(row["seed"] for row in predictions)
    for method, color in COLORS.items():
        curve = [
            row for row in history if row["method"] == method and row["seed"] == seed
        ]
        axes[0, 0].plot(
            [row["epoch"] for row in curve],
            [row["loss"] for row in curve],
            color=color,
            label=method,
        )
        values = [
            row
            for row in predictions
            if row["method"] == method and row["seed"] == seed
        ]
        axes[0, 1].scatter(
            [row["truth_kg"] for row in values],
            [row["prediction_kg"] for row in values],
            s=12,
            alpha=0.65,
            color=color,
            label=method,
        )
    axes[0, 0].set(
        xlabel="Training epoch", ylabel="Training objective (different losses)"
    )
    axes[0, 0].legend(fontsize=7)
    all_mass = [
        row[key] for row in predictions for key in ("truth_kg", "prediction_kg")
    ]
    low, high = min(all_mass) * 0.8, max(all_mass) * 1.2
    axes[0, 1].plot([low, high], [low, high], color="black", lw=0.8, ls="--")
    axes[0, 1].set(
        xscale="log",
        yscale="log",
        xlabel="Reference mass (kg)",
        ylabel="Predicted mass (kg)",
        xlim=(low, high),
        ylim=(low, high),
    )
    axes[0, 1].legend(fontsize=7)
    conditions = ("original", "seen_material_edits", "unseen_steel_edit")
    for offset, (method, color) in enumerate(COLORS.items()):
        for x, condition in enumerate(conditions):
            values = [
                row["ALDE_ln"]
                for row in metrics
                if row["method"] == method and row["condition"] == condition
            ]
            position = x + (offset - 1) * 0.22
            axes[1, 0].scatter(
                np.full(len(values), position),
                values,
                color=color,
                s=18,
                label=method if x == 0 else None,
            )
            axes[1, 0].plot(
                [position - 0.07, position + 0.07],
                [np.mean(values)] * 2,
                color=color,
                lw=2,
            )
    axes[1, 0].set(
        xticks=range(3),
        xticklabels=("Original", "Seen edits", "Unseen steel"),
        ylabel="Mean absolute log mass error (ln)",
    )
    axes[1, 0].legend(fontsize=7)
    chosen = [
        row
        for row in predictions
        if row["method"] == "shared_response" and row["seed"] == seed
    ]
    chosen = sorted(chosen, key=lambda row: row["truth_kg"])[:24]
    x = np.arange(len(chosen))
    prediction = np.array([row["prediction_kg"] for row in chosen])
    lower, upper = (
        np.array([row[key] for row in chosen]) for key in ("lower90_kg", "upper90_kg")
    )
    axes[1, 1].errorbar(
        x,
        prediction,
        yerr=(prediction - lower, upper - prediction),
        fmt="o",
        ms=3,
        color=COLORS["shared_response"],
        lw=0.8,
        label="Calibrated 90% interval",
    )
    axes[1, 1].scatter(
        x,
        [row["truth_kg"] for row in chosen],
        marker="x",
        color="black",
        s=20,
        label="Reference",
    )
    axes[1, 1].set(
        xlabel="Test object (first 24 sorted by reference)",
        ylabel="Mass (kg)",
        yscale="log",
    )
    axes[1, 1].legend(fontsize=7)
    for label, axis in zip("abcd", axes.flat, strict=True):
        axis.text(-0.12, 1.04, label, transform=axis.transAxes, fontweight="bold")
    fig.savefig(output / "learning_results.svg")
    fig.savefig(output / "learning_results.png", dpi=240)
    plt.close(fig)
