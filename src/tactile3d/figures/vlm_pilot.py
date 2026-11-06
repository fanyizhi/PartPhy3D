"""真实VLM先导图：质量对照、部件提议和材料先验，保留各自证据边界。"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
from PIL import Image

COLORS = ("#5A76A8", "#C96A6A", "#4FA38F", "#8573B5", "#D1A14B", "#747474")
DPI = 300
STYLE = {
    "font.family": "DejaVu Serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.linewidth": 0.7,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "svg.fonttype": "none",
    "savefig.facecolor": "white",
}


def save_figure(figure: Any, path: Path) -> None:
    figure.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=DPI)
    plt.close(figure)


def plot_mass(
    report: dict[str, Any], predictions: list[dict[str, Any]], output: Path
) -> None:
    """每个原始点都显示，不把主观质量范围绘制成统计置信区间。"""
    rows = report["evaluation"]
    if not rows:
        raise ValueError("没有有效预测，不能生成质量对比图")
    by_id = {p["scene_id"]: p for p in predictions}
    true = np.array([p["target_mass_kg"] for p in rows])
    predicted = np.array([p["predicted_mass_kg"] for p in rows])
    low = np.array([by_id[p["scene_id"]]["mass_kg"]["plausible_low"] for p in rows])
    high = np.array([by_id[p["scene_id"]]["mass_kg"]["plausible_high"] for p in rows])
    x = np.arange(len(rows))
    figure, axes = plt.subplots(1, 3, figsize=(10.4, 3.0), layout="constrained")
    axes[0].scatter(x - 0.10, true, color="#222222", marker="s", label="ABO mass label")
    axes[0].errorbar(
        x + 0.10,
        predicted,
        yerr=np.vstack((predicted - low, high - predicted)),
        fmt="o",
        color=COLORS[0],
        capsize=3,
        label="VLM estimate + plausibility bounds",
    )
    axes[0].set(
        xticks=x,
        xticklabels=[str(i + 1) for i in x],
        xlabel="Object index (fixed pilot)",
        ylabel="Mass (kg)",
        yscale="log",
    )
    axes[0].legend(loc="upper right", fontsize=6.5)
    lower, upper = (
        min(true.min(), predicted.min()) * 0.65,
        max(true.max(), predicted.max()) * 1.5,
    )
    axes[1].plot(
        [lower, upper], [lower, upper], "--", color="#747474", label="Identity"
    )
    for i, (target, estimate) in enumerate(zip(true, predicted, strict=True)):
        axes[1].scatter(target, estimate, color=COLORS[i % len(COLORS)], s=30)
        axes[1].annotate(
            str(i + 1),
            (target, estimate),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=8,
        )
    axes[1].set(
        xscale="log",
        yscale="log",
        xlim=(lower, upper),
        ylim=(lower, upper),
        xlabel="ABO mass label (kg)",
        ylabel="VLM predicted mass (kg)",
    )
    axes[1].legend()
    ape = np.abs(predicted - true) / true * 100
    axes[2].bar(x, ape, color=COLORS[0], alpha=0.65, label="Image-only VLM")
    axes[2].scatter(x, ape, color="#222222", s=16)
    axes[2].set(
        xticks=x,
        xticklabels=[str(i + 1) for i in x],
        xlabel="Object index (fixed pilot)",
        ylabel="Absolute percentage error (%)",
    )
    axes[2].legend()
    for label, axis in zip("abc", axes, strict=True):
        axis.text(-0.16, 1.03, label, transform=axis.transAxes, weight="bold")
        axis.spines[["top", "right"]].set_visible(False)
    save_figure(figure, output / "mass_comparison")


def plot_part_proposals(
    predictions: list[dict[str, Any]], root: Path, output: Path
) -> None:
    figure, axes = plt.subplots(
        1,
        len(predictions),
        figsize=(3.1 * len(predictions), 3.6),
        squeeze=False,
        layout="constrained",
    )
    for index, (axis, prediction) in enumerate(zip(axes[0], predictions, strict=True)):
        scene = prediction["scene_id"]
        path = root / "scenes" / scene / "images" / f"{scene}_00.png"
        with Image.open(path) as source:
            image = source.convert("RGBA")
            axis.imshow(image)
            width, height = image.size
        for i, part in enumerate(prediction["parts"]):
            box = part["bbox_first_view_1000"]
            if box is None:
                continue
            x0, y0, x1, y1 = (
                np.array(box) * np.array([width, height, width, height]) / 1000
            )
            color = COLORS[i % len(COLORS)]
            axis.add_patch(
                Rectangle(
                    (x0, y0),
                    x1 - x0,
                    y1 - y0,
                    fill=False,
                    edgecolor=color,
                    linewidth=1.1,
                )
            )
            axis.text(
                x0,
                y0,
                part["part_id"],
                color="white",
                fontsize=8,
                bbox={"facecolor": color, "edgecolor": "none", "pad": 1.5},
            )
        axis.set(
            xlabel="u (pixel)",
            ylabel="v (pixel)",
            title=f"{index+1:02d} | {scene.split('_')[0]}",
        )
    figure.suptitle(
        "VLM bounding-box proposals | not validated part segmentation", fontsize=10
    )
    save_figure(figure, output / "part_proposals")


def plot_material_priors(predictions: list[dict[str, Any]], output: Path) -> None:
    labels, density, youngs = [], [], []
    mapping = []
    for object_index, prediction in enumerate(predictions, 1):
        priors = prediction["material_priors"]
        for part_index, part in enumerate(prediction["parts"], 1):
            token = f"{object_index:02d}:{part_index:02d}"
            labels.append(token)
            materials = [part["surface_material"], part["interior_material"]]
            mapping.append(
                {
                    "figure_part_index": token,
                    "scene_id": prediction["scene_id"],
                    "model_part_id": part["part_id"],
                    "name": part.get("name", ""),
                    "surface_material_hypothesis": materials[0],
                    "interior_material_hypothesis": materials[1],
                    "parameter_source": "configuration_prior_not_measurement",
                }
            )
            density.append(
                [priors.get(m, {}).get("density_mean_kg_m3", np.nan) for m in materials]
            )
            youngs.append(
                [priors.get(m, {}).get("youngs_mean_pa", np.nan) for m in materials]
            )
    figure, axes = plt.subplots(
        1, 2, figsize=(6.7, max(4, len(labels) * 0.22)), layout="constrained"
    )
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#DDDDDD")
    for axis, values, label in zip(
        axes,
        (density, youngs),
        ("Density prior mean (kg m⁻³)", "Young's modulus prior mean (Pa)"),
        strict=True,
    ):
        data = np.asarray(values)
        valid = data[np.isfinite(data) & (data > 0)]
        if not len(valid):
            axis.text(0.5, 0.5, "No material prior available", ha="center")
            continue
        image = axis.imshow(
            data,
            aspect="auto",
            cmap=cmap,
            norm=LogNorm(vmin=valid.min() * 0.95, vmax=valid.max() * 1.05),
        )
        axis.set(
            yticks=np.arange(len(labels)),
            yticklabels=labels,
            xticks=(0, 1),
            xticklabels=("Surface", "Interior"),
            xlabel="Material role",
            ylabel="Object : part index" if axis is axes[0] else "",
        )
        axis.tick_params(axis="y", labelleft=axis is axes[0])
        axis.tick_params(axis="x", labelrotation=25)
        figure.colorbar(image, ax=axis, label=label)
    figure.suptitle(
        "Configuration priors conditioned on VLM material hypotheses\n"
        "Not measured local properties; gray = unknown / void",
        fontsize=10,
    )
    save_figure(figure, output / "material_prior_parameters")
    with (output / "material_part_mapping.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(mapping[0]))
        writer.writeheader()
        writer.writerows(mapping)


def make_vlm_figures(run: Path, data_root: Path) -> list[str]:
    """只读取已封存结果，绘图不会重新调用模型或修改预测。"""
    from tactile3d.physics.part_io import file_sha256

    matplotlib.use("Agg")
    seal = json.loads((run / "prediction_seal.json").read_text())["sha256"]
    source = run / "sealed_predictions.json"
    if file_sha256(source) != seal:
        raise ValueError("预测哈希与封存记录不符")
    predictions = json.loads(source.read_text())
    report = json.loads((run / "report.json").read_text())
    if not predictions:
        raise ValueError("全部对象失败，不生成虚构结果图")
    output = run / "figures"
    output.mkdir(exist_ok=False)
    with plt.rc_context(STYLE):
        plot_mass(report, predictions, output)
        plot_part_proposals(predictions, data_root, output)
        plot_material_priors(predictions, output)
    with (run / "mass_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(report["evaluation"][0]))
        writer.writeheader()
        writer.writerows(report["evaluation"])
    return [str(path) for path in sorted(output.glob("*"))]
