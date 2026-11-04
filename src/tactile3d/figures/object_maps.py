from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import trimesh

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from tactile3d.figures.style import PALETTE, configure_style, save_figure
from tactile3d.utils.io import ensure_dir, read_json


def generate_object_figures(
    prediction_path: str | Path,
    mesh_path: str | Path,
    face_labels_path: str | Path,
    output_dir: str | Path,
    *,
    formats: tuple[str, ...] = ("png", "svg", "pdf", "tiff"),
    max_faces: int = 50000,
) -> dict[str, Any]:
    prediction = read_json(prediction_path)
    mesh = _load_mesh(mesh_path)
    labels = np.load(face_labels_path).astype(np.int64)
    if labels.ndim != 1 or len(labels) != len(mesh.faces):
        raise ValueError("face labels must contain one value per mesh face")
    parts = {int(item["part_id"]): item for item in prediction["parts"]}
    missing = sorted(set(np.unique(labels)) - set(parts))
    if missing:
        raise ValueError(f"Prediction has no physical values for part IDs: {missing}")

    output_dir = ensure_dir(output_dir)
    source_rows = _source_rows(parts)
    _write_source_data(output_dir / "object_figure_source_data.csv", source_rows)
    files = []
    files.extend(_plot_mesh_maps(mesh, labels, parts, output_dir, formats, max_faces))
    files.extend(_plot_posterior(parts, prediction, output_dir, formats))
    return {"scene_id": prediction.get("scene_id"), "figure_files": files}


def _plot_mesh_maps(
    mesh: trimesh.Trimesh,
    labels: np.ndarray,
    parts: dict[int, dict],
    output_dir: Path,
    formats: tuple[str, ...],
    max_faces: int,
) -> list[str]:
    configure_style()
    triangles = np.asarray(mesh.triangles)
    if len(triangles) > max_faces:
        selected = np.linspace(0, len(triangles) - 1, max_faces, dtype=np.int64)
        triangles, labels = triangles[selected], labels[selected]
    center = mesh.bounds.mean(axis=0)
    scale = max(float(mesh.extents.max()), 1e-12)
    triangles = (triangles - center) / scale
    density = np.asarray([parts[int(label)]["density_kg_m3"]["mean"] for label in labels])
    youngs = np.asarray([parts[int(label)]["youngs_modulus_pa"]["mean"] for label in labels])

    figure = plt.figure(figsize=(7.2, 2.7), constrained_layout=True)
    axes = [figure.add_subplot(1, 3, index + 1, projection="3d") for index in range(3)]
    part_colors = np.asarray(
        [colors.to_rgba(PALETTE[int(label) % len(PALETTE)]) for label in labels]
    )
    density_norm = _log_norm(density)
    youngs_norm = _log_norm(youngs)
    color_sets = [
        part_colors,
        plt.colormaps["viridis"](density_norm(density)),
        plt.colormaps["magma"](youngs_norm(youngs)),
    ]
    titles = ("a  Part segmentation", "b  Effective density", "c  Effective Young's modulus")
    for axis, face_colors, title in zip(axes, color_sets, titles, strict=True):
        collection = Poly3DCollection(
            triangles, facecolors=face_colors, edgecolors="none", linewidths=0, rasterized=True
        )
        axis.add_collection3d(collection)
        axis.set(xlim=(-0.6, 0.6), ylim=(-0.6, 0.6), zlim=(-0.6, 0.6))
        axis.set_box_aspect((1, 1, 1))
        axis.view_init(elev=22, azim=-55)
        axis.set_proj_type("ortho")
        axis.set_axis_off()
        axis.set_title(title, loc="left", weight="bold", pad=0)
    legend = [Patch(facecolor=PALETTE[part % len(PALETTE)], label=f"Part {part}") for part in parts]
    axes[0].legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=2)
    figure.colorbar(
        ScalarMappable(norm=density_norm, cmap="viridis"),
        ax=axes[1],
        shrink=0.55,
        pad=0.01,
        label="kg m$^{-3}$",
    )
    figure.colorbar(
        ScalarMappable(norm=youngs_norm, cmap="magma"),
        ax=axes[2],
        shrink=0.55,
        pad=0.01,
        label="Pa",
    )
    return save_figure(figure, output_dir / "object_physics_maps", formats)


def _plot_posterior(
    parts: dict[int, dict], prediction: dict, output_dir: Path, formats: tuple[str, ...]
) -> list[str]:
    configure_style()
    structures = sorted(
        {
            name
            for part in parts.values()
            for name in part["interior"]["posterior_structure_probabilities"]
        }
    )
    part_ids = sorted(parts)
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), constrained_layout=True)
    mass_rows = [(f"Part {part}", parts[part]["mass_kg"]) for part in part_ids]
    mass_rows.append(("Object", prediction["object"]["mass_kg"]))
    for index, (label, stats) in enumerate(mass_rows):
        mean = float(stats["mean"])
        axes[0].errorbar(
            mean,
            index,
            xerr=[[mean - float(stats["q05"])], [float(stats["q95"]) - mean]],
            fmt="o",
            color=PALETTE[index % len(PALETTE)],
            capsize=2.5,
        )
    axes[0].set_yticks(range(len(mass_rows)), [item[0] for item in mass_rows])
    axes[0].set_xlabel("Mass (kg), posterior mean and 90% interval")
    axes[0].set_title("a  Mass uncertainty", loc="left", weight="bold")
    axes[0].grid(axis="x", color="#E2E2E2", linewidth=0.45)

    left = np.zeros(len(part_ids))
    for index, structure in enumerate(structures):
        probability = np.asarray(
            [
                parts[part]["interior"]["posterior_structure_probabilities"].get(structure, 0.0)
                for part in part_ids
            ]
        )
        axes[1].barh(
            range(len(part_ids)),
            probability,
            left=left,
            color=PALETTE[index % len(PALETTE)],
            label=structure,
        )
        left += probability
    axes[1].set_yticks(range(len(part_ids)), [f"Part {part}" for part in part_ids])
    axes[1].set(xlim=(0, 1), xlabel="Posterior probability")
    axes[1].set_title("b  Interior construction", loc="left", weight="bold")
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    return save_figure(figure, output_dir / "part_posterior_summary", formats)


def _source_rows(parts: dict[int, dict]) -> list[dict[str, Any]]:
    return [
        {
            "part_id": part_id,
            "density_mean_kg_m3": part["density_kg_m3"]["mean"],
            "density_q05_kg_m3": part["density_kg_m3"]["q05"],
            "density_q95_kg_m3": part["density_kg_m3"]["q95"],
            "youngs_mean_pa": part["youngs_modulus_pa"]["mean"],
            "mass_mean_kg": part["mass_kg"]["mean"],
            "mass_q05_kg": part["mass_kg"]["q05"],
            "mass_q95_kg": part["mass_kg"]["q95"],
        }
        for part_id, part in sorted(parts.items())
    ]


def _log_norm(values: np.ndarray) -> colors.LogNorm:
    positive = values[values > 0]
    if not len(positive):
        raise ValueError("Log-scaled physical maps require positive values")
    low, high = float(positive.min()), float(positive.max())
    if np.isclose(low, high):
        low, high = low / 1.1, high * 1.1
    return colors.LogNorm(vmin=low, vmax=high)


def _write_source_data(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_mesh(path: str | Path) -> trimesh.Trimesh:
    loaded = trimesh.load(Path(path), force="mesh", process=False)
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"Unsupported mesh payload in {path}")
    return loaded
