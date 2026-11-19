"""Render paired mass and shared-camera density figures from ABO-50 results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from prepare_abo50_benchmark import digest, read_json, write_json

BLUE = "#276A89"
GRAY = "#838890"


def save(figure: plt.Figure, path: Path) -> None:
    for suffix in ("png", "pdf", "svg"):
        destination = path.with_suffix(f".{suffix}")
        figure.savefig(destination, dpi=300, facecolor="white")
        if suffix == "svg":
            svg = destination.read_text(encoding="utf-8")
            destination.write_text(
                "\n".join(line.rstrip() for line in svg.splitlines()) + "\n",
                encoding="utf-8",
            )
    plt.close(figure)


def mass_figures(baseline: Path, predictions: Path, seal: Path, output: Path) -> None:
    if digest(predictions) != read_json(seal)["sha256"]:
        raise ValueError("Winding-only predictions failed seal verification")
    reference = read_json(baseline)["per_scene"]
    predicted = read_json(predictions)
    if len(reference) != 50 or [r["scene_id"] for r in reference] != [
        r["scene_id"] for r in predicted
    ]:
        raise ValueError("Mass comparison requires matching 50 objects in frozen order")
    rows = []
    for ref, method in zip(reference, predicted):
        row = {
            "scene_id": ref["scene_id"],
            "target_kg": ref["target_mass_kg"],
            "nerf_cached_kg": ref["mass_kg"],
        }
        for name in ("direct_mass_kg", "all_solid_mass_kg", "construction_mass_kg"):
            row[name] = method.get(name)
        rows.append(row)
    write_json(output / "mass_figure_data.json", rows)

    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), layout="constrained")
    for axis, labels, method_keys, group in (
        (
            axes[0],
            ("NeRF2Physics (cache)", "Direct VLM"),
            ("nerf_cached_kg", "direct_mass_kg"),
            rows,
        ),
        (
            axes[1],
            ("All-solid", "Ours (construction)"),
            ("all_solid_mass_kg", "construction_mass_kg"),
            [row for row in rows if row["construction_mass_kg"] is not None],
        ),
    ):
        target = np.array([row["target_kg"] for row in group])
        values = [np.array([row[key] for row in group]) for key in method_keys]
        low = min(target.min(), *(v.min() for v in values)) * 0.76
        high = max(target.max(), *(v.max() for v in values)) * 1.35
        axis.plot([low, high], [low, high], "--", color="#333333", lw=0.75, zorder=1)
        for estimates, color, label, marker in zip(
            values, (GRAY, BLUE), labels, ("o", "^")
        ):
            axis.scatter(
                target,
                estimates,
                s=20,
                marker=marker,
                alpha=0.76,
                c=color,
                linewidths=0.25,
                edgecolors="white",
                label=label,
                zorder=2,
            )
        axis.set(
            xscale="log",
            yscale="log",
            xlim=(low, high),
            ylim=(low, high),
            xlabel="Catalogue mass (kg)",
            ylabel="Predicted mass (kg)",
            title=f"{len(group)} paired objects",
            aspect="equal",
        )
        axis.legend(loc="upper left", fontsize=7, handletextpad=0.35)
    save(figure, output / "mass_agreement")


def density_figure(directory: Path, output: Path) -> None:
    manifest = read_json(directory / "density_manifest.json")
    keys = ["nerf", "ours"]
    titles = ["Input RGB · view 08", "NeRF2Physics (cache)", "Ours · original"]
    replay = "propagated_foreground_coverage" in manifest["scenes"][0]
    pugs = "pugs_foreground_coverage" in manifest["scenes"][0]
    if replay:
        keys.append("propagated")
        titles.append(manifest.get("density_variant_label", "Ours · local propagation"))
    if pugs:
        keys = ["nerf", "pugs", "propagated"]
        titles = [
            "Input RGB · view 08",
            "NeRF2Physics (cache)",
            "PUGS-Qwen",
            "Ours · local propagation",
        ]
    figure, axes = plt.subplots(
        len(manifest["scenes"]),
        len(titles),
        figsize=(7.2, 2.3 if pugs else max(5.7, 1.8 * len(manifest["scenes"]))),
        layout="constrained",
        squeeze=False,
    )
    norm = LogNorm(100, 8000, clip=True)
    cmap = mpl.colormaps["viridis"]
    for coord, row in zip(axes, manifest["scenes"]):
        scene = row["scene_id"]
        with np.load(directory / f"{scene}.npz") as values:
            rgb, foreground = values["rgb"], values["foreground"]
            y, x = np.where(foreground)
            pad = 10
            side = max(int(np.ptp(x)), int(np.ptp(y))) + 2 * pad
            x0 = max(
                0, min((int(x.min() + x.max()) - side) // 2, foreground.shape[1] - side)
            )
            y0 = max(
                0, min((int(y.min() + y.max()) - side) // 2, foreground.shape[0] - side)
            )
            crop = np.s_[y0 : y0 + side, x0 : x0 + side]
            coord[0].imshow(rgb[crop], interpolation="nearest")
            for axis, key in zip(coord[1:], keys):
                field = values[key]
                color = np.ones((*foreground.shape, 3))
                color[foreground] = (0.79, 0.80, 0.81)
                known = foreground & np.isfinite(field)
                color[known] = cmap(norm(field[known]))[:, :3]
                axis.imshow(color[crop], interpolation="nearest")
            for axis in coord:
                axis.set_axis_off()
            coord[0].text(
                -0.03,
                0.5,
                scene.split("_")[0],
                transform=coord[0].transAxes,
                ha="right",
                va="center",
                rotation=90,
            )
    for axis, label in zip(axes[0], titles):
        axis.set_title(label)
    bar = figure.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, shrink=0.7
    )
    bar.set_ticks(
        [100, 300, 1000, 3000, 8000], labels=["100", "300", "1,000", "3,000", "8,000"]
    )
    bar.set_label("Estimated density (kg m$^{-3}$)")
    name = manifest.get(
        "density_variant_filename",
        "density_coverage_ablation" if replay else "shared_density_maps",
    )
    save(figure, output / ("pugs_rack_density" if pugs else name))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "predictions", "seal", "density-data", "output"):
        parser.add_argument(f"--{name}", type=Path, required=name == "output")
    args = parser.parse_args()
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "legend.frameon": False,
        }
    )
    args.output.mkdir(parents=True, exist_ok=True)
    if args.baseline is not None:
        mass_figures(args.baseline, args.predictions, args.seal, args.output)
    if args.density_data is not None:
        density_figure(args.density_data, args.output)


if __name__ == "__main__":
    main()
