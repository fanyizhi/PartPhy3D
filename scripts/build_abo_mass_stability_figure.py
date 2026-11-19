"""Plot the three appliance replays and paired 50-object mass estimates."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from prepare_abo50_benchmark import read_json
from run_component_mass_stability import sealed_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("summary", "probe", "abo", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    rows = read_json(args.summary)["cohorts"][0]["per_scene"]
    by_id = {row["scene_id"]: row for row in rows}
    probes = [row for row in sealed_rows(args.probe).values() if "role_mass_kg" in row]
    labels = ("Window AC", "Portable AC", "Dehumidifier")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig = plt.figure(figsize=(7.15, 3.8), layout="constrained")
    grid = fig.add_gridspec(3, 3, width_ratios=(1.0, 1.9, 2.8))
    left = fig.add_subplot(grid[:, 1])
    right = fig.add_subplot(grid[:, 2])
    methods = (
        ("Previous Ours", "physical", "#90969E", "o"),
        ("Unconstrained proposal", "mass_kg", "#B36533", "x"),
        ("Role constraints", "physical_role_replay", "#337982", "o"),
        ("Role constraints + image prior", "physical_role_clip", "#2B506F", "D"),
        ("Catalogue", "target", "#202428", "|"),
    )
    for i, (probe, label) in enumerate(zip(probes, labels, strict=True)):
        scene = probe["scene_id"]
        ax = fig.add_subplot(grid[i, 0])
        with Image.open(
            args.abo / "scenes" / scene / "images" / f"{scene}_08.png"
        ) as img:
            rgba = img.convert("RGBA")
            crop = rgba.getbbox()
            white = Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba)
            ax.imshow(white.crop(crop))
        ax.set_title(label, fontsize=8)
        ax.set_axis_off()
        row = by_id[scene] | probe
        for j, (method, key, color, marker) in enumerate(methods):
            left.scatter(
                row[key],
                2 - i + (j - 2) * 0.126,
                label=method if i == 0 else None,
                c=color,
                marker=marker,
                s=27 if marker != "|" else 85,
            )
    left.set(
        xscale="log", xlabel="Mass (kg)", xlim=(1.7, 200), ylim=(-0.42, 2.42), yticks=[]
    )
    left.set_title("a   Appliance volume replay", weight="bold", loc="left")
    left.set_xticks([2, 5, 10, 20, 50, 100], labels=[2, 5, 10, 20, 50, 100])
    left.axhline(0.5, color="#DDE0E4", lw=0.65)
    left.axhline(1.5, color="#DDE0E4", lw=0.65)
    y = np.array([r["target"] for r in rows])
    p0 = np.array([r["physical"] for r in rows])
    p1 = np.array([r["physical_role_clip"] for r in rows])
    lo, hi = min(y.min(), p0.min(), p1.min()), max(y.max(), p0.max(), p1.max())
    right.plot([lo, hi], [lo, hi], color="#789", lw=0.7, linestyle="--")
    right.scatter(
        y, p0, s=18, marker="o", facecolors="none", edgecolors="#90969E", linewidth=0.7
    )
    right.scatter(y, p1, s=17, marker="D", color="#2B506F", alpha=0.8, linewidths=0)
    right.set(
        xscale="log",
        yscale="log",
        xlabel="Catalogue mass (kg)",
        ylabel="Predicted mass (kg)",
    )
    right.set_title("b   Paired ABO-50 results", weight="bold", loc="left")
    for ax in (left, right):
        ax.spines[["top", "right"]].set_visible(False)
    right.legend(
        *left.get_legend_handles_labels(), frameon=False, fontsize=6.6, loc="upper left"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(args.output.with_suffix(f".{suffix}"), dpi=350)
    plt.close(fig)


if __name__ == "__main__":
    main()
