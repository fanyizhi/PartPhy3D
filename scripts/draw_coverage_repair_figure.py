"""Show held-out coverage together with background leakage for the ABO rack."""

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


METHODS = ("instantmesh", "legacy_hull_material_consensus", "hull_material_consensus")
LABELS = ("Original mesh", "Dilated hull", "Subpixel hull")
INK = "#182535"
COVERED, MISSED, LEAK = "#176272", "#c6d1da", "#d88049"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--views", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.4,
            "text.color": INK,
            "axes.labelcolor": INK,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig = plt.figure(figsize=(7.2, 4.7))
    grid = fig.add_gridspec(
        3,
        4,
        left=0.07,
        right=0.97,
        bottom=0.13,
        top=0.82,
        height_ratios=(1, 1, 0.91),
        wspace=0.11,
        hspace=0.18,
    )
    fig.text(
        0.07,
        0.968,
        "Recovering thin structures with less background leakage",
        size=12.6,
        va="top",
        weight="bold",
    )
    fig.text(
        0.07,
        0.892,
        "Metal shoe rack  ·  Same six geometry views and three material views",
        size=7.5,
    )
    cmap = ListedColormap(["white", COVERED, MISSED, LEAK])
    for index, row in enumerate(audit["rows"][::2][:2]):
        view = row["view"]
        with np.load(args.views / f"view_{view:02d}.npz") as data:
            fg = data["foreground"]
            density_present = [np.isfinite(data[f"{key}_density"]) for key in METHODS]
            region = np.logical_or.reduce([fg, *density_present])
            y, x = np.nonzero(region)
            source = fig.add_subplot(grid[index, 0])
            source.imshow(data["rgb"])
            axes = [source]
            for col, exists in enumerate(density_present, start=1):
                ax = fig.add_subplot(grid[index, col])
                axes.append(ax)
                code = np.zeros_like(fg, dtype=np.uint8)
                code[fg & exists], code[fg & ~exists], code[~fg & exists] = 1, 2, 3
                ax.imshow(code, cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
        for ax in axes:
            ax.set(xlim=(x.min() - 4, x.max() + 4), ylim=(y.max() + 4, y.min() - 4))
            ax.axis("off")
        if index == 0:
            for ax, title in zip(axes, ("Held-out RGB", *LABELS)):
                ax.set_title(title, weight="bold", fontsize=8, pad=4)
        source.text(
            0, 0.03, f"View {view:02d}", transform=source.transAxes, fontsize=6.6
        )
    ax = fig.add_subplot(grid[2, :2])
    caption = fig.add_subplot(grid[2, 2:])
    position = ax.get_position()
    ax.set_position(
        [
            position.x0 + 0.073,
            position.y0 + 0.009,
            position.width - 0.073,
            position.height - 0.009,
        ]
    )
    ypos = np.arange(len(METHODS))[::-1]
    for y, key in zip(ypos, METHODS):
        stats = audit["metrics"][key]
        c, p = (100 * stats[m] for m in ("density_coverage", "geometry_precision"))
        ax.plot([c, p], [y, y], color="#bcc7ce", linewidth=1, zorder=1)
        ax.scatter(c, y, facecolor=COVERED, edgecolor="white", s=30, zorder=3)
        ax.scatter(
            p, y, facecolor="white", edgecolor=LEAK, s=26, zorder=3, linewidth=1.25
        )
    ax.set(
        yticks=ypos,
        yticklabels=LABELS,
        xlim=(20, 104),
        xticks=[25, 50, 75, 100],
        ylim=(-0.48, 2.48),
        xlabel="Held-out mean (%)",
    )
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.grid(axis="x", color="#e4e9ed", lw=0.5)
    caption.axis("off")
    caption.legend(
        handles=[
            Patch(facecolor=COVERED, label="Foreground with density"),
            Patch(facecolor=MISSED, label="Foreground without density"),
            Patch(facecolor=LEAK, label="Density outside foreground"),
        ],
        loc="upper left",
        frameon=False,
        fontsize=7.1,
    )
    caption.text(0.03, 0.10, "●", color=COVERED, transform=caption.transAxes)
    caption.text(0.075, 0.10, "Density coverage", size=7.1, transform=caption.transAxes)
    caption.text(0.44, 0.10, "○", color=LEAK, transform=caption.transAxes)
    caption.text(
        0.485, 0.10, "Silhouette precision", size=7.1, transform=caption.transAxes
    )
    fig.text(
        0.07,
        0.038,
        "Five views held out from geometry and material assignment  ·  Same steel density prior across both hulls",
        color="#526171",
        size=6.8,
    )
    for fmt in ("png", "pdf", "svg"):
        fig.savefig(args.output.with_suffix(f".{fmt}"), dpi=360)
    plt.close(fig)


if __name__ == "__main__":
    main()
