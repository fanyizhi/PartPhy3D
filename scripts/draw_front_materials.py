"""Render front-facing ABO projections and optionally append credited author panels."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from PIL import Image

from draw_wide_material_grid import FAMILIES, MATERIALS


def front_grid(fig, area, data):
    records = json.loads((data / "manifest.json").read_text())
    grid = area.subgridspec(
        4,
        5,
        width_ratios=[0.88] + [1.65] * 4,
        height_ratios=[0.69, 1, 1, 1],
        wspace=0.05,
        hspace=0.12,
    )
    for row, text in enumerate(["", "Ours", "PUGS–Qwen", "NeRF2Physics"]):
        ax = fig.add_subplot(grid[row, 0])
        ax.axis("off")
        ax.text(
            0, 0.5, text, va="center", size=8, weight="bold" if row == 1 else "normal"
        )
    for col, record in enumerate(records, start=1):
        path = data / (record["scene_id"] + ".npz")
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["raster_sha256"]:
            raise ValueError(f"Input hash mismatch for {path}")
        with np.load(path) as arrays:
            fg = arrays["foreground"]
            y, x = np.where(fg)
            crop = np.s_[y.min() : y.max() + 1, x.min() : x.max() + 1]
            for row, method in enumerate(["rgb", "ours", "pugs", "nerf"]):
                cell = grid[row, col].subgridspec(
                    1, 2, width_ratios=(1.7, 1), wspace=0.03
                )
                ax = fig.add_subplot(cell[0, 0])
                label = fig.add_subplot(cell[0, 1])
                for a in (ax, label):
                    a.axis("off")
                if row == 0:
                    raster = arrays["rgb"]
                    label.text(
                        0,
                        0.5,
                        record["object"].replace(" ", "\n"),
                        size=7.1,
                        va="center",
                        weight="semibold",
                    )
                else:
                    ids = arrays[method][fg]
                    families = ["Unassigned"] + [
                        FAMILIES.get(n.lower(), n.title())
                        for n in arrays[method + "_names"]
                    ]
                    colors = np.array([to_rgb(MATERIALS[n]) for n in families])
                    raster = np.ones(fg.shape + (3,))
                    raster[fg] = colors[ids]
                    used = {families[i] for i in np.unique(ids)}
                    handles = [
                        Patch(facecolor=color, label=n)
                        for n, color in MATERIALS.items()
                        if n in used and n != "Unassigned"
                    ]
                    label.legend(
                        handles=handles,
                        loc="center left",
                        frameon=False,
                        borderaxespad=0,
                        handlelength=0.72,
                        handletextpad=0.35,
                        labelspacing=0.35,
                        fontsize=5.7,
                    )
                ax.imshow(raster[crop], interpolation="nearest")
    foot = fig.add_subplot(grid[3, 0])
    foot.axis("off")
    foot.legend(
        handles=[Patch(color=MATERIALS["Unassigned"], label="Unassigned")],
        loc="lower left",
        frameon=False,
        fontsize=5.7,
        borderaxespad=0,
        handlelength=0.85,
    )


def extract(folder, name, box):
    # Crops are specified in the author's original aspect ratio at width 900.
    with Image.open(folder / (name + ".png")) as im:
        scale = im.width / 900
        return im.crop(tuple(round(v * scale) for v in box)).convert("RGB")


def references(fig, area, folder):
    # Photograph and prediction always come from the same source column.
    # Source colors are retained; the last panel shows continuous density.
    examples = [
        (
            "GS2Physics",
            "Fig. 8 · desk chair",
            "gs2",
            (335, 17, 410, 117),
            (336, 379, 409, 489),
        ),
        (
            "Gen2Physics",
            "Fig. 4 · folding chair",
            "gen2",
            (425, 5, 496, 83),
            (425, 295, 496, 378),
        ),
        (
            "PhysVGGT",
            "Fig. 4 · accent chair",
            "physvggt",
            (121, 157, 253, 316),
            (758, 157, 881, 316),
        ),
    ]
    grid = area.subgridspec(1, 3, wspace=0.24)
    for i, (name, caption, stem, rgb, pred) in enumerate(examples):
        g = grid[0, i].subgridspec(
            3, 2, height_ratios=(0.31, 1, 0.27), hspace=0.06, wspace=0.10
        )
        title = fig.add_subplot(g[0, :])
        title.axis("off")
        title.text(0, 0.78, name, size=8, weight="bold")
        title.text(0, 0.08, caption, size=6.3, color="#536574")
        for c, box in enumerate((rgb, pred)):
            ax = fig.add_subplot(g[1, c])
            ax.axis("off")
            ax.imshow(extract(folder, stem, box))
        ax = fig.add_subplot(g[2, :])
        ax.axis("off")
        if stem == "physvggt":
            colors = extract(folder, stem, (747, 425, 891, 433))
            ax.imshow(colors, extent=(0.12, 0.94, 0.59, 0.77), aspect="auto")
            for x, txt in zip([0.12, 0.39, 0.66, 0.94], ["10¹", "10²", "10³", "10⁴"]):
                ax.text(x, 0.31, txt, ha="center", size=5.8)
            ax.text(0.53, -0.12, "Density (kg m⁻³)", ha="center", size=6)
            ax.set(xlim=(0, 1), ylim=(-0.2, 1))
        else:
            # Sample swatches from the source legend, not from the ABO palette.
            entries = (
                [("Plastic", (427, 420)), ("Metal", (427, 435))]
                if stem == "gs2"
                else [("Metal", (222, 514)), ("Plastic", (483, 514))]
            )
            handles = [
                Patch(
                    facecolor=np.array(extract(folder, stem, (x, y, x + 1, y + 1)))[
                        0, 0
                    ]
                    / 255,
                    label=label,
                )
                for label, (x, y) in entries
            ]
            ax.legend(
                handles=handles,
                loc="center",
                frameon=False,
                ncols=2,
                fontsize=6.1,
                handlelength=0.8,
                handletextpad=0.35,
                columnspacing=1.1,
            )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("data", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument(
        "--paper-panels",
        type=Path,
        help="Directory containing original figure extracts",
    )
    args = p.parse_args()
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "text.color": "#1e2f3c",
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    combined = args.paper_panels is not None
    fig = plt.figure(figsize=(8.4, 5.6 if combined else 3.65))
    section = fig.add_gridspec(
        2 if combined else 1,
        1,
        height_ratios=[3.2, 1.44] if combined else [1],
        left=0.035,
        right=0.987,
        top=0.915,
        bottom=0.035,
        hspace=0.40,
    )
    fig.text(
        0.035,
        0.959,
        "ABO-500   /   Matched front views",
        size=10,
        weight="bold",
        va="center",
    )
    front_grid(fig, section[0], args.data)
    if combined:
        top = section[1].get_position(fig).y1
        fig.add_artist(
            Line2D(
                [0.035, 0.987],
                [top + 0.075] * 2,
                transform=fig.transFigure,
                color="#bac5cb",
                lw=0.6,
            )
        )
        fig.text(0.035, top + 0.045, "Published examples", size=9, weight="bold")
        fig.text(
            0.265,
            top + 0.045,
            "Author-selected objects and original color keys",
            size=7,
            color="#536574",
        )
        references(fig, section[1], args.paper_panels)
    for fmt in ("png", "pdf", "svg"):
        fig.savefig(args.output.with_suffix("." + fmt), dpi=360, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
