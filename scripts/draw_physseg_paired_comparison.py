"""Draw the same-view, same-palette material comparison without ablation rows."""

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch


PALETTE = {
    "fabric": "#347ca1",
    "wood": "#b18059",
    "metal": "#7666a0",
    "plastic": "#d2ab37",
    "leather": "#ad5e69",
    "foam": "#70a58a",
    "unassigned": "#cbd2d9",
}
ALIASES = {"steel": "metal", "aluminum": "metal"}
ROWS = (
    ("RGB input", "rgba"),
    ("PhysSeg annotation", "target"),
    ("NeRF2Physics", "nerf"),
    ("PUGS–Qwen", "pugs"),
    ("Ours", "ours"),
)


def plot(source, stem):
    report = json.loads((source / "paired_material.json").read_text(encoding="utf-8"))
    objects = {row["scene_id"]: row["object"] for row in report["rows"]}
    fig, axes = plt.subplots(2, 5, figsize=(7.2, 4.14))
    fig.subplots_adjust(
        left=0.04, right=0.965, top=0.76, bottom=0.17, wspace=0.165, hspace=0.50
    )
    fig.text(
        0.04,
        0.966,
        "Material segmentation on shared PhysSeg views",
        va="top",
        size=12.5,
        weight="bold",
    )
    fig.text(
        0.04,
        0.891,
        "Two ABO objects  ·  Same camera, mesh and material colors for all predictions",
        size=7.1,
        color="#536473",
    )
    used = set()
    for row, (scene, title) in enumerate(objects.items()):
        view = min(r["view"] for r in report["rows"] if r["scene_id"] == scene)
        with np.load(source / f"{scene}_{view:02d}.npz") as data:
            rgba, target = data["rgba"], data["target"]
            fg = (rgba[..., 3] >= 128) | (target > 0)
            y, x = np.nonzero(fg)
            for column, (header, key) in enumerate(ROWS):
                ax = axes[row, column]
                if key == "rgba":
                    alpha = rgba[..., 3:] / 255.0
                    layer = rgba[..., :3] / 255.0 * alpha + 1 - alpha
                else:
                    names = (
                        ("fabric", "wood") if key == "target" else data[f"{key}_names"]
                    )
                    names = ["unassigned", *[ALIASES.get(name, name) for name in names]]
                    labels = data[key]
                    colors = [to_rgb(PALETTE[name]) for name in names]
                    layer = np.ones((*fg.shape, 3))
                    region = (target > 0) if key == "target" else fg
                    layer[region] = np.array(colors)[labels[region]]
                    used.update(names[i] for i in np.unique(labels[region]))
                ax.imshow(layer, interpolation="nearest")
                ax.set(xlim=(x.min() - 8, x.max() + 8), ylim=(y.max() + 8, y.min() - 8))
                if row == 0:
                    ax.set_title(header, fontsize=7.6, weight="bold", pad=6)
                if column == 0:
                    ax.text(
                        0.5,
                        -0.06,
                        title,
                        transform=ax.transAxes,
                        ha="center",
                        va="top",
                        size=7,
                        weight="bold",
                    )
                ax.axis("off")
    fig.legend(
        handles=[
            Patch(facecolor=color, label=name.capitalize())
            for name, color in PALETTE.items()
            if name in used
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.071),
        ncols=len(used),
        fontsize=6.7,
        frameon=False,
        handlelength=1.0,
    )
    fig.text(
        0.04,
        0.03,
        "One labeled view shown per object. Quantitative results use all ten annotated views.",
        size=6.7,
        color="#536473",
    )
    for fmt in ("png", "pdf", "svg"):
        out = stem.with_suffix(f".{fmt}")
        fig.savefig(out, dpi=380, facecolor="white")
        if fmt == "svg":
            out.write_text(
                "\n".join(
                    line.rstrip()
                    for line in out.read_text(encoding="utf-8").splitlines()
                )
                + "\n",
                encoding="utf-8",
            )
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "text.color": "#172633",
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    plot(args.source, args.output)
