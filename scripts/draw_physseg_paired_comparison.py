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

from draw_wide_material_grid import MATERIALS, load_maskclip

PALETTE = {name.lower(): color for name, color in MATERIALS.items()}
ALIASES = {"steel": "metal", "aluminum": "metal"}
ROWS = (
    ("(a) RGB input", "rgba"),
    ("(b) PhysSeg annotation", "target"),
    ("(c) Ours", "ours"),
    ("(d) NeRF2Physics", "nerf"),
    ("(e) PUGS–Qwen", "pugs"),
)


def plot(source, stem, maskclip):
    report = json.loads((source / "paired_material.json").read_text(encoding="utf-8"))
    objects = {row["scene_id"]: row["object"] for row in report["rows"]}
    rows = ROWS
    if maskclip is not None:
        rows = (*rows, ("(f) MaskCLIP (2D)", "maskclip"))
    fig, axes = plt.subplots(len(rows), len(objects), figsize=(3.55, 5.26))
    fig.subplots_adjust(
        left=0.29, right=0.98, top=0.93, bottom=0.095, wspace=0.04, hspace=0.08
    )
    for (header, key), ax in zip(rows, axes[:, 0], strict=True):
        box = ax.get_subplotspec().get_position(fig)
        fig.text(
            0.025,
            (box.y0 + box.y1) / 2,
            header,
            va="center",
            size=6.5,
            weight="bold" if key == "ours" else "normal",
        )
    used = set()
    for column, (scene, title) in enumerate(objects.items()):
        box = axes[0, column].get_subplotspec().get_position(fig)
        fig.text((box.x0 + box.x1) / 2, 0.962, title, ha="center", size=6.6)
        view = min(r["view"] for r in report["rows"] if r["scene_id"] == scene)
        view_path = source / f"{scene}_{view:02d}.npz"
        with np.load(view_path) as data:
            rgba, target = data["rgba"], data["target"]
            fg = (rgba[..., 3] >= 128) | (target > 0)
            y, x = np.nonzero(fg)
            for row, (_, key) in enumerate(rows):
                ax = axes[row, column]
                if key == "rgba":
                    alpha = rgba[..., 3:] / 255.0
                    layer = rgba[..., :3] / 255.0 * alpha + 1 - alpha
                else:
                    if key == "maskclip":
                        labels, names = load_maskclip(maskclip, view_path)
                    else:
                        labels = data[key]
                        names = (
                            ("fabric", "wood")
                            if key == "target"
                            else data[f"{key}_names"]
                        )
                    names = ["unassigned", *[ALIASES.get(name, name) for name in names]]
                    colors = [to_rgb(PALETTE[name]) for name in names]
                    layer = np.ones((*fg.shape, 3))
                    region = (target > 0) if key == "target" else fg
                    layer[region] = np.array(colors)[labels[region]]
                    used.update(names[i] for i in np.unique(labels[region]))
                ax.imshow(layer, interpolation="nearest")
                ax.set(xlim=(x.min() - 8, x.max() + 8), ylim=(y.max() + 8, y.min() - 8))
                ax.axis("off")
    fig.legend(
        handles=[
            Patch(facecolor=color, label=name.capitalize())
            for name, color in PALETTE.items()
            if name in used
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncols=min(4, len(used)),
        fontsize=6.3,
        frameon=False,
        handlelength=1.0,
        columnspacing=1.0,
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
    parser.add_argument("--maskclip", type=Path)
    args = parser.parse_args()
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "text.color": "#172633",
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    plot(args.source, args.output, args.maskclip)
