"""Draw shared ABO views and projected material predictions."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch


MATERIALS = {
    "Wood": "#b98355",
    "Metal": "#74669f",
    "Fabric": "#377eaa",
    "Plastic": "#cfad35",
    "Rubber": "#2d786d",
    "Foam": "#8fbd8d",
    "Leather": "#b85f63",
    "Electronics": "#2c97a9",
    "Paint": "#87915d",
    "Glass": "#50b6c2",
    "Ceramic": "#777e8b",
    "Stone": "#626d57",
    "Unassigned": "#bcc7d1",
}
FAMILIES = {
    **{
        key: "Wood"
        for key in ("white wood", "particle board", "plywood", "mdf", "bamboo")
    },
    **{key: "Metal" for key in ("aluminum", "aluminium", "steel", "copper")},
    **{key: "Electronics" for key in ("circuit board", "wires")},
    **{key: "Paint" for key in ("black paint", "white paint")},
    "polyester": "Fabric",
}


def add_local_legend(ax, names, *, side=False):
    """Keep each method's observed families next to the prediction."""
    handles = [
        Patch(facecolor=MATERIALS[name], edgecolor="none", label=name)
        for name in MATERIALS
        if name in names and name != "Unassigned"
    ]
    if not handles:
        return
    ax.legend(
        handles=handles,
        loc="center left" if side else "upper center",
        bbox_to_anchor=(0, 0.5) if side else (0.5, 0.96),
        ncols=1 if side else min(2, len(handles)),
        frameon=False,
        fontsize=6.4 if side else 5.6,
        columnspacing=0.54,
        handlelength=0.68,
        handletextpad=0.28,
        labelspacing=0.26,
        borderaxespad=0,
    )


def load_maskclip(root, source):
    """Verify that the baseline saw the exact displayed input archive."""
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    record = next(row for row in manifest["views"] if row["file"] == source.name)
    predicted = root / source.name
    if any(
        hashlib.sha256(path.read_bytes()).hexdigest() != record[key]
        for path, key in ((source, "input_sha256"), (predicted, "labels_sha256"))
    ):
        raise ValueError(f"MaskCLIP provenance mismatch for {source.name}")
    with np.load(predicted) as data:
        return data["maskclip"], data["maskclip_names"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maskclip", type=Path)
    parser.add_argument("--top-coverage", type=int)
    args = parser.parse_args()
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "text.color": "#162535",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    manifest = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
    selected = manifest
    if args.top_coverage is not None:
        if not 1 <= args.top_coverage <= len(manifest):
            parser.error("--top-coverage must select at least one available object")
        for record in manifest:
            with np.load(args.data / f"{record['scene_id']}.npz") as data:
                record["ours_foreground_coverage"] = float(
                    (data["ours"][data["foreground"]] > 0).mean()
                )
        selected = sorted(manifest, key=lambda r: -r["ours_foreground_coverage"])[
            : args.top_coverage
        ]

    methods = [("ours", "Ours"), ("nerf", "NeRF2Physics")]
    pugs_available = ["pugs_classes" in record for record in selected]
    if any(pugs_available):
        if not all(pugs_available):
            raise ValueError("Choose cases with a complete PUGS prediction row")
        methods.insert(1, ("pugs", "PUGS–Qwen"))
    if args.maskclip is not None:
        methods.append(("maskclip", "MaskCLIP (2D)"))
    labels = ["RGB input", "Region evidence", *(label for _, label in methods)]
    side = all(pugs_available)
    heights = (1,) * len(labels) if side else (1, 1) + (1, 0.29) * len(methods)
    image_rows = range(len(labels)) if side else (0, 1, *range(2, len(heights), 2))
    fig = plt.figure(figsize=(7.2, 4.4 if side else 4.9 + 0.88 * (len(methods) - 3)))
    grid = fig.add_gridspec(len(heights), len(selected), height_ratios=heights)
    fig.subplots_adjust(
        left=0.134, right=0.987, top=0.916, bottom=0.025, wspace=0.085, hspace=0.035
    )
    for panel, (row, label) in enumerate(zip(image_rows, labels, strict=True)):
        pos = grid[row, 0].get_position(fig)
        fig.text(
            0.009,
            (pos.y0 + pos.y1) / 2,
            f"({chr(97 + panel)}) {label}",
            va="center",
            size=7.1,
            weight="bold" if label == "Ours" else "normal",
        )
    fig.legend(
        handles=[Patch(facecolor=MATERIALS["Unassigned"], label="Unassigned")],
        loc="lower left",
        bbox_to_anchor=(0.009, 0.025),
        frameon=False,
        fontsize=5.6,
        handlelength=0.68,
        handletextpad=0.28,
        borderaxespad=0,
    )
    for column, record in enumerate(selected):
        if side:
            cells = [
                grid[row, column].subgridspec(
                    1, 2, width_ratios=(0.72, 0.28), wspace=0.045
                )
                for row in image_rows
            ]
        source = args.data / f"{record['scene_id']}.npz"
        if "raster_sha256" in record:
            checksum = hashlib.sha256(source.read_bytes()).hexdigest()
            if checksum != record["raster_sha256"]:
                raise ValueError(
                    f"Raster differs from the sealed export for {source.name}"
                )
        with np.load(source) as data:
            fg = data["foreground"]
            layers = [data["rgb"], data["evidence"]]
            for step, (method, _) in enumerate(methods):
                if method == "maskclip":
                    prediction, names = load_maskclip(args.maskclip, source)
                else:
                    prediction, names = data[method], data[f"{method}_names"]
                groups = [
                    "Unassigned",
                    *[FAMILIES.get(n.lower(), n.title()) for n in names],
                ]
                record[f"{method}_display_families"] = groups[1:]
                rgb = np.array([to_rgb(MATERIALS[n]) for n in groups])
                legend = fig.add_subplot(
                    cells[2 + step][0, 1] if side else grid[3 + 2 * step, column]
                )
                legend.axis("off")
                add_local_legend(
                    legend, {groups[i] for i in np.unique(prediction[fg])}, side=side
                )
                raster = np.ones(fg.shape + (3,))
                raster[fg] = rgb[prediction[fg]]
                layers.append(raster)
            y, x = np.nonzero(fg)
        for row, layer in zip(image_rows, layers, strict=True):
            ax = fig.add_subplot(cells[row][0, 0] if side else grid[row, column])
            ax.imshow(layer, interpolation="nearest")
            ax.set(xlim=(x.min() - 5, x.max() + 5), ylim=(y.max() + 5, y.min() - 5))
            ax.axis("off")
        words = record["object"].split()
        title = (
            "\n".join((" ".join(words[:-1]), words[-1]))
            if len(words) > 2
            else record["object"]
        )
        box = (cells[0][0, 0] if side else grid[0, column]).get_position(fig)
        fig.text(
            (box.x0 + box.x1) / 2, 0.948, title, ha="center", va="center", size=6.8
        )
    for fmt in ("png", "pdf", "svg"):
        out = args.output.with_suffix(f".{fmt}")
        fig.savefig(out, dpi=360, facecolor="white")
        if fmt == "svg":
            out.write_text(
                "\n".join(
                    s.rstrip() for s in out.read_text(encoding="utf-8").splitlines()
                )
                + "\n",
                encoding="utf-8",
            )
    plt.close(fig)
    args.output.with_name(args.output.name + "_data.json").write_text(
        json.dumps(selected, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
