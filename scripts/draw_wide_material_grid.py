"""Draw seven ABO scenes with actual projected evidence and NeRF2Physics labels."""

import argparse
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
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
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
    fig, axes = plt.subplots(4, len(manifest), figsize=(9.3, 4.5))
    fig.subplots_adjust(
        left=0.135, right=0.982, top=0.80, bottom=0.18, wspace=0.14, hspace=0.16
    )
    fig.text(
        0.052,
        0.965,
        "Region evidence and material predictions across ABO-500",
        va="top",
        size=14,
        weight="bold",
    )
    for index, label in enumerate(
        ("RGB input", "Face evidence", "PartPhy3D", "NeRF2Physics")
    ):
        pos = axes[index, 0].get_position()
        fig.text(
            0.052, (pos.y0 + pos.y1) / 2, label, va="center", size=7.4, weight="bold"
        )
    rendered_materials = set()
    for column, record in enumerate(manifest):
        with np.load(args.data / f"{record['scene_id']}.npz") as data:
            fg = data["foreground"]
            layers = [data["rgb"], data["evidence"]]
            for method in ("ours", "nerf"):
                groups = [
                    "Unassigned",
                    *[
                        FAMILIES.get(n.lower(), n.title())
                        for n in data[f"{method}_names"]
                    ],
                ]
                record[f"{method}_display_families"] = groups[1:]
                rgb = np.array([to_rgb(MATERIALS[n]) for n in groups])
                rendered_materials.update(
                    groups[i] for i in np.unique(data[method][fg])
                )
                raster = np.ones(fg.shape + (3,))
                raster[fg] = rgb[data[method][fg]]
                layers.append(raster)
            y, x = np.nonzero(fg)
        for ax, layer in zip(axes[:, column], layers, strict=True):
            ax.imshow(layer, interpolation="nearest")
            ax.set(xlim=(x.min() - 5, x.max() + 5), ylim=(y.max() + 5, y.min() - 5))
            ax.axis("off")
        title = record["object"]
        if len(title.split()) > 2:
            words = title.split()
            title = " ".join(words[:-1]) + "\n" + words[-1]
        axes[0, column].set_title(title, size=7.2, weight="bold", pad=6)
    fig.legend(
        handles=[
            Patch(color=color, label=name)
            for name, color in MATERIALS.items()
            if name in rendered_materials
        ],
        loc="lower center",
        bbox_to_anchor=(0.553, 0.072),
        ncols=len(rendered_materials),
        columnspacing=0.86,
        fontsize=6.7,
        frameon=False,
        handlelength=0.9,
        handletextpad=0.4,
    )
    fig.text(
        0.052,
        0.036,
        "Shared cameras and meshes within each column  ·  Face-evidence colors show PCA of fused part votes",
        size=7,
        color="#526171",
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
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
