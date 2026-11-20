"""Compare surface labels with PhysSeg annotations at fixed annotated camera ranks."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import Patch

COLORS = np.array(
    [[0.78, 0.78, 0.78], [0.29, 0.52, 0.69], [0.78, 0.48, 0.24], [0.54, 0.41, 0.65]]
)
OUT = Path(__file__).resolve().parents[1] / "paper_assets/abo500_20261120"


def render(source):
    paths = sorted(source.glob("*.npz"))
    if len(paths) != 4:
        raise ValueError("Expected two fixed camera ranks for each of two cases")
    fig, axes = plt.subplots(4, 4, figsize=(8.0, 7.4), layout="constrained")
    for col, title in enumerate(
        ("RGB input", "PhysSeg annotation", "Ours material", "Young's modulus prior")
    ):
        axes[0, col].set_title(title, fontsize=10)
    for row, path in enumerate(paths):
        with np.load(path) as record:
            rgba = record["rgba"] / 255
            alpha = rgba[..., 3:4]
            rgb = rgba[..., :3] * alpha + 1 - alpha
            target, pred = record["target"], record["prediction"]
            keep = (alpha[..., 0] > 0.5) | (target > 0) | (pred > 0)
            foreground = np.where(keep)
            margin = 9
            crop = (
                slice(
                    max(0, int(foreground[0].min()) - margin),
                    int(foreground[0].max()) + margin + 1,
                ),
                slice(
                    max(0, int(foreground[1].min()) - margin),
                    int(foreground[1].max()) + margin + 1,
                ),
            )
            panels = [
                rgb,
                np.where(keep[..., None], COLORS[target], 1),
                np.where(keep[..., None], COLORS[pred], 1),
            ]
            for ax, panel in zip(axes[row, :3], panels):
                ax.imshow(panel[crop], interpolation="nearest")
            ax = axes[row, 3]
            ax.imshow(
                np.where(keep[..., None], COLORS[0], 1)[crop], interpolation="nearest"
            )
            layer = ax.imshow(
                record["modulus_pa"][crop] / 1e9,
                norm=LogNorm(0.1, 200),
                cmap="viridis",
                interpolation="nearest",
            )
        scene, view = path.stem.rsplit("_", 1)
        axes[row, 0].set_ylabel(scene.split("_")[0] + "\nview " + view, fontsize=8)
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines[:].set_visible(False)
    fig.colorbar(layer, ax=axes[:, 3], shrink=0.62, label="Young's modulus (GPa)")
    fig.legend(
        handles=[
            Patch(color=c, label=n)
            for c, n in zip(COLORS, ["Unassigned", "Fabric", "Wood", "Steel"])
        ],
        loc="outside lower center",
        ncol=4,
        frameon=False,
    )
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"physseg_material_modulus.{ext}", dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    render(parser.parse_args().source)
