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
    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(4, 5, figsize=(7.3, 5.9), layout="constrained")
    errors = np.array([[1., 1., 1.], [.85, .87, .88], [.71, .22, .25]])
    for col, title in enumerate(
        ("RGB input", "Annotation", "Ours", "Disagreement", "Modulus prior")
    ):
        axes[0, col].set_title(title, fontsize=8, fontweight="semibold")
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
            # Only annotated pixels can establish a disagreement. Unknown predictions
            # at annotated pixels are counted as disagreements as in the evaluation.
            evaluated = (target > 0).astype(int)
            evaluated[(target > 0) & (target != pred)] = 2
            axes[row, 3].imshow(errors[evaluated][crop], interpolation="nearest")
            ax = axes[row, 4]
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
    fig.colorbar(layer, ax=axes[:, 4], shrink=0.53, aspect=24, label="GPa")
    fig.legend(
        handles=[
            Patch(color=c, label=n)
            for c, n in zip(COLORS, ["Unassigned", "Fabric", "Wood", "Steel"])
        ] + [Patch(color=errors[1], label="Agreement"), Patch(color=errors[2], label="Disagreement")],
        loc="outside lower center",
        ncol=6,
        fontsize=7,
        columnspacing=1.2,
        frameon=False,
    )
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"physseg_material_modulus.{ext}", dpi=320)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    render(parser.parse_args().source)
