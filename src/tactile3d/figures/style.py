from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

PALETTE = (
    "#4C78A8",
    "#72B7B2",
    "#F2CF5B",
    "#E07B54",
    "#B279A2",
    "#8C8C8C",
)


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.titlesize": 7.5,
            "axes.labelsize": 7,
            "axes.linewidth": 0.7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save_figure(
    figure: plt.Figure,
    stem: str | Path,
    formats: tuple[str, ...] = ("png", "svg", "pdf", "tiff"),
) -> list[str]:
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in formats:
        suffix = suffix.lower()
        if suffix not in {"png", "svg", "pdf", "tiff"}:
            raise ValueError(f"Unsupported figure format: {suffix}")
        path = stem.with_suffix(f".{suffix}")
        dpi = 600 if suffix in {"png", "tiff"} else None
        figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        outputs.append(str(path))
    plt.close(figure)
    return outputs
