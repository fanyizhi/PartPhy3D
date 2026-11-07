"""只绘制真实计时和活跃单元计数，不外推端到端速度。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

COLORS = ("#5A76A8", "#C3916C")


def plot(source: Path, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = json.loads(source.read_text(encoding="utf-8"))
    rows = report["objects"]
    if not rows or any(not r["geometry_equivalent"] for r in rows):
        raise ValueError("存在几何不一致或空实验，不生成通过图")
    output.mkdir(parents=True, exist_ok=False)
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
    x = np.arange(len(rows))
    for j, (mode, color) in enumerate(zip(("dense", "sparse"), COLORS, strict=True)):
        samples = np.array([r["seconds"][mode] for r in rows]) * 1000
        pos = x + (j - 0.5) * 0.32
        axes[0].bar(
            pos,
            np.median(samples, axis=1),
            width=0.3,
            color=color,
            alpha=0.7,
            label=mode,
        )
        for index in range(samples.shape[1]):
            axes[0].scatter(
                pos + (index - 1) * 0.04,
                samples[:, index],
                s=9,
                color=color,
                edgecolor="black",
                linewidth=0.25,
            )
    fraction = np.array([r["sparse_counts"]["weight_cell_fraction"] for r in rows])
    axes[1].bar(x, fraction * 100, color=COLORS[1], label="Weight cells evaluated")
    for i, value in enumerate(fraction):
        axes[1].text(
            i, value * 100 + 0.03, f"{value*100:.2f}%", ha="center", fontsize=8
        )
    labels = [r["scene_id"].split("_")[0] for r in rows]
    for axis in axes:
        axis.set(xticks=x, xticklabels=labels, xlabel="Frozen development object")
        axis.tick_params(axis="x", labelrotation=25)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("Geometry decoding + extraction (ms)")
    axes[1].set_ylabel("Evaluated weight cells / all cells (%)")
    axes[1].set_ylim(0, max(fraction) * 125)
    figure.suptitle(
        "Same triplanes and resolution; identical face topology\n"
        "3 repeats after warmup; full SDF queries retained; not end-to-end speedup",
        fontsize=10,
    )
    for suffix in ("svg", "png"):
        figure.savefig(output / f"sparse_decoder_benchmark.{suffix}", dpi=300)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.source, args.output)


if __name__ == "__main__":
    main()
