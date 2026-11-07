"""从封存尺度预测绘制测试集误差与校准区间，不修改模型或删难例。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tactile3d.physics.part_io import file_sha256


def plot(run: Path, catalog: Path, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seal = json.loads((run / "prediction_seal.json").read_text(encoding="utf-8"))
    if file_sha256(run / "test_predictions.json") != seal["sha256"]:
        raise ValueError("测试尺度预测封存错误")
    protocol = json.loads((run / "protocol.json").read_text(encoding="utf-8"))
    if file_sha256(catalog) != protocol["catalog_sha256"]:
        raise ValueError("商品尺寸标签与实验协议不一致")
    rows = json.loads((run / "test_predictions.json").read_text(encoding="utf-8"))
    source = json.loads(catalog.read_text(encoding="utf-8"))["objects"]
    truth = np.array([source[r["scene_id"]]["max_dimension_m"] for r in rows])
    pred = np.array([r["max_length_m"] for r in rows])
    mean = np.array([r["train_mean_baseline_m"] for r in rows])
    if len(rows) != 100:
        raise ValueError("这张固定协议图要求完整100测试对象，不接受只画成功样本")
    output.mkdir(parents=True, exist_ok=False)
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.7), layout="constrained")
    for estimates, name, color in (
        (mean, "Train geometric mean", "#C3916C"),
        (pred, "CLIP + ridge", "#5A76A8"),
    ):
        axes[0].scatter(truth, estimates, s=15, alpha=0.65, label=name, color=color)
        error = np.sort(np.abs(estimates / truth - 1) * 100)
        axes[1].step(
            error, np.arange(1, 101) / 100, where="post", color=color, label=name
        )
    low = min(truth.min(), pred.min(), mean.min()) * 0.8
    high = max(truth.max(), pred.max(), mean.max()) * 1.2
    axes[0].plot([low, high], [low, high], "--", color="0.5", lw=1)
    axes[0].set(
        xscale="log",
        yscale="log",
        xlabel="Catalog max dimension (m)",
        ylabel="Predicted max dimension (m)",
    )
    axes[1].set(
        xscale="symlog",
        xlabel="Absolute relative scale error (%)",
        ylabel="Fraction of 100 test objects",
    )
    lo = np.array([r["conformal_scale_low_m"] for r in rows[:5]])
    hi = np.array([r["conformal_scale_high_m"] for r in rows[:5]])
    x = np.arange(5)
    axes[2].errorbar(
        x,
        pred[:5],
        yerr=np.array([pred[:5] - lo, hi - pred[:5]]),
        fmt="o",
        color="#5A76A8",
        capsize=3,
        label="90% scale interval",
    )
    axes[2].scatter(x, truth[:5], marker="x", color="black", label="Catalog")
    axes[2].set(
        yscale="log",
        xticks=x,
        xticklabels=[r["scene_id"].split("_")[0] for r in rows[:5]],
        xlabel="Five development objects",
        ylabel="Max dimension (m)",
    )
    axes[2].tick_params(axis="x", labelrotation=35)
    for axis in axes:
        axis.legend(fontsize=7, frameon=False)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        "Image-only scale | 300 train / 50 tune / 50 calibration / 100 test\n"
        "Catalog reference; uncertainty is for scale, not complete physical properties",
        fontsize=10,
    )
    for suffix in ("svg", "png"):
        figure.savefig(output / f"image_scale.{suffix}", dpi=300)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.run, args.catalog, args.output)


if __name__ == "__main__":
    main()
