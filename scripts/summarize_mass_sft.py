"""检查封存预测，报告完整测试集与未用于五物体开发的子集。"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tactile3d.models.calibration import mass_metrics
from tactile3d.physics.part_io import file_sha256, write_json

COLORS = ("#5A76A8", "#B0764F")


def plot_results(
    rows: list[dict[str, Any]], state: dict[str, Any], output: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(12, 3.6), layout="constrained")
    truth = np.array([r["target_mass_kg"] for r in rows])
    for key, color, label in zip(
        ("before", "after"), COLORS, ("NF4 base", "NF4 + QLoRA"), strict=True
    ):
        estimate = np.array([r[key] for r in rows])
        axes[0].scatter(truth, estimate, s=14, alpha=0.65, color=color, label=label)
        error = np.sort(100 * np.abs(estimate - truth) / truth)
        axes[1].step(
            error,
            np.arange(1, len(error) + 1) / len(error),
            where="post",
            color=color,
            label=label,
        )
    limits = [
        min(truth.min(), min(r["before"] for r in rows), min(r["after"] for r in rows))
        * 0.7,
        max(truth.max(), max(r["before"] for r in rows), max(r["after"] for r in rows))
        * 1.4,
    ]
    axes[0].plot(limits, limits, "--", color="0.6", lw=1)
    axes[0].set(
        xscale="log",
        yscale="log",
        xlim=(truth.min() * 0.7, truth.max() * 1.4),
        ylim=limits,
        xlabel="ABO mass label (kg)",
        ylabel="Predicted mass (kg)",
    )
    axes[1].set(
        xscale="symlog",
        xlabel="Absolute relative mass error (%)",
        ylabel="Fraction of objects",
    )
    for key, color, label in zip(
        ("loss", "eval_loss"), COLORS, ("Train", "Validation"), strict=True
    ):
        history = [v for v in state["log_history"] if key in v]
        axes[2].plot(
            [v["step"] for v in history],
            [v[key] for v in history],
            color=color,
            label=label,
            marker="." if key == "eval_loss" else None,
        )
    axes[2].set(xlabel="Optimizer update", ylabel="Answer-token cross-entropy")
    for axis in axes:
        axis.legend(frameon=False, fontsize=8)
        axis.spines[["right", "top"]].set_visible(False)
    figure.suptitle(
        "ABO mass-only supervision | one seed | "
        "10 test objects in historical development pilot",
        fontsize=10,
    )
    for suffix in ("png", "svg"):
        figure.savefig(output / f"mass_sft.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(figure)


def summarize(run: Path, dataset: Path, output: Path) -> dict[str, Any]:
    """不删难例；模型格式失败则拒绝制作全覆盖对比图。"""
    predictions, protocols = {}, {}
    for phase in ("before", "after"):
        path = run / phase
        seal = json.loads((path / "prediction_seal.json").read_text())["sha256"]
        if file_sha256(path / "predictions.json") != seal:
            raise ValueError("预测封存校验失败")
        report = json.loads((path / "report.json").read_text())
        if report["coverage"] != 1.0:
            raise ValueError("存在失败，须单独报告失败与条件指标，不制作全覆盖图")
        predictions[phase] = {
            r["scene_id"]: r["predicted_mass_kg"]
            for r in json.loads((path / "predictions.json").read_text())
        }
        protocols[phase] = json.loads((path / "protocol.json").read_text())
    for key in ("test_inputs_sha256", "dataset_manifest_sha256", "prompt", "system"):
        if protocols["before"][key] != protocols["after"][key]:
            raise ValueError(f"训练前后评价不一致：{key}")
    if predictions["before"].keys() != predictions["after"].keys():
        raise ValueError("测试对象不一致")
    labels = json.loads((dataset / "evaluation_labels.json").read_text())
    old_five = set(
        json.loads(Path("configs/five_object_pilot.json").read_text())["object_ids"]
    )
    rows = [
        {
            "scene_id": scene,
            "target_mass_kg": labels[scene],
            "prior_development": scene in list(predictions["before"])[:10],
            "before": mass,
            "after": predictions["after"][scene],
        }
        for scene, mass in predictions["before"].items()
    ]
    results = {}
    for subset, selected in (
        ("all_test", rows),
        ("outside_five_95", [r for r in rows if r["scene_id"] not in old_five]),
        ("outside_historical_ten_90", [r for r in rows if not r["prior_development"]]),
    ):
        results[subset] = {
            phase: mass_metrics(
                [r["target_mass_kg"] for r in selected], [r[phase] for r in selected]
            )
            for phase in ("before", "after")
        }
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "scope": "real_ABO_metadata_mass_only_finetuning",
        "metrics": results,
        "real_internal_material_accuracy": False,
        "three_dimensional_physics_method_validated": False,
    }
    write_json(output / "report.json", result)
    with (output / "per_object.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    state = json.loads((run / "adapter/trainer_state.json").read_text())
    plot_results(rows, state, output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.run, args.dataset, args.output), indent=2))


if __name__ == "__main__":
    main()
