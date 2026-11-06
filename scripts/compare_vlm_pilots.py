"""比较已封存的同一五物体VLM开发实验，不启动模型或重新拟合。"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tactile3d.physics.part_io import file_sha256, write_json

COLORS = ("#8198B8", "#455F86", "#D1A789", "#A56F4B")
MARKERS = ("o", "s", "^", "D", "v")
OBJECT_NAMES = ("Chaise", "USB hub", "Ladder", "Mirror", "Sleeve")


def plot_comparison(rows: list[dict[str, Any]], output: Path) -> None:
    """显示每个对象误差，避免五样本均值掩盖失败案例。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    labels = list(dict.fromkeys(row["method"] for row in rows))
    errors = np.array(
        [
            [row["ape_percent"] for row in rows if row["method"] == label]
            for label in labels
        ]
    )
    figure, axes = plt.subplots(1, 2, figsize=(10.6, 4.4), layout="constrained")
    x = np.arange(len(labels))
    axes[0].bar(x, errors.mean(axis=1), color=COLORS[: len(labels)], alpha=0.45)
    for index, (name, marker) in enumerate(zip(OBJECT_NAMES, MARKERS, strict=True)):
        axes[0].scatter(
            x + (index - 2) * 0.05,
            errors[:, index],
            marker=marker,
            s=24,
            color="#39454F",
            linewidth=0.7,
            zorder=3,
        )
    axes[0].set(
        xticks=x,
        xticklabels=[label.replace(" / ", "\n") for label in labels],
        xlabel="Direct image-only VLM baseline",
        ylabel="Absolute relative mass error (%)",
        title="a  Five development objects; bar = mean",
    )
    masses = np.array(
        [
            [row["predicted_mass_kg"] for row in rows if row["method"] == label]
            for label in labels
        ]
    )
    truth = np.array([r["target_mass_kg"] for r in rows[:5]])
    for method_index, label in enumerate(labels):
        for object_index, marker in enumerate(MARKERS):
            axes[1].scatter(
                truth[object_index],
                masses[method_index, object_index],
                color=COLORS[method_index],
                marker=marker,
                s=30,
                alpha=0.85,
            )
    lower = min(truth.min(), masses.min()) * 0.7
    upper = max(truth.max(), masses.max()) * 1.4
    axes[1].plot([lower, upper], [lower, upper], "--", color="0.5", lw=1)
    axes[1].set(
        xscale="log",
        yscale="log",
        xlim=(lower, upper),
        ylim=(lower, upper),
        xlabel="ABO mass label (kg)",
        ylabel="Predicted mass (kg)",
        title="b  Reference line: prediction = label",
    )
    axes[1].legend(
        handles=[
            Line2D([0], [0], marker="o", linestyle="none", color=COLORS[i], label=label)
            for i, label in enumerate(labels)
        ],
        frameon=False,
        fontsize=8,
        loc="upper left",
    )
    figure.legend(
        handles=[
            Line2D([0], [0], marker=m, linestyle="none", color="#39454F", label=n)
            for n, m in zip(OBJECT_NAMES, MARKERS, strict=True)
        ],
        loc="outside lower center",
        ncol=5,
        frameon=False,
        fontsize=8,
    )
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(direction="out")
    for suffix in ("svg", "png"):
        figure.savefig(
            output / f"vlm_capacity_views.{suffix}", dpi=300, bbox_inches="tight"
        )
    plt.close(figure)


def compare_runs(runs: list[str], output: Path) -> dict[str, Any]:
    """只接受相同对象、提示、质量标签和有效封存哈希的完整运行。"""
    if not 2 <= len(runs) <= len(COLORS):
        raise ValueError("比较需要2到4个运行")
    records, rows, ids, targets, prompt = [], [], None, None, None
    labels: set[str] = set()
    input_hashes: dict[tuple[str, str], str] = {}
    for value in runs:
        label, separator, raw_path = value.partition("=")
        if not separator or not label.strip() or label in labels:
            raise ValueError("--run需要唯一名称=完整实验目录")
        labels.add(label)
        path = Path(raw_path).resolve()
        report = json.loads((path / "report.json").read_text())
        protocol = json.loads((path / "protocol.json").read_text())
        normalized_prompt = protocol["prompt"].replace(
            f"{len(protocol['views'])} image(s)", "{view_count} image(s)", 1
        )
        for scene in protocol["inputs"]:
            for image in scene["images"]:
                key = (scene["scene_id"], Path(image["path"]).name)
                if key in input_hashes and input_hashes[key] != image["sha256"]:
                    raise ValueError("同一观测在不同运行中的SHA256不一致")
                input_hashes[key] = image["sha256"]
        seal = json.loads((path / "prediction_seal.json").read_text())["sha256"]
        if file_sha256(path / "sealed_predictions.json") != seal:
            raise ValueError(f"封存预测校验失败：{path}")
        if report["status"] != "complete" or report["successful"] != 5:
            raise ValueError("当前固定队列比较要求每次五个物体全部成功")
        current_ids = [row["scene_id"] for row in report["evaluation"]]
        current_targets = [row["target_mass_kg"] for row in report["evaluation"]]
        if ids is None:
            ids, targets, prompt = current_ids, current_targets, normalized_prompt
        elif (
            ids != current_ids
            or targets != current_targets
            or prompt != normalized_prompt
        ):
            raise ValueError("队列、标签或核心提示不同，拒绝直接合并")
        for index, row in enumerate(report["evaluation"]):
            rows.append(
                {
                    "method": label,
                    "object": OBJECT_NAMES[index],
                    **row,
                    "ape_percent": 100
                    * abs(row["predicted_mass_kg"] - row["target_mass_kg"])
                    / row["target_mass_kg"],
                }
            )
        records.append(
            {
                "method": label,
                "directory": str(path),
                "prediction_seal": seal,
                "report_sha256": file_sha256(path / "report.json"),
                "metrics": report["metrics_successful_objects_only"],
                "views": protocol["views"],
            }
        )
    result = {
        "scope": "fixed_five_object_development_diagnostic",
        "statistical_generalization_established": False,
        "our_complete_method_evaluated": False,
        "published_external_baselines_evaluated": False,
        "object_ids": ids,
        "runs": records,
    }
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "report.json", result)
    with (output / "per_object.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_comparison(rows, output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compare_runs(args.run, args.output), indent=2))


if __name__ == "__main__":
    main()
