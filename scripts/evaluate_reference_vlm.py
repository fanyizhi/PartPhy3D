"""Evaluate sealed VLM states against a small, explicit visible-material review."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tactile3d.physics.part_io import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--review",
        type=Path,
        default=Path("configs/evidence/chaise_surface_material_review.json"),
    )
    args = parser.parse_args()
    source = args.source
    raw_path = source / "vlm" / "records.json"
    records = json.loads(raw_path.read_text())
    seal = json.loads((source / "vlm" / "seal.json").read_text())
    if hashlib.sha256(raw_path.read_bytes()).hexdigest() != seal["sha256"]:
        raise ValueError("VLM seal mismatch")
    review = json.loads(args.review.read_text())
    known = {
        int(i): value
        for i, value in review["component_materials"].items()
        if value != "unknown"
    }
    rows = []
    for arm in ("repeated", "progressive"):
        for trial in sorted({r["trial"] for r in records}):
            for stage in range(3):
                budgeted = [
                    r
                    for r in records
                    if r["arm"] == arm and r["trial"] == trial and r["stage"] <= stage
                ]
                valid = [r for r in budgeted if r["valid"]]
                used = (
                    valid
                    if arm == "repeated"
                    else [r for r in valid if r["stage"] == stage]
                )
                predictions = {}
                for i in known:
                    votes = [
                        next(
                            c
                            for c in r["answer"]["components"]
                            if c["component_id"] == i
                        )["surface_material"]
                        for r in used
                    ]
                    predictions[i] = (
                        Counter(votes).most_common(1)[0][0] if votes else "unknown"
                    )
                correct = sum(predictions[i] == truth for i, truth in known.items())
                rows.append(
                    {
                        "arm": arm,
                        "trial": trial,
                        "calls": stage + 1,
                        "known_components": len(known),
                        "correct": correct,
                        "accuracy": correct / len(known),
                        "coverage": sum(x != "unknown" for x in predictions.values())
                        / len(known),
                        "predictions": predictions,
                        "total_seconds": sum(
                            r["response"]["seconds"] for r in budgeted
                        ),
                        "total_output_tokens": sum(
                            r["response"]["output_tokens"] for r in budgeted
                        ),
                        "peak_gpu_bytes": max(
                            r["response"]["peak_gpu_bytes"] for r in budgeted
                        ),
                    }
                )
    report = {
        "scope": review["scope"],
        "reference_sha256": hashlib.sha256(args.review.read_bytes()).hexdigest(),
        "interior_accuracy_evaluated": False,
        "raw_call_count": len(records),
        "valid_call_count": sum(r["valid"] for r in records),
        "records": rows,
    }
    write_json(source / "vlm_evaluation.json", report)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), layout="constrained")
    for arm, color, marker in (
        ("repeated", "#7b8b9b", "o"),
        ("progressive", "#bd805c", "s"),
    ):
        selected = [r for r in rows if r["arm"] == arm]
        means = [
            np.mean([r["accuracy"] for r in selected if r["calls"] == n])
            for n in (1, 2, 3)
        ]
        axes[0].plot(
            [1, 2, 3],
            np.array(means) * 100,
            marker=marker,
            color=color,
            label=arm,
            fillstyle="none" if arm == "progressive" else "full",
            markersize=7,
        )
        axes[1].plot(
            [1, 2, 3],
            [
                np.mean([r["total_seconds"] for r in selected if r["calls"] == n])
                for n in (1, 2, 3)
            ],
            marker=marker,
            color=color,
        )
    axes[0].set(
        title="Visible-material review", ylabel="Component accuracy (%)", ylim=(0, 105)
    )
    axes[1].set(title="Measured VLM generation cost", ylabel="Cumulative seconds")
    for ax in axes:
        ax.set(xlabel="Calls per trial", xticks=[1, 2, 3])
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=8)
    fig.savefig(source / "vlm_rounds.png", dpi=250)
    fig.savefig(source / "vlm_rounds.svg")
    plt.close(fig)
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
