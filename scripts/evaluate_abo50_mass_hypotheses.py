"""Score frozen construction hypotheses with the benchmark's four mass metrics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from prepare_abo50_benchmark import digest, read_json, write_json

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tactile3d.models.calibration import mass_error_arrays, mass_metrics

METHODS = ("direct_mass_kg", "all_solid_mass_kg", "construction_mass_kg")


def paired_comparison(rows: list[tuple[float, float, float]]) -> dict:
    target, prediction, baseline = np.asarray(rows, dtype=float).T
    errors = mass_error_arrays(target, prediction)
    reference = mass_error_arrays(target, baseline)
    samples = np.random.default_rng(17).integers(0, len(rows), (2000, len(rows)))
    result = {}
    for metric in errors:
        difference = errors[metric] - reference[metric]
        result[metric] = {
            "method_minus_reference": float(difference.mean()),
            "paired_bootstrap_ci95": np.quantile(
                difference[samples].mean(axis=1), [0.025, 0.975]
            ).tolist(),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    path = args.run / "predictions.json"
    if read_json(args.run / "prediction_seal.json")["sha256"] != digest(path):
        raise ValueError("Prediction seal mismatch")
    baseline = read_json(args.baseline)
    if (
        read_json(args.run / "protocol.json")["cohort_sha256"]
        != baseline["cohort_sha256"]
    ):
        raise ValueError("Baseline and predictions use different observation manifests")
    rows = read_json(path)
    reference = baseline["per_scene"]
    if len(rows) != 50 or [row["scene_id"] for row in rows] != [
        row["scene_id"] for row in reference
    ]:
        raise ValueError("Expected all 50 rows in frozen order")
    methods = {}
    for name in METHODS:
        valid = [
            (ref["target_mass_kg"], row[name], ref["mass_kg"])
            for row, ref in zip(rows, reference)
            if name in row
        ]
        targets = [item[0] for item in valid]
        estimates = [item[1] for item in valid]
        metrics = mass_metrics(targets, estimates) if valid else None
        methods[name] = {
            "coverage": len(valid),
            "metrics": metrics,
            "full_cohort": len(valid) == 50,
            "paired_delta_vs_nerf_cache": paired_comparison(valid) if valid else None,
            "failed_scene_ids": [row["scene_id"] for row in rows if name not in row],
        }
    common = [
        (ref["target_mass_kg"], row["construction_mass_kg"], row["all_solid_mass_kg"])
        for row, ref in zip(rows, reference)
        if "construction_mass_kg" in row and "all_solid_mass_kg" in row
    ]
    write_json(
        args.output,
        {
            "requested": 50,
            "methods": methods,
            "baseline": baseline["metrics"],
            "construction_vs_all_solid": {
                "matched_objects": len(common),
                "paired_delta": paired_comparison(common) if common else None,
            },
        },
    )


if __name__ == "__main__":
    main()
