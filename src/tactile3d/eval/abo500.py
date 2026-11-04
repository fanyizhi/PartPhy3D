from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from tactile3d.models.calibration import mass_metrics
from tactile3d.utils.io import read_json, write_json


def evaluate_abo500_predictions(
    predictions_path: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    payload = read_json(predictions_path)
    predictions = payload.get("predictions", []) if isinstance(payload, dict) else payload
    if not isinstance(predictions, list) or not predictions:
        raise ValueError(f"No predictions found in {predictions_path}")

    targets: list[float] = []
    means: list[float] = []
    covered: list[bool] = []
    per_scene: list[dict[str, Any]] = []
    for item in predictions:
        target = float(item["target_mass_kg"])
        stats = item["object"]["mass_kg"]
        mean = float(stats["mean"])
        targets.append(target)
        means.append(mean)
        covered.append(float(stats["q05"]) <= target <= float(stats["q95"]))
        per_scene.append(
            {
                "scene_id": item["scene_id"],
                "target_mass_kg": target,
                "predicted_mass_kg": mean,
                "absolute_error_kg": abs(target - mean),
                "ratio_error": min(target / max(mean, 1e-12), mean / max(target, 1e-12)),
            }
        )

    metrics = mass_metrics(targets, means)
    metrics["coverage_90"] = float(np.mean(covered))
    metrics["mean_interval_width_kg"] = float(
        np.mean(
            [
                float(item["object"]["mass_kg"]["q95"])
                - float(item["object"]["mass_kg"]["q05"])
                for item in predictions
            ]
        )
    )
    result = {"metrics": metrics, "per_scene": per_scene}
    if output_path is not None:
        write_json(output_path, result)
    return result
