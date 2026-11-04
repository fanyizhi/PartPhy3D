from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from tactile3d.utils.io import read_json, write_json


def import_mass_baseline(
    predictions_path: str | Path,
    targets_path: str | Path,
    output_path: str | Path,
    *,
    method: str,
    scene_id_separator: str = "_",
) -> dict[str, Any]:
    """Convert scalar or PUGS-style interval predictions to the project schema."""

    predictions = _unwrap_mapping(read_json(predictions_path), "predictions")
    targets = _unwrap_mapping(read_json(targets_path), "targets")
    rows = []
    skipped = []
    seen = set()
    for prediction_id, raw_prediction in predictions.items():
        scene_id = _match_target_id(str(prediction_id), targets, scene_id_separator)
        if scene_id is None:
            skipped.append(str(prediction_id))
            continue
        if scene_id in seen:
            raise ValueError(f"Multiple prediction IDs map to scene {scene_id}")
        seen.add(scene_id)
        mean, lower, upper, semantics = _prediction_stats(raw_prediction)
        target = _target_mass(targets[scene_id])
        if not np.isfinite([target, mean]).all() or target <= 0 or mean <= 0:
            raise ValueError(f"Masses must be finite and positive for scene {scene_id}")
        mass = {"mean": mean}
        if lower is not None and upper is not None:
            if not np.isfinite([lower, upper]).all() or lower < 0 or lower > upper:
                raise ValueError(
                    f"Prediction interval is reversed for scene {prediction_id}"
                )
            keys = (
                ("q05", "q95")
                if semantics == "reported_quantiles"
                else ("range_low", "range_high")
            )
            mass.update(dict(zip(keys, (lower, upper), strict=True)))
        rows.append(
            {
                "scene_id": scene_id,
                "target_mass_kg": target,
                "object": {"mass_kg": mass},
                "source_prediction_id": str(prediction_id),
                "interval_semantics": semantics,
            }
        )
    if not rows:
        raise ValueError("No prediction IDs could be matched to a target mass")
    payload = {
        "schema_version": 1,
        "method": method,
        "predictions": rows,
        "import_summary": {
            "num_imported": len(rows),
            "num_skipped": len(skipped),
            "skipped_prediction_ids": skipped,
        },
    }
    write_json(output_path, payload)
    return payload["import_summary"]


def _unwrap_mapping(payload: Any, key: str) -> dict[str, Any]:
    if isinstance(payload, dict) and isinstance(payload.get(key), dict):
        payload = payload[key]
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"Expected a non-empty JSON object containing {key}")
    return {str(item_key): value for item_key, value in payload.items()}


def _match_target_id(
    prediction_id: str, targets: dict[str, Any], separator: str
) -> str | None:
    if prediction_id in targets:
        return prediction_id
    candidates = [
        target_id
        for target_id in targets
        if prediction_id.startswith(f"{target_id}{separator}")
    ]
    return max(candidates, key=len) if candidates else None


def _prediction_stats(
    value: Any,
) -> tuple[float, float | None, float | None, str]:
    """区分原生范围与显式分位数，不把材料范围冒充90%概率区间。"""
    if isinstance(value, bool):
        raise ValueError("Boolean values are not mass predictions")
    if isinstance(value, (int, float)):
        return float(value), None, None, "none"
    if isinstance(value, list) and len(value) == 2:
        lower, upper = map(float, value)
        return (lower + upper) / 2.0, lower, upper, "range_bounds_not_quantiles"
    if isinstance(value, dict):
        mean_value = value.get("mean", value.get("q50", value.get("prediction")))
        quantiles = "q05" in value or "q95" in value
        lower = value.get("q05") if quantiles else value.get("low")
        upper = value.get("q95") if quantiles else value.get("high")
        if mean_value is None and lower is not None and upper is not None:
            if quantiles:
                raise ValueError(
                    "Quantile bounds alone do not define a mean prediction"
                )
            mean_value = (float(lower) + float(upper)) / 2.0
        if mean_value is None:
            raise ValueError(f"Cannot find a point prediction in {value!r}")
        lower = float(lower) if lower is not None else None
        upper = float(upper) if upper is not None else None
        if (lower is None) != (upper is None):
            raise ValueError(
                "Prediction dictionaries must provide both interval bounds"
            )
        semantics = (
            "reported_quantiles"
            if quantiles
            else ("range_bounds_not_quantiles" if lower is not None else "none")
        )
        return float(mean_value), lower, upper, semantics
    raise ValueError(f"Unsupported prediction value: {value!r}")


def _target_mass(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        for key in ("mass_kg", "mass", "weight_kg", "weight"):
            if key in value:
                return float(value[key])
    raise ValueError(f"Unsupported target mass value: {value!r}")
