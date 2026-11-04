from __future__ import annotations

from tactile3d.utils.io import read_json


def evaluate_summary(pred_path: str, target_path: str) -> dict:
    pred = read_json(pred_path)
    target = read_json(target_path)

    metrics = {}
    pred_mass = pred["object"]["mass_kg"]["mean"]
    pred_volume = pred["object"]["volume_m3"]["mean"]
    if "mass_kg" in target:
        metrics["mass_abs_error_kg"] = abs(pred_mass - float(target["mass_kg"]))
    if "volume_m3" in target:
        metrics["volume_abs_error_m3"] = abs(pred_volume - float(target["volume_m3"]))
    if "watertight" in target:
        metrics["watertight_rate"] = pred["object"].get("watertight_rate", 0.0)
    if "mass_kg" in target:
        lo = pred["object"]["mass_kg"]["q05"]
        hi = pred["object"]["mass_kg"]["q95"]
        metrics["mass_in_90pct_interval"] = float(lo <= float(target["mass_kg"]) <= hi)
    return metrics
