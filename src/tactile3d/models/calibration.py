from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize

from tactile3d.physics.materials import MaterialPrior
from tactile3d.utils.io import read_json, read_jsonl, write_json


def load_calibration(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return read_json(path)


def calibrated_probabilities(
    logits: np.ndarray,
    material_names: list[str] | tuple[str, ...],
    calibration: dict[str, Any] | None,
    default_temperature: float,
) -> np.ndarray:
    bias = np.zeros(len(material_names), dtype=np.float64)
    temperature = default_temperature
    if calibration is not None:
        expected = list(calibration["material_names"])
        if list(material_names) != expected:
            raise ValueError(
                "Calibration material order does not match configured priors"
            )
        bias = np.asarray(calibration["material_bias"], dtype=np.float64)
        temperature = float(calibration.get("temperature", temperature))
    return _softmax(
        np.asarray(logits, dtype=np.float64) / max(temperature, 1e-6) + bias
    )


def fit_mass_calibration(
    feature_file: str | Path,
    priors: list[MaterialPrior],
    output_path: str | Path,
    l2: float = 0.02,
    max_iters: int = 500,
) -> dict[str, Any]:
    rows = [
        row
        for row in read_jsonl(feature_file)
        if float(row.get("target_mass_kg", 0)) > 0
    ]
    if any("split" in row for row in rows):
        rows = [row for row in rows if row.get("split") == "train"]
    if not rows:
        raise ValueError(f"No labelled training feature rows found in {feature_file}")
    material_names = [prior.name for prior in priors]
    density_means = np.asarray(
        [prior.density_kg_m3.mean for prior in priors], dtype=np.float64
    )
    _validate_rows(rows, material_names)
    num_materials = len(material_names)

    def unpack(parameters: np.ndarray) -> tuple[np.ndarray, float, float]:
        bias = parameters[:num_materials]
        bias = bias - bias.mean()
        density_scale = float(np.exp(parameters[num_materials]))
        temperature = float(np.exp(parameters[num_materials + 1]))
        return bias, density_scale, temperature

    def objective(parameters: np.ndarray) -> float:
        bias, density_scale, temperature = unpack(parameters)
        predicted = np.asarray(
            [
                _row_mass(row, density_means, bias, density_scale, temperature)
                for row in rows
            ],
            dtype=np.float64,
        )
        target = np.asarray([row["target_mass_kg"] for row in rows], dtype=np.float64)
        residual = np.log(target) - np.log(np.maximum(predicted, 1e-12))
        regularizer = l2 * float(np.sum(bias**2))
        regularizer += 0.01 * float(parameters[num_materials] ** 2)
        regularizer += 0.01 * float(parameters[num_materials + 1] ** 2)
        return float(np.mean(residual**2) + regularizer)

    initial = np.zeros(num_materials + 2, dtype=np.float64)
    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        bounds=[(-4.0, 4.0)] * num_materials + [(-4.0, 4.0), (-2.0, 2.0)],
        options={"maxiter": max_iters},
    )
    bias, density_scale, temperature = unpack(result.x)
    predictions = [
        _row_mass(row, density_means, bias, density_scale, temperature) for row in rows
    ]
    target = [float(row["target_mass_kg"]) for row in rows]
    log_residual = np.log(np.asarray(target)) - np.log(np.maximum(predictions, 1e-12))
    sigma = max(float(np.sqrt(np.mean(log_residual**2))), 0.05)
    checkpoint = {
        "schema_version": 1,
        "material_names": material_names,
        "material_bias": bias.tolist(),
        "density_scale": density_scale,
        "residual_log_sigma": sigma,
        "temperature": temperature,
        "num_train_scenes": len(rows),
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "train_metrics": mass_metrics(target, predictions),
    }
    write_json(output_path, checkpoint)
    return checkpoint


def apply_calibration_to_density(
    value: float, calibration: dict[str, Any] | None
) -> float:
    if calibration is None:
        return value
    return value * float(calibration.get("density_scale", 1.0))


def residual_log_sigma(calibration: dict[str, Any] | None) -> float:
    return float(calibration.get("residual_log_sigma", 0.0)) if calibration else 0.0


def mass_error_arrays(
    target: list[float] | np.ndarray, predicted: list[float] | np.ndarray
) -> dict[str, np.ndarray]:
    """计算逐物体质量误差；无效预测明确拒绝，不用截断掩盖失败。"""
    gt = np.asarray(target, dtype=np.float64)
    pred = np.asarray(predicted, dtype=np.float64)
    if gt.ndim != 1 or pred.shape != gt.shape or not gt.size:
        raise ValueError("Mass metrics require equal non-empty one-dimensional arrays")
    if not np.isfinite(gt).all() or not np.isfinite(pred).all():
        raise ValueError("Mass metrics require finite targets and predictions")
    if np.any(gt <= 0) or np.any(pred <= 0):
        raise ValueError("Mass metrics require strictly positive masses")
    absolute = np.abs(gt - pred)
    return {
        "ade_kg": absolute,
        "alde": np.abs(np.log(gt) - np.log(pred)),
        "ape": absolute / gt,
        "mnre": np.minimum(gt / pred, pred / gt),
    }


def mass_metrics(target: list[float], predicted: list[float]) -> dict[str, float]:
    values = mass_error_arrays(target, predicted)
    return {
        "num_scenes": len(values["ade_kg"]),
        **{name: float(value.mean()) for name, value in values.items()},
        "rmse_kg": float(np.sqrt(np.mean(values["ade_kg"] ** 2))),
    }


def _row_mass(
    row: dict[str, Any],
    density_means: np.ndarray,
    bias: np.ndarray,
    density_scale: float,
    temperature: float,
) -> float:
    expected_volumes = row.get("expected_material_volumes_m3")
    if expected_volumes is not None:
        volumes = np.asarray(expected_volumes, dtype=np.float64)
        if volumes.shape != density_means.shape:
            raise ValueError(
                "Invalid expected material volumes in scene "
                f"{row.get('scene_id', '<unknown>')}"
            )
        return max(float(volumes @ density_means) * density_scale, 1e-12)
    mass = 0.0
    for part in row["parts"]:
        logits = np.asarray(part["mean_logits"], dtype=np.float64)
        probabilities = _softmax(logits / max(temperature, 1e-6) + bias)
        density = float(probabilities @ density_means) * density_scale
        mass += float(part["volume_m3"]) * density
    return max(mass, 1e-12)


def _validate_rows(rows: list[dict[str, Any]], material_names: list[str]) -> None:
    for row in rows:
        if list(row.get("material_names", [])) != material_names:
            raise ValueError(
                f"Material order mismatch in scene {row.get('scene_id', '<unknown>')}"
            )
        if not row.get("parts"):
            raise ValueError(
                f"No parts in feature row {row.get('scene_id', '<unknown>')}"
            )
        expected_volumes = row.get("expected_material_volumes_m3")
        if expected_volumes is not None and len(expected_volumes) != len(
            material_names
        ):
            raise ValueError(
                "Invalid expected material volumes in scene "
                f"{row.get('scene_id', '<unknown>')}"
            )
        for part in row["parts"]:
            if len(part.get("mean_logits", [])) != len(material_names):
                raise ValueError(
                    f"Invalid material logits in scene {row.get('scene_id')}"
                )


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    result = np.exp(shifted)
    return result / result.sum()
