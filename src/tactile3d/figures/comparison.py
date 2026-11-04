from __future__ import annotations

import csv
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tactile3d.figures.style import PALETTE, configure_style, save_figure
from tactile3d.models.calibration import mass_error_arrays
from tactile3d.utils.io import ensure_dir, read_json, write_json

METRICS = {
    "ade_kg": {"label": "ADE (kg)", "direction": "lower"},
    "alde": {"label": "ALDE", "direction": "lower"},
    "ape": {"label": "APE (%)", "direction": "lower"},
    "mnre": {"label": "MnRE", "direction": "higher"},
}

UNCERTAINTY_METRICS = {
    "coverage_90": {
        "label": "Reported quantile interval hit fraction",
        "direction": "not automatically calibrated",
    },
    "normalized_width_90": {
        "label": "Normalized interval width",
        "direction": "lower is better",
    },
}


def generate_comparison_figures(
    runs: dict[str, str | Path],
    output_dir: str | Path,
    *,
    bootstrap_samples: int = 2000,
    seed: int = 17,
    formats: tuple[str, ...] = ("png", "svg", "pdf", "tiff"),
    cohort_ids: list[str] | None = None,
    evaluation_context: str = "unspecified",
) -> dict[str, Any]:
    """Compare methods on exactly the same test scenes and export source data."""

    if len(runs) < 2:
        raise ValueError("At least two labelled prediction runs are required")
    if bootstrap_samples < 100:
        raise ValueError("Use at least 100 bootstrap samples")
    loaded = {name: _load_predictions(path) for name, path in runs.items()}
    common_ids = sorted(set.intersection(*(set(items) for items in loaded.values())))
    if cohort_ids is not None:
        if not cohort_ids or len(set(cohort_ids)) != len(cohort_ids):
            raise ValueError("Declared cohort must contain unique non-empty scene IDs")
        for method, items in loaded.items():
            missing = sorted(set(cohort_ids) - set(items))
            if missing:
                raise ValueError(
                    f"Method {method} is missing declared cohort IDs: {missing}"
                )
        common_ids = list(cohort_ids)
    if not common_ids:
        raise ValueError("The supplied runs have no common scene IDs")
    targets = _aligned_targets(loaded, common_ids)
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, len(common_ids), size=(bootstrap_samples, len(common_ids))
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "comparison_unit": "declared cohort"
        if cohort_ids is not None
        else "common objects",
        "evaluation_context": evaluation_context,
        "num_common_scenes": len(common_ids),
        "bootstrap_samples": bootstrap_samples,
        "confidence_interval": "paired nonparametric percentile bootstrap, 95%",
        "methods": {},
        "uncertainty": {},
        "pairwise_differences": {},
    }
    source_rows = []
    per_method_values: dict[str, dict[str, np.ndarray]] = {}
    for method, items in loaded.items():
        prediction = np.asarray(
            [items[scene]["mean"] for scene in common_ids], dtype=np.float64
        )
        values = _metric_values(targets, prediction)
        per_method_values[method] = values
        report["methods"][method] = {
            key: _bootstrap(values[key], indices) for key in METRICS
        }
        q05 = np.asarray(
            [items[scene].get("q05", np.nan) for scene in common_ids], dtype=float
        )
        q95 = np.asarray(
            [items[scene].get("q95", np.nan) for scene in common_ids], dtype=float
        )
        interval_mask = np.isfinite(q05) & np.isfinite(q95)
        if interval_mask.any():
            if np.any(q05[interval_mask] > q95[interval_mask]):
                raise ValueError(f"Invalid posterior interval for method {method}")
            uncertainty = {
                "coverage_90": (
                    (targets[interval_mask] >= q05[interval_mask])
                    & (targets[interval_mask] <= q95[interval_mask])
                ).astype(float),
                "normalized_width_90": (
                    (q95[interval_mask] - q05[interval_mask]) / targets[interval_mask]
                ),
            }
            interval_indices = rng.integers(
                0,
                int(interval_mask.sum()),
                size=(bootstrap_samples, int(interval_mask.sum())),
            )
            report["uncertainty"][method] = {
                "num_intervals": int(interval_mask.sum()),
                **{
                    key: _bootstrap(metric_values, interval_indices)
                    for key, metric_values in uncertainty.items()
                },
            }
        for row_index, scene_id in enumerate(common_ids):
            stats = items[scene_id]
            lower = stats.get("q05")
            upper = stats.get("q95")
            source_rows.append(
                {
                    "method": method,
                    "scene_id": scene_id,
                    "target_mass_kg": targets[row_index],
                    "predicted_mass_kg": prediction[row_index],
                    "q05_kg": lower,
                    "q95_kg": upper,
                    "covered_by_90pct_interval": (
                        float(lower <= targets[row_index] <= upper)
                        if lower is not None and upper is not None
                        else None
                    ),
                    "normalized_90pct_interval_width": (
                        (upper - lower) / targets[row_index]
                        if lower is not None and upper is not None
                        else None
                    ),
                    **{key: values[key][row_index] for key in METRICS},
                }
            )

    for method_a, method_b in combinations(per_method_values, 2):
        comparison_key = f"{method_a} - {method_b}"
        report["pairwise_differences"][comparison_key] = {
            "definition": "metric(method_a) minus metric(method_b), paired by scene_id",
            **{
                metric: _bootstrap(
                    per_method_values[method_a][metric]
                    - per_method_values[method_b][metric],
                    indices,
                )
                for metric in METRICS
            },
        }

    output_dir = ensure_dir(output_dir)
    write_json(output_dir / "comparison_metrics.json", report)
    _write_source_data(output_dir / "comparison_source_data.csv", source_rows)
    files = []
    files.extend(_plot_metric_grid(report, output_dir, formats))
    files.extend(
        _plot_predictions(
            loaded, common_ids, targets, output_dir, formats, evaluation_context
        )
    )
    files.extend(
        _plot_error_distribution(
            per_method_values, output_dir, formats, seed, evaluation_context
        )
    )
    if report["uncertainty"]:
        files.extend(_plot_uncertainty(report, output_dir, formats))
    caption = (
        f"Methods were evaluated on the same {len(common_ids)} objects "
        f"(context: {evaluation_context}). Points show mean "
        "performance and paired nonparametric 95% bootstrap confidence intervals "
        f"({bootstrap_samples} resamples). ADE, ALDE and APE are lower-is-better; "
        "MnRE is higher-is-better. For q05/q95 outputs, the uncertainty panel "
        "reports empirical quantile interval hit fraction and normalized width; "
        "reported quantiles, especially uncalibrated priors, are not evidence "
        "of calibration."
    )
    (output_dir / "figure_caption.txt").write_text(caption, encoding="utf-8")
    report["figure_files"] = files
    return report


def _load_predictions(path: str | Path) -> dict[str, dict[str, float]]:
    payload = read_json(path)
    rows = payload.get("predictions", []) if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"No prediction rows found in {path}")
    result = {}
    for row in rows:
        scene_id = str(row["scene_id"])
        if scene_id in result:
            raise ValueError(f"Duplicate scene ID {scene_id!r} in {path}")
        target = float(row["target_mass_kg"])
        stats = row["object"]["mass_kg"]
        mean = float(stats["mean"])
        if not np.isfinite([target, mean]).all() or target <= 0 or mean <= 0:
            raise ValueError(f"Masses must be positive for scene {scene_id}")
        lower = float(stats["q05"]) if "q05" in stats else None
        upper = float(stats["q95"]) if "q95" in stats else None
        if (lower is None) != (upper is None):
            raise ValueError(f"Both quantile bounds are required for scene {scene_id}")
        if lower is not None and (
            not np.isfinite([lower, upper]).all() or lower < 0 or lower > upper
        ):
            raise ValueError(f"Invalid quantile bounds for scene {scene_id}")
        result[scene_id] = {
            "target": target,
            "mean": mean,
            "q05": lower,
            "q95": upper,
        }
    return result


def _aligned_targets(
    loaded: dict[str, dict[str, dict[str, float]]], common_ids: list[str]
) -> np.ndarray:
    reference = next(iter(loaded.values()))
    targets = np.asarray(
        [reference[scene]["target"] for scene in common_ids], dtype=np.float64
    )
    for method, items in loaded.items():
        candidate = np.asarray([items[scene]["target"] for scene in common_ids])
        if not np.allclose(candidate, targets, rtol=1e-8, atol=1e-10):
            raise ValueError(f"Ground-truth masses disagree for method {method}")
    return targets


def _metric_values(target: np.ndarray, prediction: np.ndarray) -> dict[str, np.ndarray]:
    return mass_error_arrays(target, prediction)


def _bootstrap(values: np.ndarray, indices: np.ndarray) -> dict[str, float | str]:
    samples = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
    }


def _plot_metric_grid(
    report: dict, output_dir: Path, formats: tuple[str, ...]
) -> list[str]:
    configure_style()
    methods = list(report["methods"])
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 4.6), constrained_layout=True)
    figure.suptitle(
        f"{report['evaluation_context']} cohort; "
        f"n={report['num_common_scenes']} objects"
    )
    for panel, (axis, (metric, metadata)) in enumerate(
        zip(axes.flat, METRICS.items(), strict=True)
    ):
        for index, method in enumerate(methods):
            summary = report["methods"][method][metric]
            draw_summary(
                axis,
                index,
                summary,
                PALETTE[index % len(PALETTE)],
                scale=100.0 if metric == "ape" else 1.0,
            )
        axis.set_xticks(range(len(methods)), methods, rotation=25, ha="right")
        axis.set_ylabel(metadata["label"])
        axis.set_xlabel("Method")
        axis.set_title(
            f"{chr(97 + panel)}  {metadata['label']} "
            f"({metadata['direction']} is better)",
            loc="left",
            weight="bold",
        )
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.7)
        axis.legend(loc="best", fontsize=6)
    return save_figure(figure, output_dir / "mass_metrics_comparison", formats)


def draw_summary(
    axis: plt.Axes,
    index: int,
    summary: dict[str, float],
    color: str,
    scale: float = 1.0,
) -> None:
    """区间与均值分开画，避免偏斜分布下均值落在百分位区间外出错。"""
    low, high = summary["ci95_low"] * scale, summary["ci95_high"] * scale
    axis.vlines(
        index,
        low,
        high,
        color=color,
        linewidth=1.2,
        label="95% paired bootstrap" if index == 0 else None,
    )
    axis.plot([index, index], [low, high], marker="_", linestyle="none", color=color)
    axis.plot(
        index,
        summary["mean"] * scale,
        "o",
        markersize=5,
        color=color,
        label="Mean" if index == 0 else None,
    )


def _plot_predictions(
    loaded: dict[str, dict[str, dict[str, float]]],
    common_ids: list[str],
    targets: np.ndarray,
    output_dir: Path,
    formats: tuple[str, ...],
    evaluation_context: str = "unspecified",
) -> list[str]:
    configure_style()
    figure, axis = plt.subplots(figsize=(3.6, 3.25), constrained_layout=True)
    figure.suptitle(f"{evaluation_context} cohort; n={len(common_ids)} objects")
    for index, (method, items) in enumerate(loaded.items()):
        prediction = np.asarray([items[scene]["mean"] for scene in common_ids])
        axis.scatter(
            targets,
            prediction,
            s=22,
            color=PALETTE[index % len(PALETTE)],
            edgecolor="white",
            linewidth=0.35,
            alpha=0.85,
            label=method,
        )
    low = min(
        float(targets.min()),
        *(item["mean"] for rows in loaded.values() for item in rows.values()),
    )
    high = max(
        float(targets.max()),
        *(item["mean"] for rows in loaded.values() for item in rows.values()),
    )
    low, high = low * 0.8, high * 1.2
    axis.plot([low, high], [low, high], color="#555555", linestyle="--", linewidth=0.8)
    axis.set(xscale="log", yscale="log", xlim=(low, high), ylim=(low, high))
    axis.set_xlabel("Ground-truth mass (kg)")
    axis.set_ylabel("Predicted mass (kg)")
    axis.set_title("Prediction agreement", loc="left", weight="bold")
    axis.legend()
    axis.grid(color="#E2E2E2", linewidth=0.45, alpha=0.7)
    return save_figure(figure, output_dir / "predicted_vs_ground_truth", formats)


def _plot_error_distribution(
    values: dict[str, dict[str, np.ndarray]],
    output_dir: Path,
    formats: tuple[str, ...],
    seed: int,
    evaluation_context: str = "unspecified",
) -> list[str]:
    configure_style()
    methods = list(values)
    figure, axis = plt.subplots(figsize=(4.4, 3.0), constrained_layout=True)
    figure.suptitle(f"{evaluation_context} cohort")
    box = axis.boxplot(
        [100.0 * values[method]["ape"] for method in methods],
        patch_artist=True,
        widths=0.48,
        showfliers=False,
        medianprops={"color": "#222222", "linewidth": 1.0},
    )
    for index, patch in enumerate(box["boxes"]):
        patch.set_facecolor(PALETTE[index % len(PALETTE)])
        patch.set_alpha(0.45)
    rng = np.random.default_rng(seed)
    for index, method in enumerate(methods, start=1):
        data = 100.0 * values[method]["ape"]
        x = index + rng.normal(0, 0.045, size=len(data))
        axis.scatter(
            x, data, s=10, color=PALETTE[(index - 1) % len(PALETTE)], alpha=0.55
        )
    axis.set_xticks(range(1, len(methods) + 1), methods, rotation=25, ha="right")
    axis.set_ylabel("Absolute percentage error (%)")
    axis.set_xlabel("Method")
    axis.set_title("Per-object error distribution", loc="left", weight="bold")
    axis.grid(axis="y", color="#E2E2E2", linewidth=0.45)
    return save_figure(figure, output_dir / "per_object_ape", formats)


def _plot_uncertainty(
    report: dict, output_dir: Path, formats: tuple[str, ...]
) -> list[str]:
    configure_style()
    methods = list(report["uncertainty"])
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.65), constrained_layout=True)
    figure.suptitle(
        f"{report['evaluation_context']}: reported quantiles, not calibration evidence"
    )
    for panel, (axis, (metric, metadata)) in enumerate(
        zip(axes, UNCERTAINTY_METRICS.items(), strict=True)
    ):
        for index, method in enumerate(methods):
            summary = report["uncertainty"][method][metric]
            draw_summary(axis, index, summary, PALETTE[index % len(PALETTE)])
        axis.set_xticks(range(len(methods)), methods, rotation=25, ha="right")
        axis.set_ylabel(metadata["label"])
        axis.set_xlabel("Method")
        axis.set_title(
            f"{chr(97 + panel)}  {metadata['label']} ({metadata['direction']})",
            loc="left",
            weight="bold",
        )
        if metric == "coverage_90":
            axis.axhline(0.90, color="#666666", linestyle="--", linewidth=0.8)
            axis.set_ylim(0.0, 1.05)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.7)
        axis.legend(loc="best", fontsize=6)
    return save_figure(figure, output_dir / "uncertainty_calibration", formats)


def _write_source_data(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
