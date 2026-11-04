"""一条命令完成CPU学习、封存、校准、物理编辑和论文数据导出。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from tactile3d import __version__
from tactile3d.mesh.state_surface import part_surface
from tactile3d.physics.part_editing import EditSpec, apply_edit, verify_full_recompute
from tactile3d.physics.part_io import file_sha256, save_object, write_json
from tactile3d.physics.part_state import Material
from tactile3d.research.construction import (
    TEST_EDIT,
    TRAIN_EDITS,
    Array,
    candidate_volumes,
    decode_state,
    export_state,
    mass_forward,
)
from tactile3d.research.data import (
    FEATURE_NAMES,
    analytic_volumes,
    load_inputs,
    save_dataset,
)
from tactile3d.research.learning import (
    DEFAULT_EPOCHS,
    DEFAULT_HIDDEN,
    direct_predict,
    train_models,
)
from tactile3d.research.network import Network

DEFAULT_COUNT = 384
DEFAULT_SEEDS = (17, 29, 43)
DATA_SEED = 20261104
INTERVAL_COVERAGE = 0.90
METHODS = ("mass_only", "shared_response", "direct_response")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("不能导出空表")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def predict_models(
    inputs: dict[str, Array],
    models: dict[str, Network],
    mean: Array,
    scale: Array,
) -> dict[str, Array]:
    """推理接口不接收真值，物体尺度与材料先验属于显式条件输入。"""
    features = (inputs["features"] - mean) / scale
    volumes = candidate_volumes(inputs["pitch_m"])
    priors = inputs["priors"]
    reference = priors[:, 0] * volumes[:, 0, 0]
    result = {}
    for method, network in models.items():
        if method == "direct_response":
            result[method] = direct_predict(network, features, reference)
            result[method + "_unseen"] = direct_predict(
                network, features, reference, (TEST_EDIT,)
            )[:, 1]
        else:
            logits, _ = network.forward(features)
            result[method] = mass_forward(logits, priors, volumes)[0]
            result[method + "_hard"] = mass_forward(logits, priors, volumes, hard=True)[
                0
            ]
            result[method + "_unseen"] = mass_forward(
                logits, priors, volumes, (TEST_EDIT,)
            )[0][:, 1]
            result[method + "_logits"] = logits
    return result


def mass_metrics(predicted: Array, truth: Array) -> dict[str, float]:
    if predicted.shape != truth.shape or np.any(predicted <= 0) or np.any(truth <= 0):
        raise ValueError("质量指标需要相同形状的正数预测与真值")
    if not np.isfinite(predicted).all() or not np.isfinite(truth).all():
        raise ValueError("质量评测不能删除或跳过非有限预测")
    return {
        "ADE_kg": float(np.abs(predicted - truth).mean()),
        "ALDE_ln": float(np.abs(np.log(predicted / truth)).mean()),
        "APE_fraction": float((np.abs(predicted - truth) / truth).mean()),
        "MnRE": float(np.minimum(predicted / truth, truth / predicted).mean()),
    }


def calibrate_interval(
    predicted: Array, truth: Array, groups: Array | None = None
) -> float:
    """独立校准集的有限样本分位数；小样本不足时明确失败。

    Returns:
        对数质量的对称半宽，不是内部材料结构的置信区间。
    """
    mass_metrics(predicted, truth)
    score = np.abs(np.log(predicted / truth))
    if groups is not None:
        if groups.shape != score.shape:
            raise ValueError("校准族标识必须与质量一一对应")
        # 同族变体不独立；族内最大误差用于族级有限样本校准。
        score = np.array([score[groups == g].max() for g in np.unique(groups)])
    rank = math.ceil((len(score) + 1) * INTERVAL_COVERAGE)
    if not 1 <= rank <= len(score):
        raise ValueError("校准对象不足，不能构造要求覆盖率的有限区间")
    return float(np.sort(score)[rank - 1])


def evaluate(
    inputs: dict[str, Array],
    labels: dict[str, Array],
    predictions: dict[str, Array],
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    test, calibration = inputs["split"] == 2, inputs["split"] == 1
    truth = labels["mass_kg"]
    basis = analytic_volumes(inputs["pitch_m"])
    occupied = basis[np.arange(len(truth)), labels["construction"].astype(int)]
    edited_density = labels["density"].copy()
    edited_density[:, TEST_EDIT[0]] = TEST_EDIT[1]
    unseen = (occupied * labels["fill"] * edited_density).sum(1)
    metrics, rows = [], []
    for method in METHODS:
        predicted = predictions[method]
        quantile = calibrate_interval(
            predicted[calibration, 0],
            truth[calibration, 0],
            inputs["group"][calibration],
        )
        lower, upper = (
            predicted[:, 0] * np.exp(-quantile),
            predicted[:, 0] * np.exp(quantile),
        )
        for condition, actual, target in (
            ("original", predicted[test, 0], truth[test, 0]),
            ("seen_material_edits", predicted[test, 1:], truth[test, 1:]),
            ("unseen_steel_edit", predictions[method + "_unseen"][test], unseen[test]),
        ):
            metrics.append(
                {
                    "seed": seed,
                    "method": method,
                    "condition": condition,
                    **mass_metrics(actual, target),
                }
            )
        for index in np.flatnonzero(test):
            rows.append(
                {
                    "seed": seed,
                    "object_index": int(index),
                    "method": method,
                    "truth_kg": float(truth[index, 0]),
                    "prediction_kg": float(predicted[index, 0]),
                    "lower90_kg": float(lower[index]),
                    "upper90_kg": float(upper[index]),
                    "interval_contains_truth": bool(
                        lower[index] <= truth[index, 0] <= upper[index]
                    ),
                    "unseen_truth_kg": float(unseen[index]),
                    "unseen_prediction_kg": float(
                        predictions[method + "_unseen"][index]
                    ),
                }
            )
        if method != "direct_response":
            hard = predictions[method + "_hard"]
            metrics.append(
                {
                    "seed": seed,
                    "method": method,
                    "condition": "MAP_asset_original",
                    **mass_metrics(hard[test, 0], truth[test, 0]),
                }
            )
    return metrics, rows


def export_example(
    inputs: dict[str, Array],
    predictions: dict[str, Array],
    path: Path,
) -> dict[str, Any]:
    index = int(np.flatnonzero(inputs["split"] == 2)[0])
    logits = predictions["shared_response_logits"][index : index + 1]
    prior = inputs["priors"][index : index + 1]
    state = export_state(
        f"synthetic_{index:04d}", float(inputs["pitch_m"][index]), logits, prior
    )
    save_object(state, path / "original")
    part_surface(state, "body").export(path / "original.ply")
    spec = EditSpec(
        "body",
        "material_change",
        expected_part_version=0,
        material_name="outer",
        new_material=Material("steel", TEST_EDIT[1], "explicit_edit"),
    )
    edited = apply_edit(state, spec)
    save_object(edited.state, path / "steel")
    part_surface(edited.state, "body").export(path / "steel.ply")
    spec.to_json(path / "edit_spec.json")
    for current, column in ((state, 0), (edited.state, 1)):
        expected = mass_forward(
            logits,
            prior,
            candidate_volumes(inputs["pitch_m"][index : index + 1]),
            (TEST_EDIT,),
            hard=True,
        )[0][0, column]
        if not np.isclose(current.total.mass_kg, expected, rtol=1e-10, atol=1e-12):
            raise ValueError("学习状态导出后的积分不一致")
    return {
        "object_index": index,
        "original": state.total.properties(),
        "edited": edited.state.total.properties(),
        "full_recompute_error": verify_full_recompute(edited.state),
        "soft_mass_kg": float(predictions["shared_response"][index, 0]),
        "MAP_mass_kg": state.total.mass_kg,
        "source": "synthetic_learned_candidate_not_real_object",
    }


def state_metrics(
    inputs: dict[str, Array],
    labels: dict[str, Array],
    predictions: dict[str, Array],
    seed: int,
) -> list[dict[str, Any]]:
    test = inputs["split"] == 2
    volumes = analytic_volumes(inputs["pitch_m"])
    truth_class = labels["construction"].astype(int)
    true_volume = volumes[np.arange(len(test)), truth_class]
    active = true_volume[test] > 0
    result = []
    for method in ("mass_only", "shared_response"):
        decoded = decode_state(predictions[method + "_logits"], inputs["priors"])
        predicted_class = decoded["probability"].argmax(1)
        predicted_volume = volumes[np.arange(len(test)), predicted_class]
        result.append(
            {
                "seed": seed,
                "method": method,
                "MAP_construction_accuracy": float(
                    (predicted_class[test] == truth_class[test]).mean()
                ),
                "active_density_MAE_kg_m3": float(
                    np.abs(decoded["density"][test] - labels["density"][test])[
                        active
                    ].mean()
                ),
                "active_fill_MAE": float(
                    np.abs(decoded["fill"][test] - labels["fill"][test])[active].mean()
                ),
                "occupied_volume_MAE_m3": float(
                    np.abs(
                        predicted_volume[test] * decoded["fill"][test]
                        - true_volume[test] * labels["fill"][test]
                    ).mean()
                ),
                "soft_MAP_mass_gap_kg": float(
                    np.abs(
                        predictions[method][test, 0]
                        - predictions[method + "_hard"][test, 0]
                    ).mean()
                ),
            }
        )
    return result


def run_experiment(
    output: Path,
    count: int = DEFAULT_COUNT,
    epochs: int = DEFAULT_EPOCHS,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    hidden: int = DEFAULT_HIDDEN,
    figures: bool = True,
) -> dict[str, Any]:
    """完整CPU实验，任何失败都记录，不覆盖已有结果。

    Args:
        output: 全新的实验目录。
        count: 合成对象数，按族分割。
        epochs: 固定训练轮数，不用测试集挑选。
        seeds: 同一数据分割上的初始化种子。
        hidden: 隐藏维数。
        figures: 是否生成Matplotlib图片。

    Returns:
        数据来源、运行时间、指标路径与验证结果清单。
    """
    if epochs < 1 or hidden < 1 or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("轮数/维数/种子配置无效")
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    manifest: dict[str, Any] = {
        "status": "running",
        "source": "analytic_synthetic_boxes",
        "ABO_experiment": False,
        "VLM_training": False,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "software_version": __version__,
        "dependencies": {
            name: version(name)
            for name in ("numpy", "scipy", "trimesh", "Pillow")
            + (("matplotlib",) if figures else ())
        },
        "count": count,
        "epochs": epochs,
        "hidden": hidden,
        "seeds": list(seeds),
        "data_seed": DATA_SEED,
        "feature_names": list(FEATURE_NAMES),
        "train_edits": TRAIN_EDITS,
        "unseen_edit": TEST_EDIT,
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output / "run.json", manifest)
    package_root = Path(__file__).resolve().parents[1]
    write_json(
        output / "source_sha256.json",
        {
            path.relative_to(package_root).as_posix(): file_sha256(path)
            for path in sorted(package_root.rglob("*.py"))
        },
    )
    try:
        save_dataset(output / "dataset", count, DATA_SEED)
        inputs = load_inputs(output / "dataset/inputs.npz")
        with np.load(output / "dataset/labels.npz", allow_pickle=False) as stored:
            labels = {name: stored[name].copy() for name in stored.files}
        all_metrics, all_rows, all_history, all_states = [], [], [], []
        for seed in seeds:
            folder = output / f"seed_{seed}"
            folder.mkdir()
            models, mean, scale, history = train_models(
                inputs, labels, seed, epochs, hidden
            )
            for method, model in models.items():
                # 直接回归的编辑编码不参与基础观测特征归一化。
                model_mean = (
                    np.r_[mean, np.zeros(3)] if method == "direct_response" else mean
                )
                model_scale = (
                    np.r_[scale, np.ones(3)] if method == "direct_response" else scale
                )
                model.save(folder / f"{method}.npz", model_mean, model_scale)
            manifest["model_parameter_counts"] = {
                method: sum(value.size for value in model.parameters.values())
                for method, model in models.items()
            }
            predictions = predict_models(inputs, models, mean, scale)
            prediction_path = folder / "predictions.npz"
            np.savez_compressed(prediction_path, **predictions)
            seal = {
                "sha256": file_sha256(prediction_path),
                "inputs_sha256": file_sha256(output / "dataset/inputs.npz"),
                "sealed_utc": datetime.now(timezone.utc).isoformat(),
            }
            write_json(folder / "prediction_seal.json", seal)
            metrics, rows = evaluate(inputs, labels, predictions, seed)
            if file_sha256(prediction_path) != seal["sha256"]:
                raise ValueError("评测期间预测被修改")
            all_metrics.extend(metrics)
            all_rows.extend(rows)
            all_states.extend(state_metrics(inputs, labels, predictions, seed))
            all_history.extend({"seed": seed, **row} for row in history)
            if seed == seeds[0]:
                example = export_example(inputs, predictions, folder / "example")
                write_json(folder / "example/physical_report.json", example)
                if figures:
                    from tactile3d.research.plots import plot_physical_fields

                    plot_physical_fields(
                        inputs, labels, predictions, output / "physical_fields"
                    )
        write_csv(output / "metrics.csv", all_metrics)
        write_csv(output / "test_predictions.csv", all_rows)
        write_csv(output / "training.csv", all_history)
        write_csv(output / "state_metrics.csv", all_states)
        interval_rows = []
        for seed in seeds:
            for method in METHODS:
                values = [
                    row
                    for row in all_rows
                    if row["seed"] == seed and row["method"] == method
                ]
                groups = {int(inputs["group"][row["object_index"]]) for row in values}
                group_covered = [
                    all(
                        row["interval_contains_truth"]
                        for row in values
                        if int(inputs["group"][row["object_index"]]) == group
                    )
                    for group in groups
                ]
                interval_rows.append(
                    {
                        "seed": seed,
                        "method": method,
                        "test_object_coverage": float(
                            np.mean([row["interval_contains_truth"] for row in values])
                        ),
                        "test_family_coverage": float(np.mean(group_covered)),
                        "mean_width_kg": float(
                            np.mean(
                                [
                                    row["upper90_kg"] - row["lower90_kg"]
                                    for row in values
                                ]
                            )
                        ),
                        "calibration_families": len(
                            np.unique(inputs["group"][inputs["split"] == 1])
                        ),
                    }
                )
        write_csv(output / "interval_metrics.csv", interval_rows)
        if figures:
            from tactile3d.research.plots import plot_results

            plot_results(output, all_metrics, all_rows, all_history)
        manifest.update(
            status="complete",
            seconds=time.perf_counter() - start,
            split_counts={
                name: int((inputs["split"] == i).sum())
                for i, name in enumerate(("train", "calibration", "test"))
            },
            independent_labels="analytic_volumes_not_predicted_fields",
            learned_encoder=f"11D_proxy_features_to_{hidden}D_MLP_not_VLM",
            interval_protocol="90_percent_family_max_log_residual_conformal",
            raw_results="metrics.csv",
            limitations=[
                "Known box geometry and candidate catalogue",
                "Synthetic feature generator, no pixel encoder",
                "Construction labels supervised for all controls",
                "No real-object physical accuracy established",
            ],
        )
        write_json(output / "run.json", manifest)
        return manifest
    except Exception as error:
        manifest.update(
            status="failed",
            error_type=type(error).__name__,
            error=str(error),
            seconds=time.perf_counter() - start,
        )
        write_json(output / "run.json", manifest)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/reproduction"))
    parser.add_argument("--objects", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    try:
        report = run_experiment(
            args.output,
            args.objects,
            args.epochs,
            tuple(args.seeds),
            args.hidden,
            not args.no_figures,
        )
    except (OSError, ValueError, ImportError) as error:
        parser.exit(1, f"复现失败：{error}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
